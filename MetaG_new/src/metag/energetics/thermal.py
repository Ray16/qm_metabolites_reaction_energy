#!/usr/bin/env python
"""Fast thermal + solvation correction — the GPU-efficient replacement for the
`xtb --ohess --cosmo` bottleneck in step7b/step7c/step8.

Splits the old bundled `--ohess --cosmo` correction into its two physical pieces,
each computed the cheap+consistent way (this is the SAME method already validated
in step3b/step5c/step6, not a new shortcut):

  corr(kJ) = ΔG_thermal(UMA-Hessian, gas RRHO)      # was GFN2 --ohess (CPU, slow)
           + ΔΔG_solv(xtb --sp <model> - xtb --sp)  # was bundled COSMO in --ohess

Why this is faithful, not lossy:
- Thermal uses UMA (OMol25 DFT-quality) frequencies on the UMA geometry instead of
  GFN2 frequencies on a GFN2-re-optimized geometry -> an accuracy UPGRADE, and it
  stops the geometry drifting off the UMA electronic surface.
- Solvation as two single points on the UMA geometry is the standard single-point
  ΔG_solv convention (step5c multi_dgsolv). We only drop the in-solvent geometry
  relaxation (a few kJ), and we gain the ~0.5 s single point vs a full CPU Hessian.
- The UMA Hessian is computed FULL-CLUSTER (solute + explicit waters) so the
  cluster-cycle water reference G_wc(n) cancels the water thermal exactly the way it
  did with --ohess. (Solute-only Hessian is a further approximation; kept separate.)

Efficiency: the finite-difference Hessian's 6N displaced geometries go through ONE
batched UMA forward pass (chunked), so a Hessian is ~1-2 GPU passes, not 6N
sequential ASE calls, and not a CPU xtb --ohess.
"""
import os
import re
import subprocess
import tempfile

import numpy as np
from ase import Atoms
from ase.thermochemistry import IdealGasThermo
from ase.vibrations import VibrationsData

from metag.energetics.uma import _predict

EV2KJ = 96.485
CM2EV = 1.23984e-4
HARTREE2KJ = 2625.499639
T = 298.15
XTB = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "OMP_STACKSIZE": "4G"}


# ------------------------------------------------------------------ thermal (UMA)
_HESS_CHUNK = int(os.environ.get("UMA_HESS_CHUNK", "128"))


def _forces_batched(pu, structs, chunk=None):
    """Per-structure forces (list of (nat,3) arrays) via batched UMA passes.
    chunk defaults to UMA_HESS_CHUNK (result-preserving memory lever for big
    molecules on small GPUs; forces are independent per displaced structure)."""
    if chunk is None:
        chunk = _HESS_CHUNK
    out = [None] * len(structs)
    for s in range(0, len(structs), chunk):
        sub = structs[s:s + chunk]
        _, F, bi = _predict(pu, sub)
        F = F.detach().cpu().numpy(); bi = bi.detach().cpu().numpy()
        for k in range(len(sub)):
            out[s + k] = F[bi == k]
    return out


# |imaginary| frequencies up to this are finite-difference / loose-optimisation noise on soft modes and are
# floored like any other soft mode; above it the structure is not a minimum (reported, caller retries).
IMAG_TOL_CM = 50.0


def internal_vib_energies(atoms, H, geometry, return_modes=False):
    """Vibrational analysis with translations and rotations PROJECTED OUT (Eckart), not dropped by size.

    H = Cartesian Hessian (eV/Å², 3N x 3N). Returns (vib_energies_eV, imag_cm): the real vibrational
    quanta of the 3N - n_ext internal modes, and the |frequencies| (cm^-1) of internal modes with negative
    curvature. The old treatment took |Re(E)| of all 3N modes and dropped the n_ext smallest, so a purely
    imaginary mode (Re = 0) became a 'rotation' or a floored 50 cm^-1 mode and a saddle point received
    ordinary minimum thermochemistry."""
    from ase import units
    m = atoms.get_masses()
    x = atoms.get_positions() - atoms.get_center_of_mass()
    n = len(m)
    sm = np.repeat(np.sqrt(m), 3)
    Hmw = H / np.outer(sm, sm)                                  # eV / (Å² amu)
    ext = []
    for a in range(3):                                          # translations
        v = np.zeros((n, 3)); v[:, a] = 1.0
        ext.append((v * np.sqrt(m)[:, None]).ravel())
    for a in range(3):                                          # rotations about the centre of mass
        e = np.zeros(3); e[a] = 1.0
        ext.append((np.cross(e, x) * np.sqrt(m)[:, None]).ravel())
    n_ext = {"monatomic": 3, "linear": 5, "nonlinear": 6}[geometry]
    U, sv, _ = np.linalg.svd(np.array(ext).T, full_matrices=True)
    B = U[:, n_ext:]                                            # orthonormal basis of the internal space
    lam, vec = np.linalg.eigh(B.T @ Hmw @ B)
    s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)     # sqrt(eV/Å²/amu) -> eV (as ASE)
    e = s * np.sqrt(np.abs(lam))
    if return_modes:                                            # Cartesian displacement of each negative mode
        cart = [((B @ vec[:, k]) / sm).reshape(n, 3) for k in np.nonzero(lam < 0)[0]]
        cart = [c / np.linalg.norm(c) for c in cart]
        return e[lam >= 0], e[lam < 0] / CM2EV, cart
    return e[lam >= 0], e[lam < 0] / CM2EV


def uma_gibbs_corr(pu, symbols, coords, q, delta=0.01, chunk=None,
                   geometry=None, symmetrynumber=None, spin=1, return_info=False, imag_as_soft=False):
    """Gibbs correction Gcorr = G_gas(RRHO,ideal-gas) - E_elec (kJ/mol), UMA Hessian.

    Central-difference Hessian from UMA forces; all 6N displacements batched. Uses the same
    low-frequency floor (50 cm^-1) as step6.

    The molecular geometry (linear vs nonlinear -> drop 5 vs 6 external modes) and the rotational
    symmetry number sigma are DETECTED from the geometry (mol_symmetry.geometry_and_sigma), not
    hard-coded: sigma=1/nonlinear-for-everything biases every reaction whose net species include a
    symmetric molecule (water sigma=2 does not cancel when water is created/destroyed) and mis-treats
    the linear species (CO2/O2/H2/N2) common in ModelSEED. Pass geometry/symmetrynumber explicitly only
    to override the automatic detection (e.g. for testing).
    """
    from metag.symmetry import geometry_and_sigma
    geo_auto, sigma_auto, n_drop = geometry_and_sigma(symbols, coords)
    if geometry is None:
        geometry = geo_auto
    if symmetrynumber is None:
        symmetrynumber = sigma_auto
    base = Atoms(symbols=symbols, positions=np.asarray(coords, float),
                 info={"charge": int(q), "spin": int(spin)})
    nat = len(base); ndof = 3 * nat
    pos0 = base.get_positions()
    # electronic energy at the (already UMA-relaxed) geometry
    E_eV, _, _ = _predict(pu, [base]); E_elec = float(E_eV.detach().cpu().numpy()[0])
    # build 2*ndof displaced structures (+/- for each Cartesian DOF)
    structs = []
    for i in range(nat):
        for c in range(3):
            for sgn in (+1.0, -1.0):
                p = pos0.copy(); p[i, c] += sgn * delta
                structs.append(Atoms(symbols=symbols, positions=p,
                                     info={"charge": int(q), "spin": int(spin)}))
    F = _forces_batched(pu, structs, chunk=chunk)          # eV/Å, list of (nat,3)
    H = np.zeros((ndof, ndof))
    for d in range(ndof):
        Fp = F[2 * d].reshape(-1); Fm = F[2 * d + 1].reshape(-1)
        H[d] = -(Fp - Fm) / (2.0 * delta)                  # eV/Å²
    H = 0.5 * (H + H.T)
    vib, imag_cm, imag_vecs = internal_vib_energies(base, H, geometry, return_modes=True)
    # soft imaginary modes (<= IMAG_TOL_CM) are numerical noise on floppy torsions: floored like soft modes.
    # imag_as_soft: the caller has shown by mode following that the larger ones are artefacts too.
    soft = imag_cm if imag_as_soft else imag_cm[imag_cm <= IMAG_TOL_CM]
    n_imag = 0 if imag_as_soft else int(np.sum(imag_cm > IMAG_TOL_CM))   # genuine negative curvature
    mags_real = np.sort(np.concatenate([vib, soft * CM2EV]))
    mags = np.where(mags_real < 50 * CM2EV, 50 * CM2EV, mags_real)   # low-frequency floor
    th = IdealGasThermo(vib_energies=mags, potentialenergy=E_elec, atoms=base,
                        geometry=geometry, symmetrynumber=symmetrynumber,
                        spin=(int(spin) - 1) / 2.0,          # total electronic S = (mult-1)/2
                        vib_selection="all")                 # external modes were projected out above
    G = th.get_gibbs_energy(temperature=T, pressure=101325.0, verbose=False)
    Gcorr = float((G - E_elec) * EV2KJ)
    if qrrho_enabled():                                     # Grimme quasi-RRHO entropy for low-freq modes
        # replace the floored-harmonic vibrational entropy with the free-rotor-interpolated one (S only;
        # ZPE/enthalpy stay harmonic). Uses the REAL frequency for each mode vs the floored one the
        # harmonic G above used, so it RESTORES the entropy the 50 cm^-1 floor suppresses on floppy modes.
        Gcorr += _qrrho_S_correction(mags_real / CM2EV, mags / CM2EV, T)
    if return_info:
        order = np.argsort(-imag_cm) if imag_cm.size else []
        return Gcorr, {"n_imag": n_imag, "max_imag_cm": round(float(imag_cm.max()), 1) if imag_cm.size else 0.0,
                       "_imag_vecs": [imag_vecs[k] for k in order if imag_cm[k] > IMAG_TOL_CM]}
    return Gcorr


# ------------------------------------------------------------------ quasi-RRHO (Grimme 2012)
def qrrho_enabled():
    """QRRHO env flag (default OFF). Parsed like every other flag: QRRHO=0/off/false/no disables it
    (a bare truthiness test made QRRHO=0 switch it ON). Part of the species-cache key."""
    v = os.environ.get("QRRHO")
    return v is not None and v.strip().lower() not in ("", "0", "off", "false", "no")


_H = 6.62607015e-34; _C = 2.99792458e10; _KB = 1.380649e-23; _R = 8.314462618e-3  # kJ/mol/K
_BAV = 1e-44                                                # limiting moment of inertia (kg m^2)


def _S_HO(w_cm, T):
    x = _H * _C * w_cm / (_KB * T)
    return _R * (x / (np.exp(x) - 1.0) - np.log(1.0 - np.exp(-x)))     # kJ/mol/K


def _S_FR(w_cm, T):
    mu = _H / (8.0 * np.pi**2 * _C * w_cm)                  # moment of inertia of the mode (kg m^2)
    mup = mu * _BAV / (mu + _BAV)
    return _R * (0.5 + np.log(np.sqrt(8.0 * np.pi**3 * mup * _KB * T) / _H))


def _qrrho_S_correction(freqs_real_cm, freqs_floored_cm, T, w0=100.0):
    """G correction (kJ/mol) = -T * Σ_modes [ S_qRRHO(real) - S_HO(floored) ]. Grimme interpolation
    w = 1/(1+(w0/w)^4) between harmonic and free rotor; only low-freq (floppy) modes are shifted."""
    corr = 0.0
    for wr, wf in zip(freqs_real_cm, freqs_floored_cm):
        if wr <= 0:
            continue
        w = 1.0 / (1.0 + (w0 / wr) ** 4)
        S_q = w * _S_HO(wr, T) + (1.0 - w) * _S_FR(wr, T)   # qRRHO entropy at the real frequency
        corr += -T * (S_q - _S_HO(wf, T))                  # minus the floored-harmonic entropy used above
    return corr


# --------------------------------------------------------------- solvation (xtb sp)
def _write_xyz(path, symbols, coords):
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n\n")
        for s, (x, y, z) in zip(symbols, np.asarray(coords, float)):
            f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")


def _run_xtb(cmd, d, timeout):
    """Run xtb; a timeout or non-zero exit returns None (the caller drops that conformer) instead of
    raising through ThreadPoolExecutor.map and aborting the whole species/reaction."""
    try:
        r = subprocess.run(cmd, cwd=d, env=ENV, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", r.stdout)
    return float(m.group(1)) * HARTREE2KJ if m else None


def _xtb_sp_E(xyz, q, d, flag):
    return _run_xtb([XTB, xyz, "--gfn", "2", "--chrg", str(int(q)), "--sp"] + flag, d, 180)


def xtb_dgsolv(symbols, coords, q, model="cosmo"):
    """ΔG_solv (kJ) = E_xtb(--sp <model> water) - E_xtb(--sp gas), no Hessian.
    VERTICAL (single point) — fine for bare species; for water-decorated clusters the
    solute/water reorganization in solvent is under-captured (use xtb_dgsolv_relaxed)."""
    with tempfile.TemporaryDirectory() as d:
        xyz = os.path.join(d, "m.xyz")
        _write_xyz(xyz, symbols, coords)
        eg = _xtb_sp_E(xyz, q, d, [])
        flag = ["--gbsa", "water"] if model == "gbsa" else [f"--{model}", "water"]
        es = _xtb_sp_E(xyz, q, d, flag)
        return (es - eg) if (es is not None and eg is not None) else None


XTBCPX = os.environ.get("XTBCPX_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtbcpx/bin/xtb")


def dgsolv(symbols, coords, q, model="cosmo", mult=1):
    """Implicit ΔG_solv (kJ/mol, 1 M gas -> 1 M solution) for one geometry, any supported model:
      cosmo / alpb / gbsa : E_xtb(--sp <model> water) - E_xtb(--sp gas)   (GFN2, single points)
      cpcmx               : xtb CPCM-X (xtbcpx build), dG_solv read from fort.6
    FreeSolv (642 neutral molecules, 2026-09-26): MAE cosmo 9.1 / alpb 6.7 / cpcmx 5.5 kJ; on the polar,
    metabolite-like subset cosmo R=0.38 (slope 0.28, bias +8.6) vs alpb R=0.73, cpcmx R=0.75.
    `mult` sets --uhf (open-shell species, e.g. O2 triplet). Returns None on any xtb failure/timeout."""
    uhf = ["--uhf", str(int(mult) - 1)]
    with tempfile.TemporaryDirectory() as d:
        xyz = os.path.join(d, "m.xyz")
        _write_xyz(xyz, symbols, coords)
        if model == "cpcmx":
            try:
                subprocess.run([XTBCPX, "m.xyz", "--gfn", "2", "--chrg", str(int(q)), *uhf, "--cpcmx", "water"],
                               cwd=d, env=ENV, capture_output=True, text=True, timeout=600)
            except subprocess.TimeoutExpired:
                return None
            f6 = os.path.join(d, "fort.6")
            if not os.path.isfile(f6):
                return None
            m = re.search(r"solvation free energy \(dG_solv\):\s+(-?\d+\.\d+E[+-]\d+)",
                          open(f6, errors="replace").read())
            return float(m.group(1)) * HARTREE2KJ if m else None
        base = [XTB, xyz, "--gfn", "2", "--chrg", str(int(q)), *uhf, "--sp"]
        eg = _run_xtb(base, d, 180)
        flag = ["--gbsa", "water"] if model == "gbsa" else [f"--{model}", "water"]
        es = _run_xtb(base + flag, d, 180)
        return (es - eg) if (es is not None and eg is not None) else None


def xtb_dgsolv_relaxed(symbols, coords, q, model="cosmo"):
    """RELAXED ΔG_solv (kJ) = E_xtb(--opt <model> water) - E_xtb(--sp gas). Optimizes
    the geometry IN the continuum (captures solute + explicit-water reorganization that
    the vertical single point misses) but NO Hessian — so it recovers what step7b's
    --ohess did for water-decorated clusters, at ~seconds not minutes. Use for explicit
    clusters; bare species don't need it."""
    with tempfile.TemporaryDirectory() as d:
        xyz = os.path.join(d, "m.xyz")
        _write_xyz(xyz, symbols, coords)
        eg = _xtb_sp_E(xyz, q, d, [])
        flag = ["--gbsa", "water"] if model == "gbsa" else [f"--{model}", "water"]
        es = _run_xtb([XTB, "m.xyz", "--gfn", "2", "--chrg", str(int(q)), "--opt"] + flag, d, 600)
        return (es - eg) if (es is not None and eg is not None) else None


# ------------------------------------------------------------------ combined corr
def corr_fast(pu, symbols, coords, q, solv_model="cosmo"):
    """Replacement for step7b.xtb_corr: UMA thermal + xtb single-point solvation (kJ).
    Returns None if solvation single points fail."""
    solv = xtb_dgsolv(symbols, coords, q, model=solv_model)
    if solv is None:
        return None
    therm = uma_gibbs_corr(pu, symbols, coords, q)
    return therm + solv


# ---------------------------------------------------- slow reference (validation)
def xtb_corr_ohess(symbols, coords, q):
    """OLD bundled correction: xtb --ohess --cosmo (thermal+solv+reopt). SLOW.
    Kept only to validate that corr_fast reproduces it."""
    with tempfile.TemporaryDirectory() as d:
        xyz = os.path.join(d, "m.xyz")
        _write_xyz(xyz, symbols, coords)
        sp = subprocess.run([XTB, "m.xyz", "--gfn", "2", "--chrg", str(int(q)), "--sp"],
                            cwd=d, env=ENV, capture_output=True, text=True, timeout=180)
        e_gas = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", sp.stdout)
        oh = subprocess.run([XTB, "m.xyz", "--gfn", "2", "--chrg", str(int(q)),
                             "--ohess", "--cosmo", "water"], cwd=d, env=ENV,
                            capture_output=True, text=True, timeout=900)
        g_aq = re.search(r"TOTAL FREE ENERGY\s+(-?\d+\.\d+)", oh.stdout)
        if not e_gas or not g_aq:
            return None
        return (float(g_aq.group(1)) - float(e_gas.group(1))) * HARTREE2KJ
