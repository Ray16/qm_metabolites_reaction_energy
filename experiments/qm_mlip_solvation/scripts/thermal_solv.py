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

from batched_relax import _predict

EV2KJ = 96.485
CM2EV = 1.23984e-4
HARTREE2KJ = 2625.499639
T = 298.15
XTB = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "OMP_STACKSIZE": "4G"}


# ------------------------------------------------------------------ thermal (UMA)
def _forces_batched(pu, structs, chunk=128):
    """Per-structure forces (list of (nat,3) arrays) via batched UMA passes."""
    out = [None] * len(structs)
    for s in range(0, len(structs), chunk):
        sub = structs[s:s + chunk]
        _, F, bi = _predict(pu, sub)
        F = F.detach().cpu().numpy(); bi = bi.detach().cpu().numpy()
        for k in range(len(sub)):
            out[s + k] = F[bi == k]
    return out


def uma_gibbs_corr(pu, symbols, coords, q, delta=0.01, chunk=128,
                   geometry="nonlinear", symmetrynumber=1):
    """Gibbs correction Gcorr = G_gas(RRHO,ideal-gas) - E_elec (kJ/mol), UMA Hessian.

    Central-difference Hessian from UMA forces; all 6N displacements batched. Uses
    the same low-frequency floor (50 cm^-1) and drop-6-modes convention as step6.
    """
    base = Atoms(symbols=symbols, positions=np.asarray(coords, float),
                 info={"charge": int(q), "spin": 1})
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
                                     info={"charge": int(q), "spin": 1}))
    F = _forces_batched(pu, structs, chunk=chunk)          # eV/Å, list of (nat,3)
    H = np.zeros((ndof, ndof))
    for d in range(ndof):
        Fp = F[2 * d].reshape(-1); Fm = F[2 * d + 1].reshape(-1)
        H[d] = -(Fp - Fm) / (2.0 * delta)                  # eV/Å²
    H = 0.5 * (H + H.T)
    vd = VibrationsData.from_2d(base, H)
    en = vd.get_energies()                                  # eV, complex for imaginary
    mags = np.sort(np.abs(en.real))[6:]                    # drop 6 trans/rot
    mags = np.where(mags < 50 * CM2EV, 50 * CM2EV, mags)   # low-frequency floor
    th = IdealGasThermo(vib_energies=mags, potentialenergy=E_elec, atoms=base,
                        geometry=geometry, symmetrynumber=symmetrynumber, spin=0)
    G = th.get_gibbs_energy(temperature=T, pressure=101325.0, verbose=False)
    return float((G - E_elec) * EV2KJ)


# --------------------------------------------------------------- solvation (xtb sp)
def _write_xyz(path, symbols, coords):
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n\n")
        for s, (x, y, z) in zip(symbols, np.asarray(coords, float)):
            f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")


def _xtb_sp_E(xyz, q, d, flag):
    cmd = [XTB, xyz, "--gfn", "2", "--chrg", str(int(q)), "--sp"] + flag
    r = subprocess.run(cmd, cwd=d, env=ENV, capture_output=True, text=True, timeout=180)
    m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", r.stdout)
    return float(m.group(1)) * HARTREE2KJ if m else None


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
        r = subprocess.run([XTB, "m.xyz", "--gfn", "2", "--chrg", str(int(q)), "--opt"] + flag,
                           cwd=d, env=ENV, capture_output=True, text=True, timeout=600)
        m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", r.stdout)
        es = float(m.group(1)) * HARTREE2KJ if m else None
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
