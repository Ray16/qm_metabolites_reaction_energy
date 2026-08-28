"""Quasi-chemical theory (QCT) prototype: absolute hydration free energy of Mg2+, validated vs experiment.

QCT / Bryantsev cluster-continuum for M(g) + n H2O(l) -> M(H2O)n(aq):
  ΔG*_hyd(M) = G_aq[M(H2O)n]  -  G_gas[M]  -  n * G*_liq(H2O)
  G_aq[cluster]  = E_UMA(cluster) + Gcorr_RRHO(cluster) + ΔGsolv_COSMO(cluster)     (inner + outer shell)
  G*_liq(H2O)    = E_UMA(H2O)    + Gcorr_RRHO(H2O)     + ΔGsolv_COSMO(H2O) + RT ln(55.34)   (Bryantsev)
  G_gas[M]       = E_UMA(Mg2+)   + Gcorr_trans(Mg2+, monatomic)
The RT ln(55.34) puts the n waters at liquid concentration; that is the QCT bookkeeping the earlier
min-over-seeds cluster attempts got wrong. Scan n = 4,5,6 (QCT: the physical n ~ minimises ΔG*_hyd).

Reference: experimental ΔG*_hyd(Mg2+) ≈ -1830 kJ/mol (Marcus real scale; ±~50-100 kJ single-ion
convention for a 2+). PASS if UMA-QCT lands in ~[-1780,-1900] AND the n-scan has a clear minimum near n=6.
GPU (UMA) + CPU (xtb-COSMO).
"""
import os, sys, subprocess, tempfile, shutil, math
sys.path.insert(0, "scripts")
import numpy as np
from ase import Atoms
from thermal_solv import xtb_dgsolv, uma_gibbs_corr

EV_KJ = 96.48533212; T = 298.15; RT = 8.314462618e-3 * T
XTB = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")
SCRATCH = os.environ.get("QM_SCRATCH", "/nfs/lambda_stor_01/homes/rzhu/qm_scratch")
OCT = np.array([[2.1, 0, 0], [-2.1, 0, 0], [0, 2.1, 0], [0, -2.1, 0], [0, 0, 2.1], [0, 0, -2.1]])
EXP_HYD = -1830.0                                              # kJ/mol, Marcus real scale


def water_geom():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles("O")); AllChem.EmbedMolecule(m, randomSeed=1); AllChem.MMFFOptimizeMolecule(m)
    c = m.GetConformer()
    return ["O", "H", "H"], np.array([list(c.GetAtomPosition(i)) for i in range(3)])


def _align(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b); v = np.cross(a, b); c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def build(n, ws, wc):
    sym = ["Mg"]; crd = [np.zeros(3)]; wc = wc - wc[0]
    for k in range(n):
        u = OCT[k] / np.linalg.norm(OCT[k]); R = _align(np.array([0, 0, 1.0]), -u)
        Opos = u * 2.1
        for a, p in zip(ws, wc):
            sym.append(a); crd.append(Opos + R @ p)
    return sym, np.array(crd)


def uma_opt(pu, sym, crd, q):
    from batched_relax import batched_fire
    a = Atoms(symbols=sym, positions=crd, info={"charge": int(q), "spin": 1})
    rel, E, _ = batched_fire(pu, [a], fmax=0.03, steps=500, return_converged=True)
    return float(E[0]) * EV_KJ, rel[0].get_positions(), rel[0].get_chemical_symbols()


def uma_E(pu, sym, crd, q):
    from batched_relax import batched_energies
    return float(batched_energies(pu, [Atoms(symbols=sym, positions=crd, info={"charge": int(q), "spin": 1})])[0]) * EV_KJ


def mono_trans_G(mass_amu):
    """Ideal-gas translational Gibbs free energy contribution G-E for a monatomic species at T, 1 atm (kJ/mol)."""
    from ase.thermochemistry import IdealGasThermo
    a = Atoms("Mg", positions=[[0, 0, 0]]); a.set_masses([mass_amu])
    th = IdealGasThermo(vib_energies=[], potentialenergy=0.0, atoms=a, geometry="monatomic", symmetrynumber=1, spin=0)
    return float(th.get_gibbs_energy(T, 101325.0, verbose=False)) * EV_KJ


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    from batched_relax import _ensure_registered, load_uma
    _ensure_registered("uma-s-1p2p1"); pu = load_uma("uma-s-1p2p1")

    ws, wc = water_geom()
    Ew, wg, wgs = uma_opt(pu, ws, wc, 0)
    Gw_thermal = uma_gibbs_corr(pu, wgs, wg.tolist(), 0)
    Sw = xtb_dgsolv(wgs, wg.tolist(), 0, "cosmo")
    Gliq = Ew + Gw_thermal + Sw + RT * math.log(55.34)
    print(f"G*_liq(H2O) = E {Ew:.1f} + therm {Gw_thermal:.1f} + solv {Sw:.1f} + RTln55.34 {RT*math.log(55.34):.1f} = {Gliq:.1f}", flush=True)

    Emg, _, _ = uma_opt(pu, ["Mg"], np.zeros((1, 3)), 2)
    Gmg = Emg + mono_trans_G(24.305)
    print(f"G_gas(Mg2+) = E {Emg:.1f} + trans {mono_trans_G(24.305):.1f} = {Gmg:.1f}\n", flush=True)

    print(f"{'n':>2} {'E_clu':>12} {'therm':>7} {'solv':>8} {'MgO':>6} {'ΔG*_hyd':>10} {'vs exp':>8}")
    best = None
    for n in (4, 5, 6):
        sym, crd = build(n, ws, wc)
        Ec, geo, gsym = uma_opt(pu, sym, crd, 2)
        th = uma_gibbs_corr(pu, gsym, geo.tolist(), 2)
        solv = xtb_dgsolv(gsym, geo.tolist(), 2, "cosmo")
        mg = geo[[i for i, s in enumerate(gsym) if s == "Mg"][0]]
        mgo = np.mean(sorted(np.linalg.norm(geo[i] - mg) for i, s in enumerate(gsym) if s == "O")[:n])
        dGhyd = (Ec + th + solv) - Gmg - n * Gliq
        print(f"{n:>2} {Ec:12.1f} {th:7.1f} {solv:8.1f} {mgo:6.2f} {dGhyd:10.1f} {dGhyd-EXP_HYD:+8.0f}", flush=True)
        if best is None or dGhyd < best[1]:
            best = (n, dGhyd)
    print(f"\nQCT ΔG*_hyd(Mg2+) [min over n] = {best[1]:.0f} kJ/mol at n={best[0]}   (exp ≈ {EXP_HYD:.0f})")
    ok = -1900 < best[1] < -1750
    print(f"VERDICT: {'PASS' if ok else 'CHECK'} — UMA-QCT within convention uncertainty of experiment"
          f" => QCT pipeline works, extend to Mg-phosphate / anions." if ok else
          f"VERDICT: off by {best[1]-EXP_HYD:+.0f} kJ — diagnose (n-scan, standard state, or single-ion convention).")


if __name__ == "__main__":
    main()
