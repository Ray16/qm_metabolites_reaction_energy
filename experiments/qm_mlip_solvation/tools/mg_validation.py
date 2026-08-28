"""UMA-on-Mg VALIDATION (retry, done well-conditioned). The prior explicit-Mg failures all computed
ILL-CONDITIONED absolute reaction energies (small ΔG from ~millions-of-kJ solvation terms). Here we ask the
prerequisite question in a SMALL, GAS-PHASE, well-conditioned way: does UMA reproduce Mg²⁺–ligand binding
against a trusted reference? If yes, the anion/Mg wall is FEP-reachable; if not, FEP inherits UMA's Mg error
and we stop.

Test: Mg²⁺ + n H2O -> Mg(H2O)n²⁺ (n=1,6) — the canonical Mg-coordination benchmark.
  - UMA: optimize cluster, binding E = E(cluster) - E(Mg²⁺) - n*E(H2O); report Mg–O distance.
  - REFERENCE: pyscf PBE0-D3(BJ)/def2-TZVP single point on the UMA geometry (BSSE-uncorrected; hybrid DFT
    is a sound reference for closed-shell Mg²⁺–O electrostatics/coordination).
  - LITERATURE anchors (experimental/high-level, kJ/mol): 1st-water ΔH ≈ -335; six-water total ≈ -1495;
    octahedral Mg–O ≈ 2.08 Å.
PASS if UMA tracks DFT within ~10-15 kJ/mol per water AND Mg–O within ~0.05 Å. GPU (UMA) + CPU (pyscf).
"""
import os, sys, subprocess, tempfile, shutil
sys.path.insert(0, "scripts")
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

EV_KJ = 96.48533212
XTB = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")
SCRATCH = os.environ.get("QM_SCRATCH", "/nfs/lambda_stor_01/homes/rzhu/qm_scratch")
OCT = np.array([[2.1, 0, 0], [-2.1, 0, 0], [0, 2.1, 0], [0, -2.1, 0], [0, 0, 2.1], [0, 0, -2.1]])


def water_geom():
    m = Chem.AddHs(Chem.MolFromSmiles("O")); AllChem.EmbedMolecule(m, randomSeed=1); AllChem.MMFFOptimizeMolecule(m)
    c = m.GetConformer()
    return ["O", "H", "H"], np.array([list(c.GetAtomPosition(i)) for i in range(3)])


def build_cluster(n):
    """Mg²⁺ at origin, n waters octahedrally coordinated with O pointing at Mg (~2.1 Å)."""
    sym = ["Mg"]; crd = [np.zeros(3)]
    ws, wc = water_geom()
    wc = wc - wc[0]                                       # O at origin
    for k in range(n):
        axis = OCT[k]; d = np.linalg.norm(axis); u = axis / d
        Opos = u * 2.1
        # orient water: O toward Mg (O along -u from its local frame). translate rigid water so O=Opos,
        # H's point away from Mg
        R = _align(np.array([0, 0, 1.0]), -u)            # rotate local +z (bisector) to point away from Mg
        for a, p in zip(ws, wc):
            sym.append(a); crd.append(Opos + R @ p)
    return sym, np.array(crd)


def _align(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b); v = np.cross(a, b); c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def uma_opt_energy(pu, sym, crd, q):
    from ase import Atoms
    from batched_relax import batched_fire, batched_energies
    a = Atoms(symbols=sym, positions=crd, info={"charge": int(q), "spin": 1})
    rel, E, conv = batched_fire(pu, [a], fmax=0.03, steps=400, return_converged=True)
    return float(E[0]) * EV_KJ, rel[0].get_positions(), rel[0].get_chemical_symbols()


def pyscf_energy(sym, crd, q, xc="PBE0"):
    from pyscf import gto, dft
    mol = gto.M(atom=[(s, tuple(p)) for s, p in zip(sym, crd)], basis="def2-tzvp", charge=int(q), spin=0, verbose=0)
    mf = dft.RKS(mol); mf.xc = xc
    try:
        from pyscf import dftd3
        mf = dftd3.dftd3(mf)
    except Exception:
        pass
    return float(mf.kernel()) * 2625.499639                # Hartree -> kJ/mol


def mg_o_dist(sym, crd):
    mg = crd[[i for i, s in enumerate(sym) if s == "Mg"][0]]
    dO = sorted(np.linalg.norm(crd[i] - mg) for i, s in enumerate(sym) if s == "O")
    return dO


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    from batched_relax import _ensure_registered, load_uma
    _ensure_registered("uma-s-1p2p1"); pu = load_uma("uma-s-1p2p1")

    ws, wc = water_geom()
    Ew_uma, wg, wgs = uma_opt_energy(pu, ws, wc, 0)
    Emg_uma, _, _ = uma_opt_energy(pu, ["Mg"], np.zeros((1, 3)), 2)
    Ew_ref = pyscf_energy(wgs, wg, 0); Emg_ref = pyscf_energy(["Mg"], np.zeros((1, 3)), 2)
    print(f"refs: E(H2O) UMA {Ew_uma:.1f} / DFT {Ew_ref:.1f} ; E(Mg2+) UMA {Emg_uma:.1f} / DFT {Emg_ref:.1f}\n")
    print(f"{'n':>2} {'UMA_bind':>9} {'DFT_bind':>9} {'ΔUMA-DFT':>9} {'per-water':>9} {'Mg-O(UMA)':>20}")
    LIT = {1: -335.0, 6: -1495.0}
    for n in (1, 6):
        sym, crd = build_cluster(n)
        Ec_uma, geo, gsym = uma_opt_energy(pu, sym, crd, 2)
        b_uma = Ec_uma - Emg_uma - n * Ew_uma
        Ec_ref = pyscf_energy(gsym, geo, 2)
        b_ref = Ec_ref - Emg_ref - n * Ew_ref
        d = mg_o_dist(gsym, geo)
        print(f"{n:>2} {b_uma:9.1f} {b_ref:9.1f} {b_uma-b_ref:+9.1f} {(b_uma-b_ref)/n:+9.1f} "
              f"  {[round(x,2) for x in d[:n]]}")
        print(f"     literature ΔH ≈ {LIT[n]:+.0f} kJ/mol ; UMA err vs lit {b_uma-LIT[n]:+.0f}")
    print("\nVERDICT: UMA reliable on Mg-O if |ΔUMA-DFT|/water < ~15 kJ AND Mg-O ~2.05-2.10 Å -> FEP viable.")


if __name__ == "__main__":
    main()
