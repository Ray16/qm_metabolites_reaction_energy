"""Verify (not assert) the PHYSICAL CAUSE of the adenylylate anchor offset: is it an ELECTRONIC bond-type
error (UMA gas-phase ΔE disagrees with high-level DFT) or a SOLVATION error (electronic fine, ΔGsolv
carries it)? Mirrors experiments/qm_mlip_solvation/tools/verify_anchor_physics.py for the new class.

Minimal, charge-balanced, NEUTRAL gas-phase model that isolates the P-O-P -> C(=O)-O-P bond swap
(the acyl-adenylate mixed-anhydride formation, stripped of the adenosine/anion machinery):
    pyrophosphate + acetic acid  ->  acetyl phosphate + phosphoric acid

ΔE_UMA vs ΔE_DFT (PBE0/def2-TZVP) on the SAME UMA-relaxed geometry.
 |UMA-DFT| ~ +20  => the anchor offset IS an electronic bond error (UMA model limit; a solvation-style
                     anchor would be mislabelled, and UMA-being-right is ruled out).
 |UMA-DFT| ~ 0    => electronic is fine => the ~+20 solution error is SOLVATION (like the 3 earned anchors),
                     and the anchor is the right KIND of correction.
Run (uma env, GPU):  CUDA_VISIBLE_DEVICES=N python analysis/verify_adenylylate_physics.py
"""
import os, subprocess, tempfile, shutil
import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

EV_KJ = 96.48533212; HARTREE_KJ = 2625.499639
XTB = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")
SCRATCH = os.environ.get("QM_SCRATCH", "/nfs/lambda_stor_01/homes/rzhu/qm_scratch")

REAC = [("OP(=O)(O)OP(=O)(O)O", 0), ("CC(=O)O", 0)]     # pyrophosphate + acetic acid
PROD = [("CC(=O)OP(=O)(O)O", 0), ("OP(=O)(O)O", 0)]     # acetyl phosphate + phosphoric acid


def xtb_opt(smi, q):
    os.makedirs(SCRATCH, exist_ok=True)
    wd = tempfile.mkdtemp(dir=SCRATCH)
    try:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=1); AllChem.MMFFOptimizeMolecule(m)
        Chem.MolToXYZFile(m, f"{wd}/in.xyz")
        subprocess.run([XTB, "in.xyz", "--gfn", "2", "--chrg", str(q), "--opt", "tight"], cwd=wd,
                       env={**os.environ, "OMP_NUM_THREADS": "4"}, capture_output=True, text=True, timeout=600)
        g = open(f"{wd}/xtbopt.xyz").read()
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    sym = [l.split()[0] for l in g.splitlines()[2:] if l.strip()]
    crd = np.array([[float(x) for x in l.split()[1:4]] for l in g.splitlines()[2:] if l.strip()])
    return sym, crd


def pyscf_E(sym, crd, q):
    from pyscf import gto, dft
    mol = gto.M(atom=[(s, tuple(p)) for s, p in zip(sym, crd)], basis="def2-tzvp",
                charge=int(q), spin=0, verbose=0)
    mf = dft.RKS(mol); mf.xc = "PBE0"
    return float(mf.kernel()) * HARTREE_KJ


def main():
    from metag.energetics.uma import load_uma, batched_fire
    pu = load_uma("uma-s-1p2p1")

    def species_E(smi, q):
        s, c = xtb_opt(smi, q)
        a = Atoms(symbols=s, positions=c, info={"charge": int(q), "spin": 1})
        rel, E = batched_fire(pu, [a], fmax=0.03, steps=400)
        geo = rel[0].get_positions(); gsym = rel[0].get_chemical_symbols()
        return float(E[0]) * EV_KJ, pyscf_E(gsym, geo, q)          # UMA and DFT on the SAME geometry

    du = dd = 0.0
    print(f"{'species':26s} {'q':>2s} {'E_UMA(kJ)':>14s} {'E_DFT(kJ)':>14s}")
    for smi, q in PROD:
        eu, ed = species_E(smi, q); du += eu; dd += ed
        print(f"  +{smi:24s} {q:>2d} {eu:14.1f} {ed:14.1f}")
    for smi, q in REAC:
        eu, ed = species_E(smi, q); du -= eu; dd -= ed
        print(f"  -{smi:24s} {q:>2d} {eu:14.1f} {ed:14.1f}")
    gap = du - dd
    print(f"\nΔE_UMA = {du:+.1f}   ΔE_DFT = {dd:+.1f}   UMA-DFT = {gap:+.1f} kJ/mol")
    if abs(gap) > 10:
        print("VERDICT: ELECTRONIC bond error -> UMA mis-ranks the mixed anhydride; anchor is a bond-type"
              " correction (NOT solvation). UMA-being-right is ruled out; reference accuracy still matters.")
    else:
        print("VERDICT: electronic FINE (UMA≈DFT) -> the ~+20 SOLUTION error is SOLVATION, same KIND as the"
              " 3 earned anchors. Anchor is the right form; magnitude still rests on the reference.")


if __name__ == "__main__":
    main()
