"""Verify (not assert) the PHYSICAL CAUSE of each anchor offset: is it an ELECTRONIC bond-type error
(UMA gas ΔE disagrees with high-level DFT) or a SOLVATION error (electronic is fine, ΔGsolv carries it)?
Compute gas-phase ΔE for a MINIMAL model of each class with UMA and with pyscf DFT (PBE0/def2-TZVP) on the
same geometry. |ΔE_UMA - ΔE_DFT| ~ offset => electronic bond error; ~0 => the offset is NOT electronic
(solvation / other), and any 'bond-type reference error' label is wrong.
"""
import os, sys, subprocess, tempfile, shutil
sys.path.insert(0, "scripts")
import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

EV_KJ = 96.48533212; HARTREE_KJ = 2625.499639
XTB = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")
SCRATCH = os.environ.get("QM_SCRATCH", "/nfs/lambda_stor_01/homes/rzhu/qm_scratch")

# minimal charge-balanced models of each anchor class (gas-phase ΔE, neutral where possible)
MODELS = {
    # thioester formation: acetic acid + methanethiol -> S-methyl thioacetate + water (C(=O)O -> C(=O)S)
    "thioester":   ([("CC(=O)O", 0), ("CS", 0)], [("CC(=O)SC", 0), ("O", 0)]),
    # phosphagen P-N: methylphosphate + methylamine -> methanol + N-methyl phosphoramidate (phosphoryl
    # transfer O->N; leaving group is METHANOL, not water -> mass-balanced C2H10NO4P both sides)
    "phosphagen":  ([("COP(=O)(O)O", 0), ("CN", 0)], [("CO", 0), ("CNP(=O)(O)O", 0)]),
}


def xtb_opt(smi, q):
    wd = tempfile.mkdtemp(dir=SCRATCH)
    try:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=1); AllChem.MMFFOptimizeMolecule(m)
        Chem.MolToXYZFile(m, f"{wd}/in.xyz")
        subprocess.run([XTB, "in.xyz", "--gfn", "2", "--chrg", str(q), "--opt", "tight"], cwd=wd,
                       env={**os.environ, "OMP_NUM_THREADS": "1"}, capture_output=True, text=True, timeout=400)
        g = open(f"{wd}/xtbopt.xyz").read()
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    sym = [l.split()[0] for l in g.splitlines()[2:] if l.strip()]
    crd = np.array([[float(x) for x in l.split()[1:4]] for l in g.splitlines()[2:] if l.strip()])
    return sym, crd


def pyscf_E(sym, crd, q):
    from pyscf import gto, dft
    mol = gto.M(atom=[(s, tuple(p)) for s, p in zip(sym, crd)], basis="def2-tzvp", charge=int(q), spin=0, verbose=0)
    mf = dft.RKS(mol); mf.xc = "PBE0"
    return float(mf.kernel()) * HARTREE_KJ


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    from batched_relax import _ensure_registered, load_uma, batched_fire, batched_energies
    _ensure_registered("uma-s-1p2p1"); pu = load_uma("uma-s-1p2p1")

    def species_E(smi, q):
        s, c = xtb_opt(smi, q)
        a = Atoms(symbols=s, positions=c, info={"charge": int(q), "spin": 1})
        rel, E, _ = batched_fire(pu, [a], fmax=0.03, steps=400, return_converged=True)
        geo = rel[0].get_positions(); gsym = rel[0].get_chemical_symbols()
        return float(E[0]) * EV_KJ, pyscf_E(gsym, geo, q)

    print(f"{'class':12s} {'ΔE_UMA':>9s} {'ΔE_DFT':>9s} {'UMA-DFT':>8s}  verdict")
    for name, (reac, prod) in MODELS.items():
        du = dd = 0.0
        for smi, q in prod:
            eu, ed = species_E(smi, q); du += eu; dd += ed
        for smi, q in reac:
            eu, ed = species_E(smi, q); du -= eu; dd -= ed
        gap = du - dd
        verdict = "ELECTRONIC bond error" if abs(gap) > 10 else "electronic FINE -> offset is NOT a bond error (solvation/other)"
        print(f"{name:12s} {du:9.1f} {dd:9.1f} {gap:+8.1f}  {verdict}", flush=True)
    print("\nInterpretation: if UMA-DFT ~ 0 for a class, its anchor offset is NOT an electronic bond-type"
          " error -- it is solvation (or another term), and the 'bond-type reference error' label is wrong.")


if __name__ == "__main__":
    main()
