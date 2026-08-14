"""Verify batched FIRE == sequential BFGS (energies) and measure the speedup,
on real conformer ensembles. Trust the batched relaxer only if the min energy
matches to <~1 kJ/mol.
"""
import time
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.optimize import BFGS
from fairchem.core import FAIRChemCalculator

from batched_relax import load_uma, batched_fire

EV2KJ = 96.485
TESTS = [("MeUDPGlc", -2, "OC[C@H]1O[C@@H](OP(=O)([O-])OP(=O)([O-])OC)[C@H](O)[C@@H](O)[C@@H]1O"),
         ("MNA+", +1, "C[n+]1cccc(C(N)=O)c1")]
NCONF = 24


def conformers(smiles, seed=1, nconf=NCONF):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3(); p.randomSeed = seed; p.pruneRmsThresh = 0.3
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=nconf, params=p))
    AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=300)
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    return [Atoms(symbols=syms, positions=m.GetConformer(c).GetPositions()) for c in cids]


def main():
    pu = load_uma()
    calc = FAIRChemCalculator(pu, task_name="omol")
    for name, q, smi in TESTS:
        base = conformers(smi, seed=1)
        # sequential BFGS
        seq = [a.copy() for a in base]
        t0 = time.time()
        seqE = []
        for a in seq:
            a.info = {"charge": int(q), "spin": 1}; a.calc = calc
            BFGS(a, logfile=None).run(fmax=0.03, steps=300)
            seqE.append(a.get_potential_energy() * EV2KJ)
        t_seq = time.time() - t0
        seqE = np.array(sorted(seqE))
        # batched FIRE
        bat = [a.copy() for a in base]
        for a in bat:
            a.info = {"charge": int(q), "spin": 1}
        t0 = time.time()
        _, batE_ev = batched_fire(pu, bat, fmax=0.03, steps=300)
        t_bat = time.time() - t0
        batE = np.array(sorted(batE_ev * EV2KJ))
        print(f"\n=== {name} (q{q:+d}, {len(base)} conformers, {len(base[0])} atoms) ===")
        print(f"  min energy   seq {seqE.min():.2f}   batched {batE.min():.2f}   Δ {abs(seqE.min()-batE.min()):.2f} kJ")
        print(f"  mean|Δ| over sorted conformers: {np.abs(seqE-batE).mean():.2f} kJ")
        print(f"  TIME  sequential {t_seq:.1f}s   batched {t_bat:.1f}s   speedup {t_seq/max(t_bat,1e-9):.1f}x")


if __name__ == "__main__":
    main()
