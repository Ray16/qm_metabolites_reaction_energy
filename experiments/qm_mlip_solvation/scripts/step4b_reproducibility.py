#!/usr/bin/env python
"""Step 4b: is the glycosyl-transfer ΔG (rxn00579, Step 4 gave -3.0 vs exp -4.2)
REPRODUCIBLE across conformer seeds, or was the 1.2 kJ agreement luck?

The within-species conformer spread was ~85 kJ. The real question is whether the
REACTION ΔG is stable when we re-sample conformers with different random seeds.
Physical reason it might be: the glucosyl & fructosyl groups are conserved across
the transfer (glucose in UDP-glucose AND sucrose; fructose free AND in sucrose),
so their conformer contributions can correlate and cancel even if each species is
individually noisy.

For each seed: regenerate conformers (ETKDG seed), UMA-opt, take min electronic
energy + best geometry, xTB-ALPB ΔGsolv. ΔG_aq(seed) = ΔE_elec + ΔΔGsolv + thermal.
Thermal held fixed at the Step-4 value (-0.8 kJ; geometry-insensitive) — this test
isolates the CONFORMER noise in the electronic + solvation terms.

Verdict: std(ΔG across seeds) small (~<5 kJ) ⇒ noise cancels, -3.0 is real, the
transfer class works. Large ⇒ the agreement was fortuitous ⇒ need matched-transfer
conformers (Step 5).

Run (uma env):  CUDA_VISIBLE_DEVICES=1 python scripts/step4b_reproducibility.py
"""
import argparse
import json
import os
import re
import subprocess
import tempfile

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.optimize import BFGS
from fairchem.core import pretrained_mlip, FAIRChemCalculator

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "artifacts")
EV2KJ = 96.485
HARTREE2KJ = 2625.4996
THERMAL_FIXED = -0.8   # kJ/mol, reaction thermal shift from Step 4 (geometry-insensitive)
EXP = -4.2
NCONF = 24

SPECIES = {
    "MeUDPGlc": (-2, "OC[C@H]1O[C@@H](OP(=O)([O-])OP(=O)([O-])OC)[C@H](O)[C@@H](O)[C@@H]1O"),
    "Fructose": (0,  "OC[C@H]1OC(O)(CO)[C@@H](O)[C@@H]1O"),
    "MeUDP":    (-2, "COP(=O)([O-])OP(=O)([O-])O"),
    "Suc":      (0,  "OC[C@H]1O[C@@H](O[C@]2(CO)O[C@H](CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"),
}


def uma_min(smiles, q, calc, seed, nconf=NCONF):
    """Return (E_min_kJ, best_atoms) over a fresh conformer ensemble for `seed`."""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3(); p.randomSeed = seed; p.pruneRmsThresh = 0.3
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=nconf, params=p))
    if not cids:
        AllChem.EmbedMolecule(m, randomSeed=seed, useRandomCoords=True); cids = [0]
    try:
        AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=300)
    except Exception:
        pass
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    best = (1e18, None)
    for cid in cids:
        a = Atoms(symbols=syms, positions=m.GetConformer(cid).GetPositions())
        a.info = {"charge": int(q), "spin": 1}; a.calc = calc
        try:
            BFGS(a, logfile=None).run(fmax=0.03, steps=250)
            e = a.get_potential_energy() * EV2KJ
            if e < best[0]:
                best = (e, a.copy())
        except Exception:
            continue
    return best


def xtb_dgsolv(atoms, q):
    with tempfile.TemporaryDirectory() as d:
        xyz = os.path.join(d, "m.xyz")
        with open(xyz, "w") as f:
            f.write(f"{len(atoms)}\n\n")
            for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
                f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")

        def e(solv):
            cmd = ["conda", "run", "-n", "xtb", "xtb", xyz, "--gfn", "2",
                   "--chrg", str(int(q)), "--sp"] + (["--alpb", solv] if solv else [])
            r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=400)
            return float(re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", r.stdout).group(1))
        return (e("water") - e(None)) * HARTREE2KJ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5", help="comma-separated ETKDG seeds")
    ap.add_argument("--tag", default="", help="suffix for the output json (per-shard runs)")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    print("loading uma-s-1p2 ...", flush=True)
    pu = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")
    calc = FAIRChemCalculator(pu, task_name="omol")

    per_seed = []
    emin = {n: [] for n in SPECIES}
    for seed in seeds:
        E, S = {}, {}
        for name, (q, smi) in SPECIES.items():
            e, atoms = uma_min(smi, q, calc, seed)
            E[name] = e; S[name] = xtb_dgsolv(atoms, q)
            emin[name].append(e)
        dE = (E["MeUDP"] + E["Suc"]) - (E["MeUDPGlc"] + E["Fructose"])
        dS = (S["MeUDP"] + S["Suc"]) - (S["MeUDPGlc"] + S["Fructose"])
        dG = dE + dS + THERMAL_FIXED
        per_seed.append(dict(seed=seed, dE_elec=dE, dSolv=dS, dG_aq=dG))
        print(f"  seed {seed}:  ΔE_elec {dE:7.1f}  ΔΔGsolv {dS:7.1f}  ->  ΔG_aq {dG:7.1f} kJ  "
              f"(err vs exp {dG-EXP:+.1f})", flush=True)

    dGs = np.array([r["dG_aq"] for r in per_seed])
    print(f"\n==== reproducibility of rxn00579 ΔG across {len(seeds)} conformer seeds ====")
    print(f"  ΔG_aq  mean {dGs.mean():.1f}  std {dGs.std():.1f}  min {dGs.min():.1f}  max {dGs.max():.1f} kJ")
    print(f"  experiment {EXP};  spread of the ANSWER = {dGs.max()-dGs.min():.1f} kJ")
    # which species drives the noise?
    print("  per-species min-E spread across seeds (kJ):")
    for n in SPECIES:
        print(f"     {n:9s} {max(emin[n])-min(emin[n]):6.1f}")
    verdict = ("REPRODUCIBLE — noise cancels (conserved sugar moieties), -3.0 is real"
               if dGs.std() < 5 else
               "NOT reproducible — Step-4 agreement was fortuitous; need matched-transfer conformers")
    print(f"  VERDICT: {verdict}")

    json.dump(dict(seeds=SEEDS, nconf=NCONF, per_seed=per_seed,
                   mean=float(dGs.mean()), std=float(dGs.std()),
                   range=float(dGs.max() - dGs.min()), exp=EXP,
                   emin_spread={n: float(max(v) - min(v)) for n, v in emin.items()},
                   verdict=verdict),
              open(os.path.join(OUT, f"step4b_reproducibility{a.tag}.json"), "w"), indent=2)
    print(f"wrote artifacts/step4b_reproducibility{a.tag}.json")


if __name__ == "__main__":
    main()
