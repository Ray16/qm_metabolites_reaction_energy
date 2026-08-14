#!/usr/bin/env python
"""Step 4c: Boltzmann ENSEMBLE free energy over the 24 conformers, vs the min —
does the correct statistic make rxn00579 ΔG reproducible across seeds?

Step 4b used the MIN conformer energy (worst statistic for reproducibility: a
single extreme, sensitive to whether the global min was sampled) → std 12.5 kJ.
Two fixes bundled here:
  1. Boltzmann free energy  G_ens = -kT ln Σ_i exp(-G_i/kT)  over the conformers
     (includes conformational entropy; more correct than min).
  2. Average BOTH gas AND solvation per conformer — the Step-4b solvation term
     swung 36 kJ because xTB ΔGsolv was taken on a single (varying) min geometry.
     Here G_i = E_gas(UMA, conf i) + ΔGsolv(xTB, conf i).

Reports reproducibility of the reaction ΔG across seeds for BOTH min and Boltzmann,
so we see whether Boltzmann-24 is enough or we still need denser sampling.
Vibrational thermal held fixed at the Step-4 value (-0.8 kJ; separate DOF from the
conformational entropy captured here — no double count).

Parallelize: one seed per GPU, e.g. --seeds 3 --tag _s3.
Run (uma env):  CUDA_VISIBLE_DEVICES=0 python scripts/step4c_boltzmann.py --seeds 1 --tag _s1
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
KT = 2.4789   # RT at 298.15 K, kJ/mol
THERMAL_FIXED = -0.8
EXP = -4.2
NCONF = 24

SPECIES = {
    "MeUDPGlc": (-2, "OC[C@H]1O[C@@H](OP(=O)([O-])OP(=O)([O-])OC)[C@H](O)[C@@H](O)[C@@H]1O"),
    "Fructose": (0,  "OC[C@H]1OC(O)(CO)[C@@H](O)[C@@H]1O"),
    "MeUDP":    (-2, "COP(=O)([O-])OP(=O)([O-])O"),
    "Suc":      (0,  "OC[C@H]1O[C@@H](O[C@]2(CO)O[C@H](CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"),
}


def boltz_free_energy(Gs):
    """Ensemble free energy -kT ln Σ exp(-Gi/kT), numerically stable."""
    Gs = np.asarray(Gs); ref = Gs.min()
    return float(ref - KT * np.log(np.exp(-(Gs - ref) / KT).sum()))


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
            m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", r.stdout)
            return float(m.group(1)) if m else None
        eg, ew = e(None), e("water")
        return (ew - eg) * HARTREE2KJ if (eg is not None and ew is not None) else None


def species_ensemble(smiles, q, calc, seed, nconf=NCONF):
    """Per-conformer (E_gas, ΔGsolv, G_total) lists; returns dict of aggregates."""
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
    Eg, Gt = [], []
    for cid in cids:
        a = Atoms(symbols=syms, positions=m.GetConformer(cid).GetPositions())
        a.info = {"charge": int(q), "spin": 1}; a.calc = calc
        try:
            BFGS(a, logfile=None).run(fmax=0.03, steps=250)
            eg = a.get_potential_energy() * EV2KJ
            if not np.isfinite(eg):
                continue
            ds = xtb_dgsolv(a, q)
            if ds is None:
                continue
            Eg.append(eg); Gt.append(eg + ds)
        except Exception:
            continue
    Eg = np.array(Eg); Gt = np.array(Gt)
    i_min = int(np.argmin(Gt))
    return dict(n=len(Gt),
               G_min=float(Gt[i_min]),                    # total G of the min-total conformer
               G_boltz=boltz_free_energy(Gt),             # ensemble free energy (total)
               Eg_spread=float(Eg.max() - Eg.min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    pu = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")
    calc = FAIRChemCalculator(pu, task_name="omol")

    rows = []
    for seed in seeds:
        agg = {n: species_ensemble(smi, q, calc, seed) for n, (q, smi) in SPECIES.items()}
        # ΔG = [MeUDP + Suc] - [MeUDPGlc + Fru] + thermal   (G already includes solvation)
        def dG(kind):
            return ((agg["MeUDP"][kind] + agg["Suc"][kind])
                    - (agg["MeUDPGlc"][kind] + agg["Fructose"][kind])) + THERMAL_FIXED
        r = dict(seed=seed, dG_min=dG("G_min"), dG_boltz=dG("G_boltz"),
                 nconf={n: agg[n]["n"] for n in SPECIES})
        rows.append(r)
        print(f"  seed {seed}:  ΔG_min {r['dG_min']:7.1f}   ΔG_boltz {r['dG_boltz']:7.1f} kJ  "
              f"(exp {EXP})", flush=True)

    if len(rows) > 1:
        mn = np.array([r["dG_min"] for r in rows]); bz = np.array([r["dG_boltz"] for r in rows])
        print(f"\n  ΔG_min   mean {mn.mean():6.1f}  std {mn.std():5.1f}  range {mn.max()-mn.min():5.1f}")
        print(f"  ΔG_boltz mean {bz.mean():6.1f}  std {bz.std():5.1f}  range {bz.max()-bz.min():5.1f}")

    json.dump(dict(seeds=seeds, rows=rows, exp=EXP),
              open(os.path.join(OUT, f"step4c_boltzmann{a.tag}.json"), "w"), indent=2)
    print(f"wrote artifacts/step4c_boltzmann{a.tag}.json")


if __name__ == "__main__":
    main()
