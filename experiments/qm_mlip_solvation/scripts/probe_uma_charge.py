#!/usr/bin/env python
"""Step 0 feasibility: does UMA (OMol25) run on the CHARGED / polyanionic species
biochemical thermo needs -- the regime xTB/ALPB mangled?

For a charge ladder (neutral -> -1 -> -2 -> -3) of common-element (C/H/O/N/P/S)
species, we:
  1. build a 3D geometry with RDKit (add Hs, ETKDG, MMFF tidy),
  2. set the TOTAL CHARGE + spin (UMA's OMol task reads atoms.info),
  3. get a single-point UMA energy + max force,
  4. run a short geometry optimization (BFGS) to confirm the forces are usable
     (energy decreases, converges) -- the real test of "can we do QM with this".

Green light = finite energies, sane forces, stable opt across the WHOLE charge
ladder including PO4^3-.  That means UMA covers the polyanion regime and we can
proceed to reaction ΔE (step 1) and cluster-continuum solvation (step 2).

Run (uma env):  CUDA_VISIBLE_DEVICES=1 python scripts/probe_uma_charge.py
"""
import json
import os
import sys
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.optimize import BFGS

from fairchem.core import pretrained_mlip, FAIRChemCalculator

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "artifacts")
os.makedirs(OUT, exist_ok=True)

# charge ladder: (name, SMILES, expected total charge).  Common biochemical
# elements only; spans neutral -> trianion, the xTB-ALPB failure regime.
SPECIES = [
    ("water",              "O",                    0),
    ("methanol",           "CO",                   0),
    ("acetate",            "CC(=O)[O-]",          -1),
    ("ammonium",           "[NH4+]",              +1),
    ("methanethiolate",    "C[S-]",               -1),
    ("dihydrogenphosphate","OP(=O)(O)[O-]",       -1),
    ("hydrogenphosphate",  "OP(=O)([O-])[O-]",    -2),
    ("phosphate",          "[O-]P(=O)([O-])[O-]", -3),   # the hard polyanion
    ("pyrophosphate_3",    "OP(=O)([O-])OP(=O)([O-])[O-]", -3),
]


def build(smiles):
    """RDKit 3D geometry; returns (ase.Atoms, formal_charge, n_atoms) or raises."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError("rdkit parse failed")
    m = Chem.AddHs(m)
    if AllChem.EmbedMolecule(m, randomSeed=1, useRandomCoords=False) != 0:
        AllChem.EmbedMolecule(m, randomSeed=1, useRandomCoords=True)
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=200)
    except Exception:
        pass  # MMFF may lack params for some anions; geometry is only a seed
    conf = m.GetConformer()
    pos = conf.GetPositions()
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    q = Chem.GetFormalCharge(m)
    return Atoms(symbols=syms, positions=pos), q, m.GetNumAtoms()


def main():
    dev = "cuda"
    print(f"loading uma-s-1p2 on {dev} ...", flush=True)
    t0 = time.time()
    pu = pretrained_mlip.get_predict_unit("uma-s-1p2", device=dev)
    calc = FAIRChemCalculator(pu, task_name="omol")
    print(f"  loaded in {time.time()-t0:.1f}s\n", flush=True)

    rows = []
    for name, smi, q_expect in SPECIES:
        rec = {"name": name, "smiles": smi, "charge": q_expect}
        try:
            atoms, q, nat = build(smi)
            if q != q_expect:
                rec["warn"] = f"rdkit charge {q} != expected {q_expect}"
            atoms.info = {"charge": int(q), "spin": 1}  # closed-shell singlets (int required)
            atoms.calc = calc

            e0 = atoms.get_potential_energy()               # eV
            f0 = atoms.get_forces()
            fmax0 = float(np.linalg.norm(f0, axis=1).max())

            # short geometry optimization -- the real usability test
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=0.05, steps=60)
            e1 = atoms.get_potential_energy()
            f1 = atoms.get_forces()
            fmax1 = float(np.linalg.norm(f1, axis=1).max())

            rec.update(n_atoms=nat, E0_eV=round(e0, 4), fmax0=round(fmax0, 4),
                       E_opt_eV=round(e1, 4), fmax_opt=round(fmax1, 4),
                       dE_opt_eV=round(e1 - e0, 4), converged=bool(fmax1 <= 0.05),
                       finite=bool(np.isfinite(e1) and np.isfinite(f1).all()),
                       ok=True)
            print(f"  {name:22s} q{q:+d} nat={nat:3d}  E {e1:12.3f} eV  "
                  f"fmax {fmax0:.3f}->{fmax1:.3f}  {'conv' if rec['converged'] else 'noconv'}",
                  flush=True)
        except Exception as e:
            rec.update(ok=False, error=f"{type(e).__name__}: {e}")
            print(f"  {name:22s} q{q_expect:+d}  FAILED: {type(e).__name__}: {e}", flush=True)
        rows.append(rec)

    nok = sum(r.get("ok") and r.get("finite") for r in rows)
    nconv = sum(r.get("converged", False) for r in rows)
    print(f"\n==== feasibility: {nok}/{len(rows)} species ran with finite E+F; "
          f"{nconv}/{len(rows)} optimized to fmax<=0.05 ====")
    anions = [r for r in rows if r["charge"] < 0]
    aok = sum(r.get("ok") and r.get("finite") for r in anions)
    print(f"  anions specifically: {aok}/{len(anions)} ran (incl. PO4^3-, pyrophosphate)")
    verdict = ("GREEN: UMA covers the polyanion regime -> proceed to reaction ΔE (step 1)"
               if aok == len(anions) else
               "PARTIAL/RED: some anions failed -> inspect before proceeding")
    print(f"  VERDICT: {verdict}")

    json.dump({"species": rows, "n_ok": nok, "n_conv": nconv,
               "anions_ok": aok, "anions_total": len(anions), "verdict": verdict},
              open(os.path.join(OUT, "probe_uma_charge.json"), "w"), indent=2)
    print(f"\nwrote artifacts/probe_uma_charge.json")


if __name__ == "__main__":
    main()
