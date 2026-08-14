#!/usr/bin/env python
"""Step 3: add implicit solvation to the matched-scaffold redox model -> full
aqueous ΔG vs experiment.

UMA gives gas-phase electronic energies; it has no solvent.  We add ΔGsolv per
species from xTB-ALPB(water) SINGLE POINTS on the UMA-optimized geometries
(ΔGsolv = E_ALPB - E_gas at fixed geometry).  This is the reliable regime for
these models: MNA+ is a CATION and the rest are NEUTRAL -- no polyanions, so the
ALPB anion-undersolvation wall that sank the old composite does not apply.

  ΔG_aq = ΔE_elec(UMA) + [Σ ΔGsolv(products) - Σ ΔGsolv(reactants)] + G(H+,aq,pH7)

for  2 RSH + MNA+ -> RSSR + MNAH + H+   (both rxn00070 NAD and rxn00086 NADP).

Runs UMA in-process (uma env) and calls the xtb 6.7.1 binary via `conda run -n
xtb`.  Run (uma env):  CUDA_VISIBLE_DEVICES=1 python scripts/step3_redox_solvation.py
"""
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
GEOM = os.path.join(OUT, "geom_redox")
os.makedirs(GEOM, exist_ok=True)
EV2KJ = 96.485
HARTREE2KJ = 2625.4996

MODELS = {
    "MNA+":   (+1, "C[n+]1cccc(C(N)=O)c1"),
    "MNAH":   (0,  "O=C(N)C1=CN(C)C=CC1"),
    "MeSH":   (0,  "CS"),
    "MeSSMe": (0,  "CSSC"),
}
G_H_GAS, DGSOLV_H = -26.3, -1104.5
RT_LN10 = 2.303 * 8.314e-3 * 298.15
def g_proton(pH):
    return G_H_GAS + DGSOLV_H - RT_LN10 * pH


def uma_opt_best(smiles, q, calc, nconf=16):
    """Return (best ase.Atoms optimized, E_gas_kJ) over a conformer ensemble."""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3(); p.randomSeed = 1; p.pruneRmsThresh = 0.3
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=nconf, params=p)) or [
        AllChem.EmbedMolecule(m, randomSeed=1, useRandomCoords=True)]
    try:
        AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=200)
    except Exception:
        pass
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    best = (1e18, None)
    for cid in cids:
        a = Atoms(symbols=syms, positions=m.GetConformer(cid).GetPositions())
        a.info = {"charge": int(q), "spin": 1}; a.calc = calc
        try:
            BFGS(a, logfile=None).run(fmax=0.05, steps=150)
            e = a.get_potential_energy() * EV2KJ
            if e < best[0]:
                best = (e, a.copy())
        except Exception:
            continue
    return best[1], best[0]


def write_xyz(atoms, path):
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n\n")
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")


def xtb_energy(xyz, q, solvent=None):
    """xtb single-point TOTAL ENERGY (Hartree) via the xtb env binary."""
    with tempfile.TemporaryDirectory() as d:
        cmd = ["conda", "run", "-n", "xtb", "xtb", os.path.abspath(xyz),
               "--gfn", "2", "--chrg", str(int(q)), "--sp"]
        if solvent:
            cmd += ["--alpb", solvent]
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=300)
        m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", r.stdout)
        if not m:
            raise RuntimeError(f"xtb parse fail for {xyz} (solv={solvent}):\n{r.stdout[-500:]}")
        return float(m.group(1))


def main():
    print("loading uma-s-1p2 ...", flush=True)
    pu = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")
    calc = FAIRChemCalculator(pu, task_name="omol")

    E_gas, dGsolv, spread_info = {}, {}, {}
    for name, (q, smi) in MODELS.items():
        atoms, e_gas = uma_opt_best(smi, q, calc)
        xyz = os.path.join(GEOM, f"{name.replace('+','plus')}.xyz")
        write_xyz(atoms, xyz)
        e_xtb_gas = xtb_energy(xyz, q, solvent=None)
        e_xtb_sol = xtb_energy(xyz, q, solvent="water")
        ds = (e_xtb_sol - e_xtb_gas) * HARTREE2KJ
        E_gas[name] = e_gas
        dGsolv[name] = ds
        print(f"  {name:7s} q{q:+d}  UMA_gas {e_gas:14.1f} kJ  xTB ΔGsolv {ds:8.1f} kJ", flush=True)

    gH = g_proton(7.0)
    dE_elec = (E_gas["MeSSMe"] + E_gas["MNAH"]) - (2 * E_gas["MeSH"] + E_gas["MNA+"])
    dSolv = (dGsolv["MeSSMe"] + dGsolv["MNAH"]) - (2 * dGsolv["MeSH"] + dGsolv["MNA+"])
    dG_aq = dE_elec + dSolv + gH

    exp = {"rxn00070_NAD": 18.0, "rxn00086_NADP": 11.9}
    err = {k: dG_aq - v for k, v in exp.items()}
    print(f"\n==== full aqueous ΔG: matched-scaffold redox (UMA + xTB-ALPB solv + CHE) ====")
    print(f"  ΔE_elec (UMA gas)          {dE_elec:9.1f} kJ")
    print(f"  ΔΔGsolv (xTB-ALPB water)   {dSolv:9.1f} kJ   (dominated by MNA+ cation desolvation)")
    print(f"  G(H+, aq, pH7)             {gH:9.1f} kJ")
    print(f"  ------------------------------------")
    print(f"  ΔG_aq (predicted)          {dG_aq:9.1f} kJ/mol")
    print(f"  experiment: rxn00070(NAD) {exp['rxn00070_NAD']}, rxn00086(NADP) {exp['rxn00086_NADP']} kJ")
    print(f"  error vs NAD  {err['rxn00070_NAD']:+.1f} kJ   vs NADP {err['rxn00086_NADP']:+.1f} kJ")
    print(f"  (dGP-retrained on these: 101.6/104.9 kJ, off by ~90; prior QM composite MAE ~38)")

    json.dump(dict(E_gas_kJ=E_gas, dGsolv_kJ=dGsolv, dE_elec=dE_elec, dSolv=dSolv,
                   g_proton=gH, dG_aq=dG_aq, exp=exp, err=err),
              open(os.path.join(OUT, "step3_redox_solvation.json"), "w"), indent=2)
    print("wrote artifacts/step3_redox_solvation.json")


if __name__ == "__main__":
    main()
