#!/usr/bin/env python
"""Step 5: diagnose the +28 kJ glycosyl bias (rxn00579, energy-targeted ΔG +24 vs
exp −4.2). Decomposition (Step 4e) showed ΔE_elec +113 nearly cancels ΔΔGsolv −88;
the reaction solvation is dominated by two q−2 diphosphate anions (MeUDPGlc −780,
MeUDP −841) whose xTB-ALPB under-solvation need not cancel. Two cheap tests:

(A) SOLVATION-MODEL sensitivity: recompute ΔΔGsolv with ALPB / GBSA / COSMO on the
    same UMA geometries. Large spread ⇒ implicit anion solvation is the culprit
    (and we need cluster-continuum). All agree ⇒ bias is electronic/truncation
    (or a shared anion under-solvation all implicit models miss).
(B) TRUNCATION test: swap the uridine cap methyl→ethyl. ΔE_elec should be cap-
    invariant if the cap is a clean spectator; a shift means truncation matters.

Batched UMA relax (keep lowest 10 of pool 96), min-G geometry per species, direct
xtb binary (OMP=1). Run (uma env):
  CUDA_VISIBLE_DEVICES=0 python scripts/step5_bias_diag.py
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batched_relax import load_uma, batched_fire, batched_energies

EV2KJ = 96.485
HARTREE2KJ = 2625.4996
XTB_BIN = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")
XTB_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

# methyl-capped (baseline) and ethyl-capped (truncation test) UDP models
SPECIES = {
    "MeUDPGlc": (-2, "OC[C@H]1O[C@@H](OP(=O)([O-])OP(=O)([O-])OC)[C@H](O)[C@@H](O)[C@@H]1O"),
    "Fructose": (0,  "OC[C@H]1OC(O)(CO)[C@@H](O)[C@@H]1O"),
    "MeUDP":    (-2, "COP(=O)([O-])OP(=O)([O-])O"),
    "Suc":      (0,  "OC[C@H]1O[C@@H](O[C@]2(CO)O[C@H](CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"),
    "EtUDPGlc": (-2, "OC[C@H]1O[C@@H](OP(=O)([O-])OP(=O)([O-])OCC)[C@H](O)[C@@H](O)[C@@H]1O"),
    "EtUDP":    (-2, "CCOP(=O)([O-])OP(=O)([O-])O"),
}


def best_geom(pu, smi, q, pool=96, keep=10, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    p = AllChem.ETKDGv3(); p.randomSeed = seed; p.pruneRmsThresh = 0.3
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=pool, params=p))
    try:
        AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=200)
    except Exception:
        pass
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    cands = [Atoms(symbols=syms, positions=m.GetConformer(c).GetPositions(),
                   info={"charge": int(q), "spin": 1}) for c in cids]
    order = np.argsort(batched_energies(pu, cands))[:keep]
    sel = [cands[i] for i in order]
    rel, E, conv = batched_fire(pu, sel, fmax=0.05, steps=300, stop_frac=0.9,
                                return_converged=True, label=smi[:8])
    E = E[conv] * EV2KJ; rel = [a for a, c in zip(rel, conv) if c]
    i = int(np.argmin(E))
    return rel[i], float(E[i])   # min-energy geometry + its gas E (kJ)


def xtb_E(atoms, q, model=None, solvent="water"):
    with tempfile.TemporaryDirectory() as d:
        xyz = os.path.join(d, "m.xyz")
        with open(xyz, "w") as f:
            f.write(f"{len(atoms)}\n\n")
            for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
                f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")
        cmd = [XTB_BIN, xyz, "--gfn", "2", "--chrg", str(int(q)), "--sp"]
        if model == "alpb":
            cmd += ["--alpb", solvent]
        elif model == "gbsa":
            cmd += ["--gbsa", solvent]
        elif model == "cosmo":
            cmd += ["--cosmo", solvent]
        r = subprocess.run(cmd, cwd=d, env=XTB_ENV, capture_output=True, text=True, timeout=120)
        m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", r.stdout)
        return float(m.group(1)) * HARTREE2KJ if m else None


def dgsolv(atoms, q, model):
    eg = xtb_E(atoms, q, model=None)
    es = xtb_E(atoms, q, model=model)
    return (es - eg) if (eg is not None and es is not None) else None


def main():
    print("loading UMA...", flush=True)
    pu = load_uma()
    geom, Egas = {}, {}
    for n, (q, smi) in SPECIES.items():
        geom[n], Egas[n] = best_geom(pu, smi, q)
        print(f"  {n:9s} q{q:+d}  gas E {Egas[n]:.1f} kJ", flush=True)

    # (A) solvation-model sensitivity on the baseline (methyl) reaction
    print("\n=== (A) solvation-model sensitivity: rxn ΔΔGsolv & ΔG ===")
    print(f"  per-species ΔGsolv (kJ):  {'model':6}  " + "  ".join(f"{n:9s}" for n in
          ["MeUDPGlc", "Fructose", "MeUDP", "Suc"]))
    for model in ("alpb", "gbsa", "cosmo"):
        ds = {n: dgsolv(geom[n], SPECIES[n][0], model) for n in ["MeUDPGlc", "Fructose", "MeUDP", "Suc"]}
        if any(v is None for v in ds.values()):
            print(f"  {model:6}: FAILED ({[n for n,v in ds.items() if v is None]})"); continue
        dSolv = (ds["MeUDP"] + ds["Suc"]) - (ds["MeUDPGlc"] + ds["Fructose"])
        dE = (Egas["MeUDP"] + Egas["Suc"]) - (Egas["MeUDPGlc"] + Egas["Fructose"])
        dG = dE + dSolv - 0.8
        print(f"  {model:6}: " + "  ".join(f"{ds[n]:9.1f}" for n in
              ["MeUDPGlc", "Fructose", "MeUDP", "Suc"]) +
              f"   ΔΔGsolv {dSolv:7.1f}  ΔG {dG:6.1f}  (exp -4.2, err {dG+4.2:+.1f})")

    # (B) truncation test: methyl vs ethyl cap on ΔE_elec (gas)
    print("\n=== (B) truncation test: ΔE_elec(gas), methyl vs ethyl cap ===")
    dE_me = (Egas["MeUDP"] + Egas["Suc"]) - (Egas["MeUDPGlc"] + Egas["Fructose"])
    dE_et = (Egas["EtUDP"] + Egas["Suc"]) - (Egas["EtUDPGlc"] + Egas["Fructose"])
    print(f"  ΔE_elec methyl {dE_me:.1f}   ethyl {dE_et:.1f}   Δ(cap) {dE_et-dE_me:+.1f} kJ")
    print(f"  -> cap-invariant (<~5 kJ) means truncation is a clean spectator; "
          "large shift means truncation contributes to the bias")


if __name__ == "__main__":
    main()
