#!/usr/bin/env python
"""Step 7b: charge-balanced explicit-water cluster-continuum (UMA), no mu_water.

Fixes the grand-canonical pinning (mu_water mis-calibration made every added water
favorable -> filled to the cap). Instead, REASON the water count from charge:

    n_water(species) = WPC * |charge|     (WPC = waters per unit charge)

Because the reaction conserves charge (MeP-2 + MePPP-3 -> MePPMe-2 + PPi-3, -5 both
sides), the TOTAL explicit waters are equal on both sides (5*WPC each) -> the bulk
water reference CANCELS in the reaction. No mu_water calibration needed -> robust.
Waters seeded on anionic O/S sites; per species take the min-G cluster over seeds.

  G_aq(cluster) = E_UMA(cluster, UMA-relaxed) + [G_xtb(--ohess --cosmo) - E_xtb(--sp gas)]
  ΔG = Σ G_aq(products) - Σ G_aq(reactants)                      (waters cancel)

Sweep WPC to check convergence. exp ~+2 (nucleotidyl). Implicit-only was -28..-52.
Run (uma env): CUDA_VISIBLE_DEVICES=1 python scripts/step7b_charge_balanced_waters.py --wpc 2,3
"""
import argparse
import json
import math
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
_THERMO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_THERMO, "backup", "explicit_water"))
import grand_canonical_clusters as gc
from batched_relax import load_uma, batched_energies, batched_fire
from step4e_targeted import pool_confs

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "artifacts")
EV2KJ = 96.485
HARTREE2KJ = 2625.499639
XTB = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")
ENV = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "1", "OMP_STACKSIZE": "4G"}

SPECIES = {"MeP": (-2, "COP(=O)([O-])[O-]"),
           "MePPP": (-3, "COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])O"),
           "MePPMe": (-2, "COP(=O)([O-])OP(=O)([O-])OC"),
           "PPi": (-3, "O=P([O-])([O-])OP(=O)([O-])O")}
EXP = 1.85


def bare_geom(pu, q, smi, pool=48, keep=8):
    cands = pool_confs(smi, q, 1, pool)
    sel = [cands[i] for i in np.argsort(batched_energies(pu, cands))[:keep]]
    rel, E, conv = batched_fire(pu, sel, fmax=0.05, steps=300, stop_frac=0.9, return_converged=True)
    rel = [a for a, c in zip(rel, conv) if c]; E = E[conv]
    a = rel[int(np.argmin(E))]
    return a.get_chemical_symbols(), a.get_positions()


def xtb_corr(symbols, coords, q):
    """xtb (RRHO thermal + COSMO solvation) correction to UMA electronics (kJ)."""
    with tempfile.TemporaryDirectory() as d:
        gc.write_xyz(symbols, coords, os.path.join(d, "m.xyz"))
        sp = subprocess.run([XTB, "m.xyz", "--gfn", "2", "--chrg", str(int(q)), "--sp"],
                            cwd=d, env=ENV, capture_output=True, text=True, timeout=180)
        e_gas = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", sp.stdout)
        oh = subprocess.run([XTB, "m.xyz", "--gfn", "2", "--chrg", str(int(q)),
                             "--ohess", "--cosmo", "water"], cwd=d, env=ENV,
                            capture_output=True, text=True, timeout=600)
        g_aq = re.search(r"TOTAL FREE ENERGY\s+(-?\d+\.\d+)", oh.stdout)
        if not e_gas or not g_aq:
            return None
        return (float(g_aq.group(1)) - float(e_gas.group(1))) * HARTREE2KJ


def species_Gaq(pu, name, q, smi, wpc, seeds, log):
    """min-G_aq explicit-water cluster: n = wpc*|charge| waters seeded on anionic sites."""
    bsym, bcoord = bare_geom(pu, q, smi)
    n_water = wpc * abs(q)
    rng = np.random.default_rng(abs(hash((name, wpc))) % (2**32))
    clusters = []
    for s in range(seeds):
        csym, ccoord = gc.seed_waters(bsym, bcoord, n_water, rng)
        clusters.append(Atoms(symbols=csym, positions=ccoord, info={"charge": int(q), "spin": 1}))
    rel, E, conv = batched_fire(pu, clusters, fmax=0.06, steps=350, stop_frac=0.8,
                                return_converged=True, label=f"{name}w{n_water}")
    rel = [a for a, c in zip(rel, conv) if c]; E = E[conv] * EV2KJ
    if not len(E):
        log(f"    {name}: NO converged cluster (n_water={n_water})"); return None
    order = np.argsort(E)
    # take the lowest UMA-electronic cluster, add xtb (thermal+cosmo) correction
    for i in order:
        a = rel[i]
        corr = xtb_corr(a.get_chemical_symbols(), a.get_positions(), q)
        if corr is not None:
            g = float(E[i]) + corr
            log(f"    {name:7s} q{q:+d} n_water={n_water}: E_UMA {E[i]:.1f} + corr {corr:.1f} = G_aq {g:.1f}")
            return g
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wpc", default="2,3", help="comma list of waters-per-unit-charge")
    ap.add_argument("--seeds", type=int, default=6)
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)
    log("loading UMA...")
    pu = load_uma()
    results = {}
    for wpc in [int(x) for x in a.wpc.split(",")]:
        log(f"\n=== WPC = {wpc} waters/charge (waters balance: 5x{wpc}={5*wpc} each side) ===")
        G = {n: species_Gaq(pu, n, q, smi, wpc, a.seeds, log) for n, (q, smi) in SPECIES.items()}
        if any(v is None for v in G.values()):
            log(f"  WPC {wpc}: incomplete"); continue
        dG = (G["MePPMe"] + G["PPi"]) - (G["MeP"] + G["MePPP"])
        results[wpc] = dG
        log(f"  ==> ΔG (WPC {wpc}) = {dG:+.1f} kJ/mol   vs exp {EXP:+.1f}   err {dG-EXP:+.1f}")
    log(f"\n==== nucleotidyl ΔG vs waters-per-charge (implicit-only was -28..-52) ====")
    for wpc, dG in results.items():
        log(f"  WPC {wpc}: ΔG {dG:+.1f}  err {dG-EXP:+.1f}")
    json.dump(dict(results=results, exp=EXP, seeds=a.seeds),
              open(os.path.join(OUT, "step7b_charge_balanced.json"), "w"), indent=2)
    log("wrote artifacts/step7b_charge_balanced.json")


if __name__ == "__main__":
    main()
