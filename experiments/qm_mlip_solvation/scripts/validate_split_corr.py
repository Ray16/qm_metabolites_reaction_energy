#!/usr/bin/env python
"""Validate that the fast split correction (UMA-Hessian thermal + xtb --sp solvation)
reproduces the old bundled `xtb --ohess --cosmo` correction, on real UMA geometries.

Reports, per species/cluster:
  thermal:  GFN2 gas --ohess   vs  UMA Hessian        (frequency-source swap)
  total:    old --ohess --cosmo vs corr_fast          (the pipeline-level number)

Pass criterion: total corr agrees within a few kJ (the split only drops in-solvent
geometry relaxation), and UMA thermal is sane (same ballpark as GFN2 thermal, no
blow-up from bad frequencies).

Run (uma env): CUDA_VISIBLE_DEVICES=7 python scripts/validate_split_corr.py
"""
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
from ase import Atoms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_THERMO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_THERMO, "backup", "explicit_water"))
import grand_canonical_clusters as gc
from batched_relax import load_uma
from step7b_charge_balanced_waters import bare_geom
from thermal_solv import (uma_gibbs_corr, xtb_dgsolv, corr_fast, xtb_corr_ohess,
                          XTB, ENV, HARTREE2KJ, _write_xyz)


def xtb_thermal_gas(symbols, coords, q):
    """GFN2 gas-phase RRHO thermal only: (G_gas --ohess) - (E_gas --sp), kJ."""
    with tempfile.TemporaryDirectory() as d:
        xyz = os.path.join(d, "m.xyz"); _write_xyz(xyz, symbols, coords)
        sp = subprocess.run([XTB, "m.xyz", "--gfn", "2", "--chrg", str(int(q)), "--sp"],
                            cwd=d, env=ENV, capture_output=True, text=True, timeout=180)
        eg = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", sp.stdout)
        oh = subprocess.run([XTB, "m.xyz", "--gfn", "2", "--chrg", str(int(q)),
                             "--ohess"], cwd=d, env=ENV, capture_output=True, text=True, timeout=900)
        gg = re.search(r"TOTAL FREE ENERGY\s+(-?\d+\.\d+)", oh.stdout)
        return (float(gg.group(1)) - float(eg.group(1))) * HARTREE2KJ if (eg and gg) else None


CASES = [
    ("acetate",  -1, "CC(=O)[O-]",        0),
    ("MeP",      -2, "COP(=O)([O-])[O-]", 0),
    ("MeP",      -2, "COP(=O)([O-])[O-]", 4),   # + 4 explicit waters (cluster)
]


def main():
    log = lambda s: print(s, flush=True)
    log("loading UMA...")
    pu = load_uma()
    log(f"\n{'case':22s} {'th_GFN2gas':>11s} {'th_UMA':>9s} {'solv':>8s} "
        f"{'corr_old':>9s} {'corr_new':>9s} {'Δ(new-old)':>10s}  {'t_old':>7s} {'t_new':>7s}")
    for name, q, smi, nw in CASES:
        bsym, bcoord = bare_geom(pu, q, smi)
        if nw > 0:
            rng = np.random.default_rng(42)
            bsym, bcoord = gc.seed_waters(bsym, bcoord, nw, rng)
            # relax the cluster so we score a real minimum, not the seed
            from batched_relax import batched_fire
            a = Atoms(symbols=bsym, positions=bcoord, info={"charge": int(q), "spin": 1})
            rel, _, conv = batched_fire(pu, [a], fmax=0.06, steps=350, return_converged=True)
            if not conv[0]:
                log(f"{name}+{nw}w: cluster did not converge, skipping"); continue
            bsym = rel[0].get_chemical_symbols(); bcoord = rel[0].get_positions()
        tag = f"{name}({q:+d})" + (f"+{nw}w" if nw else "")

        t0 = time.time(); th_uma = uma_gibbs_corr(pu, bsym, bcoord, q); t_uma = time.time() - t0
        solv = xtb_dgsolv(bsym, bcoord, q, model="cosmo")
        corr_new = (th_uma + solv) if solv is not None else float("nan")

        t0 = time.time(); th_gfn = xtb_thermal_gas(bsym, bcoord, q)
        corr_old = xtb_corr_ohess(bsym, bcoord, q); t_old = time.time() - t0

        d = (corr_new - corr_old) if (corr_old is not None) else float("nan")
        log(f"{tag:22s} {th_gfn:11.1f} {th_uma:9.1f} {solv:8.1f} "
            f"{corr_old:9.1f} {corr_new:9.1f} {d:10.1f}  {t_old:6.1f}s {t_uma:6.1f}s")
    log("\nPASS if |Δ(new-old)| within a few kJ and th_UMA ~ th_GFN2gas (freqs sane).")
    log("t_new is JUST the UMA Hessian (GPU); solvation single points add ~1s (CPU).")


if __name__ == "__main__":
    main()
