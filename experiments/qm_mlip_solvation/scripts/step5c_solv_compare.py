#!/usr/bin/env python
"""Step 5c: systematic solvation-model comparison in the FULL batched Boltzmann
pipeline (glycosyl rxn00579), all on the SAME UMA-relaxed geometries so only the
solvation model differs. Tests the commonly-used implicit models:
  ALPB, GBSA, ddCOSMO (xtb --cosmo), CPCM-X (xtbcpx --cpcmx).
Per conformer: 1 gas + 3 solv (alpb/gbsa/cosmo) via xtb + 1 cpcmx via xtbcpx.
Per model: Boltzmann ensemble ΔG. Report mean/std/err over 5 seeds for each.

Run: CUDA_VISIBLE_DEVICES=0 python scripts/step5c_solv_compare.py --seeds 1,2,3,4,5 --keep 10
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ase import Atoms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batched_relax import load_uma, batched_fire, batched_energies
from step4e_targeted import pool_confs, SPECIES, boltz, EXP, THERMAL_FIXED

EV2KJ = 96.485
HARTREE2KJ = 2625.4996
XTB = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")
XTBCPX = os.environ.get("XTBCPX_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtbcpx/bin/xtb")
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "OMP_STACKSIZE": "4G"}
MODELS = ["alpb", "gbsa", "cosmo", "cpcmx"]


def _xtb_E(xyz, q, d, flag):
    cmd = [XTB, xyz, "--gfn", "2", "--chrg", str(int(q)), "--sp"] + flag
    r = subprocess.run(cmd, cwd=d, env=ENV, capture_output=True, text=True, timeout=180)
    m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", r.stdout)
    return float(m.group(1)) * HARTREE2KJ if m else None


def multi_dgsolv(atoms, q):
    """dict of dGsolv (kJ) for all models on this geometry."""
    with tempfile.TemporaryDirectory() as d:
        xyz = os.path.join(d, "m.xyz")
        with open(xyz, "w") as f:
            f.write(f"{len(atoms)}\n\n")
            for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
                f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")
        eg = _xtb_E(xyz, q, d, [])
        out = {}
        for mdl in ("alpb", "gbsa", "cosmo"):
            es = _xtb_E(xyz, q, d, [f"--{mdl}", "water"] if mdl != "gbsa" else ["--gbsa", "water"])
            out[mdl] = (es - eg) if (es is not None and eg is not None) else None
        # cpcmx: separate build, dG_solv direct from fort.6
        with tempfile.TemporaryDirectory() as dc:
            xyz2 = os.path.join(dc, "in.xyz"); open(xyz2, "w").write(open(xyz).read())
            subprocess.run([XTBCPX, "in.xyz", "--gfn", "2", "--chrg", str(int(q)),
                            "--uhf", "0", "--cpcmx", "water"], cwd=dc, env=ENV,
                           capture_output=True, text=True, timeout=600)
            f6 = os.path.join(dc, "fort.6")
            m = re.search(r"solvation free energy \(dG_solv\):\s+(-?\d+\.\d+E[+-]\d+)",
                          open(f6, errors="replace").read()) if os.path.isfile(f6) else None
            out["cpcmx"] = float(m.group(1)) * HARTREE2KJ if m else None
        return out


def species_run(pu, n, q, smi, seed, pool, keep):
    cands = pool_confs(smi, q, seed, pool)
    order = np.argsort(batched_energies(pu, cands))[:keep]
    sel = [cands[i] for i in order]
    _, E_ev, conv = batched_fire(pu, sel, fmax=0.05, steps=300, stop_frac=0.9,
                                 return_converged=True, label=f"{n}s{seed}")
    sel = [a for a, c in zip(sel, conv) if c]; Eg = E_ev[conv] * EV2KJ
    with ThreadPoolExecutor(max_workers=8) as ex:
        solv = list(ex.map(lambda a: multi_dgsolv(a, q), sel))
    # per model: list of G_total over conformers
    G = {m: [] for m in MODELS}
    for e, sv in zip(Eg, solv):
        if not np.isfinite(e):
            continue
        for m in MODELS:
            if sv[m] is not None:
                G[m].append(float(e + sv[m]))
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5"); ap.add_argument("--pool", type=int, default=128)
    ap.add_argument("--keep", type=int, default=10)
    a = ap.parse_args(); seeds = [int(s) for s in a.seeds.split(",")]
    print(f"loading UMA... seeds={seeds} keep={a.keep} models={MODELS}", flush=True)
    pu = load_uma()

    per = {m: [] for m in MODELS}
    for seed in seeds:
        agg = {n: species_run(pu, n, q, smi, seed, a.pool, a.keep)
               for n, (q, smi) in SPECIES.items()}
        for m in MODELS:
            dG = (boltz(agg["MeUDP"][m]) + boltz(agg["Suc"][m])) \
                 - (boltz(agg["MeUDPGlc"][m]) + boltz(agg["Fructose"][m])) + THERMAL_FIXED
            per[m].append(dG)
        print(f"  seed {seed}: " + "  ".join(f"{m} {per[m][-1]:6.1f}" for m in MODELS), flush=True)

    print(f"\n==== solvation-model comparison, rxn00579, {len(seeds)} seeds (exp {EXP}) ====")
    for m in MODELS:
        v = np.array(per[m])
        print(f"  {m:6}: mean {v.mean():6.1f}  std {v.std():5.1f}  range {v.max()-v.min():5.1f}  err {v.mean()-EXP:+.1f}")
    json.dump({m: per[m] for m in MODELS} | {"exp": EXP, "seeds": seeds},
              open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "artifacts", "step5c_solv_compare.json"), "w"), indent=2)
    print("wrote artifacts/step5c_solv_compare.json")


if __name__ == "__main__":
    main()
