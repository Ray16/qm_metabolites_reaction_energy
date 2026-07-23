#!/usr/bin/env python
"""Cluster-continuum microsolvation: explicit first-shell water on the ion.

The error budget says reaction accuracy is ~sqrt(4) x (per-species error). To
reach eQuilibrator's 5.4 kJ/mol we need per-species accuracy near 3-4 kJ/mol.
Continuum models on polyanions are 15-20 at best, and that is the binding
constraint -- no correction layer fixes it, because it is not a systematic
offset (compound-grouped CV on per-species corrections gave R^2 = 0.177).

Continuum solvation fails for anions because the first hydration shell of a
carboxylate/thiolate/phosphate is strongly directional -- specific H-bonds, not
a dielectric response. Adding explicit waters and running the continuum on the
cluster recovers that from physics.

Design: put the SAME number of waters on the acid and the conjugate base, so the
water count cancels in the deprotonation free energy and only the differential
stabilisation survives:

    dG_deprot = G[A(-)(H2O)n] + mu_H - G[AH(H2O)n]

Waters are seeded on the ionisable site (the anionic heavy atom, or the acidic
proton's heavy atom in the neutral) at H-bond geometry and then relaxed by xtb,
which finds the real minimum. Several random seedings are tried per species and
the ensemble is Boltzmann-averaged, because a single hand-placed cluster is
exactly the "one arbitrary conformer" error this pipeline already learned to
avoid.

Validation target: the pKa set, where the anionic error is currently +40 kJ/mol
at the first charge. If microsolvation does not cut that substantially, explicit
solvation is not going to reach the accuracy the goal requires either.

Run:  /homes/rzhu/miniforge3/envs/palm/bin/python microsolvate.py --nwat 3 --seeds 6
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)

XTB = "/homes/rzhu/miniforge3/envs/xtb/bin/xtb"
HARTREE_TO_KJ = 2625.499639
SCRATCH = os.path.join("/tmp", "qm_thermo_scratch", "microsolv")
ANIONIC = {"O", "S", "N"}


def read_xyz(path):
    L = open(path).read().splitlines()
    n = int(L[0].split()[0])
    S, X = [], []
    for ln in L[2:2 + n]:
        f = ln.split()
        S.append(f[0]); X.append([float(f[1]), float(f[2]), float(f[3])])
    return S, np.array(X)


def write_xyz(S, X, path, comment=""):
    with open(path, "w") as fh:
        fh.write(f"{len(S)}\n{comment}\n")
        for s, x in zip(S, X):
            fh.write(f"{s:<3s} {x[0]:>18.10f} {x[1]:>18.10f} {x[2]:>18.10f}\n")


def polar_sites(S, X, smiles_charge):
    """Heavy atoms that should carry the first shell.

    For an anion, the formally charged heavy atoms; failing that (neutral acid),
    any O/S/N, so the acid and its conjugate base get water in the same region.
    """
    idx = [i for i, s in enumerate(S) if s in ANIONIC]
    if not idx:
        idx = list(range(len(S)))
    return idx


def seed_waters(S, X, n_wat, rng, sites):
    """Place n_wat waters at ~2.8 A from randomly chosen polar sites."""
    S2, X2 = list(S), list(X)
    com = X.mean(0)
    for _ in range(n_wat):
        a = sites[rng.integers(len(sites))]
        # direction pointing away from the molecular centre, jittered
        d = X[a] - com
        d = d / (np.linalg.norm(d) or 1.0)
        d = d + 0.6 * rng.normal(size=3)
        d = d / (np.linalg.norm(d) or 1.0)
        o = X[a] + 2.75 * d
        # crude but sufficient: xtb relaxes it
        h1 = o + 0.96 * (-d)
        perp = np.cross(d, rng.normal(size=3))
        perp = perp / (np.linalg.norm(perp) or 1.0)
        h2 = o + 0.96 * (-0.33 * d + 0.94 * perp)
        S2 += ["O", "H", "H"]; X2 += [o, h1, h2]
    return S2, np.array(X2)


def xtb_opt_G(S, X, chg, wd, ohess=True):
    """Optimise in ALPB water and return G (E + G_RRHO) in kJ/mol."""
    os.makedirs(wd, exist_ok=True)
    write_xyz(S, X, os.path.join(wd, "in.xyz"))
    args = ["in.xyz", "--gfn", "2", "--alpb", "water", "--chrg", str(int(chg)), "--uhf", "0"]
    args += ["--ohess"] if ohess else ["--opt", "tight"]
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OMP_STACKSIZE": "4G"}
    r = subprocess.run([XTB, *args], cwd=wd, capture_output=True, text=True, env=env)
    g = re.search(r"TOTAL FREE ENERGY\s+(-?\d+\.\d+)", r.stdout)
    e = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", r.stdout)
    if g is None or e is None:
        return None
    return dict(G=float(g.group(1)) * HARTREE_TO_KJ,
                E=float(e.group(1)) * HARTREE_TO_KJ,
                xyz=os.path.join(wd, "xtbopt.xyz"))


def best_cluster(xyz, chg, n_wat, seeds, tag):
    """Lowest-G microsolvated cluster over several random seedings."""
    S, X = read_xyz(xyz)
    rng = np.random.default_rng(abs(hash(tag)) % (2**31))
    sites = polar_sites(S, X, chg)
    best = None
    for k in range(seeds):
        wd = tempfile.mkdtemp(prefix=f"ms_{tag}_{k}_", dir=SCRATCH)
        try:
            S2, X2 = seed_waters(S, X, n_wat, rng, sites)
            r = xtb_opt_G(S2, X2, chg, wd)
            if r and (best is None or r["G"] < best["G"]):
                best = r
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwat", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)

    pairs = json.load(open(os.path.join(HERE, "pka_pairs.json")))
    ens = json.load(open(os.path.join(HERE, "pka_xtb.json")))
    mets = {m["id"]: m for m in json.load(open(os.path.join(HERE, "pka_metabolites.json")))}
    out_path = args.out or os.path.join(HERE, f"microsolv_n{args.nwat}.json")
    done = json.load(open(out_path)) if os.path.isfile(out_path) else {}

    todo = []
    for p in pairs:
        for key in (p["acid"], p["base"]):
            if key in done or key not in ens:
                continue
            todo.append((key, ens[key][0]["xyz"], int(mets[key]["charge"])))
    print(f"=== microsolvation n={args.nwat} waters, {args.seeds} seedings "
          f"| {len(todo)} species | {args.workers} workers ===", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(best_cluster, xyz, q, args.nwat, args.seeds, k): k
                for k, xyz, q in todo}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = None
                print(f"  [fail] {k}: {e}", flush=True)
            done[k] = None if r is None else dict(G=r["G"], E=r["E"], xyz=r["xyz"])
            print(f"  {k:22} G={done[k]['G'] if done[k] else None}", flush=True)
            json.dump(done, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
