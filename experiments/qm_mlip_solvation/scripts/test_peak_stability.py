#!/usr/bin/env python
"""step8 validation — TEST 1 (peak reproducibility) + TEST 2 (converges below cap?).

Batched + parallel. The whole occupancy ladder (all n, all cluster-seeds) is relaxed
in ONE batched_fire call (independent rungs -> one GPU batch, not 20 sequential
calls); solvation single points are threaded across CPU cores; run the two backends
on two GPUs concurrently (--backend fast|ohess, one process each). Cache disabled so
conformer-seed trials stay independent (no false reproducibility).

  TEST 1: is the self-selected peak stable across conformer seeds? (bounces -> not a
          valid observable; fall back to coordination rule + step7b)
  TEST 2: does a localized anion (acetate) peak LOW, not pinned at nmax?

Run (uma env), in parallel:
  CUDA_VISIBLE_DEVICES=0 python scripts/test_peak_stability.py --backend fast  --nmax 6 &
  CUDA_VISIBLE_DEVICES=1 python scripts/test_peak_stability.py --backend ohess --nmax 6 &
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ase import Atoms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_THERMO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_THERMO, "backup", "explicit_water"))
import grand_canonical_clusters as gc
from batched_relax import load_uma, batched_energies, batched_fire
from step4e_targeted import pool_confs
from thermal_solv import uma_gibbs_corr, xtb_dgsolv, xtb_corr_ohess

OUT = os.path.join(_THERMO, "experiments", "qm_mlip_solvation", "artifacts")
EV2KJ = 96.485
RT = gc.RT

SMALL = [
    ("acetate",        -1, "CC(=O)[O-]"),          # one carboxylate -> expect ~2-3
    ("methylphosphate", -2, "COP(=O)([O-])[O-]"),  # MeP -> a few, < nmax
]


def bare_geom_seed(pu, q, smi, seed, pool=48, keep=8):
    cands = pool_confs(smi, q, seed, pool)
    sel = [cands[i] for i in np.argsort(batched_energies(pu, cands))[:keep]]
    rel, E, conv = batched_fire(pu, sel, fmax=0.05, steps=300, stop_frac=0.9,
                                return_converged=True)
    rel = [a for a, c in zip(rel, conv) if c]; E = E[conv]
    a = rel[int(np.argmin(E))]
    return a.get_chemical_symbols(), a.get_positions()


def ladder_occ(pu, name, q, smi, nmax, seeds, cseed, backend, Gwc, log):
    """One species ladder at conformer seed `cseed`. Batches ALL rung relaxations in
    one call; threads solvation. Returns (peak, mean_n, occ)."""
    bsym, bcoord = bare_geom_seed(pu, q, smi, cseed)
    # build every cluster (all n, all seeds) up front -> one batched relaxation
    tagged = [(0, Atoms(symbols=bsym, positions=bcoord, info={"charge": int(q), "spin": 1}))]
    for n in range(1, nmax + 1):
        rng = np.random.default_rng((cseed * 100003 + n * 97) % (2**32))
        for _ in range(seeds):
            cs, cc = gc.seed_waters(bsym, bcoord, n, rng)
            tagged.append((n, Atoms(symbols=cs, positions=cc, info={"charge": int(q), "spin": 1})))
    atoms = [a for _, a in tagged]
    rel, E, conv = batched_fire(pu, atoms, fmax=0.06, steps=250, stop_frac=0.8,
                                return_converged=True, label=f"{name}s{cseed}")
    E = E * EV2KJ
    # per n: lowest-E converged cluster
    best = {}
    for (n, _), a, e, c in zip(tagged, rel, E, conv):
        if not c:
            continue
        if n not in best or e < best[n][0]:
            best[n] = (float(e), a.get_chemical_symbols(), a.get_positions())
    if 0 not in best:
        return None, None, {}
    # corr per rung: thermal (GPU, sequential) + solvation (CPU, threaded)
    if backend == "ohess":
        with ThreadPoolExecutor(max_workers=8) as ex:
            corr = dict(ex.map(lambda n: (n, xtb_corr_ohess(best[n][1], best[n][2], q)), best))
    else:
        therm = {n: uma_gibbs_corr(pu, best[n][1], best[n][2], q) for n in best}
        with ThreadPoolExecutor(max_workers=8) as ex:
            solv = dict(ex.map(lambda n: (n, xtb_dgsolv(best[n][1], best[n][2], q)), best))
        corr = {n: (therm[n] + solv[n] if solv[n] is not None else None) for n in best}
    G = {n: best[n][0] + corr[n] for n in best if corr[n] is not None}
    g0 = G.get(0)
    ref = {n: (G[n] - Gwc[n] - g0) for n in G if n in Gwc and Gwc[n] is not None and g0 is not None}
    if not ref:
        return None, None, {}
    lo = min(ref.values())
    Z = sum(math.exp(-(v - lo) / RT) for v in ref.values())
    occ = {n: math.exp(-(ref[n] - lo) / RT) / Z for n in ref}
    peak = max(occ, key=occ.get)
    mean_n = sum(k * v for k, v in occ.items())
    log(f"    {name:16s} seed{cseed}: peak {peak}  <n>={mean_n:.1f}  "
        f"occ {[(k, round(v,2)) for k,v in sorted(occ.items(), key=lambda kv:-kv[1])[:3]]}")
    return peak, mean_n, occ


def water_ladder(pu, nmax, seeds, backend, log):
    """G_wc(n), n=0..nmax, batched in one relaxation; same backend corr as species."""
    tagged = [(0, None)]
    Gwc = {0: 0.0}
    all_clusters = []
    idx = []
    for n in range(1, nmax + 1):
        rng = np.random.default_rng(1000 + n)
        R = 1.6 * (n ** (1.0 / 3.0)) + 1.0
        for _ in range(seeds):
            sym, coord = [], []
            for _w in range(n):
                c = rng.uniform(-R, R, 3)
                sym += ["O", "H", "H"]; coord += [c, c + [0.96, 0, 0], c + [-0.24, 0.93, 0]]
            all_clusters.append(Atoms(symbols=sym, positions=np.array(coord),
                                      info={"charge": 0, "spin": 1}))
            idx.append(n)
    rel, E, conv = batched_fire(pu, all_clusters, fmax=0.06, steps=250, stop_frac=0.8,
                                return_converged=True, label="wc")
    E = E * EV2KJ
    best = {}
    for n, a, e, c in zip(idx, rel, E, conv):
        if not c:
            continue
        if n not in best or e < best[n][0]:
            best[n] = (float(e), a.get_chemical_symbols(), a.get_positions())
    if backend == "ohess":
        with ThreadPoolExecutor(max_workers=8) as ex:
            corr = dict(ex.map(lambda n: (n, xtb_corr_ohess(best[n][1], best[n][2], 0)), best))
    else:
        therm = {n: uma_gibbs_corr(pu, best[n][1], best[n][2], 0) for n in best}
        with ThreadPoolExecutor(max_workers=8) as ex:
            solv = dict(ex.map(lambda n: (n, xtb_dgsolv(best[n][1], best[n][2], 0)), best))
        corr = {n: (therm[n] + solv[n] if solv[n] is not None else None) for n in best}
    for n in best:
        Gwc[n] = best[n][0] + corr[n] if corr[n] is not None else None
        log(f"    G_wc({n}) = {Gwc[n]:.1f} kJ" if Gwc.get(n) is not None else f"    G_wc({n}) FAILED")
    return Gwc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["fast", "ohess"], default="fast")
    ap.add_argument("--nmax", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=2)          # cluster seeds per rung
    ap.add_argument("--cseeds", default="1,2,3")             # conformer seeds (repro)
    a = ap.parse_args()
    cseeds = [int(s) for s in a.cseeds.split(",")]
    log = lambda s: print(s, flush=True)
    log(f"loading UMA... step8 validation backend={a.backend} nmax={a.nmax} "
        f"cluster-seeds={a.seeds} conformer-seeds={cseeds}")
    pu = load_uma()
    log(f"  water-cluster reference ladder ({a.backend}):")
    Gwc = water_ladder(pu, a.nmax, a.seeds, a.backend, log)

    rows = []
    for name, q, smi in SMALL:
        peaks, means = [], []
        for cs in cseeds:
            pk, mn, _ = ladder_occ(pu, name, q, smi, a.nmax, a.seeds, cs, a.backend, Gwc, log)
            peaks.append(pk); means.append(round(mn, 2) if mn is not None else None)
        rows.append(dict(backend=a.backend, nmax=a.nmax, name=name, charge=q,
                         peaks=peaks, means=means))

    log(f"\n==== SUMMARY ({a.backend}) ====")
    log(f"  {'species':16s} {'peaks(per seed)':18s} {'<n>'}")
    for r in rows:
        log(f"  {r['name']:16s} {str(r['peaks']):18s} {r['means']}")
    log("  TEST1 = peaks agree across seeds (small spread). "
        "TEST2 = acetate peaks LOW (<~4), not pinned at nmax.")
    json.dump(rows, open(os.path.join(OUT, f"test_peak_stability_{a.backend}.json"), "w"), indent=2)
    log(f"wrote artifacts/test_peak_stability_{a.backend}.json")


if __name__ == "__main__":
    main()
