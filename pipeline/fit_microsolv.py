#!/usr/bin/env python
"""Choose the explicit-water count per functional group, and report what the
resulting per-species accuracy implies for reaction-level accuracy.

Bare continuum leaves the anions +46.5 kJ/mol off (pKa set, mu_H refit on the
cationic pairs so the number is anion physics, not bookkeeping). Three explicit
waters cut that to 21.2 -- but the optimum is clearly group-dependent: 3 waters
essentially fix thiols and phenol (57->3, 80->-1) and overshoot carboxylates
(44->-24), which is what you would expect since a thiolate is one large soft
anion while a carboxylate's two oxygens are already reasonably described by a
continuum.

So: pick n per group on the calibration set, then report the residual. The
number that matters is not the fitted bias -- it is the SCATTER, because
reaction error is ~sqrt(4) x per-species scatter and the bias is absorbable.

Leave-one-out over the calibration pairs, so the reported residual is not the
same data the water count was chosen on.

Run:  /homes/rzhu/miniforge3/envs/palm/bin/python fit_microsolv.py
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RT = 8.314462618e-3 * 298.15
RTLN10 = RT * math.log(10)
NS = (0, 1, 2, 3, 4)


def load():
    pairs = json.load(open(os.path.join(HERE, "pka_pairs.json")))
    bare = json.load(open(os.path.join(HERE, "bare_xtb_G.json")))
    ms = {}
    for n in NS:
        if n == 0:
            continue
        p = os.path.join(HERE, f"microsolv_n{n}.json")
        if os.path.isfile(p):
            ms[n] = json.load(open(p))
    return pairs, bare, ms


def dG(pair, n, bare, ms):
    a, b = pair["acid"], pair["base"]
    if n == 0:
        if not (bare.get(a) and bare.get(b)):
            return None
        return bare[b] - bare[a]
    t = ms.get(n)
    if not t or not t.get(a) or not t.get(b):
        return None
    return t[b]["G"] - t[a]["G"]


def main():
    pairs, bare, ms = load()
    have = [0] + sorted(ms)
    print(f"water counts available: {have}\n")

    # mu_H is fitted per water count on the CATIONIC pairs only
    mu = {}
    for n in have:
        cat = [p for p in pairs if p["kind"] == "cationic" and dG(p, n, bare, ms) is not None]
        if not cat:
            continue
        mu[n] = np.mean([p["pKa_exp"] * RTLN10 - dG(p, n, bare, ms) for p in cat])

    def err(p, n):
        d = dG(p, n, bare, ms)
        if d is None or n not in mu:
            return None
        e = (d + mu[n]) - p["pKa_exp"] * RTLN10
        return None if abs(e) > 200 else e          # failed cluster

    groups = sorted({p["group"] for p in pairs if p["kind"] == "anionic"})
    print(f"{'group':11}{'n_pairs':>8}" + "".join(f"{'n='+str(n):>10}" for n in have))
    best_n = {}
    for g in groups:
        sub = [p for p in pairs if p["group"] == g and p["kind"] == "anionic"]
        line = f"{g:11}{len(sub):8d}"
        scores = {}
        for n in have:
            e = [err(p, n) for p in sub]
            e = [x for x in e if x is not None]
            scores[n] = np.mean(np.abs(e)) if e else np.inf
            line += f"{scores[n]:10.1f}" if e else f"{'-':>10}"
        best_n[g] = min(scores, key=scores.get)
        print(line + f"   -> n={best_n[g]}")

    print("\nleave-one-out (water count chosen without the held-out pair):")
    res = []
    for p in [q for q in pairs if q["kind"] == "anionic"]:
        others = [q for q in pairs if q["kind"] == "anionic"
                  and q["group"] == p["group"] and q["key"] != p["key"]]
        if not others:
            continue
        sc = {}
        for n in have:
            e = [err(q, n) for q in others]
            e = [x for x in e if x is not None]
            if e:
                sc[n] = np.mean(np.abs(e))
        if not sc:
            continue
        n = min(sc, key=sc.get)
        e = err(p, n)
        if e is not None:
            res.append((p["key"], p["group"], abs(int(p.get("q_base", -1))), n, e))
    if res:
        e = np.array([r[4] for r in res])
        print(f"  n = {len(res)} anionic pairs")
        print(f"  bias {e.mean():+.1f}   scatter {e.std(ddof=1):.1f}   MAE {np.abs(e).mean():.1f}")
        # residual after removing a per-charge bias (absorbable, systematic)
        adj = []
        for q in (1, 2, 3, 4):
            g = [r[4] for r in res if r[2] == q]
            if len(g) >= 2:
                adj += list(np.array(g) - np.mean(g))
        if adj:
            adj = np.array(adj)
            print(f"  after removing per-charge bias: scatter {adj.std(ddof=1):.1f}"
                  f"   MAE {np.abs(adj).mean():.1f}")
            print(f"\n  => implied reaction error ~ sqrt(4) x {adj.std(ddof=1):.1f}"
                  f" = {2*adj.std(ddof=1):.1f} kJ/mol")
            print(f"     (eQuilibrator 5.4 | predict-zero 11.0 | current QM 38.4)")
        print(f"\n{'pair':16}{'group':11}{'|q|':>4}{'n_wat':>7}{'err':>9}")
        for k, g, q, n, e_ in sorted(res, key=lambda r: (r[1], r[2])):
            print(f"{k:16}{g:11}{q:4d}{n:7d}{e_:9.1f}")


if __name__ == "__main__":
    main()
