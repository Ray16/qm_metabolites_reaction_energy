#!/usr/bin/env python
"""Full-benchmark evaluation: absolute QM, and every correction scheme, under
both CV regimes. Runs as soon as the 797-reaction scoring exists.

Why these schemes and not others. Per-SPECIES corrections were ruled out at
compound-grouped CV R^2 = 0.177 -- that killed mu_H/mu_water/cofactor anchoring
and the reference-compound reformulation in one measurement. What survived is
that the QM error is a property of the TRANSFORMATION: within a tight
bond-change class it is nearly constant (leave-one-out R^2 = 0.75), which is
also why isodesmic referencing keyed on bond-change similarity gained 2.6x while
referencing on shared species made things worse (errors added in quadrature).

Two CV regimes, because they answer different questions:
  random          -- new reaction among metabolites we have already computed
  compound-grouped-- new metabolite entirely (the novel-compound use case)

Reported against the two baselines that matter: predict-zero (11.0 on the full
set, the bar any method must clear to be contributing information) and
eQuilibrator/dGPredictor where they have coverage.

Run:  /homes/rzhu/miniforge3/envs/palm/bin/python analyze_full.py [--scored FILE]
"""
import argparse
import csv
import json
import os
from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as fpg
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))


def krr_cv(K, y, lam, folds, groups=None, seed=0):
    n = len(y)
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = np.array_split(rng.permutation(n), folds)
    else:
        gs = sorted(set(groups))
        rng.shuffle(gs)
        idx = [np.array([i for i in range(n) if groups[i] in set(p)])
               for p in np.array_split(np.array(gs, dtype=object), folds)]
    pred = np.full(n, np.nan)
    for te in idx:
        te = np.asarray(te, int)
        if len(te) == 0:
            continue
        tr = np.setdiff1d(np.arange(n), te)
        if len(tr) < 5:
            continue
        a = np.linalg.solve(K[np.ix_(tr, tr)] + lam * np.eye(len(tr)), y[tr])
        pred[te] = K[np.ix_(te, tr)] @ a
    return pred


def report(lab, y, pred, z):
    m = ~np.isnan(pred)
    if m.sum() < 10:
        print(f"  {lab:34} (too few)")
        return
    res = y[m] - pred[m]
    r2 = 1 - (res ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
    print(f"  {lab:34} n={m.sum():4d}  R2={r2:+.3f}  MAE {np.abs(res).mean():6.1f}"
          f"   (uncorr {np.abs(y[m]).mean():5.1f}, predict-0 {z[m].mean():5.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="bench802_scored.json")
    args = ap.parse_args()
    p = os.path.join(HERE, args.scored)
    if not os.path.isfile(p):
        raise SystemExit(f"{args.scored} not found -- score the ensembles first")

    scored = {x["r"]: x for x in json.load(open(p))}
    rx = json.load(open(os.path.join(HERE, "bench802_reactions.json")))
    mets = {m["id"]: m for m in json.load(open(os.path.join(HERE, "bench802_metabolites.json")))}
    meta = {r["modelseed_rxn"]: r
            for r in csv.DictReader(open(os.path.join(HERE, "bench802_meta.csv")))}

    ids = [r for r in scored if r in rx]
    y = np.array([scored[r]["err"] for r in ids])
    z = np.array([abs(scored[r]["e"]) for r in ids])
    print(f"n = {len(ids)} distinct reactions")
    print(f"\nABSOLUTE composite: MAE {np.abs(y).mean():.1f}  bias {y.mean():+.1f} "
          f"scatter {y.std(ddof=1):.1f}   |   predict-zero {z.mean():.1f}")

    gen = fpg.GetMorganGenerator(radius=3, fpSize=2048)
    FP = {}
    for c, m in mets.items():
        mol = Chem.MolFromSmiles(m["smiles"])
        if mol is not None:
            FP[c] = np.array(gen.GetCountFingerprintAsNumPy(mol), float)
    V = []
    for r in ids:
        v = np.zeros(2048)
        for c, k in rx[r].items():
            v += k * FP.get(c, 0)
        n = np.linalg.norm(v)
        V.append(v / n if n else v)
    V = np.array(V)
    K = np.clip(V @ V.T, -1, 1)

    # per-species design, for the comparison that was decisive before
    freq = Counter(c for r in ids for c in rx[r])
    sp = [c for c, n in freq.items() if n >= 3]
    si = {c: i for i, c in enumerate(sp)}
    Xs = np.zeros((len(ids), len(sp)))
    for i, r in enumerate(ids):
        for c, v in rx[r].items():
            if c in si:
                Xs[i, si[c]] += v
    Ks = Xs @ Xs.T
    Ks = Ks / (np.abs(Ks).max() or 1)

    hub = [c for c, n in freq.items() if n >= 3]
    grp = []
    for r in ids:
        cs = [c for c in rx[r] if c in hub]
        grp.append(hub.index(cs[0]) % 10 if cs else -1)

    for lab, groups in (("RANDOM CV (new reaction, known metabolites)", None),
                        ("COMPOUND-GROUPED CV (novel metabolite)", grp)):
        print(f"\n{lab}")
        for lam in (0.03, 0.1, 0.3):
            report(f"reaction-fingerprint KRR (lam={lam})", y,
                   krr_cv(K, y, lam, 10, groups), z)
        report("per-species linear kernel", y, krr_cv(Ks, y, 0.1, 10, groups), z)
        report("class+species (sum kernel)", y,
               krr_cv(K + Ks, y, 0.1, 10, groups), z)

    print("\nreference points on the same reactions:")
    for col, nm in (("other_eQuilibrator_dG_kJ", "eQuilibrator"),
                    ("dGpredictor_modelseed_dG_kJ", "dGPredictor")):
        v = [(float(meta[r][col]), scored[r]["e"]) for r in ids
             if meta[r].get(col) not in ("", "nan", None)]
        if v:
            print(f"  {nm:20} n={len(v):4d}  MAE {np.mean([abs(a - b) for a, b in v]):5.1f}")


if __name__ == "__main__":
    main()
