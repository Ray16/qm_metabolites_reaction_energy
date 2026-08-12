#!/usr/bin/env python
"""Does QC, used correctly, help EXTRAPOLATION (compound-disjoint CV)?

Reasoning: QC computes a physical dG for any reaction (universal coverage) but
its absolute error is dominated by anion under-solvation, ~ Born (z^2). That
bias is a function of the reaction's CHARGE CHANGE (Sum nu*z, nu*|z|, nu*z^2),
which is computable for any reaction and TRANSFERS across compounds -- unlike
group features. So QC + charge-gated correction should help most exactly where
group models are weakest: novel compounds.

Compare under BOTH CV schemes:
  linear-CC            group-difference ridge (the dGPredictor model class)
  QC+scale             a*QC + b
  QC+charge            ridge on [QC, dz, d|z|, dz2]
  CC + QC+charge       group prior + QC/charge correction (stacked)
"""
import json, os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
d = torch.load(f"{HERE}/data.pt")
y = d["y"].numpy(); Xg = d["Xgroup"].numpy()
rxn_ids = d["rxn_ids"]; rxn_comps = d["rxn_comps"]; n_comp = d["n_comp"]; N = len(rxn_ids)
S = d["S"].numpy()

mets = json.load(open(f"{os.path.dirname(os.path.dirname(HERE))}/pipeline/tecrdb_full_metabolites.json"))
z = np.array([m.get("charge", 0) for m in mets], float)          # compound charges
qc_all = json.load(open(f"{os.path.dirname(os.path.dirname(HERE))}/results/benchmark/tecrdb_full_scored.json"))["scored_kJ"]
QC = np.array([qc_all.get(r, 0.0) for r in rxn_ids], float)

# reaction charge-change features (transfer across compounds)
dz = S @ z                    # net charge change
dabs = S @ np.abs(z)          # change in total |charge|
dz2 = S @ (z ** 2)            # change in sum z^2  (Born solvation proxy)
CH = np.column_stack([dz, dabs, dz2])


def kfold(n, k, s):
    p = np.random.default_rng(s).permutation(n)
    return [(np.setdiff1d(np.arange(n), q), q) for q in (p[i::k] for i in range(k))]


def cpd(rc, nc, k, s):
    g = np.random.default_rng(s).integers(0, k, size=nc); F = []
    for j in range(k):
        h = set(np.where(g == j)[0].tolist())
        te = [i for i, c in enumerate(rc) if h & set(c)]
        tr = [i for i, c in enumerate(rc) if not (h & set(c))]
        F.append((np.array(tr), np.array(te)))
    return F


def ridge_cv(X, folds, lam, resid_of=None):
    """OOF ridge on features X; if resid_of given, fit residual on top of it."""
    e = []
    Xb = np.column_stack([X, np.ones(len(X))])   # intercept
    for tr, te in folds:
        if len(te) == 0:
            continue
        target = y - resid_of if resid_of is not None else y
        A = Xb[tr].T @ Xb[tr] + lam * np.eye(Xb.shape[1])
        c = np.linalg.solve(A, Xb[tr].T @ target[tr])
        pred = Xb[te] @ c + (resid_of[te] if resid_of is not None else 0.0)
        e.append(np.abs(pred - y[te]))
    return np.concatenate(e).mean()


def group_oof(folds, lam):
    pred = np.zeros(N)
    for tr, te in folds:
        if len(te) == 0:
            continue
        A = Xg[tr].T @ Xg[tr] + lam * np.eye(Xg.shape[1])
        c = np.linalg.solve(A, Xg[tr].T @ y[tr])
        pred[te] = Xg[te] @ c
    return pred


for name, folds in [("RANDOM", kfold(N, 5, 0)), ("CPD-DISJOINT", cpd(rxn_comps, n_comp, 5, 0))]:
    print(f"=== {name} (held-out MAE) ===")
    gpred = group_oof(folds, 30.0)
    print(f"  linear-CC (group)              {np.abs(gpred - y).mean():.2f}")
    print(f"  QC + scale/offset              {ridge_cv(QC[:, None], folds, 1e-3):.2f}")
    print(f"  QC + charge correction         {ridge_cv(np.column_stack([QC, CH]), folds, 1.0):.2f}")
    print(f"  charge only (no QC)            {ridge_cv(CH, folds, 1.0):.2f}")
    print(f"  CC prior + (QC+charge) resid   {ridge_cv(np.column_stack([QC, CH]), folds, 1.0, resid_of=gpred):.2f}")
    print()
