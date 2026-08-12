#!/usr/bin/env python
"""Baseline / QC-reasoning comparisons (base env; numpy only).

1) Fair held-out dGPredictor group-linear model (removes the incumbents'
   in-sample advantage) at several regularizations.
2) QC-as-charge-gated-prior test: does QC + Born(Delta z^2) correction help,
   and can it beat group-additivity? (Answer: improves QC-alone, can't beat CC.)
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import torch

from gnn import paths
from gnn.training import kfold, compound_disjoint

d = torch.load(paths.artifact("data.pt"))
y = d["y"].numpy(); Xg = d["Xgroup"].numpy(); S = d["S"].numpy()
rxn_ids = d["rxn_ids"]; rxn_comps = d["rxn_comps"]; n_comp = d["n_comp"]; N = len(rxn_ids)

mets = json.load(open(f"{paths.PIPE}/tecrdb_full_metabolites.json"))
z = np.array([m.get("charge", 0) for m in mets], float)
qc_all = json.load(open(f"{paths.RESULTS}/benchmark/tecrdb_full_scored.json"))["scored_kJ"]
QC = np.array([qc_all.get(r, 0.0) for r in rxn_ids], float)
CH = np.column_stack([S @ z, S @ np.abs(z), S @ (z ** 2)])   # dz, d|z|, dz^2 (Born)

SCHEMES = [("RANDOM", kfold(N, 5, 0)), ("CPD-DISJOINT", compound_disjoint(rxn_comps, n_comp, 5, 0))]


def oof(X, folds, lam, resid=None):
    Xb = np.column_stack([X, np.ones(len(X))])
    pred = np.full(N, np.nan)
    for tr, te in folds:
        if len(te) == 0:
            continue
        t = y - resid if resid is not None else y
        c = np.linalg.solve(Xb[tr].T @ Xb[tr] + lam * np.eye(Xb.shape[1]), Xb[tr].T @ t[tr])
        pred[te] = Xb[te] @ c + (resid[te] if resid is not None else 0.0)
    m = np.isfinite(pred)
    return np.abs(pred[m] - y[m]).mean()


def group_oof(folds, lam):
    pred = np.zeros(N)
    for tr, te in folds:
        if len(te):
            c = np.linalg.solve(Xg[tr].T @ Xg[tr] + lam * np.eye(Xg.shape[1]), Xg[tr].T @ y[tr])
            pred[te] = Xg[te] @ c
    return pred


print("=== Fair held-out dGPredictor group-linear (lam sweep) ===")
for name, folds in SCHEMES:
    print(f"  {name}:", "  ".join(f"lam{int(l)}={np.abs(group_oof(folds, l) - y).mean():.2f}"
                                   for l in (3, 10, 30, 100)))
print("\n=== QC as charge-gated prior (held-out MAE) ===")
for name, folds in SCHEMES:
    gp = group_oof(folds, 30.0)
    print(f"  {name}:  group-CC {np.abs(gp - y).mean():.2f}   "
          f"QC+scale {oof(QC[:, None], folds, 1e-3):.2f}   "
          f"QC+charge {oof(np.column_stack([QC, CH]), folds, 1.0):.2f}   "
          f"CC+(QC+charge)resid {oof(np.column_stack([QC, CH]), folds, 1.0, resid=gp):.2f}")
