#!/usr/bin/env python
"""FAIR held-out comparison: dGPredictor-style linear group-contribution model
evaluated under the SAME CV folds as the GNN (train_gpu.py).

The incumbents' 3.0 MAE is in-sample. This removes that advantage: we refit the
dGPredictor group-difference linear model on each training fold and score the
held-out reactions -- apples-to-apples with the GNN's held-out numbers.

Uses dgp_group_features.json (reaction-level group-change vectors X) + the same
targets/fold logic (seed 0) as train_gpu.py.
"""
import json, os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "..", "results", "eq"))

d = torch.load(f"{HERE}/data.pt")
rxn_ids = d["rxn_ids"]; y = d["y"].numpy(); rxn_comps = d["rxn_comps"]
n_comp = d["n_comp"]; N = len(rxn_ids)

gf = json.load(open(f"{RES}/dgp_group_features.json"))
X = np.array([gf["X"][r] for r in rxn_ids], float)   # (367, n_features)
# drop all-zero columns (as the dGPredictor retrain did)
X = X[:, X.any(axis=0)]
print(f"group-feature matrix: {X.shape}")


def kfold(n, k, seed):
    perm = np.random.default_rng(seed).permutation(n)
    return [perm[i::k] for i in range(k)]


def compound_disjoint(rxn_comps, n_comp, k, seed):
    rng = np.random.default_rng(seed)
    grp = rng.integers(0, k, size=n_comp)
    folds = []
    for j in range(k):
        held = set(np.where(grp == j)[0].tolist())
        te = [i for i, cs in enumerate(rxn_comps) if held & set(cs)]
        tr = [i for i, cs in enumerate(rxn_comps) if not (held & set(cs))]
        folds.append((np.array(tr), np.array(te)))
    return folds


def ridge_cv(folds, lam):
    errs = []
    for tr, te in folds:
        if len(te) == 0:
            continue
        A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
        coef = np.linalg.solve(A, X[tr].T @ y[tr])   # fit_intercept=False (like dGP)
        errs.append(np.abs(X[te] @ coef - y[te]))
    return np.concatenate(errs)


def rpt(name, e):
    e = np.asarray(e)
    print(f"  {name:<42s} MAE {e.mean():6.2f}   RMSE {np.sqrt((e**2).mean()):6.2f}   n={len(e)}")


for scheme, folds in [
    ("RANDOM 5-fold", kfoldwrap := [(np.setdiff1d(np.arange(N), te), te)
                                    for te in kfold(N, 5, 0)]),
    ("COMPOUND-DISJOINT", compound_disjoint(rxn_comps, n_comp, 5, 0)),
]:
    print(f"=== dGPredictor group-linear, HELD-OUT, {scheme} ===")
    for lam in (1.0, 10.0, 100.0):
        rpt(f"ridge(lam={lam})", ridge_cv(folds, lam))
    print()
