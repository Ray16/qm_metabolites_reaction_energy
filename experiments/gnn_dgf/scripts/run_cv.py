#!/usr/bin/env python
"""Main held-out cross-validation (GPU / uma env, no rdkit).

Evaluates, under RANDOM and COMPOUND-DISJOINT CV, on identical folds:
  linear         ridge on dGPredictor group features (the CC prior)
  GNN[level]     from-scratch per-compound GNN, dG = S@f      (no group anchor)
  GNN-delta      prior + GNN residual                          (Delta-learning)
for feature levels full and rich.  Writes artifacts/results_v2.json.
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np
import torch

from gnn import paths
from gnn.model import Graph, DEV
from gnn.training import (ridge_fit, gnn_predict, kfold, compound_disjoint, mae_rmse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ens", type=int, default=2)
    ap.add_argument("--levels", default="full,rich")
    ap.add_argument("--lam", type=float, default=30.0)
    a = ap.parse_args()

    d = torch.load(paths.artifact("data.pt"))
    S = d["S"].to(DEV); y = d["y"].to(DEV); n = d["n"].to(DEV)
    w = torch.log1p(n); w = w / w.mean()
    yn = y.cpu().numpy(); Xg = d["Xgroup"].numpy()
    N = len(d["rxn_ids"])
    graphs = {lvl: Graph(d["graphs"][lvl]) for lvl in a.levels.split(",")}
    print(f"device={DEV}  N={N}  lam={a.lam}  ens={a.ens}\n")

    schemes = [("RANDOM", kfold(N, a.folds, a.seed)),
               ("CPD-DISJOINT", compound_disjoint(d["rxn_comps"], d["n_comp"], a.folds, a.seed))]
    out = {}
    for scheme, folds in schemes:
        rows, lin_full = {}, np.zeros(N)
        e_zero, e_lin = [], []
        for tr, te in folds:
            if len(te) == 0:
                continue
            lp = Xg @ ridge_fit(Xg, yn, tr, a.lam)
            lin_full[te] = lp[te]
            e_zero.append(np.abs(yn[te])); e_lin.append(np.abs(lp[te] - yn[te]))
        rows["predict-zero"] = mae_rmse(np.concatenate(e_zero))
        rows["linear (group CC)"] = mae_rmse(np.concatenate(e_lin))
        for lvl, g in graphs.items():
            e_gnn, e_delta = [], []
            for tr, te in folds:
                if len(te) == 0:
                    continue
                te_t = torch.as_tensor(te, device=DEV)
                p = gnn_predict(g, S, y, w, tr, a.epochs, seed=a.seed, n_ens=a.ens)
                e_gnn.append((p[te_t] - y[te_t]).abs().cpu().numpy())
                lp = Xg @ ridge_fit(Xg, yn, tr, a.lam)
                resid = torch.as_tensor(yn - lp, dtype=torch.float32, device=DEV)
                pr = gnn_predict(g, S, resid, w, tr, a.epochs, seed=a.seed + 7, n_ens=a.ens)
                final = torch.as_tensor(lp, device=DEV) + pr
                e_delta.append((final[te_t] - y[te_t]).abs().cpu().numpy())
            rows[f"GNN[{lvl}]"] = mae_rmse(np.concatenate(e_gnn))
            rows[f"GNN-delta[{lvl}]"] = mae_rmse(np.concatenate(e_delta))
        print(f"=== {scheme} (held-out) ===")
        for k, (m, r) in rows.items():
            print(f"  {k:<22s} MAE {m:6.2f}  RMSE {r:6.2f}")
        print()
        out[scheme] = {k: {"mae": m, "rmse": r} for k, (m, r) in rows.items()}
    json.dump(out, open(paths.artifact("results_v2.json"), "w"), indent=2)
    print("wrote artifacts/results_v2.json")


if __name__ == "__main__":
    main()
