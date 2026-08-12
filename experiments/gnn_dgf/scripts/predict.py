#!/usr/bin/env python
"""Honest held-out (out-of-fold) predictions for all 367 TECRDB reactions.

Each reaction predicted by a model that never trained on it (5-fold OOF,
seed-ensembled). --mode scratch (default, no dGP anchor) or delta.
Writes artifacts/predictions.json {rxn_id: {exp, gnn, linear}} and prints MAE.
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np
import torch

from gnn import paths
from gnn.model import Graph, DEV
from gnn.training import ridge_fit, gnn_predict, kfold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["scratch", "delta"], default="scratch")
    ap.add_argument("--ens", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lam", type=float, default=30.0)
    a = ap.parse_args()

    d = torch.load(paths.artifact("data.pt"))
    S = d["S"].to(DEV); y = d["y"].to(DEV); n = d["n"].to(DEV)
    w = torch.log1p(n); w = w / w.mean()
    yn = y.cpu().numpy(); Xg = d["Xgroup"].numpy()
    rxn_ids = d["rxn_ids"]; N = len(rxn_ids)
    g = Graph(d["graphs"]["rich"])

    gnn_oof, lin_oof = np.zeros(N), np.zeros(N)
    for fi, (tr, te) in enumerate(kfold(N, 5, 0)):
        lp = Xg @ ridge_fit(Xg, yn, tr, a.lam)
        lin_oof[te] = lp[te]
        if a.mode == "delta":
            resid = torch.as_tensor(yn - lp, dtype=torch.float32, device=DEV)
            pr = gnn_predict(g, S, resid, w, tr, a.epochs, seed=0, n_ens=a.ens)
            final = torch.as_tensor(lp, device=DEV) + pr
        else:
            final = gnn_predict(g, S, y, w, tr, a.epochs, seed=0, n_ens=a.ens)
        gnn_oof[te] = final[torch.as_tensor(te, device=DEV)].cpu().numpy()
        print(f"fold {fi}: |te|={len(te)}", flush=True)

    json.dump({r: {"exp": float(yn[i]), "gnn": float(gnn_oof[i]), "linear": float(lin_oof[i])}
               for i, r in enumerate(rxn_ids)},
              open(paths.artifact("predictions.json"), "w"), indent=1)
    print(f"\nOOF MAE  GNN[{a.mode}] {np.abs(gnn_oof - yn).mean():.2f}  "
          f"linear {np.abs(lin_oof - yn).mean():.2f}  "
          f"predict-zero {np.abs(yn).mean():.2f}  -> wrote predictions.json")


if __name__ == "__main__":
    main()
