#!/usr/bin/env python
"""Honest held-out (out-of-fold) predictions for ALL 367 TECRDB reactions,
using the best model: GNN-delta on the rich (CPCM-X) feature level.

Each reaction is predicted by a model that never trained on it (5-fold OOF,
seed-ensembled). Also emits the linear group-CC OOF for reference. Saves
predictions.json {rxn_id: {exp, gnn, linear}} and prints full-367 MAE.
"""
import json, os
import numpy as np
import torch
import train_v2 as T

HERE = os.path.dirname(os.path.abspath(__file__))
LEVEL = "rich"; LAM = 30.0; ENS = 4; EPOCHS = 400
HP = dict(hidden=96, layers=3, drop=0.1, lr=3e-3, wd=1e-4)

d = torch.load(f"{HERE}/data.pt")
S = d["S"].to(T.DEV); y = d["y"].to(T.DEV); n = d["n"].to(T.DEV)
w = torch.log1p(n); w = w / w.mean()
yn = y.cpu().numpy(); Xg = d["Xgroup"].numpy()
rxn_ids = d["rxn_ids"]; N = len(rxn_ids)
g = T.Graph(d["graphs"][LEVEL])

gnn_oof = np.zeros(N); lin_oof = np.zeros(N)
for fi, (tr, te) in enumerate(T.kfold(N, 5, 0)):
    coef = T.ridge_fit(Xg, yn, tr, LAM)
    lp = Xg @ coef
    lin_oof[te] = lp[te]
    resid = torch.as_tensor(yn - lp, dtype=torch.float32, device=T.DEV)
    pr = T.gnn_predict(g, S, resid, w, tr, EPOCHS, 0, ENS, HP)
    final = torch.as_tensor(lp, device=T.DEV) + pr
    gnn_oof[te] = final[torch.as_tensor(te, device=T.DEV)].cpu().numpy()
    print(f"fold {fi}: |te|={len(te)}", flush=True)

out = {r: {"exp": float(yn[i]), "gnn": float(gnn_oof[i]), "linear": float(lin_oof[i])}
       for i, r in enumerate(rxn_ids)}
json.dump(out, open(f"{HERE}/predictions.json", "w"), indent=1)

gnn_mae = np.abs(gnn_oof - yn).mean()
lin_mae = np.abs(lin_oof - yn).mean()
print(f"\nFULL-367 out-of-fold MAE:  GNN-delta[{LEVEL}] {gnn_mae:.2f}   "
      f"linear-CC {lin_mae:.2f}   predict-zero {np.abs(yn).mean():.2f}")
print("wrote predictions.json")
