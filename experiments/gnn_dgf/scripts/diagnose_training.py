#!/usr/bin/env python
"""Sanity-check training: prints the device (GPU?) and the full convergence
curve (train loss + inner-val MAE) for one model, plus where early-stopping
fires. Confirms the model is properly trained (loss descends, val plateaus,
stop is sensible) and that it ran on GPU.
"""
import _bootstrap  # noqa: F401
import time

import numpy as np
import torch

from gnn import paths
from gnn.model import MPNN, Graph, DEV
from gnn.training import DEFAULT_HP

d = torch.load(paths.artifact("data.pt"))
S = d["S"].to(DEV); y = d["y"].to(DEV); n = d["n"].to(DEV)
w = torch.log1p(n); w = w / w.mean()
g = Graph(d["graphs"]["rich"])
N = len(d["rxn_ids"])

print(f"device        : {DEV}")
if DEV.type == "cuda":
    print(f"gpu           : {torch.cuda.get_device_name(0)}")
print(f"reactions     : {N}   compounds: {g.n_comp}   atoms: {g.x.shape[0]}")

hp = DEFAULT_HP
torch.manual_seed(0)
model = MPNN(g.atom_dim, g.bond_dim, g.qm.size(1), hp["hidden"], hp["layers"], hp["drop"]).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])

perm = np.random.default_rng(0).permutation(N)
nval = N // 6
vi = torch.as_tensor(perm[:nval], device=DEV); ti = torch.as_tensor(perm[nval:], device=DEV)
best = (1e9, -1)
t0 = time.time()
print("\n epoch |  train-loss | train-MAE | inner-val-MAE   (early stop = best val)")
for ep in range(600):
    model.train(); opt.zero_grad()
    tr_loss = (w[ti] * (S[ti] @ model(g) - y[ti]) ** 2).mean()
    tr_loss.backward(); opt.step()
    if ep % 20 == 0:
        model.eval()
        with torch.no_grad():
            f = model(g)
            tr_mae = (S[ti] @ f - y[ti]).abs().mean().item()
            v_mae = (S[vi] @ f - y[vi]).abs().mean().item()
        flag = ""
        if v_mae < best[0] - 1e-4:
            best = (v_mae, ep); flag = "  <- best"
        print(f" {ep:5d} | {tr_loss.item():10.2f} | {tr_mae:8.2f} | {v_mae:12.2f}{flag}")
        if ep - best[1] > 200:
            print(f" early stop at epoch {ep} (best val {best[0]:.2f} @ epoch {best[1]})")
            break
print(f"\nwall time     : {time.time() - t0:.1f} s   ({'GPU' if DEV.type == 'cuda' else 'CPU'})")
print(f"predict-zero MAE (reference): {y.abs().mean().item():.2f}")
