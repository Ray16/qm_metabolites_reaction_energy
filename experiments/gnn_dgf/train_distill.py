#!/usr/bin/env python
"""Distillation test: train the GNN on 8,765 eQ-labeled ModelSEED reactions,
evaluate on the 367 TECRDB *experimental* reactions (never in training).

Does 25x more (pseudo-labeled) training signal break the n=367 ceiling (6.78)?
Reports:
  test experimental MAE  -- end accuracy vs TECRDB experiment
  test-vs-eQ fidelity    -- how well it reproduces eQ's own pH7 predictions
  coverage               -- fraction of test compounds seen in training
"""
import json, os
import numpy as np
import torch
import torch.nn as nn
import train_v2 as T

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = T.DEV

d = torch.load(f"{HERE}/distill_data.pt")
g = T.Graph(d["graph"])
S_tr = d["S_tr"].to(DEV); y_tr = d["y_tr"].to(DEV)
S_te = d["S_te"].to(DEV); y_te = d["y_te"].to(DEV)
test_ids = d["test_ids"]

# eQ pH7 predictions on the 367 (for distillation-fidelity metric)
eqf = json.load(open(f"{HERE}/../../results/eq/equilibrator_full.json"))
eq_te = np.array([eqf.get(r, {}).get("dG_kJ", np.nan) for r in test_ids])

# coverage: test compounds seen in any training reaction
seen = (S_tr.abs().sum(0) > 0).cpu().numpy()
test_comp = (S_te.abs().sum(0) > 0).cpu().numpy()
cov = (seen & test_comp).sum() / max(1, test_comp.sum())
print(f"device={DEV}  train={S_tr.shape[0]}  test={S_te.shape[0]}  "
      f"compounds={g.n_comp}  test-compound coverage={cov*100:.0f}%\n")

torch.manual_seed(0)
model = T.MPNN(g.atom_dim, g.bond_dim, g.qm.size(1), hidden=128, layers=4, drop=0.1).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)

# inner val split of the eQ training reactions for early stopping
N = S_tr.shape[0]
perm = np.random.default_rng(0).permutation(N)
nval = N // 10
vi = torch.as_tensor(perm[:nval], device=DEV); ti = torch.as_tensor(perm[nval:], device=DEV)
Sti, yti = S_tr[ti], y_tr[ti]; Sv, yv = S_tr[vi], y_tr[vi]

yn_te = y_te.cpu().numpy()
best = (1e9, None)
for ep in range(1200):
    model.train(); opt.zero_grad()
    loss = ((Sti @ model(g) - yti) ** 2).mean()
    loss.backward(); opt.step()
    if ep % 25 == 0:
        model.eval()
        with torch.no_grad():
            f = model(g)
            vmae = (Sv @ f - yv).abs().mean().item()
            te_pred = (S_te @ f).cpu().numpy()
        exp_mae = np.abs(te_pred - yn_te).mean()
        fid_mae = np.nanmean(np.abs(te_pred - eq_te))
        print(f"ep{ep:4d}  eQ-val {vmae:6.1f}   TEST-exp {exp_mae:6.2f}   "
              f"vs-eQ {fid_mae:6.2f}", flush=True)
        if vmae < best[0] - 1e-3:
            best = (vmae, te_pred.copy())

# best-by-eQ-val test predictions
te_pred = best[1]
exp_mae = np.abs(te_pred - yn_te).mean()
fid_mae = np.nanmean(np.abs(te_pred - eq_te))
print(f"\n=== DISTILLATION RESULT (early-stopped on eQ-val) ===")
print(f"  TEST experimental MAE   {exp_mae:6.2f}  kJ/mol   (367-only GNN was 6.78)")
print(f"  TEST vs eQ (fidelity)   {fid_mae:6.2f}  kJ/mol")
print(f"  eQ vs experiment ref     3.0   kJ/mol   (in-sample)")
json.dump({r: float(p) for r, p in zip(test_ids, te_pred)},
          open(f"{HERE}/distill_test_pred.json", "w"))
print("wrote distill_test_pred.json")
