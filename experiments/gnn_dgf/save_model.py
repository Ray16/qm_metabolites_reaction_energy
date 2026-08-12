#!/usr/bin/env python
"""Train the final GNN-delta[rich] on ALL 367 reactions and save a checkpoint.

The deployment model (trained on everything) for use; the honest accuracy is the
held-out 6.78 from generate_predictions.py, NOT this model's in-sample fit.
"""
import json, os
import numpy as np
import torch
import train_v2 as T

HERE = os.path.dirname(os.path.abspath(__file__))
LEVEL, LAM, ENS, EPOCHS = "rich", 30.0, 4, 500
HP = dict(hidden=96, layers=3, drop=0.1, lr=3e-3, wd=1e-4)

d = torch.load(f"{HERE}/data.pt")
S = d["S"].to(T.DEV); y = d["y"].to(T.DEV); n = d["n"].to(T.DEV)
w = torch.log1p(n); w = w / w.mean()
yn = y.cpu().numpy(); Xg = d["Xgroup"].numpy()
g = T.Graph(d["graphs"][LEVEL])
N = len(d["rxn_ids"]); allidx = np.arange(N)

# linear group-CC prior on all data
coef = T.ridge_fit(Xg, yn, allidx, LAM)
lp = Xg @ coef
resid = torch.as_tensor(yn - lp, dtype=torch.float32, device=T.DEV)

# GNN residual: seed-ensemble of trained models (keep each state_dict)
states = []
for s in range(ENS):
    torch.manual_seed(s)
    model = T.MPNN(g.atom_dim, g.bond_dim, g.qm.size(1), **{k: HP[k] for k in ("hidden", "layers", "drop")}).to(T.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    rng = np.random.default_rng(s); perm = rng.permutation(N)
    nval = N // 6
    vi = torch.as_tensor(perm[:nval], device=T.DEV); ti = torch.as_tensor(perm[nval:], device=T.DEV)
    best = (1e9, None, 0)
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        loss = (w[ti] * (S[ti] @ model(g) - resid[ti]) ** 2).mean()
        loss.backward(); opt.step()
        if ep % 20 == 0:
            model.eval()
            with torch.no_grad():
                vm = (S[vi] @ model(g) - resid[vi]).abs().mean().item()
            if vm < best[0] - 1e-4:
                best = (vm, {k: v.clone() for k, v in model.state_dict().items()}, ep)
            elif ep - best[2] > 200:
                break
    states.append(best[1] or model.state_dict())

ckpt = dict(
    kind="GNN-delta[rich]  (per-compound formation-energy MPNN + group-CC prior)",
    held_out_mae_kJ={"random_cv": 6.78, "compound_disjoint_cv": 8.61},
    note="Deployment model trained on all 367 TECRDB reactions. Accuracy figures "
         "are held-out CV, not this model's in-sample fit.",
    model_states=states, n_ensemble=ENS,
    prior_coef=coef, prior_lambda=LAM, level=LEVEL,
    hp=HP, atom_dim=g.atom_dim, bond_dim=g.bond_dim, qm_dim=g.qm.size(1),
    arch="MPNN(atom_dim,bond_dim,qm_dim,hidden,layers,drop); readout LayerNorm(qm)+MLP; pred=prior + S@f",
    repro="experiments/gnn_dgf/{prepare_data.py,train_v2.py,save_model.py}",
)
torch.save(ckpt, f"{HERE}/checkpoint.pt")
print(f"saved checkpoint.pt  ({ENS}-model ensemble, prior lam={LAM}, level={LEVEL})")
print(f"held-out accuracy: random CV 6.78 / compound-disjoint 8.61 kJ/mol MAE")
