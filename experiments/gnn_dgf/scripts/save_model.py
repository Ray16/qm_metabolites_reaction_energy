#!/usr/bin/env python
"""Train the final model on ALL 367 reactions and save artifacts/checkpoint.pt.

  --mode scratch  (default)  from-scratch GNN[rich], NO dGPredictor anchor.
                             Coverage-appropriate: no group-decomposition
                             dependence, extends to novel compounds.
  --mode delta               prior (group CC) + GNN residual.

Accuracy figures embedded in the checkpoint are held-out CV, not this model's
in-sample fit.
"""
import _bootstrap  # noqa: F401
import argparse

import numpy as np
import torch

from gnn import paths
from gnn.model import MPNN, Graph, DEV
from gnn.training import ridge_fit, DEFAULT_HP

MAE = {"scratch": {"random_cv": 6.80, "compound_disjoint_cv": 8.59},
       "delta": {"random_cv": 6.75, "compound_disjoint_cv": 8.61}}


def train_ensemble(g, S, target, w, ens, epochs, hp):
    N = target.shape[0]
    states = []
    for s in range(ens):
        torch.manual_seed(s)
        m = MPNN(g.atom_dim, g.bond_dim, g.qm.size(1), hp["hidden"], hp["layers"], hp["drop"]).to(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
        perm = np.random.default_rng(s).permutation(N)
        vi = torch.as_tensor(perm[:N // 6], device=DEV); ti = torch.as_tensor(perm[N // 6:], device=DEV)
        best = (1e9, None, 0)
        for ep in range(epochs):
            m.train(); opt.zero_grad()
            loss = (w[ti] * (S[ti] @ m(g) - target[ti]) ** 2).mean()
            loss.backward(); opt.step()
            if ep % 20 == 0:
                m.eval()
                with torch.no_grad():
                    vm = (S[vi] @ m(g) - target[vi]).abs().mean().item()
                if vm < best[0] - 1e-4:
                    best = (vm, {k: v.cpu().clone() for k, v in m.state_dict().items()}, ep)
                elif ep - best[2] > 200:
                    break
        states.append(best[1] or {k: v.cpu() for k, v in m.state_dict().items()})
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["scratch", "delta"], default="scratch")
    ap.add_argument("--ens", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--level", choices=["none", "solv", "full", "rich"], default="rich",
                    help="'none' = graph-only (RDKit features, xtb-free; deployable "
                         "for a full-database sweep). 'rich' = with xtb QM (research).")
    ap.add_argument("--out", default="checkpoint.pt", help="artifact filename")
    a = ap.parse_args()
    hp = DEFAULT_HP

    d = torch.load(paths.artifact("data.pt"))
    S = d["S"].to(DEV); y = d["y"].to(DEV); n = d["n"].to(DEV)
    w = torch.log1p(n); w = w / w.mean()
    g = Graph(d["graphs"][a.level])
    yn = y.cpu().numpy(); Xg = d["Xgroup"].numpy(); N = len(d["rxn_ids"])

    coef = None
    target = y
    if a.mode == "delta":
        coef = ridge_fit(Xg, yn, np.arange(N), a.lam)
        target = torch.as_tensor(yn - Xg @ coef, dtype=torch.float32, device=DEV)
    states = train_ensemble(g, S, target, w, a.ens, a.epochs, hp)

    tag = "graph-only, RDKit features, xtb-FREE" if a.level == "none" else f"{a.level} (QC features)"
    ckpt = dict(
        kind=(f"GNN[{a.level}] from-scratch ({tag}, NO group anchor)"
              if a.mode == "scratch" else f"GNN-delta[{a.level}] (group-CC prior + GNN residual)"),
        mode=a.mode, level=a.level, held_out_mae_kJ=MAE[a.mode],
        note="Prediction: dG = " + ("S@f" if a.mode == "scratch" else "Xgroup@coef + S@f")
             + "; f = mean over ensemble. Accuracy is held-out CV, not in-sample."
             + (" held_out_mae_kJ is the rich-level reference; graph-only is ~7.0/8.9."
                if a.level == "none" else ""),
        model_states=states, n_ensemble=a.ens,
        prior_coef=coef, prior_lambda=(a.lam if a.mode == "delta" else None),
        hp=hp, atom_dim=g.atom_dim, bond_dim=g.bond_dim, qm_dim=g.qm.size(1),
        arch="MPNN; sum-readout + LayerNorm(QC) + MLP; pred = "
             + ("S@f" if a.mode == "scratch" else "prior + S@f"),
        repro="scripts/{prepare_data.py,save_model.py}")
    torch.save(ckpt, paths.artifact(a.out))
    print(f"saved artifacts/{a.out}  mode={a.mode}  level={a.level}  {a.ens}-model ensemble")


if __name__ == "__main__":
    main()
