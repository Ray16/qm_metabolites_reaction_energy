#!/usr/bin/env python
"""Improve held-out accuracy by AUGMENTING the 367 TECRDB reactions with
in-distribution eQuilibrator reactions (|dG| < THRESH), distilling eQ's higher
accuracy without the distribution mismatch that sank naive distillation.

5-fold CV on the 367: each fold trains on (its 367-train reactions) + (all
filtered eQ reactions, down-weighted), tests on held-out 367. Compares to the
same-space from-scratch baseline (367-train only). GPU / gnndgf env.
"""
import _bootstrap  # noqa: F401
import argparse

import numpy as np
import torch

from gnn import paths
from gnn.model import MPNN, Graph, DEV
from gnn.training import kfold


def fit(g, S, y, w, ti, vi, epochs, hp, patience=12, val_every=15):
    torch.manual_seed(0)
    m = MPNN(g.atom_dim, g.bond_dim, g.qm.size(1), hp["hidden"], hp["layers"], hp["drop"]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    best = (1e9, None, 0)
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        (w[ti] * (S[ti] @ m(g) - y[ti]) ** 2).mean().backward(); opt.step()
        if ep % val_every == 0:
            m.eval()
            with torch.no_grad():
                vm = (S[vi] @ m(g) - y[vi]).abs().mean().item()
            if vm < best[0] - 1e-3:
                best = (vm, {k: v.clone() for k, v in m.state_dict().items()}, ep)
            elif ep - best[2] > patience * val_every:
                break
    if best[1]:
        m.load_state_dict(best[1])
    m.eval()
    with torch.no_grad():
        return (S @ m(g)).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=50.0)
    ap.add_argument("--waug", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--ens", type=int, default=3)
    a = ap.parse_args()
    hp = dict(hidden=128, layers=4, drop=0.1, lr=2e-3, wd=1e-4)

    d = torch.load(paths.artifact("distill_data.pt"))
    g = Graph(d["graph"])
    S_tr = d["S_tr"].to(DEV); y_tr = d["y_tr"].to(DEV)      # eQ ModelSEED
    S_te = d["S_te"].to(DEV); y_te = d["y_te"].to(DEV)      # 367 TECRDB
    yn = y_te.cpu().numpy(); N = S_te.shape[0]

    keep = (y_tr.abs() < a.thresh)
    S_aug, y_aug = S_tr[keep], y_tr[keep]
    print(f"device={DEV}  eQ in-distribution (|dG|<{a.thresh:g}): {int(keep.sum())}/{S_tr.shape[0]}  "
          f"aug weight={a.waug}")

    def cv(augment):
        pred = np.zeros(N)
        for tr, te in kfold(N, 5, 0):
            tr = np.asarray(tr); p = np.random.default_rng(0).permutation(len(tr))
            nval = max(4, len(tr) // 6)
            vi = torch.as_tensor(tr[p[:nval]], device=DEV)
            ti_tec = tr[p[nval:]]
            # stack TECRDB-train + (optional) eQ augmentation into one problem
            if augment:
                S = torch.cat([S_te, S_aug], 0); y = torch.cat([y_te, y_aug], 0)
                w = torch.cat([torch.ones(N, device=DEV),
                               torch.full((S_aug.shape[0],), a.waug, device=DEV)])
                ti = torch.cat([torch.as_tensor(ti_tec, device=DEV),
                                torch.arange(N, N + S_aug.shape[0], device=DEV)])
                viz = torch.as_tensor(tr[p[:nval]], device=DEV)
            else:
                S, y, w = S_te, y_te, torch.ones(N, device=DEV)
                ti, viz = torch.as_tensor(ti_tec, device=DEV), vi
            fold_preds = torch.stack([fit(g, S, y, w, ti, viz, a.epochs, {**hp}, )
                                      for _ in range(a.ens)]).mean(0)
            pred[te] = fold_preds[torch.as_tensor(te, device=DEV)].cpu().numpy()
        return np.abs(pred - yn).mean()

    base = cv(False)
    print(f"  from-scratch (367 only, this space)   held-out MAE {base:.2f}")
    aug = cv(True)
    print(f"  + in-distribution eQ augmentation      held-out MAE {aug:.2f}")
    print(f"\n=== {'IMPROVED' if aug < base - 0.05 else 'no gain'} ===  "
          f"{base:.2f} -> {aug:.2f}  (367-only+xtb reference 6.78, eQ in-sample 3.0)")


if __name__ == "__main__":
    main()
