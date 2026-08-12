#!/usr/bin/env python
"""Data-scale experiments (GPU / uma env). Both were NEGATIVE results.

--mode distill   train on 8,765 eQ ModelSEED reactions, test on experimental 367
                 (naive distillation)  -> ~18 MAE, distribution mismatch.
--mode finetune  pretrain on eQ, fine-tune on 367 held-out CV vs from-scratch
                 -> 7.23 vs 7.14, no benefit.
"""
import _bootstrap  # noqa: F401
import argparse

import numpy as np
import torch

from gnn import paths
from gnn.model import MPNN, Graph, DEV
from gnn.training import kfold

d = torch.load(paths.artifact("distill_data.pt"))
g = Graph(d["graph"])
S_tr = d["S_tr"].to(DEV); y_tr = d["y_tr"].to(DEV)
S_te = d["S_te"].to(DEV); y_te = d["y_te"].to(DEV)
yn = y_te.cpu().numpy(); N = S_te.shape[0]
HP = dict(hidden=128, layers=4, drop=0.1)


def fit(S, y, ti, vi, epochs, lr, wd, init=None, patience=150, val_every=25):
    torch.manual_seed(0)
    m = MPNN(g.atom_dim, g.bond_dim, g.qm.size(1), **HP).to(DEV)
    if init:
        m.load_state_dict(init)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    best = (1e9, None, 0)
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        ((S[ti] @ m(g) - y[ti]) ** 2).mean().backward(); opt.step()
        if ep % val_every == 0:
            m.eval()
            with torch.no_grad():
                vm = (S[vi] @ m(g) - y[vi]).abs().mean().item()
            if vm < best[0] - 1e-3:
                best = (vm, {k: v.clone() for k, v in m.state_dict().items()}, ep)
            elif ep - best[2] > patience:
                break
    if best[1]:
        m.load_state_dict(best[1])
    return m


def pretrain():
    Ntr = S_tr.shape[0]; perm = np.random.default_rng(0).permutation(Ntr)
    vi = torch.as_tensor(perm[:Ntr // 10], device=DEV); ti = torch.as_tensor(perm[Ntr // 10:], device=DEV)
    m = fit(S_tr, y_tr, ti, vi, 500, 2e-3, 1e-5)
    return {k: v.clone() for k, v in m.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["distill", "finetune"], required=True)
    a = ap.parse_args()
    seen = (S_tr.abs().sum(0) > 0).cpu().numpy(); tc = (S_te.abs().sum(0) > 0).cpu().numpy()
    print(f"device={DEV} train={S_tr.shape[0]} test={N} coverage={(seen & tc).sum() / tc.sum() * 100:.0f}%")

    if a.mode == "distill":
        Ntr = S_tr.shape[0]; perm = np.random.default_rng(0).permutation(Ntr)
        vi = torch.as_tensor(perm[:Ntr // 10], device=DEV); ti = torch.as_tensor(perm[Ntr // 10:], device=DEV)
        m = fit(S_tr, y_tr, ti, vi, 1200, 2e-3, 1e-5)
        with torch.no_grad():
            te = (S_te @ m(g)).cpu().numpy()
        print(f"DISTILL  test-experimental MAE {np.abs(te - yn).mean():.2f}  (367-only was 6.78)")
    else:
        state = pretrain()

        def cv(init):
            pred = np.zeros(N)
            for tr, te in kfold(N, 5, 0):
                tr = np.asarray(tr); p = np.random.default_rng(0).permutation(len(tr))
                nval = max(4, len(tr) // 6)
                vi = torch.as_tensor(tr[p[:nval]], device=DEV); ti = torch.as_tensor(tr[p[nval:]], device=DEV)
                m = fit(S_te, y_te, ti, vi, 300, 1e-3, 1e-4, init=init, patience=80, val_every=10)
                with torch.no_grad():
                    pred[te] = (S_te @ m(g)).cpu().numpy()[te]
            return np.abs(pred - yn).mean()
        print(f"FINETUNE  from-scratch {cv(None):.2f}  vs  pretrained {cv(state):.2f}  (367-only+xtb 6.78)")


if __name__ == "__main__":
    main()
