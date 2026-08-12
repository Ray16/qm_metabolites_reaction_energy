#!/usr/bin/env python
"""Transfer learning: pretrain the GNN on 8,765 eQ ModelSEED reactions to learn
compound representations, then fine-tune + evaluate on the 367 TECRDB
experimental reactions under held-out 5-fold CV.

Isolates the pretraining effect by comparing, in the SAME feature space
(graph + RDKit descriptors, 3415-compound index, no xtb):
  from-scratch   : 367 CV, random init
  pretrained     : 367 CV, init from eQ-pretrained weights (fine-tuned per fold)
Question: does pretraining on 25x more compounds beat the from-scratch OOF MAE?
"""
import json, os
import numpy as np
import torch
import train_v2 as T

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = T.DEV

d = torch.load(f"{HERE}/distill_data.pt")
g = T.Graph(d["graph"])
S_tr = d["S_tr"].to(DEV); y_tr = d["y_tr"].to(DEV)
S_te = d["S_te"].to(DEV); y_te = d["y_te"].to(DEV)
yn = y_te.cpu().numpy(); N = S_te.shape[0]


def new_model():
    torch.manual_seed(0)
    return T.MPNN(g.atom_dim, g.bond_dim, g.qm.size(1), hidden=128, layers=4, drop=0.1).to(DEV)


def pretrain(epochs=500):
    """Pretrain on eQ reactions; return state_dict."""
    m = new_model()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-5)
    Ntr = S_tr.shape[0]
    perm = np.random.default_rng(0).permutation(Ntr)
    vi = torch.as_tensor(perm[:Ntr // 10], device=DEV); ti = torch.as_tensor(perm[Ntr // 10:], device=DEV)
    best = (1e9, None, 0)
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        loss = ((S_tr[ti] @ m(g) - y_tr[ti]) ** 2).mean()
        loss.backward(); opt.step()
        if ep % 25 == 0:
            m.eval()
            with torch.no_grad():
                vmae = (S_tr[vi] @ m(g) - y_tr[vi]).abs().mean().item()
            if vmae < best[0] - 1e-3:
                best = (vmae, {k: v.clone() for k, v in m.state_dict().items()}, ep)
            elif ep - best[2] > 150:
                break
    print(f"  pretrain done: eQ-val MAE {best[0]:.1f}")
    return best[1]


def finetune_fold(init_state, tr, te, epochs=300, lr=1e-3):
    m = new_model()
    if init_state is not None:
        m.load_state_dict(init_state)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    tr = np.asarray(tr); rng = np.random.default_rng(0); p = rng.permutation(len(tr))
    nval = max(4, len(tr) // 6)
    vi = torch.as_tensor(tr[p[:nval]], device=DEV); ti = torch.as_tensor(tr[p[nval:]], device=DEV)
    best = (1e9, None, 0)
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        loss = ((S_te[ti] @ m(g) - y_te[ti]) ** 2).mean()
        loss.backward(); opt.step()
        if ep % 10 == 0:
            m.eval()
            with torch.no_grad():
                vmae = (S_te[vi] @ m(g) - y_te[vi]).abs().mean().item()
            if vmae < best[0] - 1e-3:
                best = (vmae, {k: v.clone() for k, v in m.state_dict().items()}, ep)
            elif ep - best[2] > 80:
                break
    if best[1]:
        m.load_state_dict(best[1])
    m.eval()
    with torch.no_grad():
        return (S_te[torch.as_tensor(te, device=DEV)] @ m(g)).cpu().numpy()


def cv_oof(init_state, folds):
    pred = np.zeros(N)
    for tr, te in folds:
        pred[te] = finetune_fold(init_state, tr, te)
    return np.abs(pred - yn).mean()


folds = T.kfold(N, 5, 0)
print(f"device={DEV}  test={N}  compounds={g.n_comp}\n")
print("pretraining on eQ...")
state = pretrain()
print("\nfrom-scratch 367 CV (this feature space, no xtb)...")
scratch = cv_oof(None, folds)
print(f"  from-scratch OOF MAE  {scratch:.2f}")
print("pretrained -> fine-tune 367 CV...")
pre = cv_oof(state, folds)
print(f"  pretrained  OOF MAE  {pre:.2f}")
print(f"\n=== TRANSFER RESULT ===  from-scratch {scratch:.2f}  vs  pretrained {pre:.2f}  "
      f"(367-only+xtb reference 6.78)")
