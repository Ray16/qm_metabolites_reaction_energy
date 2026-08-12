"""Training + cross-validation utilities (pure torch/numpy, no rdkit).

train_once / gnn_predict     early-stopped, seed-ensembled GNN training
ridge_fit                    closed-form ridge (linear group-CC / prior)
kfold / compound_disjoint    the two held-out CV schemes
delta_targets                residual target for Delta-learning on a prior
"""
import numpy as np
import torch

from .model import MPNN, Graph, DEV

DEFAULT_HP = dict(hidden=96, layers=3, drop=0.1, lr=3e-3, wd=1e-4)


def ridge_fit(X, y, tr, lam):
    A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X[tr].T @ y[tr])


def train_once(g, S, y, w, tr, epochs=500, hp=None, seed=0, patience=8, val_every=20):
    """Train one GNN on training reactions `tr`; return the full S@f prediction
    vector (float tensor on DEV), early-stopped on an inner validation split."""
    hp = {**DEFAULT_HP, **(hp or {})}
    torch.manual_seed(seed)
    model = MPNN(g.atom_dim, g.bond_dim, g.qm.size(1),
                 hp["hidden"], hp["layers"], hp["drop"]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    tr = np.asarray(tr)
    perm = np.random.default_rng(seed).permutation(len(tr))
    nval = max(4, len(tr) // 6)
    vi = torch.as_tensor(tr[perm[:nval]], device=DEV)
    ti = torch.as_tensor(tr[perm[nval:]], device=DEV)
    Sti, yti, wti = S[ti], y[ti], w[ti]
    Sv, yv = S[vi], y[vi]
    best = (1e9, None, 0)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = (wti * (Sti @ model(g) - yti) ** 2).mean()
        loss.backward(); opt.step()
        if ep % val_every == 0:
            model.eval()
            with torch.no_grad():
                vmae = (Sv @ model(g) - yv).abs().mean().item()
            if vmae < best[0] - 1e-4:
                best = (vmae, {k: v.clone() for k, v in model.state_dict().items()}, ep)
            elif ep - best[2] > patience * val_every:
                break
    if best[1] is not None:
        model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        return (S @ model(g)).detach()


def gnn_predict(g, S, y, w, tr, epochs=500, hp=None, seed=0, n_ens=3):
    """Seed-ensemble mean of train_once."""
    return torch.stack([train_once(g, S, y, w, tr, epochs, hp, seed + 100 * s)
                        for s in range(n_ens)]).mean(0)


def kfold(n, k, seed):
    perm = np.random.default_rng(seed).permutation(n)
    return [(np.setdiff1d(np.arange(n), p), p) for p in (perm[i::k] for i in range(k))]


def compound_disjoint(rxn_comps, n_comp, k, seed):
    """Held-out COMPOUNDS: a reaction tests iff it touches a held-out compound;
    trains iff it touches none.  Strict extrapolation / coverage proxy."""
    grp = np.random.default_rng(seed).integers(0, k, size=n_comp)
    folds = []
    for j in range(k):
        held = set(np.where(grp == j)[0].tolist())
        te = [i for i, cs in enumerate(rxn_comps) if held & set(cs)]
        tr = [i for i, cs in enumerate(rxn_comps) if not (held & set(cs))]
        folds.append((np.array(tr), np.array(te)))
    return folds


def mae_rmse(err):
    err = np.asarray(err, float)
    return float(err.mean()), float(np.sqrt((err ** 2).mean()))
