"""Training + cross-validation utilities (pure torch/numpy, no rdkit).

train_once / gnn_predict     early-stopped, seed-ensembled GNN training
ridge_fit                    closed-form ridge (linear group-CC / prior)
kfold / compound_disjoint    the two held-out CV schemes
delta_targets                residual target for Delta-learning on a prior
"""
import numpy as np
import torch

from .model import MPNN, CondHead, Graph, DEV

DEFAULT_HP = dict(hidden=96, layers=3, drop=0.1, lr=3e-3, wd=1e-4)


def ridge_fit(X, y, tr, lam):
    A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X[tr].T @ y[tr])


def _resid_loss(r, w, loss, delta):
    """Weighted MSE or Huber over residual r (both reduce to a scalar mean)."""
    if loss == "huber":
        a = r.abs()
        per = torch.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta))
        return (w * per).mean()
    return (w * r ** 2).mean()


def train_once(g, S, y, w, tr, epochs=500, hp=None, seed=0, patience=8, val_every=20,
               loss="mse", huber_delta=6.0, cond=None):
    """Train one GNN on training reactions `tr`; return the full prediction vector
    (float tensor on DEV), early-stopped (on val MAE) on an inner split.

    All ablation variants share this one code path:
      hp['qm_in_messages']  -> QM injected before message passing (variant #4)
      loss='huber'          -> robust loss (variant #6)
      w                     -> per-reaction weights, e.g. inverse-variance (#6)
      cond (N,cond_dim)     -> also fit a CondHead; pred = S@f + h(cond) (#1)
    """
    hp = {**DEFAULT_HP, **(hp or {})}
    torch.manual_seed(seed)
    model = MPNN(g.atom_dim, g.bond_dim, g.qm.size(1), hp["hidden"], hp["layers"],
                 hp["drop"], qm_in_messages=hp.get("qm_in_messages", False)).to(DEV)
    params = list(model.parameters())
    head = None
    if cond is not None:
        head = CondHead(cond.size(1), drop=hp["drop"]).to(DEV)
        params += list(head.parameters())
    opt = torch.optim.Adam(params, lr=hp["lr"], weight_decay=hp["wd"])
    tr = np.asarray(tr)
    perm = np.random.default_rng(seed).permutation(len(tr))
    nval = max(4, len(tr) // 6)
    vi = torch.as_tensor(tr[perm[:nval]], device=DEV)
    ti = torch.as_tensor(tr[perm[nval:]], device=DEV)
    yti, wti, yv = y[ti], w[ti], y[vi]

    def predict():
        p = S @ model(g)
        return p if head is None else p + head(cond)

    best = (1e9, None, 0)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        pred = predict()
        _resid_loss(pred[ti] - yti, wti, loss, huber_delta).backward(); opt.step()
        if ep % val_every == 0:
            model.eval()
            with torch.no_grad():
                vmae = (predict()[vi] - yv).abs().mean().item()
            state = ({k: v.clone() for k, v in model.state_dict().items()},
                     None if head is None else {k: v.clone() for k, v in head.state_dict().items()})
            if vmae < best[0] - 1e-4:
                best = (vmae, state, ep)
            elif ep - best[2] > patience * val_every:
                break
    if best[1] is not None:
        model.load_state_dict(best[1][0])
        if head is not None:
            head.load_state_dict(best[1][1])
    model.eval()
    with torch.no_grad():
        return predict().detach()


def gnn_predict(g, S, y, w, tr, epochs=500, hp=None, seed=0, n_ens=3,
                return_stack=False, **kw):
    """Seed-ensemble of train_once. Returns the mean, or the (n_ens, N) stack when
    return_stack=True (per-seed spread is the uncertainty signal for variant #7).
    Extra kwargs (loss, huber_delta, cond) pass through to train_once."""
    stack = torch.stack([train_once(g, S, y, w, tr, epochs, hp, seed + 100 * s, **kw)
                         for s in range(n_ens)])
    return stack if return_stack else stack.mean(0)


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
