#!/usr/bin/env python
"""Improved trainer: Delta-learning GNN on a group-contribution prior + richer
physical features, evaluated fairly (held-out CV) against the linear prior.

Modes per feature level:
  linear      : ridge on dGPredictor group-difference features (the CC prior)
  gnn         : standalone per-compound GNN, dG = S @ f          (v1)
  gnn-delta   : dG = linear_prior + S @ f_residual              (component
                contribution + learned nonlinear correction)

Feature levels: none / solv / full / rich  (rich = full + CPCM-X solvation).
QM graph features are LayerNorm-scaled inside the model.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Graph:
    def __init__(self, d):
        for k in ("x", "edge_index", "edge_attr", "qm", "batch"):
            setattr(self, k, d[k].to(DEV))
        self.atom_dim = d["atom_dim"]; self.bond_dim = d["bond_dim"]; self.n_comp = d["n_comp"]


class MPNN(nn.Module):
    def __init__(self, atom_dim, bond_dim, qm_dim, hidden=96, layers=3, drop=0.1):
        super().__init__()
        self.embed = nn.Linear(atom_dim, hidden)
        self.edge = nn.Linear(bond_dim, hidden)
        self.msg = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.upd = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                          nn.Dropout(drop), nn.Linear(hidden, hidden))
            for _ in range(layers))
        self.qm_norm = nn.LayerNorm(qm_dim) if qm_dim > 1 else nn.Identity()
        self.readout = nn.Sequential(
            nn.Linear(hidden + qm_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, g):
        h = torch.relu(self.embed(g.x))
        e = self.edge(g.edge_attr)
        src, dst = g.edge_index
        for msg, upd in zip(self.msg, self.upd):
            m = torch.relu(msg(h)[src] + e)
            agg = torch.zeros_like(h).index_add_(0, dst, m)
            h = h + upd(torch.cat([h, agg], dim=-1))
        pooled = torch.zeros(g.n_comp, h.size(1), device=h.device).index_add_(0, g.batch, h)
        return self.readout(torch.cat([pooled, self.qm_norm(g.qm)], dim=-1)).squeeze(-1)


def ridge_fit(X, y, tr, lam):
    A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X[tr].T @ y[tr])


def _train_once(g, S, y, w, tr, epochs, lr, wd, seed, hidden, layers, drop,
                patience=8, val_every=20):
    torch.manual_seed(seed)
    model = MPNN(g.atom_dim, g.bond_dim, g.qm.size(1), hidden, layers, drop).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tr))
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


def gnn_predict(g, S, y, w, tr, epochs, seed, n_ens, hp):
    preds = torch.stack([_train_once(g, S, y, w, tr, epochs, hp["lr"], hp["wd"],
                                     seed + 100 * s, hp["hidden"], hp["layers"], hp["drop"])
                         for s in range(n_ens)]).mean(0)
    return preds


def kfold(n, k, seed):
    perm = np.random.default_rng(seed).permutation(n)
    return [(np.setdiff1d(np.arange(n), p), p) for p in (perm[i::k] for i in range(k))]


def compound_disjoint(rxn_comps, n_comp, k, seed):
    rng = np.random.default_rng(seed)
    grp = rng.integers(0, k, size=n_comp)
    folds = []
    for j in range(k):
        held = set(np.where(grp == j)[0].tolist())
        te = [i for i, cs in enumerate(rxn_comps) if held & set(cs)]
        tr = [i for i, cs in enumerate(rxn_comps) if not (held & set(cs))]
        folds.append((np.array(tr), np.array(te)))
    return folds


def mae(e):
    e = np.asarray(e, float)
    return e.mean(), np.sqrt((e ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ens", type=int, default=3)
    ap.add_argument("--levels", default="full,rich")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    args = ap.parse_args()
    hp = dict(hidden=args.hidden, layers=args.layers, drop=args.drop, lr=args.lr, wd=args.wd)

    d = torch.load(f"{HERE}/data.pt")
    S = d["S"].to(DEV); y = d["y"].to(DEV); n = d["n"].to(DEV)
    w = torch.log1p(n); w = w / w.mean()
    yn = y.cpu().numpy()
    Xg = d["Xgroup"].numpy()
    rxn_comps = d["rxn_comps"]; n_comp = d["n_comp"]; N = len(d["rxn_ids"])
    graphs = {lvl: Graph(d["graphs"][lvl]) for lvl in args.levels.split(",")}

    print(f"device={DEV}  N={N}  hp={hp}  lam={args.lam}  ens={args.ens}\n")

    schemes = [("RANDOM", kfold(N, args.folds, args.seed)),
               ("CPD-DISJOINT", compound_disjoint(rxn_comps, n_comp, args.folds, args.seed))]
    out = {}
    for scheme, folds in schemes:
        rows = {}
        # predict-zero + linear prior
        e_zero, e_lin = [], []
        lin_pred_full = np.zeros(N)   # store OOF linear preds for delta
        for tr, te in folds:
            if len(te) == 0:
                continue
            coef = ridge_fit(Xg, yn, tr, args.lam)
            lp = Xg @ coef
            lin_pred_full[te] = lp[te]
            e_zero.append(np.abs(yn[te]))
            e_lin.append(np.abs(lp[te] - yn[te]))
        rows["predict-zero"] = mae(np.concatenate(e_zero))
        rows["linear (group CC prior)"] = mae(np.concatenate(e_lin))

        for lvl, g in graphs.items():
            e_gnn, e_delta = [], []
            for tr, te in folds:
                if len(te) == 0:
                    continue
                te_t = torch.as_tensor(te, device=DEV)
                # standalone GNN
                p = gnn_predict(g, S, y, w, tr, args.epochs, args.seed, args.ens, hp)
                e_gnn.append((p[te_t] - y[te_t]).abs().cpu().numpy())
                # delta: GNN on residual over the linear prior (refit on tr only)
                coef = ridge_fit(Xg, yn, tr, args.lam)
                lp = Xg @ coef
                resid = torch.as_tensor(yn - lp, dtype=torch.float32, device=DEV)
                pr = gnn_predict(g, S, resid, w, tr, args.epochs, args.seed + 7, args.ens, hp)
                final = torch.as_tensor(lp, device=DEV) + pr
                e_delta.append((final[te_t] - y[te_t]).abs().cpu().numpy())
            rows[f"GNN[{lvl}]"] = mae(np.concatenate(e_gnn))
            rows[f"GNN-delta[{lvl}]"] = mae(np.concatenate(e_delta))

        print(f"=== {scheme} (held-out) ===")
        for k, (m, r) in rows.items():
            print(f"  {k:<28s} MAE {m:6.2f}  RMSE {r:6.2f}")
        print()
        out[scheme] = {k: {"mae": m, "rmse": r} for k, (m, r) in rows.items()}
    json.dump(out, open(f"{HERE}/results_v2.json", "w"), indent=2)
    print("wrote results_v2.json")


if __name__ == "__main__":
    main()
