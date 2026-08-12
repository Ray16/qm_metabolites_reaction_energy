#!/usr/bin/env python
"""GPU trainer for the per-compound formation-energy GNN (rdkit-free).

Loads precomputed tensors (data.pt) and evaluates the GNN under two CV schemes
against predict-zero, a linear group-additivity ridge, and the real incumbents
(eQuilibrator, dGPredictor standard + retrained). Runs on CUDA.

Honesty notes baked into the report:
  * eQ / dGP numbers are IN-SAMPLE (both were trained on TECRDB) -> optimistic.
  * The GNN numbers are held-out CV. Compound-disjoint CV is the extrapolation
    test that proxies the out-of-coverage regime where the new method must earn
    its keep; random CV is the interpolation regime.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Graph:
    def __init__(self, d):
        self.x = d["x"].to(DEV); self.edge_index = d["edge_index"].to(DEV)
        self.edge_attr = d["edge_attr"].to(DEV); self.qm = d["qm"].to(DEV)
        self.batch = d["batch"].to(DEV)
        self.atom_dim = d["atom_dim"]; self.bond_dim = d["bond_dim"]
        self.n_comp = d["n_comp"]


class MPNN(nn.Module):
    def __init__(self, atom_dim, bond_dim, qm_dim, hidden=64, layers=3, drop=0.1):
        super().__init__()
        self.embed = nn.Linear(atom_dim, hidden)
        self.edge = nn.Linear(bond_dim, hidden)
        self.msg = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.upd = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                          nn.Dropout(drop), nn.Linear(hidden, hidden))
            for _ in range(layers))
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
        return self.readout(torch.cat([pooled, g.qm], dim=-1)).squeeze(-1)


def _train_once(g, S, y, w, tr, te, epochs, lr, wd, seed, patience=40):
    torch.manual_seed(seed)
    model = MPNN(g.atom_dim, g.bond_dim, g.qm.size(1)).to(DEV)
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
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                vmae = (Sv @ model(g) - yv).abs().mean().item()
            if vmae < best[0] - 1e-4:
                best = (vmae, {k: v.clone() for k, v in model.state_dict().items()}, ep)
            elif ep - best[2] > patience * 5:
                break
    if best[1] is not None:
        model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        return (S @ model(g)).detach()


def train_fold(g, S, y, w, tr, te, epochs, seed, n_ens=3, lr=3e-3, wd=1e-4):
    te_t = torch.as_tensor(te, device=DEV)
    preds = torch.stack([_train_once(g, S, y, w, tr, te, epochs, lr, wd, seed + 100 * s)
                         for s in range(n_ens)]).mean(0)
    return (preds[te_t] - y[te_t]).abs().cpu().numpy()


def ridge_baseline(S, y, feats, tr, te, lam=10.0):
    X = (S.cpu() @ feats).numpy()
    yn = y.cpu().numpy()
    A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
    coef = np.linalg.solve(A, X[tr].T @ yn[tr])
    return np.abs(X[te] @ coef - yn[te])


def kfold(n, k, seed):
    perm = np.random.default_rng(seed).permutation(n)
    return [perm[i::k] for i in range(k)]


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


def report(name, err, note=""):
    err = np.asarray(err, float)
    line = (f"  {name:<32s} MAE {err.mean():6.2f}   RMSE {np.sqrt((err**2).mean()):6.2f}"
            f"   |err|>20 {(err>20).mean()*100:4.1f}%   n={len(err)}  {note}")
    print(line)
    return float(err.mean())


def baseline_err(pred_list, y, idx):
    """abs error of a fixed per-reaction baseline over reaction indices idx,
    skipping reactions the baseline could not score (None)."""
    yn = y.cpu().numpy()
    e = [abs(pred_list[i] - yn[i]) for i in idx if pred_list[i] is not None]
    return np.array(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ens", type=int, default=3)
    args = ap.parse_args()

    d = torch.load(f"{HERE}/data.pt")
    S = d["S"].to(DEV); y = d["y"].to(DEV); n = d["n"].to(DEV)
    w = torch.log1p(n); w = w / w.mean()
    feats = d["feats"]
    graphs = {lvl: Graph(gd) for lvl, gd in d["graphs"].items()}
    rxn_comps = d["rxn_comps"]; n_comp = d["n_comp"]; base = d["baselines"]
    N = len(d["rxn_ids"])
    allidx = list(range(N))

    print(f"device={DEV}  reactions={N}  compounds={n_comp}  "
          f"predict-zero MAE={y.abs().mean():.2f}\n")

    print("=== INCUMBENTS on all 367 (IN-SAMPLE — trained on TECRDB, optimistic) ===")
    for name, preds in base.items():
        e = baseline_err(preds, y, allidx)
        report(name + " [in-sample]", e, note=f"scored {len(e)}/{N}")
    print()

    results = {}
    schemes = [
        ("RANDOM 5-fold (interpolation)",
         [(np.setdiff1d(np.arange(N), te), te) for te in kfold(N, args.folds, args.seed)]),
        ("COMPOUND-DISJOINT (extrapolation)",
         compound_disjoint(rxn_comps, n_comp, args.folds, args.seed)),
    ]
    for scheme, folds in schemes:
        print(f"=== {scheme} ===")
        acc = {k: [] for k in ("zero", "ridge", "none", "solv", "full")}
        test_union = []
        for fi, (tr, te) in enumerate(folds):
            if len(te) == 0:
                continue
            test_union += list(te)
            acc["zero"].append(y[torch.as_tensor(te, device=DEV)].abs().cpu().numpy())
            acc["ridge"].append(ridge_baseline(S, y, feats, tr, te))
            for lvl in ("none", "solv", "full"):
                acc[lvl].append(train_fold(graphs[lvl], S, y, w, tr, te,
                                           args.epochs, args.seed + fi, n_ens=args.ens))
        cat = np.concatenate
        r = {}
        r["predict-zero"] = report("predict-zero", cat(acc["zero"]))
        r["ridge-linear"] = report("ridge (linear group additivity)", cat(acc["ridge"]))
        r["gnn-graph-only"] = report("GNN graph-only [held-out]", cat(acc["none"]))
        r["gnn+dGsolv"] = report("GNN + dGsolv [held-out]", cat(acc["solv"]))
        r["gnn+xtb"] = report("GNN + xtb charges+orbitals [held-out]", cat(acc["full"]))
        # incumbents restricted to the exact tested reactions (still in-sample)
        for name, preds in base.items():
            e = baseline_err(preds, y, test_union)
            r[name + "[in-sample,ref]"] = report(name + " [in-sample ref]", e)
        results[scheme] = r
        print()

    json.dump(results, open(f"{HERE}/results_gpu.json", "w"), indent=2)
    print("wrote results_gpu.json")


if __name__ == "__main__":
    main()
