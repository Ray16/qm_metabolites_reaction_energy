#!/usr/bin/env python
"""GNN for per-compound formation energy, trained on TECRDB reaction dG.

"dGPredictor done right": instead of a linear group-additivity lookup, a
message-passing GNN maps each compound's molecular graph -> a scalar formation
energy f(compound). A reaction's dG is the stoichiometric difference S @ f.
The GNN never sees per-compound labels (none exist); it is trained end-to-end
on reaction dG only. QM physics is injected as a per-compound graph feature
(the precomputed ALPB solvation free energy dGsolv), which is exactly the
signal absolute QM gets wrong but which is informative as a *feature* the model
calibrates against the experimental scale.

Data (all in ../../pipeline):
  tecrdb_full_reactions.json    rxn_id -> {cpd_id: stoich}     (367 rxns)
  tecrdb_full_metabolites.json  [{id, smiles, charge, ...}]    (453 cpds)
  tecrdb_full_experiment.json   rxn_id -> {dG_kJ, n, sd_kJ}    (targets)
  ensemble_tecrdb_full.json     cpd_id -> [{dGsolv_kJ, G_RRHO_kJ, ...}]  (QM)

Run:  python gnn_dgf.py            # full CV comparison, both schemes, ablation
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.normpath(os.path.join(HERE, "..", "..", "pipeline"))
torch.set_num_threads(min(16, os.cpu_count() or 8))


# --------------------------------------------------------------------------- #
# Featurization
# --------------------------------------------------------------------------- #
ATOM_VOCAB = ["C", "N", "O", "P", "S", "H", "F", "Cl", "Br", "I", "Co", "*"]
HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
       Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
       Chem.HybridizationType.SP3D2]
BOND_VOCAB = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
              Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


def _onehot(x, vocab):
    v = [0.0] * (len(vocab) + 1)
    v[vocab.index(x)] = 1.0 if x in vocab else 0.0
    if x not in vocab:
        v[-1] = 1.0
    return v


def atom_features(atom):
    f = _onehot(atom.GetSymbol(), ATOM_VOCAB)
    f += _onehot(atom.GetHybridization(), HYB)
    f += [
        atom.GetDegree() / 4.0,
        atom.GetTotalNumHs() / 4.0,
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        atom.GetTotalValence() / 6.0,
    ]
    return f


def bond_features(bond):
    return _onehot(bond.GetBondType(), BOND_VOCAB) + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


ATOM_DIM = len(atom_features(Chem.MolFromSmiles("CC").GetAtomWithIdx(0)))
BOND_DIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


class CompoundGraphs:
    """All compounds packed into one disconnected graph for scatter MPNN.

    QM feature levels (ablation ladder):
      'none' : molecular graph only (no QM physics)
      'solv' : + graph-level dGsolv (ALPB solvation free energy)
      'full' : + per-atom xtb Mulliken charge (node) + HOMO/LUMO/gap (graph)
    """

    def __init__(self, mets, ensemble, qmfeat=None, level="full"):
        self.ids = [m["id"] for m in mets]
        self.idx = {c: i for i, c in enumerate(self.ids)}
        qmfeat = qmfeat or {}
        node_feats, batch, src, dst, edge_feats = [], [], [], [], []
        offset = 0
        qm = []
        use_atom_qm = level == "full"
        for i, m in enumerate(mets):
            mol = Chem.MolFromSmiles(m["smiles"])
            n = mol.GetNumAtoms()
            qf = qmfeat.get(m["id"]) or {}
            mull = qf.get("mulliken") or []
            for a in mol.GetAtoms():
                af = atom_features(a)
                if use_atom_qm:
                    q = mull[a.GetIdx()] if a.GetIdx() < len(mull) else 0.0
                    af = af + [q]                      # QM partial charge node feat
                node_feats.append(af)
                batch.append(i)
            for b in mol.GetBonds():
                u, v = b.GetBeginAtomIdx() + offset, b.GetEndAtomIdx() + offset
                bf = bond_features(b)
                src += [u, v]; dst += [v, u]; edge_feats += [bf, bf]
            for a in range(n):                          # self-loops
                src.append(a + offset); dst.append(a + offset)
                edge_feats.append([0.0] * BOND_DIM)
            offset += n
            # graph-level QM vector
            e = (ensemble.get(m["id"]) or [{}])[0]
            dgsolv = e.get("dGsolv_kJ")
            gvec = []
            if level in ("solv", "full"):
                gvec.append((dgsolv if dgsolv is not None else 0.0) / 100.0)
            if level == "full":
                gvec += [
                    (qf.get("homo") if qf.get("homo") is not None else 0.0),
                    (qf.get("lumo") if qf.get("lumo") is not None else 0.0),
                    (qf.get("gap") if qf.get("gap") is not None else 0.0) / 5.0,
                ]
            if not gvec:                                # 'none' -> single zero col
                gvec = [0.0]
            qm.append(gvec)
        self.atom_dim = len(node_feats[0])
        self.x = torch.tensor(node_feats, dtype=torch.float32)
        self.batch = torch.tensor(batch, dtype=torch.long)
        self.edge_index = torch.tensor([src, dst], dtype=torch.long)
        self.edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        self.qm = torch.tensor(qm, dtype=torch.float32)
        self.n_comp = len(self.ids)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
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
            m = torch.relu(msg(h)[src] + e)               # message per edge
            agg = torch.zeros_like(h).index_add_(0, dst, m)  # sum into targets
            h = h + upd(torch.cat([h, agg], dim=-1))      # residual update
        # sum-readout per compound
        pooled = torch.zeros(g.n_comp, h.size(1)).index_add_(0, g.batch, h)
        f = self.readout(torch.cat([pooled, g.qm], dim=-1)).squeeze(-1)
        return f  # (n_comp,) formation energies


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
def load():
    rxns = json.load(open(f"{PIPE}/tecrdb_full_reactions.json"))
    mets = json.load(open(f"{PIPE}/tecrdb_full_metabolites.json"))
    exp = json.load(open(f"{PIPE}/ensemble_tecrdb_full.json"))
    tgt = json.load(open(f"{PIPE}/tecrdb_full_experiment.json"))
    rxn_ids = [r for r in rxns if r in tgt]
    return rxns, mets, exp, tgt, rxn_ids


def stoich_matrix(rxns, rxn_ids, cidx):
    S = torch.zeros(len(rxn_ids), len(cidx))
    for i, r in enumerate(rxn_ids):
        for c, v in rxns[r].items():
            S[i, cidx[c]] = v
    return S


def targets(tgt, rxn_ids):
    y = torch.tensor([tgt[r]["dG_kJ"] for r in rxn_ids], dtype=torch.float32)
    n = torch.tensor([tgt[r].get("n", 1) for r in rxn_ids], dtype=torch.float32)
    return y, n


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def _train_once(g, S, y, w, tr, te, epochs, lr, wd, seed, patience=40):
    torch.manual_seed(seed)
    model = MPNN(g.atom_dim, BOND_DIM, g.qm.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # inner val split of the training reactions for early stopping
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tr))
    nval = max(4, len(tr) // 6)
    vi, ti = tr[perm[:nval]], tr[perm[nval:]]
    Sti, yti, wti = S[ti], y[ti], w[ti]
    Sv, yv = S[vi], y[vi]
    best = (1e9, None, 0)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        pred = Sti @ model(g)
        loss = (wti * (pred - yti) ** 2).mean()
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
        return (S @ model(g))[te]


def train_fold(g, S, y, w, tr, te, epochs=600, lr=3e-3, wd=1e-4, seed=0, n_ens=3):
    """Seed-ensemble of early-stopped models; return |pred-truth| on test."""
    preds = torch.stack([
        _train_once(g, S, y, w, tr, te, epochs, lr, wd, seed + 100 * s)
        for s in range(n_ens)]).mean(0)
    return (preds - y[te]).abs()


def kfold_indices(n, k, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [perm[i::k] for i in range(k)]


def compound_disjoint_folds(rxns, rxn_ids, cidx, k, seed):
    """Assign compounds to k groups; a reaction tests in fold j only if it
    touches a compound of group j and none from other test-eligible groups.
    Train = reactions with no held-out compound. Strict extrapolation."""
    rng = np.random.default_rng(seed)
    comp_group = {c: int(rng.integers(k)) for c in cidx}
    folds = []
    for j in range(k):
        held = {c for c, gp in comp_group.items() if gp == j}
        te, tr = [], []
        for i, r in enumerate(rxn_ids):
            cs = set(rxns[r])
            if cs & held:
                te.append(i)
            else:
                tr.append(i)
        folds.append((np.array(tr), np.array(te)))
    return folds


def ridge_baseline(S, y, tr, te, feats, lam=10.0):
    """Component-contribution analog: reaction feature = S @ compound_feats,
    ridge-fit to dG. Linear group additivity, the honest baseline."""
    X = (S @ feats).numpy()
    Xtr, ytr = X[tr], y[tr].numpy()
    A = Xtr.T @ Xtr + lam * np.eye(X.shape[1])
    b = Xtr.T @ ytr
    coef = np.linalg.solve(A, b)
    pe = X[te] @ coef
    return np.abs(pe - y[te].numpy())


def compound_count_feats(mets):
    """Atom-type-count fingerprint per compound (group-contribution style)."""
    rows = []
    for m in mets:
        mol = Chem.MolFromSmiles(m["smiles"])
        c = {s: 0 for s in ATOM_VOCAB}
        for a in mol.GetAtoms():
            s = a.GetSymbol()
            c[s if s in c else "*"] += 1
        rows.append([c[s] for s in ATOM_VOCAB] + [float(m.get("charge", 0)), 1.0])
    return torch.tensor(rows, dtype=torch.float32)


def report(name, err):
    err = np.asarray(err)
    print(f"  {name:<34s} MAE {err.mean():6.2f}   RMSE {np.sqrt((err**2).mean()):6.2f}   "
          f"|err|>20: {(err>20).mean()*100:4.1f}%")
    return err.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rxns, mets, exp, tgt, rxn_ids = load()
    cidx = {m["id"]: i for i, m in enumerate(mets)}
    S = stoich_matrix(rxns, rxn_ids, cidx)
    y, n = targets(tgt, rxn_ids)
    w = torch.log1p(n); w = w / w.mean()
    feats = compound_count_feats(mets)

    qmf_path = f"{HERE}/qm_features.json"
    qmf = json.load(open(qmf_path)) if os.path.exists(qmf_path) else {}
    graphs = {lvl: CompoundGraphs(mets, exp, qmf, level=lvl)
              for lvl in ("none", "solv", "full")}
    if not qmf:
        print("[warn] qm_features.json missing -> 'full' level has no xtb feats")

    print(f"reactions={len(rxn_ids)}  compounds={len(mets)}  "
          f"predict-zero MAE={y.abs().mean():.2f}  "
          f"xtb feats for {sum(1 for v in qmf.values() if v and v.get('mulliken'))} cpds\n")

    results = {}
    for scheme, folds in [
        ("RANDOM 5-fold (interpolation)",
         [(np.setdiff1d(np.arange(len(rxn_ids)), te), te)
          for te in kfold_indices(len(rxn_ids), args.folds, args.seed)]),
        ("COMPOUND-DISJOINT (extrapolation)",
         compound_disjoint_folds(rxns, rxn_ids, cidx, args.folds, args.seed)),
    ]:
        print(f"=== {scheme} ===")
        acc = {k: [] for k in ("zero", "ridge", "none", "solv", "full")}
        for fi, (tr, te) in enumerate(folds):
            if len(te) == 0:
                continue
            acc["zero"].append(y[te].abs().numpy())
            acc["ridge"].append(ridge_baseline(S, y, tr, te, feats))
            for lvl in ("none", "solv", "full"):
                acc[lvl].append(train_fold(graphs[lvl], S, y, w, tr, te,
                                           epochs=args.epochs,
                                           seed=args.seed + fi).numpy())
        cat = lambda L: np.concatenate(L)
        results[scheme] = {}
        results[scheme]["predict-zero"] = report("predict-zero", cat(acc["zero"]))
        results[scheme]["ridge-linear"] = report("ridge (linear group additivity)", cat(acc["ridge"]))
        results[scheme]["gnn-graph-only"] = report("GNN (graph only)", cat(acc["none"]))
        results[scheme]["gnn+dGsolv"] = report("GNN + dGsolv", cat(acc["solv"]))
        results[scheme]["gnn+xtb(full)"] = report("GNN + xtb charges+orbitals", cat(acc["full"]))
        print()
    json.dump(results, open(f"{HERE}/results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
