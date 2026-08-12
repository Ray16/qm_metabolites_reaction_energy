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
    """All compounds packed into one disconnected graph for scatter MPNN."""

    def __init__(self, mets, ensemble, use_qm=True):
        self.ids = [m["id"] for m in mets]
        self.idx = {c: i for i, c in enumerate(self.ids)}
        node_feats, batch, src, dst, edge_feats = [], [], [], [], []
        offset = 0
        qm = []
        for i, m in enumerate(mets):
            mol = Chem.MolFromSmiles(m["smiles"])
            mol = Chem.AddHs(mol) if False else mol  # heavy-atom graph
            n = mol.GetNumAtoms()
            for a in mol.GetAtoms():
                node_feats.append(atom_features(a))
                batch.append(i)
            for b in mol.GetBonds():
                u, v = b.GetBeginAtomIdx() + offset, b.GetEndAtomIdx() + offset
                bf = bond_features(b)
                src += [u, v]; dst += [v, u]; edge_feats += [bf, bf]
            # self-loops so isolated atoms (single-atom species) still update
            for a in range(n):
                src.append(a + offset); dst.append(a + offset)
                edge_feats.append([0.0] * BOND_DIM)
            offset += n
            # per-compound QM + trivial descriptors
            e = ensemble.get(m["id"]) or [{}]
            dgsolv = e[0].get("dGsolv_kJ")
            qm.append([
                (dgsolv if dgsolv is not None else 0.0) / 100.0,
                float(m.get("charge", 0)),
                n / 50.0,
            ])
        self.x = torch.tensor(node_feats, dtype=torch.float32)
        self.batch = torch.tensor(batch, dtype=torch.long)
        self.edge_index = torch.tensor([src, dst], dtype=torch.long)
        self.edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        qm = torch.tensor(qm, dtype=torch.float32)
        if not use_qm:
            qm = qm[:, 1:2] * 0.0  # keep shape (n,1) of zeros -> QM ablated
        self.qm = qm
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
def train_fold(g, S, y, w, tr, te, epochs=400, lr=3e-3, wd=1e-4, seed=0):
    torch.manual_seed(seed)
    model = MPNN(ATOM_DIM, BOND_DIM, g.qm.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Str, ytr, wtr = S[tr], y[tr], w[tr]
    Ste, yte = S[te], y[te]
    best = (1e9, None)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        f = model(g)
        pred = Str @ f
        loss = (wtr * (pred - ytr) ** 2).mean()
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pe = Ste @ model(g)
    return (pe - yte).abs()


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

    g_qm = CompoundGraphs(mets, exp, use_qm=True)
    g_noqm = CompoundGraphs(mets, exp, use_qm=False)

    print(f"reactions={len(rxn_ids)}  compounds={len(mets)}  "
          f"predict-zero MAE={y.abs().mean():.2f}\n")

    for scheme, folds in [
        ("RANDOM 5-fold (interpolation)",
         [(np.setdiff1d(np.arange(len(rxn_ids)), te), te)
          for te in kfold_indices(len(rxn_ids), args.folds, args.seed)]),
        ("COMPOUND-DISJOINT (extrapolation)",
         compound_disjoint_folds(rxns, rxn_ids, cidx, args.folds, args.seed)),
    ]:
        print(f"=== {scheme} ===")
        e_zero, e_ridge, e_gnn, e_gnn0 = [], [], [], []
        for fi, (tr, te) in enumerate(folds):
            if len(te) == 0:
                continue
            e_zero.append(y[te].abs().numpy())
            e_ridge.append(ridge_baseline(S, y, tr, te, feats))
            e_gnn.append(train_fold(g_qm, S, y, w, tr, te,
                                    epochs=args.epochs, seed=args.seed + fi).numpy())
            e_gnn0.append(train_fold(g_noqm, S, y, w, tr, te,
                                     epochs=args.epochs, seed=args.seed + fi).numpy())
        cat = lambda L: np.concatenate(L)
        report("predict-zero", cat(e_zero))
        report("ridge (linear group additivity)", cat(e_ridge))
        report("GNN (graph only, QM ablated)", cat(e_gnn0))
        report("GNN + QM feature (dGsolv)", cat(e_gnn))
        print()


if __name__ == "__main__":
    main()
