#!/usr/bin/env python
"""Precompute all tensors (base env w/ rdkit) so training can run rdkit-free on GPU.

Dumps data.pt with: per-level compound graphs, stoichiometry S, target y,
measurement counts n, ridge features, rxn/compound bookkeeping, and the
eQuilibrator / dGPredictor baseline predictions aligned to rxn_ids.
"""
import json, os, torch
import gnn_dgf as G

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = G.PIPE
RES = os.path.normpath(os.path.join(HERE, "..", "..", "results", "eq"))

rxns, mets, exp, tgt, rxn_ids = G.load()
cidx = {m["id"]: i for i, m in enumerate(mets)}
S = G.stoich_matrix(rxns, rxn_ids, cidx)
y, n = G.targets(tgt, rxn_ids)
feats = G.compound_count_feats(mets)
qmf = json.load(open(f"{HERE}/qm_features.json"))

# CPCM-X solvation (better theory than ALPB) -> mean over conformers, kJ/mol
cpcmx_raw = json.load(open(f"{PIPE}/cpcmx_dgsolv_tecrdb_full.json"))
cpcmx = {k: (sum(v) / len(v)) for k, v in cpcmx_raw.items() if v}

graphs = {}
for lvl in ("none", "solv", "full", "rich"):
    g = G.CompoundGraphs(mets, exp, qmf, level=lvl, cpcmx_solv=cpcmx)
    graphs[lvl] = dict(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr,
                       qm=g.qm, batch=g.batch, atom_dim=g.atom_dim, n_comp=g.n_comp,
                       bond_dim=G.BOND_DIM)

# dGPredictor group-difference features (reaction-level) for Delta-learning prior
gf = json.load(open(f"{RES}/dgp_group_features.json"))
import numpy as _np
Xg = _np.array([gf["X"][r] for r in rxn_ids], float)
Xg = Xg[:, Xg.any(axis=0)]          # drop all-zero columns (as dGP retrain did)

# baselines aligned to rxn_ids (None where unscorable)
def load_pred(path):
    d = json.load(open(path))
    return [d.get(r, {}).get("dG_kJ") for r in rxn_ids]

baselines = {}
for name, fn in [("eQuilibrator", "equilibrator_full.json"),
                 ("dGP-standard", "dgpredictor_full.json"),
                 ("dGP-retrained", "dgpredictor_retrained_full.json")]:
    p = os.path.join(RES, fn)
    if os.path.exists(p):
        baselines[name] = load_pred(p)

# per-reaction compound-index lists (for compound-disjoint CV, rdkit-free)
rxn_comps = [[cidx[c] for c in rxns[r]] for r in rxn_ids]

torch.save(dict(graphs=graphs, S=S, y=y, n=n, feats=feats,
                rxn_ids=rxn_ids, n_comp=len(mets), rxn_comps=rxn_comps,
                baselines=baselines, Xgroup=torch.tensor(Xg, dtype=torch.float32)),
           f"{HERE}/data.pt")
print("saved data.pt  reactions", len(rxn_ids), "compounds", len(mets),
      "baselines", list(baselines),
      "| levels", list(graphs), "| Xgroup", tuple(Xg.shape),
      "| cpcmx cpds", len(cpcmx))
