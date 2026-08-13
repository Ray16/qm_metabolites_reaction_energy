#!/usr/bin/env python
"""Build artifacts/data.pt (base env, needs rdkit).

Packs the 367 TECRDB reactions / 453 compounds into tensors: 4 feature levels
(none/solv/full/rich), stoichiometry S, target y, measurement counts n, ridge
features, the dGPredictor group-difference prior Xgroup, and the eQ/dGP baselines.
"""
import _bootstrap  # noqa: F401
import glob
import json

import numpy as np
import torch

from gnn import data, paths
from gnn.features import CompoundGraphs, compound_count_feats, BOND_DIM

rxns, mets, ens, tgt, rxn_ids = data.load_tecrdb()
cidx = {m["id"]: i for i, m in enumerate(mets)}
S = data.stoich_matrix(rxns, rxn_ids, cidx)
y, n, sd = data.targets(tgt, rxn_ids)
feats = compound_count_feats(mets)
qmf = json.load(open(paths.artifact("qm_features.json")))

cpcmx_raw = json.load(open(f"{paths.PIPE}/cpcmx_dgsolv_tecrdb_full.json"))
cpcmx = {k: sum(v) / len(v) for k, v in cpcmx_raw.items() if v}

graphs = {}
for lvl in ("none", "solv", "full", "rich"):
    graphs[lvl] = CompoundGraphs(mets, ens, qmf, level=lvl, cpcmx_solv=cpcmx).pack()

gf = json.load(open(f"{paths.RESULTS}/eq/dgp_group_features.json"))
Xg = np.array([gf["X"][r] for r in rxn_ids], float)
Xg = Xg[:, Xg.any(axis=0)]

baselines = {}
for name, fn in [("eQuilibrator", "equilibrator_full.json"),
                 ("dGP-standard", "dgpredictor_full.json"),
                 ("dGP-retrained", "dgpredictor_retrained_full.json")]:
    d = json.load(open(f"{paths.RESULTS}/eq/{fn}"))
    baselines[name] = [d.get(r, {}).get("dG_kJ") for r in rxn_ids]

rxn_comps = [[cidx[c] for c in rxns[r]] for r in rxn_ids]
torch.save(dict(graphs=graphs, S=S, y=y, n=n, sd=sd, feats=feats, rxn_ids=rxn_ids,
                n_comp=len(mets), rxn_comps=rxn_comps, baselines=baselines,
                Xgroup=torch.tensor(Xg, dtype=torch.float32)),
           paths.artifact("data.pt"))
print(f"saved data.pt  reactions={len(rxn_ids)} compounds={len(mets)} "
      f"levels={list(graphs)} Xgroup={tuple(Xg.shape)} cpcmx={len(cpcmx)}")
