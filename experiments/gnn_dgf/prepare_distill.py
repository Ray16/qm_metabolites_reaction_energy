#!/usr/bin/env python
"""Build the eQuilibrator-distillation dataset.

Train the GNN on eQ predictions for ~9k confident ModelSEED reactions
(uncertainty < 15 kJ), evaluate on the 367 TECRDB *experimental* reactions
(held out). Tests whether more training signal (25x the data) breaks the
n=367 ceiling AND yields a universal-coverage model.

Featurizes graph + RDKit descriptors only (no xtb -- thousands of compounds).
Saves distill_data.pt for the GPU trainer.
"""
import csv, glob, json, os, sys
import numpy as np
import torch
from rdkit import Chem, RDLogger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnn_dgf as G

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.normpath(os.path.join(HERE, "..", "..", "..", "ModelSEEDDatabase", "Biochemistry"))
PIPE = G.PIPE
PROTON = "cpd00067"
UNC_MAX = 15.0
csv.field_size_limit(1 << 24)

# --- compound SMILES + charge from ModelSEED shards -----------------------
smiles, charge = {}, {}
for f in sorted(glob.glob(f"{DB}/compound_*.tsv")):
    with open(f) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            s = row.get("smiles") or ""
            if s and s != "null":
                smiles[row["id"]] = s
                charge[row["id"]] = int(float(row.get("charge") or 0))
print(f"compounds with SMILES: {len(smiles)}")

# --- reaction stoichiometry from ModelSEED shards -------------------------
def parse_stoich(s):
    out = {}
    for term in s.split(";"):
        p = term.split(":")
        if len(p) >= 2:
            try:
                out[p[1]] = out.get(p[1], 0.0) + float(p[0])
            except ValueError:
                pass
    return out

rxn_stoich = {}
for f in sorted(glob.glob(f"{DB}/reaction_*.tsv")):
    with open(f) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("stoichiometry"):
                rxn_stoich[row["id"]] = parse_stoich(row["stoichiometry"])
print(f"reactions parsed: {len(rxn_stoich)}")

# --- eQ labels (confident) ------------------------------------------------
eq = json.load(open(f"{HERE}/../../results/eq/modelseed_all_dG.json"))

# --- TECRDB test set (experimental) --------------------------------------
tec_rxn = json.load(open(f"{PIPE}/tecrdb_full_reactions.json"))
tec_exp = json.load(open(f"{PIPE}/tecrdb_full_experiment.json"))
tec_mets = json.load(open(f"{PIPE}/tecrdb_full_metabolites.json"))
for m in tec_mets:                      # ensure TECRDB compounds have SMILES/charge
    smiles.setdefault(m["id"], m["smiles"]); charge.setdefault(m["id"], m.get("charge", 0))
test_ids = list(tec_rxn.keys())
test_set = set(test_ids)

def usable(st):
    cs = [c for c in st if c != PROTON]
    return cs and all(c in smiles and Chem.MolFromSmiles(smiles[c]) is not None for c in cs)

# training reactions: confident eQ, parseable, NOT in the TECRDB test set
train_ids = []
for rid, v in eq.items():
    if rid in test_set:
        continue
    if v.get("dG_prime_kJ") is None or (v.get("uncertainty_kJ") or 99) >= UNC_MAX:
        continue
    st = rxn_stoich.get(rid)
    if st and usable(st):
        train_ids.append(rid)
print(f"train reactions (eQ, unc<{UNC_MAX}, disjoint from test): {len(train_ids)}")

# --- unified compound index ----------------------------------------------
def stoich_of(rid, src):
    return {c: v for c, v in (src[rid].items()) if c != PROTON}

used = set()
for rid in train_ids:
    used |= set(stoich_of(rid, rxn_stoich))
for rid in test_ids:
    used |= set(c for c in tec_rxn[rid])
used = [c for c in sorted(used) if c in smiles and Chem.MolFromSmiles(smiles[c])]
cidx = {c: i for i, c in enumerate(used)}
print(f"unified compounds: {len(used)}")

# --- featurize compounds (graph + descriptors, no xtb) --------------------
mets = [{"id": c, "smiles": smiles[c], "charge": charge.get(c, 0)} for c in used]
graph = G.CompoundGraphs(mets, ensemble={}, qmfeat={}, level="rich", cpcmx_solv={})
gpack = dict(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr,
             qm=graph.qm, batch=graph.batch, atom_dim=graph.atom_dim,
             n_comp=graph.n_comp, bond_dim=G.BOND_DIM)

def build_S(ids, src, get):
    S = torch.zeros(len(ids), len(used))
    y = torch.zeros(len(ids))
    for i, rid in enumerate(ids):
        for c, v in get(rid).items():
            if c in cidx:
                S[i, cidx[c]] = v
        y[i] = src(rid)
    return S, y

S_tr, y_tr = build_S(train_ids, lambda r: eq[r]["dG_prime_kJ"],
                     lambda r: stoich_of(r, rxn_stoich))
S_te, y_te = build_S(test_ids, lambda r: tec_exp[r]["dG_kJ"],
                     lambda r: {c: v for c, v in tec_rxn[r].items() if c != PROTON})

torch.save(dict(graph=gpack, S_tr=S_tr, y_tr=y_tr, S_te=S_te, y_te=y_te,
                train_ids=train_ids, test_ids=test_ids),
           f"{HERE}/distill_data.pt")
print(f"saved distill_data.pt  train={len(train_ids)} test={len(test_ids)} "
      f"compounds={len(used)}  |  train dG sd={y_tr.std():.1f} test dG sd={y_te.std():.1f}")
