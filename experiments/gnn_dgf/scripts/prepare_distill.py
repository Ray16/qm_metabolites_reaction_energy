#!/usr/bin/env python
"""Build artifacts/distill_data.pt: ~9k confident eQ-labeled ModelSEED reactions
(train) + the 367 TECRDB experimental reactions (test), shared compound index.
Graph + RDKit descriptors only (no xtb for thousands of compounds). Base env.
"""
import _bootstrap  # noqa: F401
import csv
import glob
import json

import numpy as np
import torch
from rdkit import Chem, RDLogger

from gnn import paths
from gnn.features import CompoundGraphs

RDLogger.DisableLog("rdApp.*")
csv.field_size_limit(1 << 24)
PROTON = "cpd00067"
UNC_MAX = 15.0

smiles, charge = {}, {}
for f in sorted(glob.glob(f"{paths.DB}/compound_*.tsv")):
    for row in csv.DictReader(open(f), delimiter="\t"):
        s = row.get("smiles") or ""
        if s and s != "null":
            smiles[row["id"]] = s
            charge[row["id"]] = int(float(row.get("charge") or 0))


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


rxn_stoich = {row["id"]: parse_stoich(row["stoichiometry"])
              for f in sorted(glob.glob(f"{paths.DB}/reaction_*.tsv"))
              for row in csv.DictReader(open(f), delimiter="\t") if row.get("stoichiometry")}
eq = json.load(open(f"{paths.RESULTS}/eq/modelseed_all_dG.json"))
tec_rxn = json.load(open(f"{paths.PIPE}/tecrdb_full_reactions.json"))
tec_exp = json.load(open(f"{paths.PIPE}/tecrdb_full_experiment.json"))
for m in json.load(open(f"{paths.PIPE}/tecrdb_full_metabolites.json")):
    smiles.setdefault(m["id"], m["smiles"]); charge.setdefault(m["id"], m.get("charge", 0))
test_ids = list(tec_rxn); test_set = set(test_ids)


def usable(st):
    cs = [c for c in st if c != PROTON]
    return cs and all(c in smiles and Chem.MolFromSmiles(smiles[c]) is not None for c in cs)


train_ids = [rid for rid, v in eq.items()
             if rid not in test_set and v.get("dG_prime_kJ") is not None
             and (v.get("uncertainty_kJ") or 99) < UNC_MAX and rxn_stoich.get(rid)
             and usable(rxn_stoich[rid])]
print(f"train={len(train_ids)} test={len(test_ids)}")


def sof(rid):
    return {c: v for c, v in rxn_stoich[rid].items() if c != PROTON}


used = set()
for rid in train_ids:
    used |= set(sof(rid))
for rid in test_ids:
    used |= set(tec_rxn[rid])
used = [c for c in sorted(used) if c in smiles and Chem.MolFromSmiles(smiles[c])]
cidx = {c: i for i, c in enumerate(used)}
mets = [{"id": c, "smiles": smiles[c], "charge": charge.get(c, 0)} for c in used]
gpack = CompoundGraphs(mets, level="rich").pack()


def build(ids, tgt, get):
    S = torch.zeros(len(ids), len(used)); y = torch.zeros(len(ids))
    for i, rid in enumerate(ids):
        for c, v in get(rid).items():
            if c in cidx:
                S[i, cidx[c]] = v
        y[i] = tgt(rid)
    return S, y


S_tr, y_tr = build(train_ids, lambda r: eq[r]["dG_prime_kJ"], sof)
S_te, y_te = build(test_ids, lambda r: tec_exp[r]["dG_kJ"],
                   lambda r: {c: v for c, v in tec_rxn[r].items() if c != PROTON})
torch.save(dict(graph=gpack, S_tr=S_tr, y_tr=y_tr, S_te=S_te, y_te=y_te,
                train_ids=train_ids, test_ids=test_ids), paths.artifact("distill_data.pt"))
print(f"saved distill_data.pt  compounds={len(used)}  train sd={y_tr.std():.0f} test sd={y_te.std():.0f}")
