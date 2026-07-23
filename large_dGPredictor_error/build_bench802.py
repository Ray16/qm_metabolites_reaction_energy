#!/usr/bin/env python
"""Build QM inputs for the full stereo-exact TECRDB benchmark (226 reactions).

The 10-reaction set was too small to falsify anything: 8 independent points after
removing mirrored duplicates, dominated by 2 redox chemistries. This assembles
the whole stereo-exact-matched set so every later experiment is measurable.

Sources:
  stereo_exact_significant.csv  -- TECRDB vs dGPredictor, matched to ModelSEED,
                                   with equation_ids (stoichiometry), measured pH
                                   and dG. Curated by the collaborator.
  ModelSEEDDatabase Structures  -- "Charged" (pH-7) SMILES per compound id.

Writes bench802_{reactions,metabolites,species}.json in the same schema the
existing pipeline consumes, plus a report of what could not be built.

Run:  /homes/rzhu/miniforge3/envs/palm/bin/python build_bench226.py
"""
import csv
import json
import os
import re
from collections import Counter

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
STRUCT = os.path.join(ROOT, "ModelSEEDDatabase", "Biochemistry", "Structures",
                      "All_ModelSEED_Structures.txt")
# UNSELECTED stereo-exact set. stereo_exact_significant.csv is filtered to
# abs_diff > combined_err ("the confident-disagreement set"), i.e. selected ON
# dGPredictor error -- benchmarking against it would inherit that bias.
SRC = os.path.join(HERE, "tecrdb_vs_dgpredictor_modelseed.csv")
TIER = "stereo_exact"
PROTON = "cpd00067"
WATER = "cpd00001"

# "(1) cpd00006[0] + (2) cpd00042[0] <=> (1) cpd00005[0]"
TERM = re.compile(r"\((\d+(?:\.\d+)?)\)\s*(cpd\d+)")


def load_smiles():
    """cpd_id -> pH-7 ('Charged') SMILES, falling back to the neutral entry."""
    charged, neutral = {}, {}
    with open(STRUCT) as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) < 8 or f[1] != "SMILE":
                continue
            cid, kind, smi = f[0], f[2], f[7]
            (charged if kind == "Charged" else neutral)[cid] = smi
    return charged, neutral


def parse_equation(eq):
    """{cpd_id: coeff}; products positive, reactants negative. H+ dropped.

    ModelSEED writes directionality three ways (<=>, =>, <=). The arrow encodes
    reversibility, not stoichiometry, so all three are parsed identically -- the
    TECRDB dG already carries the measured direction.
    """
    arrow = next((a for a in ("<=>", "=>", "<=") if a in eq), None)
    if arrow is None:
        return None
    lhs, rhs = eq.split(arrow)
    st = {}
    for side, sign in ((lhs, -1.0), (rhs, +1.0)):
        for coeff, cid in TERM.findall(side):
            if cid == PROTON:
                continue
            st[cid] = st.get(cid, 0.0) + sign * float(coeff)
    return {k: v for k, v in st.items() if v != 0.0}


def main():
    charged, neutral = load_smiles()
    print(f"ModelSEED structures: {len(charged)} charged, {len(neutral)} neutral")

    rows = [r for r in csv.DictReader(open(SRC)) if r.get("match_tier") == TIER]
    reactions, wanted, skipped = {}, set(), Counter()
    meta_rows = {}
    for r in rows:
        rid = r["modelseed_rxn"]
        st = parse_equation(r["equation_ids"])
        if not st:
            skipped["unparseable equation"] += 1
            continue
        missing = [c for c in st if c not in charged and c not in neutral]
        if missing:
            skipped["no structure"] += 1
            continue
        reactions[rid] = st
        wanted |= set(st)
        meta_rows[rid] = r

    mets, bad = [], Counter()
    for cid in sorted(wanted):
        smi = charged.get(cid) or neutral.get(cid)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad["unparseable SMILES"] += 1
            continue
        mets.append(dict(id=cid, name=cid, smiles=smi,
                         charge=int(Chem.GetFormalCharge(mol)),
                         formula="", inchikey="", opentecr_species=""))
    ok = {m["id"] for m in mets}

    # drop reactions that lost a compound at the SMILES stage
    for rid in list(reactions):
        if not set(reactions[rid]) <= ok:
            del reactions[rid]
            skipped["compound failed RDKit"] += 1

    species = {}
    for m in mets:
        mol = Chem.AddHs(Chem.MolFromSmiles(m["smiles"]))
        species[m["id"]] = dict(name=m["name"], charge=m["charge"],
                                n_hydrogens=sum(1 for a in mol.GetAtoms()
                                                if a.GetSymbol() == "H"))

    json.dump(reactions, open(os.path.join(HERE, "bench802_reactions.json"), "w"), indent=1)
    json.dump(mets, open(os.path.join(HERE, "bench802_metabolites.json"), "w"), indent=1)
    json.dump(species, open(os.path.join(HERE, "bench802_species.json"), "w"), indent=1)
    keep = {k: v for k, v in meta_rows.items() if k in reactions}
    with open(os.path.join(HERE, "bench802_meta.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(keep.values())

    print(f"\nreactions built : {len(reactions)} / {len(rows)}")
    for k, v in skipped.items():
        print(f"  skipped ({k}): {v}")
    print(f"unique metabolites: {len(mets)}")
    sizes = Counter(Chem.MolFromSmiles(m['smiles']).GetNumHeavyAtoms() // 10 * 10
                    for m in mets)
    print("heavy-atom histogram:",
          " ".join(f"{k}-{k+9}:{v}" for k, v in sorted(sizes.items())))
    print(f"water present in {sum(1 for st in reactions.values() if WATER in st)} reactions")


if __name__ == "__main__":
    main()
