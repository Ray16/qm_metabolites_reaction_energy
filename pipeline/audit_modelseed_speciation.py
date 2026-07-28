#!/usr/bin/env python
"""Build a review queue from ModelSEED's pKa/pKb annotations.

This is a triage tool.  ModelSEED stores ChemAxon-predicted atom-level values;
the output must not be used as a thermodynamic population model until a family
has independent pKa/site validation and explicitly calculated microspecies.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)
from qm_thermo.modelseed_pka import load_compounds, sites_near_window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reactions", default=os.path.join(HERE, "reactions.json"))
    parser.add_argument("--metadata", default=os.path.join(HERE, "top10_reactions_stereo_significant.csv"))
    parser.add_argument("--modelseed", default=os.path.join(os.path.dirname(THERMO), "ModelSEEDDatabase"))
    parser.add_argument("--margin", type=float, default=1.0,
                        help="pKa units added around the reported pH window")
    parser.add_argument("--out", default=os.path.join(THERMO, "results", "benchmark",
                                                        "modelseed_speciation_audit.json"))
    args = parser.parse_args()

    reactions = json.load(open(args.reactions))
    metadata = {row["modelseed_rxn"]: row for row in csv.DictReader(open(args.metadata))}
    compound_ids = {compound for stoich in reactions.values() for compound in stoich}
    compounds = load_compounds(args.modelseed, compound_ids)

    memberships = defaultdict(list)
    for reaction_id, stoich in reactions.items():
        row = metadata[reaction_id]
        for compound, coefficient in stoich.items():
            memberships[compound].append({"reaction_id": reaction_id, "coefficient": coefficient,
                                          "pH_min": float(row["pH_min"]),
                                          "pH_max": float(row["pH_max"])})
    queue = []
    for compound_id in sorted(compound_ids):
        compound = compounds.get(compound_id)
        if compound is None:
            queue.append({"compound_id": compound_id, "status": "missing_from_modelseed"})
            continue
        hits = []
        for membership in memberships[compound_id]:
            near = sites_near_window(tuple(compound["sites"]), membership["pH_min"],
                                     membership["pH_max"], args.margin)
            for site in near:
                hits.append({**membership, "kind": site.kind, "fragment": site.fragment,
                             "atom": site.atom, "value": site.value})
        if hits:
            queue.append({"compound_id": compound_id, "name": compound["name"],
                          "stored_charge": compound["charge"], "status": "review_required",
                          "source": "ModelSEED atom-level ChemAxon prediction",
                          "reason": "candidate transition near a measured pH window; site/value unvalidated",
                          "candidate_sites": hits})
    queue.sort(key=lambda row: (-len(row.get("candidate_sites", ())), row["compound_id"]))
    output = {"scope": "triage only; no pKa values were used to score reactions",
              "margin_pKa_units": args.margin, "n_compounds": len(compound_ids),
              "n_review_required": len(queue), "review_queue": queue}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(output, open(args.out, "w"), indent=2)
    print(f"ModelSEED pKa/pKb triage: {len(queue)}/{len(compound_ids)} compounds need review")
    for row in queue:
        values = sorted({hit["value"] for hit in row.get("candidate_sites", ())})
        print(f"{row['compound_id']:10} {row.get('name', '?')[:30]:30} candidates={values}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
