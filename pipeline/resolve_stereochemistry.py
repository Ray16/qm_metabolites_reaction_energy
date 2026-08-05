#!/usr/bin/env python
"""Turn diastereomeric ambiguity into explicit, computable isomer states.

An undefined anomeric centre is not averaged by the conformer generator -- it
is resolved arbitrarily and differently for different conformers, so a single
"ensemble" silently mixes two substances (measured: D-glucose 6-phosphate
embeds 14 conformers across both anomers, 5/9, seed-dependent).  This script
replaces that with named states that can each be computed on their own.

Every enumerated state is identified by matching its InChIKey against the full
ModelSEED compound table, so an anomer is labelled by ModelSEED's own
alpha/beta entry rather than by a guess made here.  Equilibrium populations are
deliberately left ``null``: they must be curated from measurement before the
states can be collapsed into one number.  Until then the states carry an
explicit spread instead of a hidden arbitrary choice.

Output feeds the ensemble builder; no quantum chemistry runs here.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)

from qm_thermo import stereochemistry as stereo  # noqa: E402
from qm_thermo.modelseed_pka import load_compounds  # noqa: E402


def _database_index(database_root: str) -> dict[str, list[dict[str, str]]]:
    """Map InChIKey -> ModelSEED records, so enumerated states can be named."""
    index: dict[str, list[dict[str, str]]] = {}
    pattern = os.path.join(database_root, "Biochemistry", "compound_*.tsv")
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                key = (row.get("inchikey") or "").strip()
                if key:
                    index.setdefault(key, []).append(
                        {"id": row["id"], "name": row["name"],
                         "is_obsolete": row.get("is_obsolete", "0")})
    return index


def _load_reactions(paths: list[str]) -> dict[str, dict[str, float]]:
    reactions: dict[str, dict[str, float]] = {}
    for path in paths:
        with open(path) as handle:
            for reaction_id, stoich in json.load(handle).items():
                reactions.setdefault(reaction_id, {k: float(v) for k, v in stoich.items()})
    return reactions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reactions", nargs="+",
                        default=[os.path.join(HERE, "reactions.json"),
                                 os.path.join(HERE, "bench226_reactions.json")])
    parser.add_argument("--modelseed",
                        default=os.path.join(os.path.dirname(THERMO), "ModelSEEDDatabase"))
    parser.add_argument("--max-isomers", type=int, default=8)
    parser.add_argument("--out", default=os.path.join(HERE, "stereo_resolved_structures.json"))
    args = parser.parse_args()

    reactions = _load_reactions(args.reactions)
    compound_ids = {compound for stoich in reactions.values() for compound in stoich}
    compounds = load_compounds(args.modelseed, compound_ids)
    structures = {cid: row["smiles"] for cid, row in compounds.items()}

    assessments = stereo.assess_many(structures)
    ambiguous = stereo.ambiguous_compounds(assessments)
    index = _database_index(args.modelseed)

    resolved: dict[str, dict] = {}
    for compound_id in ambiguous:
        assessment = assessments[compound_id]
        states = []
        for smiles in stereo.enumerate_resolved(assessment.smiles, args.max_isomers):
            key = stereo.structure_key(smiles)
            matches = [m for m in index.get(key, []) if m["id"] != compound_id]
            preferred = next((m for m in matches if m["is_obsolete"] in ("0", "", "False")),
                            matches[0] if matches else None)
            states.append({
                "label": preferred["name"] if preferred else f"unnamed_{key[:14]}",
                "smiles": smiles,
                "inchikey": key,
                "modelseed_match": preferred["id"] if preferred else None,
                "other_modelseed_matches": [m["id"] for m in matches
                                            if not preferred or m["id"] != preferred["id"]],
            })
        resolved[compound_id] = {
            "name": compounds[compound_id]["name"],
            "parent_smiles": assessment.smiles,
            "ambiguity": assessment.ambiguity,
            "undefined_centres": [element.detail for element in assessment.undefined],
            "anomeric": bool(assessment.anomeric_undefined),
            "state_labels": [state["label"] for state in states],
            "states": states,
            "populations": None,
            "source": "",
            "citation": "",
            "status": "populations_required",
        }

    payload = {
        "scope": ("explicit isomer states for compounds whose undefined stereochemistry "
                  "changes a free energy; populations must be curated before use"),
        "n_compounds": len(resolved),
        "compounds": resolved,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"diastereomerically ambiguous compounds resolved: {len(resolved)}")
    for compound_id, record in resolved.items():
        print(f"\n  {compound_id}  {record['name']}"
              f"{'  [anomeric]' if record['anomeric'] else ''}")
        for state in record["states"]:
            match = state["modelseed_match"] or "no ModelSEED entry"
            print(f"      {state['label'][:52]:54s} {match}")
    print(f"\nwrote {args.out}")
    print("populations are null by design -- curate them before collapsing the states")


if __name__ == "__main__":
    main()
