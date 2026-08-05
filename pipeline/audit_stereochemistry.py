#!/usr/bin/env python
"""Audit the stereochemical integrity of the structures we score.

This answers three questions, in order of how badly they break a prediction:

1. Which reactions are **degenerate by construction** -- both sides carry the
   identical structure, so the computed energy is exactly zero no matter what
   method produces it?  These are not predictions and must not be scored.
2. Which compounds carry **diastereomeric ambiguity** -- an unresolved centre
   that changes the free energy?  Enantiomeric ambiguity is reported separately
   because mirror images share a free energy in water and cost nothing.
3. Which distinct compound ids **collide onto one structure**?  Some are
   genuine synonyms; the inventory is for curation, not an accusation.

Counting raw "undefined stereocentres" would drown all three in false
positives: every nucleotide reports phosphate phosphorus atoms that are
resonance equivalent, not stereogenic.  ``qm_thermo.stereochemistry`` excludes
those explicitly and this script reports how many it removed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)

from qm_thermo import stereochemistry as stereo  # noqa: E402
from qm_thermo.modelseed_pka import load_compounds  # noqa: E402


def _load_reactions(paths: list[str]) -> dict[str, dict[str, float]]:
    reactions: dict[str, dict[str, float]] = {}
    for path in paths:
        with open(path) as handle:
            payload = json.load(handle)
        for reaction_id, stoich in payload.items():
            reactions.setdefault(reaction_id, {k: float(v) for k, v in stoich.items()})
    return reactions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reactions", nargs="+",
                        default=[os.path.join(HERE, "reactions.json"),
                                 os.path.join(HERE, "bench226_reactions.json")])
    parser.add_argument("--scored", default=os.path.join(HERE, "bench226_scored.json"),
                        help="restrict the reaction report to reactions actually scored")
    parser.add_argument("--modelseed",
                        default=os.path.join(os.path.dirname(THERMO), "ModelSEEDDatabase"))
    parser.add_argument("--out", default=os.path.join(THERMO, "results", "benchmark",
                                                      "stereochemistry_audit.json"))
    args = parser.parse_args()

    reactions = _load_reactions(args.reactions)
    compound_ids = {compound for stoich in reactions.values() for compound in stoich}
    compounds = load_compounds(args.modelseed, compound_ids)
    structures = {cid: row["smiles"] for cid, row in compounds.items()}
    names = {cid: row["name"] for cid, row in compounds.items()}

    assessments = stereo.assess_many(structures)
    ambiguous = stereo.ambiguous_compounds(assessments)
    degenerate = stereo.degenerate_reactions(reactions, structures)
    collisions = stereo.find_collisions(structures, names)
    affected = stereo.affected_reactions(reactions, ambiguous)

    scored: set[str] = set()
    if args.scored and os.path.exists(args.scored):
        with open(args.scored) as handle:
            scored = {row["r"] for row in json.load(handle)}

    tally = Counter(a.ambiguity for a in assessments.values())
    artifacts_removed = sum(len(a.artifacts) for a in assessments.values())
    missing = sorted(compound_ids - set(compounds))

    report = {
        "scope": "stereochemical integrity of scored structures; no configuration is inferred",
        "n_reactions": len(reactions),
        "n_compounds": len(compound_ids),
        "n_compounds_missing_from_modelseed": len(missing),
        "compounds_missing_from_modelseed": missing,
        "phantom_stereocentres_excluded": artifacts_removed,
        "ambiguity_tally": dict(tally),
        "degenerate_reactions": {
            "note": "identical structure multiset on both sides; dG is 0 by construction",
            "n_total": len(degenerate),
            "n_scored": len(set(degenerate) & scored) if scored else None,
            "reactions": {rid: {"shared_inchikey": key,
                                "scored": rid in scored if scored else None,
                                "compounds": {c: names.get(c, "") for c in reactions[rid]}}
                          for rid, key in sorted(degenerate.items())},
        },
        "diastereomeric_compounds": {
            cid: {**assessments[cid].summary(), "name": names.get(cid, "")}
            for cid in ambiguous
        },
        "enantiomeric_only_compounds": sorted(
            cid for cid, a in assessments.items()
            if a.ambiguity == stereo.AMBIGUITY_ENANTIOMERIC),
        "reactions_touching_ambiguous_compounds": {
            rid: list(hits) for rid, hits in sorted(affected.items())
            if not scored or rid in scored
        },
        "structure_collisions": [c.as_dict() for c in collisions],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"compounds assessed              {len(compounds)}/{len(compound_ids)}")
    print(f"phantom centres excluded        {artifacts_removed} "
          f"(resonance-equivalent P/S oxo centres)")
    for kind in (stereo.AMBIGUITY_NONE, stereo.AMBIGUITY_ENANTIOMERIC,
                 stereo.AMBIGUITY_DIASTEREOMERIC):
        print(f"  ambiguity={kind:16s}      {tally.get(kind, 0)}")
    n_scored_deg = len(set(degenerate) & scored) if scored else 0
    print(f"degenerate reactions            {len(degenerate)}"
          + (f"  ({n_scored_deg} of them scored)" if scored else ""))
    for rid in sorted(degenerate):
        mark = " [SCORED]" if rid in scored else ""
        print(f"    {rid}{mark}: "
              + " = ".join(", ".join(names.get(c, c) for c, v in reactions[rid].items()
                                     if (v < 0) == first)
                           for first in (True, False)))
    print(f"structure collisions            {len(collisions)}")
    print(f"diastereomerically ambiguous    {len(ambiguous)}")
    for cid in ambiguous:
        summary = assessments[cid].summary()
        print(f"    {cid:10s} {names.get(cid, '')[:36]:38s} "
              f"{summary['n_real_undefined']} undefined "
              f"({summary['anomeric_undefined']} anomeric)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
