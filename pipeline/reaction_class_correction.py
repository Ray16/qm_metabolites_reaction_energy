#!/usr/bin/env python
"""Train/evaluate a leakage-safe reaction-class residual correction.

The input is a CSV with ``reaction_id,predicted_kJ,experimental_kJ`` and
optionally ``reaction_class,signature``.  Classes may instead be supplied in a
JSON mapping.  This is an experimental calibration layer, never the reported
QM baseline.  It refuses to fit sparse classes in held-out scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)
from qm_thermo.reaction_correction import CalibrationRow, leave_signature_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--classes", default=os.path.join(HERE, "reaction_classes.json"))
    parser.add_argument("--out", default=os.path.join(THERMO, "results", "benchmark",
                                                        "reaction_class_oof.json"))
    parser.add_argument("--min-signatures", type=int, default=4)
    parser.add_argument("--shrinkage", type=float, default=3.0)
    args = parser.parse_args()

    overrides = json.load(open(args.classes)) if args.classes else {}
    rows = []
    for value in csv.DictReader(open(args.input)):
        reaction_id = value["reaction_id"]
        reaction_class = value.get("reaction_class") or overrides.get(reaction_id)
        if not reaction_class:
            raise ValueError(f"{reaction_id}: no reaction class; add it to --classes or input")
        rows.append(CalibrationRow(
            reaction_id=reaction_id,
            signature=value.get("signature") or reaction_id,
            reaction_class=reaction_class,
            predicted_kJ=float(value["predicted_kJ"]),
            experimental_kJ=float(value["experimental_kJ"]),
        ))
    scored = leave_signature_out(rows, min_signatures=args.min_signatures,
                                 shrinkage=args.shrinkage)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"method": "leave-signature-out reaction-class residual correction",
               "min_signatures": args.min_signatures, "shrinkage": args.shrinkage,
               "rows": scored}, open(args.out, "w"), indent=2)
    n = sum(row["calibrated"] for row in scored)
    print(f"wrote {args.out}; calibrated OOF rows: {n}/{len(scored)}")


if __name__ == "__main__":
    main()
