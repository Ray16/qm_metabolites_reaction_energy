"""Audit native pipeline checkpoints without changing energies or references.

python -m metag.tools.audit_network checkpoint.json -o network_audit.json
Older checkpoints lacking original_species may supply --reactions input.json.
Missing predictions remain in the panel when reaction inputs are supplied.
"""
import argparse
import hashlib
import json
from pathlib import Path

from metag.execution.filesystem import atomic_write_json


def checkpoint_records(checkpoint, reactions=None):
    if isinstance(checkpoint, list):
        rows = checkpoint
    elif isinstance(checkpoint, dict) and ("reaction" in checkpoint or "id" in checkpoint):
        rows = [checkpoint]
    elif isinstance(checkpoint, dict):
        rows = checkpoint.get("results", checkpoint)
    else:
        raise ValueError("Expected a reaction checkpoint list or mapping")
    if isinstance(rows, dict):
        if any(not isinstance(value, dict) for value in rows.values()):
            raise ValueError("Results mapping must contain reaction records")
        rows = [dict(value, reaction=key) for key, value in rows.items()]
    if not isinstance(rows, list):
        raise ValueError("Expected a checkpoint list or results mapping")
    output, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Checkpoint entries must be reaction records")
        result = row.get("result")
        record = dict(result) if isinstance(result, dict) else dict(row)
        rid = record.get("reaction", row.get("reaction", record.get("id", row.get("id"))))
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("Checkpoint requires unique reaction identifiers")
        if reactions is not None and rid not in reactions:
            raise ValueError("Checkpoint reaction outside supplied panel: " + rid)
        seen.add(rid)
        record["reaction"] = rid
        if result is not None or "result" in row:
            record["status"] = row.get("status", record.get("status"))
            record["reason"] = row.get("error", record.get("reason"))
        if "original_species" not in record and reactions is not None:
            record["original_species"] = reactions[rid]["species"]
        # Quantity/conditions/configuration are never invented for historical rows.
        output.append(record)
    if reactions is not None:
        output.extend({"reaction": rid, "original_species": reaction["species"],
                       "status": "missing_prediction", "dG": None, "reason": "no checkpoint result"}
                      for rid, reaction in reactions.items() if rid not in seen)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--reactions", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists; choose a new evidence path")
    from metag.validation.network_consistency import audit_network
    reactions = json.loads(args.reactions.read_text()) if args.reactions else None
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    records = checkpoint_records(json.loads(args.checkpoint.read_text()), reactions)
    report = audit_network(records, calibration_artifact=calibration)
    inputs = [p for p in (args.checkpoint, args.reactions, args.calibration, Path(__file__)) if p]
    report["sources_sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    atomic_write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
