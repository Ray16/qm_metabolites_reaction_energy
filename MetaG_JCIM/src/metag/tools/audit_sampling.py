"""Audit expected independent-seed replicas without averaging away a failed seed.

python -m metag.tools.audit_sampling records.json --expected expected.json \
    --max-equilibrium-fold-change 2 -o sampling_audit.json

Expected JSON maps reaction IDs to seed offsets (integers) or explicit replica
IDs (strings). Input is an envelope, envelope list, or a bundle with ``records``
and optional ``expected_replicates``. No identity is guessed from a filename.
"""
import argparse
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
from metag.execution.filesystem import atomic_write_json
from metag.validation.sampling_replicates import audit_sampling_replicates


def _read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(path.read_text(), parse_constant=reject)


def input_records(value):
    """Read explicit envelopes without inferring scientific/replica metadata."""
    if isinstance(value, list):
        rows = value
    elif isinstance(value, Mapping) and "records" in value:
        rows = value["records"]
    elif isinstance(value, Mapping) and "reaction_id" in value:
        rows = [value]
    else:
        raise ValueError("Input must contain explicit replica envelopes, as a list or records bundle")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("Replica records must be a JSON list of objects")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--expected", type=Path, help="Explicit expected reaction/replica manifest")
    parser.add_argument("--max-equilibrium-fold-change", type=float,
                        help="Maximum observed equilibrium-constant ratio; >=1; omitted means descriptive only")
    parser.add_argument("--require-within-tolerance", action="store_true",
                        help="After writing the audit, exit 1 unless every expected set meets the declared tolerance")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.require_within_tolerance and args.max_equilibrium_fold_change is None:
        parser.error("--require-within-tolerance needs --max-equilibrium-fold-change")
    if args.output.exists():
        parser.error("Output exists; choose a new evidence path")
    paths = args.inputs + ([args.expected] if args.expected else [])
    if args.output.resolve() in {path.resolve() for path in paths}:
        parser.error("Output must not replace an input")
    try:
        values = [_read(path) for path in args.inputs]
        rows = [row for value in values for row in input_records(value)]
        embedded = [value["expected_replicates"] for value in values
                    if isinstance(value, Mapping) and "expected_replicates" in value]
        if args.expected:
            expected = _read(args.expected)
            if isinstance(expected, Mapping) and "expected_replicates" in expected:
                expected = expected["expected_replicates"]
        elif embedded:
            expected = embedded[0]
        else:
            raise ValueError("Supply --expected or an explicit expected_replicates bundle")
        if any(value != expected for value in embedded):
            raise ValueError("Conflicting expected-replicate manifests")
        report = audit_sampling_replicates(rows, expected,
                    maximum_equilibrium_fold_change=args.max_equilibrium_fold_change)
        from metag.validation import sampling_replicates as implementation
        from metag.validation import network_consistency, transfer
        sources = paths + [Path(__file__), Path(implementation.__file__),
                           Path(network_consistency.__file__), Path(transfer.__file__)]
        report["sources_sha256"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        json.dumps(report, allow_nan=False)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    atomic_write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2))
    if args.require_within_tolerance and not report["summary"]["all_within_observed_replicate_tolerance"]:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()
