"""Audit an existing measurement manifest and holdout definitions on CPU.

Usage: python -m metag.tools.audit_transfer manifest.json splits.json -o audit.json
No prediction values, model weights or experimental-reference rewrites enter.
"""
import argparse
import hashlib
import json
from pathlib import Path

from metag.execution.filesystem import atomic_write_json
def build_audit(manifest, splits, *, check_span=False):
    from metag.validation.transfer import audit_holdout, training_span_exposure

    training = manifest["training"]
    index = {row["row"]: row for row in training}
    evaluation = manifest["evaluation"]
    reports = []
    for category in ("reaction_family_folds", "compound_folds"):
        for fold in splits.get(category, ()):
            if fold.get("status", "defined") != "defined":
                reports.append({"id": fold["id"], "status": fold["status"], "passed": None})
                continue
            targets = [{"id": rid, **evaluation[rid]} for rid in fold["test_reactions"]]
            result = audit_holdout(training, targets, fold["train_rows"],
                                   heldout_compounds=fold.get("heldout_compounds", ()),
                                   heldout_families=([fold["family"]] if "family" in fold else ()))
            result.update(id=fold["id"], status="audited", category=category)
            if check_span:
                result["training_span"] = {}
                for target in targets:
                    equivalents = target.get("published_training_rows", ())
                    pairs = target.get("signature")
                    if pairs is None and equivalents:
                        pairs = index[equivalents[0]]["signature"]
                    result["training_span"][target["id"]] = (
                        training_span_exposure(training, fold["train_rows"], pairs)
                        if pairs is not None else {"status": "target_stoichiometry_not_in_manifest"})
            reports.append(result)
    return {"scope": splits.get("protocol", "unspecified split protocol"),
            "summary": {"n_folds": len(reports), "n_passed": sum(r["passed"] is True for r in reports),
                        "n_failed": sum(r["passed"] is False for r in reports),
                        "n_undefined": sum(r["passed"] is None for r in reports)},
            "normalization": "manifest baseline normalization; explicit H canonicalized; stereo collapsed for exposure",
            "folds": reports,
            "limitations": ["A passed split audit is not prospective validation.",
                            "MetaG has prior development exposure to TECRDB.",
                            "Family holdouts retaining formation labels can constrain familiar-compound reactions."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("splits", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--span", action="store_true", help="Also audit Hess-law training-span exposure")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; choose a new output to preserve evidence")
    report = build_audit(json.loads(args.manifest.read_text()), json.loads(args.splits.read_text()),
                         check_span=args.span)
    report["sources_sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (args.manifest, args.splits, Path(__file__))}
    atomic_write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2))
    return 1 if report["summary"]["n_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
