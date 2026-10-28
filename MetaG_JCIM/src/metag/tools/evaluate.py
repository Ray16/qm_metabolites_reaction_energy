"""Evaluate a fixed reference panel with explicit conditions and method provenance.

Usage: python -m metag.tools.evaluate panel.json -o evaluation.json
Input: {references: {id: {eligible, dG, conditions, source}},
        methods: {name: {provenance, predictions: {id: prediction_record}}}}.
"""
import argparse
import hashlib
import json
from pathlib import Path

from metag.execution.filesystem import atomic_write_json
def evaluate_panel(panel):
    from metag.validation.evaluation import evaluate_predictions, selective_error_curve
    from metag.validation.evaluation_conditions import condition_issues

    references = panel["references"]
    for rid, reference in references.items():
        if reference.get("eligible") and (not reference.get("conditions") or not reference.get("source")):
            raise ValueError(f"Eligible reference {rid} requires conditions and source provenance")
        if reference.get("eligible") and condition_issues(reference["conditions"]):
            raise ValueError(f"Eligible reference {rid} has incomplete conditions: {condition_issues(reference['conditions'])}")
    methods = {}
    for name, method in panel["methods"].items():
        if not method.get("provenance"):
            raise ValueError(f"Method {name} requires explicit provenance")
        issues_by_reaction = {rid: issues for rid, pred in method["predictions"].items()
            if pred.get("dG") is not None and (issues := condition_issues(pred.get("conditions")))}
        checked_predictions = {}
        for rid, pred in method["predictions"].items():
            checked = dict(pred)
            if rid in issues_by_reaction:
                checked.update(accepted=False, status="invalid_conditions",
                               source_status=pred.get("source_status", pred.get("status")),
                               reason="Invalid comparison conditions: " + ", ".join(issues_by_reaction[rid]))
            checked_predictions[rid] = checked
        report = evaluate_predictions(references, checked_predictions)
        report["condition_schema_issues"] = issues_by_reaction
        report["selective_error"] = selective_error_curve(report)
        report["provenance"] = method["provenance"]
        methods[name] = report
    comparison_methods = panel.get("comparison_methods", list(methods))
    if len(comparison_methods) != len(set(comparison_methods)) or set(comparison_methods) - methods.keys():
        raise ValueError("Comparison methods must be unique and present in the panel")
    common = set(references) if comparison_methods else set()
    for name in comparison_methods:
        report = methods[name]
        common &= {row["reaction"] for row in report["records"] if row["evaluated"]}
    matched = {}
    for name in comparison_methods:
        method = panel["methods"][name]
        matched[name] = evaluate_predictions({rid: references[rid] for rid in sorted(common)},
                                            {rid: method["predictions"][rid] for rid in sorted(common)})["summary"]
    return {"scope": panel.get("scope", "descriptive evaluation with explicit conditions"),
            "methods": methods, "comparison_methods": comparison_methods,
            "common_evaluated_reactions": sorted(common),
            "common_set_metrics": matched,
            "limitations": ["Method-specific coverage and common-set accuracy answer different questions.",
                            "References and predictions with different conditions are not paired.",
                            "Uncertainty curves do not select a production threshold from test errors."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists; use a new destination to preserve evidence")
    report = evaluate_panel(json.loads(args.panel.read_text()))
    report["sources_sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (args.panel, Path(__file__))}
    atomic_write_json(args.output, report)
    print(json.dumps({name: result["summary"] for name, result in report["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
