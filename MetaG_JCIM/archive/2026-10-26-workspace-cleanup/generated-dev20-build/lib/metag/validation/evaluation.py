"""Prediction evaluation retaining coverage, failures and unresolved references."""
import math

import numpy as np


def _finite(value):
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def evaluate_predictions(references, predictions, *, severe_threshold_kj=20.0):
    """Evaluate one method on a fixed panel; no-estimates remain in denominators.

    References are keyed records with ``eligible`` and ``dG``. Prediction
    records use ``dG``, optional ``suspect``, ``sigma_pred`` and ``ci95``.
    Point estimates with explicit failure flags are counted as withheld.
    Warnings/diagnostics remain visible; warnings alone do not choose acceptance.
    Empirical interval inclusion is reported without assuming calibration.
    """
    if not _finite(severe_threshold_kj) or severe_threshold_kj <= 0:
        raise ValueError("Severe-error threshold must be positive and finite")
    unknown = set(predictions) - references.keys()
    if unknown:
        raise ValueError(f"Predictions outside the fixed panel: {sorted(unknown)}")
    records = []
    for rid, ref in references.items():
        eligible = bool(ref.get("eligible", False)) and _finite(ref.get("dG"))
        pred = predictions.get(rid)
        if pred is None:
            status = "missing_prediction"
        elif (pred.get("suspect") or pred.get("accepted") is False
              or pred.get("status") not in (None, "ok", "estimated", "success")):
            status = "withheld_prediction"
        elif not _finite(pred.get("dG")):
            status = "invalid_or_no_estimate"
        else:
            status = "estimated"
        if status == "estimated" and "conditions" in ref:
            if pred.get("conditions") != ref["conditions"]:
                status = "condition_mismatch"
        row = {"reaction": rid, "reference_eligible": eligible, "prediction_status": status,
               "evaluated": eligible and status == "estimated", "error_kj": None}
        if pred is not None:
            route = pred.get("routes") or {}
            warnings = route.get("warnings", [])
            row["warnings"] = [warnings] if isinstance(warnings, str) else list(warnings)
            row["failure_reason"] = pred.get("suspect") or pred.get("reason") or pred.get("error")
            row["source_status"] = pred.get("source_status", pred.get("status"))
            row["diagnostics"] = {key: route[key] for key in (
                "species_diagnostics", "hydration_diagnostics", "incomplete_hydration",
                "errors", "reverted") if key in route}
            row["training_exposure"] = pred.get("training_exposure")
        if row["evaluated"]:
            row["error_kj"] = float(pred["dG"]) - float(ref["dG"])
            sigma = pred.get("sigma_pred")
            row["sigma_pred"] = float(sigma) if _finite(sigma) and float(sigma) >= 0 else None
            interval = pred.get("ci95")
            valid = (isinstance(interval, (tuple, list)) and len(interval) == 2
                     and all(_finite(x) for x in interval) and float(interval[0]) <= float(interval[1]))
            row["interval_contains_reference"] = (
                float(interval[0]) <= float(ref["dG"]) <= float(interval[1]) if valid else None)
            row["interval_width_kj"] = float(interval[1]) - float(interval[0]) if valid else None
            row["interval_calibration_claimed"] = bool((pred.get("ci_info") or {}).get("coverage_calibrated", False))
        records.append(row)
    usable = [row for row in records if row["evaluated"]]
    errors = np.asarray([row["error_kj"] for row in usable])
    intervals = [row for row in usable if row["interval_contains_reference"] is not None]
    eligible_count = sum(row["reference_eligible"] for row in records)
    counts = {status: sum(row["prediction_status"] == status for row in records)
              for status in ("estimated", "missing_prediction", "withheld_prediction",
                             "invalid_or_no_estimate", "condition_mismatch")}
    severe = int(np.sum(np.abs(errors) > severe_threshold_kj))
    summary = {"n_panel": len(records), "n_eligible_references": eligible_count,
               "n_unresolved_references": len(records) - eligible_count,
               "n_evaluated": len(usable), "prediction_status_counts": counts,
               "evaluation_coverage": len(usable) / eligible_count if eligible_count else None,
               "mae_kj": float(np.mean(np.abs(errors))) if errors.size else None,
               "median_ae_kj": float(np.median(np.abs(errors))) if errors.size else None,
               "rmse_kj": float(np.sqrt(np.mean(errors ** 2))) if errors.size else None,
               "bias_kj": float(np.mean(errors)) if errors.size else None,
               "severe_threshold_kj": severe_threshold_kj, "n_severe_errors": severe,
               "severe_error_fraction_of_evaluated": severe / len(usable) if usable else None,
               "n_intervals": len(intervals),
               "empirical_interval_coverage": (sum(row["interval_contains_reference"] for row in intervals)
                                               / len(intervals)) if intervals else None,
               "mean_interval_width_kj": float(np.mean([row["interval_width_kj"] for row in intervals])) if intervals else None,
               "scope": "descriptive evaluation; empirical inclusion does not establish held-out interval coverage"}
    return {"summary": summary, "records": records}


def selective_error_curve(evaluation, *, severe_threshold_kj=20.0):
    """Accept by predicted uncertainty only; tied uncertainties stay together.

    Coverage uses all eligible references, including failures and estimates
    without uncertainty. Thresholds are descriptive, never tuned to test errors.
    """
    if not _finite(severe_threshold_kj) or severe_threshold_kj <= 0:
        raise ValueError("Severe-error threshold must be positive and finite")
    rows = [row for row in evaluation["records"] if row["evaluated"]
            and row.get("sigma_pred") is not None]
    denominator = evaluation["summary"]["n_eligible_references"]
    out = []
    for threshold in sorted({row["sigma_pred"] for row in rows}):
        accepted = [row for row in rows if row["sigma_pred"] <= threshold]
        errors = np.asarray([row["error_kj"] for row in accepted])
        out.append({"sigma_threshold_kj": threshold, "n_accepted": len(accepted),
                    "coverage_of_eligible": len(accepted) / denominator if denominator else None,
                    "mae_kj": float(np.mean(np.abs(errors))),
                    "severe_error_fraction": float(np.mean(np.abs(errors) > severe_threshold_kj))})
    return out
