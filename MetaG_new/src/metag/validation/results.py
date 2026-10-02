"""Validation helpers for cached and assembled results."""

import numpy as np


def cache_settings_match(record_settings, required, source):
    """Require an exact physics generation and requested solvation provenance."""
    actual = dict(record_settings)
    via = actual.pop("via", None)
    expected_via = None if source == "primary" else "solv_also"
    return via == expected_via and actual == required


def error_metrics(results):
    """Summarize valid estimates, excluding suspect and missing records."""
    errors = np.asarray(
        [
            row["err"]
            for row in results.values()
            if row["err"] is not None and row["suspect"] is None
        ],
        dtype=float,
    )
    if not len(errors):
        return {"n": 0, "mae": None, "median_ae": None, "rmse": None, "bias": None}
    return {
        "n": len(errors),
        "mae": round(float(np.mean(np.abs(errors))), 3),
        "median_ae": round(float(np.median(np.abs(errors))), 3),
        "rmse": round(float(np.sqrt(np.mean(errors**2))), 3),
        "bias": round(float(np.mean(errors)), 3),
    }

