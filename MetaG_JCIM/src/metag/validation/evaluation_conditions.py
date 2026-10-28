"""Completeness checks for publication-oriented reaction-energy comparisons.

Declarations are required before comparing numbers. This validates the schema,
not the accuracy of a condition transform or the provenance of its constants.
"""
import math
from collections.abc import Mapping

NUMERIC_FIELDS = ("temperature_K", "pH", "ionic_strength_M", "pMg", "solute_standard_state_M")
CONVENTIONS = {"water_standard_state": "pure_liquid", "energy_units": "kJ/mol",
               "quantity": "standard_transformed_reaction_gibbs"}


def condition_issues(conditions):
    if not isinstance(conditions, Mapping):
        return ["missing_condition_record"]
    issues = []
    for field in NUMERIC_FIELDS:
        value = conditions.get(field)
        try:
            valid = not isinstance(value, bool) and value is not None and math.isfinite(float(value))
        except (TypeError, ValueError):
            valid = False
        if not valid:
            issues.append("missing_or_invalid_" + field)
        elif (field in ("temperature_K", "solute_standard_state_M") and float(value) <= 0
              or field == "ionic_strength_M" and float(value) < 0):
            issues.append("invalid_" + field)
    for field, expected in CONVENTIONS.items():
        if conditions.get(field) != expected:
            issues.append("missing_or_unsupported_" + field)
    return issues
