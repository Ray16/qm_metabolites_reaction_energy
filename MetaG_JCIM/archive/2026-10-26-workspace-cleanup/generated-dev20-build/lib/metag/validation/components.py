"""Matched-coordinate component comparisons with explicit incomplete outcomes."""
import math

EH_TO_KJ = 2625.499639
MATCH_FIELDS = ("xyz_sha256", "charge", "spin_multiplicity", "functional", "basis", "grid_level", "nlc_grid_level")


def _energy(row, component):
    needed = ("gas",) if component == "gas" else ("gas", "smd")
    for name in needed:
        value = row.get(name + "_Eh")
        if (not row.get(name + "_converged") or value is None
                or not math.isfinite(float(value))):
            return None
    if not row.get("nlc_enabled"):
        return None
    if component != "gas" and (row.get("smd_cds_Eh") is None
                               or not math.isfinite(float(row["smd_cds_Eh"]))):
        return None
    if component == "gas":
        return float(row["gas_Eh"]) * EH_TO_KJ
    if component == "solution":
        return float(row["smd_Eh"]) * EH_TO_KJ
    if component == "solvation":
        return (float(row["smd_Eh"]) - float(row["gas_Eh"])) * EH_TO_KJ
    raise ValueError(f"Unknown component {component}")


def compare_geometry_pair(reference, check, *, tolerance_kj=0.2):
    """Compare second-minus-first gaps, never absolute cross-engine energies.

    Both ordered pairs must represent identical geometries/states and matching
    functional, basis and grids. Overlap treatment may differ intentionally;
    its provenance remains in the source records. A completed gas comparison
    cannot establish agreement of unfinished solvent calculations.
    """
    if len(reference) != 2 or len(check) != 2:
        raise ValueError("Exactly two matched geometries are required per method")
    if not math.isfinite(tolerance_kj) or tolerance_kj <= 0:
        raise ValueError("Tolerance must be positive and finite")
    mismatches = []
    for i, (left, right) in enumerate(zip(reference, check)):
        for field in MATCH_FIELDS:
            if left.get(field) is None or left.get(field) != right.get(field):
                mismatches.append({"geometry": i, "field": field})
    if mismatches:
        return {"matched_inputs": False, "mismatches": mismatches, "full_pair_agreement": False,
                "components": {}}
    components = {}
    for component in ("gas", "solvation", "solution"):
        baseline = [_energy(row, component) for row in reference]
        trial = [_energy(row, component) for row in check]
        if any(value is None for value in baseline + trial):
            components[component] = {"status": "incomplete", "passed": None}
            continue
        before, after = baseline[1] - baseline[0], trial[1] - trial[0]
        delta = after - before
        components[component] = {"status": "compared", "reference_gap_kj": before,
                                 "check_gap_kj": after, "change_kj": delta,
                                 "passed": abs(delta) <= tolerance_kj}
    return {"matched_inputs": True, "mismatches": [], "components": components,
            "tolerance_kj": tolerance_kj,
            "full_pair_agreement": all(row["passed"] is True for row in components.values()),
            "scope": "same-geometry numerical robustness only; no experimental accuracy claim"}
