"""Audit destination ensembles without promoting borrowed or partial thermodynamics.

Pure analysis utilities: no sampling, solvation calls, production routing or
uncertainty calibration. Energies are kJ/mol, forces eV/Angstrom. Each record
has a unique start_id, source/destination geometry SHA256, electronic/solvation
geometry and model IDs, condition_id, status and SCF/identity evidence. Thermal
terms additionally identify their origin, geometry, model and temperature.
Destination minimum validation requires a matching documented solvent Hessian.

The expected start set is fixed by the caller before filtering. Completeness
refers to that sampled set, never all possible basins. Diagnostic log-sums and
weights are mathematical comparisons, not validated aqueous G or populations.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import re

R_KJ = 8.314462618e-3
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)): return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _integer(value, *, positive=False):
    return not isinstance(value, bool) and isinstance(value, int) and (not positive or value > 0)


def _diagnostics_copy(value):
    """Preserve caller data; represent unusable numeric diagnostics as missing."""
    invalid = []
    def clean(item, path):
        if isinstance(item, Mapping):
            return {key: clean(val, path + [str(key)]) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val, path + [str(i)]) for i, val in enumerate(item)]
        if not isinstance(item, bool) and isinstance(item, (int, float)) and not _finite(item):
            invalid.append({"path": path, "reason": "nonfinite_or_out_of_numeric_range", "original": repr(item)})
            return None
        return deepcopy(item)
    return clean(value, []), invalid


def _ids(values, name):
    result = list(values)
    if not result or any(not isinstance(x, str) or not x for x in result) or len(set(result)) != len(result):
        raise ValueError(name + " must contain unique nonempty IDs")
    return result


def _hash(value):
    return isinstance(value, str) and bool(_HASH.fullmatch(value))


def _index(records, expected):
    result = {}
    for record in records:
        if not isinstance(record, Mapping): raise ValueError("Each start record must be a mapping")
        key = record.get("start_id")
        if key not in expected:
            raise ValueError("Unexpected start_id: " + str(key))
        if key in result:
            raise ValueError("Duplicate start_id: " + key)
        result[key] = deepcopy(dict(record))
    return result


def _inspect(row, model, temperature):
    component_errors, admission_errors = [], []
    for key in ("source_geometry_sha256", "destination_geometry_sha256"):
        if not _hash(row.get(key)): component_errors.append("invalid_" + key)
    destination = row.get("destination_geometry_sha256")
    for component in ("electronic", "solvation"):
        if not _finite(row.get(component + "_kj")): component_errors.append("invalid_" + component + "_energy")
        if row.get(component + "_geometry_sha256") != destination:
            component_errors.append(component + "_geometry_mismatch")
        if row.get(component + "_model_id") != model.get(component + "_model_id"):
            component_errors.append(component + "_model_mismatch")
    if row.get("model_id") != model.get("model_id") or row.get("condition_id") != model.get("condition_id"):
        component_errors.append("model_or_condition_mismatch")
    if not _integer(row.get("charge")): component_errors.append("invalid_record_charge")
    if not _integer(row.get("spin_multiplicity"), positive=True): component_errors.append("invalid_record_spin_multiplicity")
    if (not isinstance(row.get("species_id"), str) or not row["species_id"]
            or any(row.get(key) != model.get(key) for key in ("species_id", "charge", "spin_multiplicity"))):
        component_errors.append("chemical_state_mismatch")
    if row.get("identity_preserved") is not True: component_errors.append("identity_not_preserved")
    if row.get("scf_converged") is not True: component_errors.append("scf_not_converged")
    if row.get("status") != "complete": admission_errors.append("incomplete_start")
    force = row.get("fmax_eV_A")
    threshold = model.get("force_threshold_eV_A")
    force_passed = (not component_errors and _finite(force) and force >= 0
                    and _finite(threshold) and threshold > 0 and force <= threshold
                    and row.get("force_converged") is True)
    if not force_passed: admission_errors.append("destination_force_not_validated")
    hessian = row.get("minimum_validation", {})
    hessian_passed = (isinstance(hessian, Mapping) and hessian.get("status") == "passed"
                      and hessian.get("kind") == "destination_hessian"
                      and hessian.get("geometry_sha256") == destination
                      and hessian.get("model_id") == model.get("model_id")
                      and hessian.get("condition_id") == model.get("condition_id")
                      and _hash(hessian.get("provenance_sha256")))
    if not hessian_passed: admission_errors.append("destination_hessian_minimum_not_validated")
    thermal = row.get("thermal_kj")
    thermal_present = _finite(thermal)
    thermal_temperature = row.get("thermal_temperature_K")
    thermal_temperature_matches = (_finite(thermal_temperature) and thermal_temperature > 0
                                   and thermal_temperature == temperature)
    thermal_coherent = (thermal_present and row.get("thermal_geometry_sha256") == destination
                        and isinstance(model.get("thermal_model_id"), str) and bool(model["thermal_model_id"])
                        and row.get("thermal_model_id") == model.get("thermal_model_id")
                        and thermal_temperature_matches)
    thermal_destination = (thermal_coherent and row.get("thermal_origin") == "destination"
                           and _hash(row.get("thermal_provenance_sha256")))
    if not thermal_present: admission_errors.append("missing_thermal_entropy")
    elif not thermal_coherent: admission_errors.append("thermal_geometry_model_or_temperature_mismatch")
    elif not thermal_destination: admission_errors.append("borrowed_or_unvalidated_thermal_entropy")
    potential = row["electronic_kj"] + row["solvation_kj"] if not component_errors else None
    if potential is not None and not math.isfinite(potential):
        component_errors.append("nonfinite_component_sum"); potential = None
    # A known source-gas term can be used as a declared diagnostic substitution;
    # it cannot establish destination entropy or Hessian minimum validation.
    diagnostic_thermal = thermal_present and row.get("thermal_origin") in ("source_gas", "destination")
    if diagnostic_thermal:
        diagnostic_thermal = (thermal_temperature_matches
                              and isinstance(model.get("thermal_model_id"), str) and bool(model["thermal_model_id"])
                              and row.get("thermal_model_id") == model.get("thermal_model_id")
                              and row.get("thermal_geometry_sha256") in
                              (row.get("source_geometry_sha256"), destination))
    diagnostic_total = potential + thermal if potential is not None and diagnostic_thermal else None
    if diagnostic_total is not None and not math.isfinite(diagnostic_total):
        admission_errors.append("nonfinite_thermal_sum"); diagnostic_total = None
    return {"component_errors": component_errors, "admission_errors": admission_errors,
            "coherent_components": not component_errors, "force_passed": force_passed,
            "hessian_passed": hessian_passed, "destination_thermal_passed": thermal_destination,
            "potential_kj": potential,
            "diagnostic_total_kj": diagnostic_total,
            "admitted_minimum": not component_errors and not admission_errors}


def _statistics(energies, temperature):
    if not energies: return None
    floor = min(energies.values())
    rt = R_KJ * temperature
    if not math.isfinite(rt) or rt <= 0: raise ValueError("Temperature outside numeric partition range")
    raw = {key: math.exp(-(value - floor) / rt) for key, value in energies.items()}
    total = sum(raw.values())
    score = floor - rt * math.log(total)
    if not math.isfinite(score): raise ValueError("Partition score outside finite numeric range")
    return {"diagnostic_logsum_kj": score,
            "conditional_weights": {key: value / total for key, value in raw.items()},
            "scope": "fixed supplied basins/terms; no unsampled-basin, population or accuracy claim"}


def assemble_aqueous_ensemble(records, expected_starts, model, *, temperature_K=298.15,
                              selected_starts=(), basin_equivalence=None,
                              source_sampling=None, destination_sampling=None,
                              merge_tolerance_kj=1e-6):
    """Retain every start, merge proved duplicate basins, and gate sampled G.

    Records mapping to exactly identical destination hashes are deduplicated.
    Different geometries merge only with a supplied basin_id and independently
    documented equivalence proof {model_id, condition_id, provenance_sha256}.
    Disagreeing energies/thermal donors within a merged basin are a conflict;
    this utility never selects the donor that gives a favorable energy.
    """
    expected = _ids(expected_starts, "expected_starts")
    if not _finite(temperature_K) or temperature_K <= 0: raise ValueError("Positive temperature required")
    if not _finite(merge_tolerance_kj) or merge_tolerance_kj < 0: raise ValueError("Invalid merge tolerance")
    if not isinstance(model, Mapping) or any(not isinstance(model.get(k), str) or not model[k] for k in
            ("model_id", "condition_id", "species_id", "electronic_model_id", "solvation_model_id")):
        raise ValueError("Explicit model and condition identities required")
    if not _integer(model.get("charge")):
        raise ValueError("Explicit integer charge required")
    if not _integer(model.get("spin_multiplicity"), positive=True):
        raise ValueError("Explicit spin multiplicity required")
    selected = list(selected_starts)
    if len(set(selected)) != len(selected) or any(x not in expected for x in selected):
        raise ValueError("selected_starts must be a unique subset of expected starts")
    indexed = _index(records, expected)
    audit = {key: _inspect(row, model, temperature_K) for key, row in indexed.items()}
    groups, merge_errors = {}, []
    if basin_equivalence is not None and not isinstance(basin_equivalence, Mapping):
        raise ValueError("basin_equivalence must be a mapping")
    proof_map = deepcopy(dict(basin_equivalence)) if basin_equivalence is not None else {}
    for key in expected:
        if key not in indexed: continue
        row, checked = indexed[key], audit[key]
        if not checked["coherent_components"] or row.get("status") != "complete": continue
        basin_id = row.get("basin_id")
        if basin_id is not None and (not isinstance(basin_id, str) or not basin_id):
            merge_errors.append({"start_id": key, "reason": "invalid_basin_id"}); continue
        if basin_id is not None:
            proof = proof_map.get(basin_id, {})
            if (not isinstance(proof, Mapping) or proof.get("model_id") != model["model_id"] or proof.get("condition_id") != model["condition_id"]
                    or not _hash(proof.get("provenance_sha256"))):
                merge_errors.append({"start_id": key, "reason": "unproved_basin_equivalence"})
                continue
            group = "basin:" + basin_id
        else:
            group = "geometry:" + row["destination_geometry_sha256"]
        groups.setdefault(group, []).append(key)
    basins, conflicts = {}, []
    potential_energies, diagnostic_energies, admitted_energies = {}, {}, {}
    for group, members in groups.items():
        potentials = [audit[key]["potential_kj"] for key in members]
        thermal_values = [indexed[key].get("thermal_kj") for key in members]
        potential_conflict = max(potentials) - min(potentials) > merge_tolerance_kj
        donor_conflict = (len({(indexed[key].get("thermal_origin"), indexed[key].get("thermal_model_id"),
                                indexed[key].get("thermal_geometry_sha256"), indexed[key].get("thermal_temperature_K"),
                                indexed[key].get("thermal_provenance_sha256")) for key in members}) > 1
                          or (all(_finite(v) for v in thermal_values)
                              and max(thermal_values) - min(thermal_values) > merge_tolerance_kj)
                          or any(_finite(v) != _finite(thermal_values[0]) for v in thermal_values))
        basins[group] = {"start_ids": members, "deduplicated_start_count": len(members),
                         "potential_conflict": potential_conflict, "thermal_donor_conflict": donor_conflict}
        if potential_conflict or donor_conflict:
            conflicts.append({"basin": group, **basins[group]}); continue
        # Equal components can be represented once. No energy-based donor choice.
        first = members[0]
        potential_energies[group] = audit[first]["potential_kj"]
        if audit[first]["diagnostic_total_kj"] is not None:
            diagnostic_energies[group] = audit[first]["diagnostic_total_kj"]
        if all(audit[key]["admitted_minimum"] for key in members):
            admitted_energies[group] = audit[first]["diagnostic_total_kj"]
    missing = [key for key in expected if key not in indexed]
    incomplete = [key for key in expected if key in indexed and indexed[key].get("status") != "complete"]
    all_coherent = not missing and not incomplete and all(audit[key]["coherent_components"] for key in indexed)
    admitted = (all_coherent and not conflicts and not merge_errors and bool(groups)
                and len(admitted_energies) == len(groups))
    potential_stats = _statistics(potential_energies, temperature_K)
    thermal_stats = _statistics(diagnostic_energies, temperature_K) if len(diagnostic_energies) == len(groups) else None
    admitted_stats = _statistics(admitted_energies, temperature_K) if admitted else None
    def selected_fraction(stats):
        if stats is None: return None
        return sum(weight for group, weight in stats["conditional_weights"].items()
                   if any(key in selected for key in groups[group]))
    source_diagnostic, source_invalid = _diagnostics_copy(source_sampling)
    destination_diagnostic, destination_invalid = _diagnostics_copy(destination_sampling)
    return {"status": "eligible_conditional_sampled_ensemble" if admitted else "diagnostic_only",
            "model": deepcopy(dict(model)), "temperature_K": temperature_K,
            "expected_start_ids": expected, "missing_start_ids": missing, "incomplete_start_ids": incomplete,
            "all_expected_component_records_complete": all_coherent,
            "complete_common_sampled_set": all_coherent and not conflicts and not merge_errors,
            "start_audit": audit, "basins": basins, "merge_errors": merge_errors, "basin_conflicts": conflicts,
            "destination_force_validated_start_count": sum(a["force_passed"] for a in audit.values()),
            "destination_hessian_validated_start_count": sum(a["hessian_passed"] for a in audit.values()),
            "admitted_start_count": sum(a["admitted_minimum"] for a in audit.values()),
            "potential_diagnostic": potential_stats, "thermal_diagnostic": thermal_stats,
            "conditional_sampled_aqueous_G_kj": admitted_stats["diagnostic_logsum_kj"] if admitted_stats else None,
            "selected_start_ids": selected,
            "selected_fraction_in_calculated_destination_potential_set": selected_fraction(potential_stats),
            "selected_fraction_in_calculated_destination_thermal_set": selected_fraction(thermal_stats),
            "selected_fraction_in_complete_expected_destination_potential_set": (selected_fraction(potential_stats)
                    if all_coherent and not conflicts and not merge_errors else None),
            "selected_fraction_in_complete_expected_destination_thermal_set": (selected_fraction(thermal_stats)
                    if all_coherent and not conflicts and not merge_errors else None),
            "unobserved_destination_mass": None,
            "coverage_scope": "fractions within calculated supplied basins; omitted starts/basins have unknown mass",
            "uncertainty": {"source_sampling": source_diagnostic, "destination_sampling": destination_diagnostic,
                            "source_invalid_diagnostics": source_invalid, "destination_invalid_diagnostics": destination_invalid,
                            "validated_CI_kj": None,
                            "scope": "source/gas diagnostics are not destination uncertainty; no interval calibrated here"},
            "aqueous_deltaG_kj": None, "validated_CI_kj": None,
            "scope": "one chemical species and a fixed sampled set; not all basins, reaction DeltaG or accuracy validation"}


def compare_common_sampled_models(records_by_model, expected_starts, models, *, temperature_K=298.15,
                                  selected_starts=(), require_same_destination_geometry=True):
    """Reassemble the exact expected sampled set across models, never silently intersect.

    Original model components/thermal terms remain explicit. Borrowed gas RRHO
    produces diagnostic log-sums only. Missing models/starts prohibit whole-set
    scores even when a small intersection could be evaluated successfully.
    """
    expected = _ids(expected_starts, "expected_starts")
    if not _finite(temperature_K) or temperature_K <= 0: raise ValueError("Positive temperature required")
    if set(records_by_model) != set(models) or not models: raise ValueError("Exact model definitions required")
    selected = list(selected_starts)
    if len(set(selected)) != len(selected) or any(key not in expected for key in selected):
        raise ValueError("Invalid selected starts")
    indexed = {name: _index(records, expected) for name, records in records_by_model.items()}
    audited = {name: {key: _inspect(row, models[name], temperature_K) for key, row in rows.items()}
               for name, rows in indexed.items()}
    available = {name: {key for key, row in rows.items() if row.get("status") == "complete"
                         and audited[name][key]["coherent_components"] and audited[name][key]["diagnostic_total_kj"] is not None}
                 for name, rows in indexed.items()}
    common = set.intersection(*available.values())
    geometry_mismatches = []
    for key in expected:
        rows = [indexed[name].get(key) for name in models]
        if any(row is None for row in rows): continue
        if len({row.get("source_geometry_sha256") for row in rows}) != 1:
            geometry_mismatches.append({"start_id": key, "kind": "source_geometry_mismatch"})
        if require_same_destination_geometry and len({row.get("destination_geometry_sha256") for row in rows}) != 1:
            geometry_mismatches.append({"start_id": key, "kind": "destination_geometry_mismatch"})
    complete = common == set(expected) and not geometry_mismatches
    # Comparisons must not change which physical thermal donors are held fixed.
    donor_mismatches = []
    for key in common:
        rows = [indexed[name][key] for name in models]
        if len({(r.get("thermal_origin"), r.get("thermal_model_id"), r.get("thermal_geometry_sha256"),
                 r.get("thermal_kj"), r.get("thermal_temperature_K"), r.get("thermal_provenance_sha256")) for r in rows}) != 1:
            donor_mismatches.append(key)
    complete = complete and not donor_mismatches
    assembled = {name: assemble_aqueous_ensemble(rows.values(), expected, models[name],
                                                temperature_K=temperature_K, selected_starts=selected)
                 for name, rows in indexed.items()}
    complete = complete and all(item["complete_common_sampled_set"] for item in assembled.values())
    selected_common = (set(selected) <= common and not any(item["start_id"] in selected for item in geometry_mismatches)
                       and not any(key in selected for key in donor_mismatches))
    outputs = {}
    for name, rows in indexed.items():
        whole = assembled[name]["thermal_diagnostic"] if complete else None
        subset_audit = (assemble_aqueous_ensemble([rows[key] for key in selected], selected, models[name],
                                                  temperature_K=temperature_K) if selected and selected_common else None)
        subset = (subset_audit["thermal_diagnostic"] if subset_audit and subset_audit["complete_common_sampled_set"] else None)
        outputs[name] = {"missing_or_invalid_start_ids": [key for key in expected if key not in available[name]],
                         "whole_expected_set_diagnostic": whole, "selected_set_diagnostic": subset,
                         "selected_fraction_in_whole_expected_set": (assembled[name]["selected_fraction_in_calculated_destination_thermal_set"] if whole else None),
                         "basins": assembled[name]["basins"], "basin_conflicts": assembled[name]["basin_conflicts"],
                         "start_audit": audited[name]}
    return {"complete_common_sampled_set": complete, "expected_start_ids": expected,
            "common_available_start_ids": [key for key in expected if key in common],
            "geometry_mismatches": geometry_mismatches, "thermal_donor_mismatch_start_ids": donor_mismatches,
            "models": outputs, "temperature_K": temperature_K,
            "aqueous_deltaG_kj": None, "validated_CI_kj": None,
            "scope": "matched finite sampled-set component diagnostics; borrowed source thermal terms remain diagnostic"}
