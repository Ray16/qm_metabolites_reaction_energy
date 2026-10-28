"""Ground-truth-free network integrity checks; never correct or fit energies.

Species identities retain charge, isotopes and specified stereochemistry. No
automatic tautomer, protonation or compartment pooling occurs. Callers may give
explicit state IDs, but each ID must identify the same chemical state throughout.
Closed cycles are tested only within common declared conditions and settings.
Passing these checks establishes consistency on available constraints, not
accuracy, complete sampling, or coverage on unfamiliar chemistry.
"""
from collections import Counter, defaultdict, deque
from collections.abc import Mapping
import hashlib
import json
import math

import numpy as np
from rdkit import Chem

from metag.assembly import reaction_covariance
from metag.validation.transfer import compound_identity


def _finite(value):
    try:
        return value is not None and not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def protocol_fingerprint(value):
    """Type-aware hash of an explicit JSON specification; NaN is forbidden."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def condition_fingerprint(conditions):
    """Normalize numeric condition declarations; preserve names and conventions.

    Boolean condition values remain distinct/invalid. Different schemas, recipes
    and reservoir declarations are not silently bridged or treated as equivalent.
    """
    if not isinstance(conditions, Mapping):
        return protocol_fingerprint(conditions)
    normalized = dict(conditions)
    for key in ("temperature_K", "T_K", "pH", "ionic_strength_M", "pMg", "pMg_free",
                "solute_standard_state_M", "water_activity"):
        if key in normalized and _finite(normalized[key]):
            normalized[key] = float(normalized[key])
            if normalized[key] == 0:
                normalized[key] = 0.0
    return protocol_fingerprint(normalized)


def state_identity(smiles, charge):
    """Canonical chemical-state identity, never a neutralized parent identity."""
    if not _finite(charge) or float(charge) != int(float(charge)):
        raise ValueError("Species charge must be a finite integer")
    canonical = compound_identity(smiles, stereo_sensitive=True)
    mol = Chem.MolFromSmiles(canonical)
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise ValueError("A generic fragment does not define a network chemical state")
    if sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) != int(float(charge)):
        raise ValueError("Declared species charge differs from its molecular structure")
    return json.dumps([canonical, int(float(charge))], separators=(",", ":"))


def _species(species, state_ids=None):
    if not isinstance(species, Mapping) or not species:
        raise ValueError("Original/scored species must be a nonempty mapping")
    state_ids = state_ids or {}
    if set(state_ids) - species.keys():
        raise ValueError("State IDs must refer to known original species labels")
    parsed = {}
    for label, value in species.items():
        if not isinstance(value, (list, tuple)) or len(value) != 3 or not _finite(value[0]):
            raise ValueError(f"Invalid species entry {label!r}; expected coefficient, charge, SMILES")
        coefficient, charge, smiles = value
        identity = state_identity(smiles, charge)
        state_id = state_ids.get(label, identity)
        if not isinstance(state_id, str) or not state_id:
            raise ValueError("Explicit state IDs must be nonempty strings")
        parsed[label] = {"coefficient": float(coefficient), "chemical_identity": identity,
                         "state_id": state_id, "proton": compound_identity(smiles) == "[H+]"}
    return parsed


def _column(parsed):
    column = defaultdict(float)
    # Fixed-pH transformed quantities exclude the buffered proton reservoir.
    # Water remains an explicit state. Chemical (untransformed) ΔG is unsupported.
    for value in parsed.values():
        if not value["proton"]:
            column[value["state_id"]] += value["coefficient"]
    return {key: value for key, value in column.items() if value != 0}


def condition_issues(conditions):
    """Check declarations used by this audit, including the deployed legacy schema.

    ``pMg=None`` explicitly means no Mg-binding states in the deployed model;
    it is not substituted with a finite free-Mg concentration. Completeness of
    these declarations does not validate experimental condition transformations.
    """
    if not isinstance(conditions, Mapping) or not conditions:
        return ["conditions_unavailable"]
    issues = []
    temperature = conditions.get("temperature_K", conditions.get("T_K"))
    for key, value in (("temperature", temperature), ("pH", conditions.get("pH")),
                       ("ionic_strength", conditions.get("ionic_strength_M"))):
        if not _finite(value):
            issues.append("invalid_or_missing_" + key)
        elif (key == "temperature" and float(value) <= 0
              or key == "ionic_strength" and float(value) < 0):
            issues.append("invalid_" + key)
    if "pMg" not in conditions or (conditions["pMg"] is not None and not _finite(conditions["pMg"])):
        issues.append("invalid_or_missing_Mg_convention")
    if "standard_state" in conditions:
        if not isinstance(conditions["standard_state"], str) or not conditions["standard_state"].strip():
            issues.append("invalid_standard_state")
    else:
        if not _finite(conditions.get("solute_standard_state_M")) or float(conditions["solute_standard_state_M"]) <= 0:
            issues.append("invalid_or_missing_solute_standard_state")
        if conditions.get("water_standard_state") != "pure_liquid":
            issues.append("unsupported_or_missing_water_standard_state")
    if "water_activity" in conditions and (not _finite(conditions["water_activity"]) or float(conditions["water_activity"]) <= 0):
        issues.append("invalid_water_activity")
    try:
        protocol_fingerprint(conditions)
    except (TypeError, ValueError):
        issues.append("conditions_not_finite_JSON")
    return issues


def uncertainty_provenance(record, calibration_artifact=None):
    """Record interval/config availability without promoting a calibration claim.

    This function does not inspect validation measurements; it cannot establish
    experimental coverage. Reaction-wise widths do not supply species covariance.
    """
    info = record.get("ci_info") or {}
    info = info if isinstance(info, Mapping) else {}
    interval = record.get("ci95")
    available = (isinstance(interval, (list, tuple)) and len(interval) == 2
                 and all(_finite(x) for x in interval) and float(interval[0]) <= float(interval[1]))
    config, calibrated_config = record.get("config"), (calibration_artifact or {}).get("config")
    mismatch = None
    if isinstance(config, Mapping) and config and isinstance(calibrated_config, Mapping) and calibrated_config:
        try:
            mismatch = [key for key in sorted(set(config) | set(calibrated_config))
                        if protocol_fingerprint(config.get(key)) != protocol_fingerprint(calibrated_config.get(key))]
        except (TypeError, ValueError):
            mismatch = None
    claimed_value = info.get("coverage_calibrated", record.get("coverage_calibrated",
                        info.get("externally_calibrated", record.get("externally_calibrated", False))))
    claimed = claimed_value is True
    rid = record.get("reaction", record.get("id"))
    targets = (calibration_artifact or {}).get("per_reaction", {})
    if isinstance(targets, Mapping):
        exposure = rid in targets
    elif isinstance(targets, (list, tuple)):
        target_ids = {row.get("rid", row.get("reaction", row.get("id"))) for row in targets if isinstance(row, Mapping)}
        target_ids.discard(None)
        exposure = rid in target_ids if target_ids else None
    else:
        exposure = None
    return {"sigma_available": _finite(record.get("sigma_pred")) and float(record["sigma_pred"]) >= 0,
            "interval_available": available, "reported_coverage_calibrated": claimed,
            "calibration_claim_flag_valid": isinstance(claimed_value, bool),
            "calibration_artifact_available": calibration_artifact is not None,
            "configuration_match": None if mismatch is None else not mismatch,
            "configuration_mismatch_keys": mismatch,
            "recorded_configuration_mismatch": info.get("config_mismatch"),
            "calibration_basis": info.get("calibration_basis", record.get("calibration_basis")),
            "recorded_scope": info.get("calibration_scope", record.get("calibration_scope")),
            "calibration_target_exposure": exposure,
            "ood": bool(info.get("ood", False)), "ood_reasons": info.get("ood_reasons", []),
            "extrapolated_features": info.get("extrapolated_features", []),
            "external_coverage_established_by_audit": False,
            "species_covariance_available_from_reaction_widths": False,
            "scope": "availability/provenance only; internal database calibration is not external or novel-chemistry coverage"}


def _support_components(columns):
    """Prune leaf reactions sparsely, then find connected cycle-support components."""
    incidence = defaultdict(set)
    for j, column in enumerate(columns):
        for key in column:
            incidence[key].add(j)
    alive = set(range(len(columns)))
    leaves = deque(key for key, rows in incidence.items() if len(rows) == 1)
    while leaves:
        key = leaves.popleft()
        if len(incidence[key]) != 1:
            continue
        j = next(iter(incidence[key]))
        if j not in alive:
            continue
        alive.remove(j)
        for member in columns[j]:
            incidence[member].discard(j)
            if len(incidence[member]) == 1:
                leaves.append(member)
    components = []
    unseen = set(alive)
    while unseen:
        pending, component = [min(unseen)], set()
        while pending:
            j = pending.pop()
            if j not in unseen:
                continue
            unseen.remove(j)
            component.add(j)
            for key in columns[j]:
                pending.extend(incidence[key] & unseen)
        components.append(sorted(component))
    return components, len(columns) - len(alive)


def _null_basis(matrix, tolerance=1e-10):
    """Fundamental null vectors by scaled elimination; no energy projection."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Stoichiometry must be a finite matrix")
    scales = np.max(np.abs(matrix), axis=0) if matrix.shape[0] else np.zeros(matrix.shape[1])
    scales = np.where(scales > 0, scales, 1.0)
    reduced = matrix / scales
    pivot_columns, row = [], 0
    for j in range(reduced.shape[1]):
        if row >= reduced.shape[0]:
            break
        pivot = row + int(np.argmax(np.abs(reduced[row:, j])))
        if abs(reduced[pivot, j]) <= tolerance:
            continue
        reduced[[row, pivot]] = reduced[[pivot, row]]
        reduced[row] /= reduced[row, j]
        factors = reduced[:, j].copy()
        factors[row] = 0
        reduced -= factors[:, None] * reduced[row]
        pivot_columns.append(j)
        row += 1
    free = sorted(set(range(matrix.shape[1])) - set(pivot_columns))
    vectors = []
    for j in free:
        vector = np.zeros(matrix.shape[1])
        vector[j] = 1
        for i, pivot in enumerate(pivot_columns):
            vector[pivot] = -reduced[i, j]
        vector /= scales
        if not np.isfinite(vector).all():
            raise ValueError("Cycle coefficients overflowed")
        vector /= np.max(np.abs(vector))
        vector[np.abs(vector) < tolerance] = 0
        first = np.flatnonzero(vector)[0]
        if vector[first] < 0:
            vector *= -1
        residual = np.linalg.norm(matrix @ vector)
        bound = tolerance * max(1.0, np.linalg.norm(matrix) * np.linalg.norm(vector)) * 10
        if not np.isfinite(residual) or residual > bound:
            raise ValueError("Numerical null-space residual exceeds tolerance")
        vectors.append(vector)
    return vectors, len(pivot_columns)


def cycle_audit(columns, energies, reaction_ids, *, tolerance_kj=1e-6, rounding_kj=None,
                maximum_component_reactions=1500, maximum_matrix_entries=2_000_000,
                maximum_cycles_listed=20):
    """Direct cycle sums; rounding bounds and unknown capped components stay visible.

    Basis vectors have maximum absolute coefficient one. Basis closure magnitudes
    depend on this chosen basis; they are not the worst of all possible cycles.
    No fitted species energies, independent reaction sigma, or corrected ΔG is used.
    """
    if len(columns) != len(energies) or len(columns) != len(reaction_ids):
        raise ValueError("Stoichiometry, energies and reaction IDs must align")
    if len(set(reaction_ids)) != len(reaction_ids) or any(not _finite(x) for x in energies):
        raise ValueError("Cycle targets require unique IDs and finite energies")
    if not _finite(tolerance_kj) or float(tolerance_kj) < 0:
        raise ValueError("Cycle tolerance must be finite and nonnegative")
    rounding = np.zeros(len(columns)) if rounding_kj is None else np.asarray(rounding_kj, dtype=float)
    if rounding.shape != (len(columns),) or not np.isfinite(rounding).all() or (rounding < 0).any():
        raise ValueError("Rounding resolutions must be finite, nonnegative and aligned")
    components, pruned = _support_components(columns)
    reports, cycles = [], []
    for component in components:
        keys = sorted({key for j in component for key in columns[j]})
        if len(component) > maximum_component_reactions or len(keys) * len(component) > maximum_matrix_entries:
            reports.append({"status": "budget_exceeded", "reaction_ids": [reaction_ids[j] for j in component],
                            "n_species": len(keys)})
            continue
        lookup = {key: i for i, key in enumerate(keys)}
        matrix = np.zeros((len(keys), len(component)))
        for k, j in enumerate(component):
            for key, value in columns[j].items():
                matrix[lookup[key], k] = value
        vectors, rank = _null_basis(matrix)
        reports.append({"status": "audited", "n_reactions": len(component), "n_species": len(keys),
                        "rank": rank, "n_independent_cycles": len(vectors)})
        for vector in vectors:
            closure = float(vector @ np.asarray(energies)[component])
            rounding_bound = float(.5 * np.abs(vector) @ rounding[component])
            cycles.append({"reactions": {reaction_ids[j]: float(vector[k]) for k, j in enumerate(component) if vector[k]},
                           "closure_kj": closure, "rounding_bound_kj": rounding_bound,
                           "violates_closure": abs(closure) > float(tolerance_kj) + rounding_bound,
                           "stoichiometric_residual_norm": float(np.linalg.norm(matrix @ vector))})
    cycles.sort(key=lambda row: -abs(row["closure_kj"]))
    return {"status": "incomplete" if any(row["status"] != "audited" for row in reports) else "audited",
            "n_reactions": len(columns), "n_pruned_noncycle_reactions": pruned,
            "n_independent_cycles_audited": len(cycles),
            "n_cycle_basis_violations": sum(row["violates_closure"] for row in cycles),
            "max_abs_basis_closure_kj": max((abs(row["closure_kj"]) for row in cycles), default=None),
            "tolerance_kj": float(tolerance_kj), "components": reports,
            "worst_cycles": cycles[:maximum_cycles_listed],
            "scope": "direct state-function diagnostic; basis-dependent closure, no projection correction or accuracy inference"}


def coherent_covariance(stoichiometry, species_covariance, *, species_keys,
                        covariance_species_keys, provenance, conditions, settings):
    """Propagate a justified shared-state covariance and quantify cycle invariance.

    Covariance keys must match the species-row order exactly. The provenance is
    required but not verified experimental evidence; no covariance is inferred
    from per-reaction intervals, and means are never adjusted to close cycles.
    """
    if not species_keys or len(set(species_keys)) != len(species_keys):
        raise ValueError("Covariance requires unique, nonempty ordered species keys")
    if list(species_keys) != list(covariance_species_keys):
        raise ValueError("Species covariance key order must match stoichiometry rows")
    if not isinstance(provenance, Mapping) or any(not provenance.get(key) for key in ("source", "scope", "units")):
        raise ValueError("Species covariance requires source, scope and units provenance")
    if provenance["units"] != "(kJ/mol)^2":
        raise ValueError("Species covariance units must be (kJ/mol)^2")
    if condition_issues(conditions) or not isinstance(settings, Mapping) or not settings:
        raise ValueError("Covariance propagation requires complete network conditions and settings")
    condition_id = condition_fingerprint(conditions)
    settings_id = protocol_fingerprint(settings)
    if (provenance.get("conditions_fingerprint") != condition_id
            or provenance.get("settings_fingerprint") != settings_id):
        raise ValueError("Species covariance conditions/settings must match the network protocol")
    matrix = np.asarray(stoichiometry, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(species_keys):
        raise ValueError("Stoichiometry rows must match the declared species keys")
    propagated = reaction_covariance(matrix, species_covariance)
    vectors, _ = _null_basis(matrix)
    cycles = np.asarray(vectors) if vectors else np.zeros((0, matrix.shape[1]))
    cycle_covariance = cycles @ propagated @ cycles.T
    reference_scale = float(np.max(np.abs(propagated))) if propagated.size else 0.0
    cycle_scale = float(np.max(np.abs(cycle_covariance))) if cycle_covariance.size else 0.0
    return {"reaction_covariance": propagated.tolist(), "cycle_coefficients": cycles.tolist(),
            "cycle_covariance": cycle_covariance.tolist(), "max_abs_cycle_covariance": cycle_scale,
            "cycle_covariance_relative_to_reaction_scale": cycle_scale / reference_scale if reference_scale else 0.0,
            "provenance": dict(provenance), "coverage_guaranteed": False,
            "conditions_fingerprint": condition_id, "settings_fingerprint": settings_id,
            "scope": "S.T C S for supplied shared-state covariance; cycle invariance is algebraic consistency, not accuracy or calibrated coverage"}


def audit_network(records, *, calibration_artifact=None, tolerance_kj=1e-6,
                  maximum_component_reactions=1500, maximum_cycles_listed=20,
                  state_groups=None):
    """Audit records retaining original and scored species plus full denominators.

    Required conventions: energy_units='kJ/mol', quantity=
    'standard_transformed_reaction_gibbs', nonempty conditions/config. Records
    require reaction/id and original_species. Optional representation_map maps
    each original label to ONE scored label; changed/ambiguous unmapped species
    are reported unknown rather than guessed. Optional state_ids name separate
    pools/compartments; each ID must retain the same canonical chemical state.
    Explicit pseudoisomer pooling additionally requires a ``state_groups`` entry
    declaring members, conditions, source, reference state and reservoir convention.
    This validates the declaration, not the underlying state-equilibrium model.
    """
    records = list(records)
    ids = [row.get("reaction", row.get("id")) for row in records]
    if any(not isinstance(rid, str) or not rid for rid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Network records require unique nonempty reaction IDs")
    state_groups = state_groups or {}
    for key, group in state_groups.items():
        if (not isinstance(key, str) or not isinstance(group, Mapping)
            or any(not group.get(field) for field in ("member_identities", "source", "reference_state_id", "reservoir_convention"))
            or condition_issues(group.get("conditions"))):
            raise ValueError("Pseudoisomer groups require member identities, source, reference state, reservoir convention and conditions")
        for identity in group["member_identities"]:
            try:
                smiles, charge = json.loads(identity)
            except (TypeError, ValueError):
                raise ValueError("Group members must be canonical microscopic state identities") from None
            if state_identity(smiles, charge) != identity:
                raise ValueError("Group member identity must use state_identity canonical form")
    rows, parsed_records, identities = [], {}, defaultdict(lambda: defaultdict(set))
    groups = defaultdict(list)
    representations = defaultdict(lambda: defaultdict(list))
    for record, rid in zip(records, ids):
        issues = condition_issues(record.get("conditions"))
        if record.get("energy_units") != "kJ/mol" or record.get("quantity") != "standard_transformed_reaction_gibbs":
            issues.append("quantity_or_energy_units_unavailable_or_unsupported")
        if not isinstance(record.get("config"), Mapping) or not record["config"]:
            issues.append("settings_unavailable")
        try:
            settings_hash = protocol_fingerprint(record.get("config"))
            conditions_hash = condition_fingerprint(record.get("conditions"))
        except (TypeError, ValueError):
            issues.append("protocol_not_finite_JSON")
            settings_hash = conditions_hash = None
        original = None
        try:
            original = _species(record.get("original_species"), record.get("state_ids"))
            for member in original.values():
                group = state_groups.get(member["state_id"])
                if group is not None and (member["chemical_identity"] not in group["member_identities"]
                    or condition_fingerprint(group["conditions"]) != conditions_hash):
                    issues.append("pseudoisomer_group_member_or_conditions_mismatch")
                identities[member["state_id"]][member["chemical_identity"]].add(rid)
            if record.get("state_ids") and not record.get("state_identity_provenance") and any(
                member["state_id"] not in state_groups for label, member in original.items() if label in record["state_ids"]):
                issues.append("explicit_state_ID_provenance_unavailable")
        except (TypeError, ValueError) as error:
            issues.append("original_species_invalid: " + str(error))
        routes = record.get("routes") or {}
        route_failures = {key: routes[key] for key in ("errors", "incomplete_hydration", "incomplete_ring_chain") if routes.get(key)}
        if routes.get("input_balanced") is False:
            issues.append("original_reaction_declared_unbalanced")
        if record.get("suspect") or record.get("accepted") is False or route_failures or record.get("status") not in (None, "ok", "estimated", "success"):
            status = "withheld"
        elif not _finite(record.get("dG")):
            status = "no_estimate"
        else:
            status = "estimated"
        row = {"reaction": rid, "prediction_status": status, "metadata_issues": issues,
               "network_admitted": status == "estimated" and not issues,
               "conditions_fingerprint": conditions_hash, "settings_fingerprint": settings_hash,
               "failure_reason": record.get("suspect") or record.get("reason") or record.get("error"),
               "warnings": (record.get("routes") or {}).get("warnings", []),
               "failure_diagnostics": route_failures,
               "reference_state_id": record.get("reference_state_id"),
               "state_identity_provenance": record.get("state_identity_provenance"),
               "uncertainty": uncertainty_provenance(record, calibration_artifact),
               "representation_changes": [], "representation_unknown": []}
        if original is not None:
            parsed_records[rid] = original
            scored = None
            try:
                scored = _species(record.get("species_scored"))
            except (TypeError, ValueError):
                pass
            mapping = record.get("representation_map") or {}
            for label, member in original.items():
                if member["coefficient"] == 0 or member["proton"]:
                    continue
                mapped = mapping.get(label, label)
                image = None if scored is None else scored.get(mapped)
                if image is None:
                    row["representation_unknown"].append(label)
                    continue
                if not math.isclose(member["coefficient"], image["coefficient"], rel_tol=1e-10, abs_tol=1e-12):
                    row["representation_unknown"].append(label)
                    continue
                if image["chemical_identity"] != member["chemical_identity"]:
                    row["representation_changes"].append({"original_label": label, "scored_label": mapped,
                        "original_identity": member["chemical_identity"], "scored_identity": image["chemical_identity"]})
                key = (conditions_hash, settings_hash, member["state_id"])
                representations[key][image["chemical_identity"]].append({"reaction": rid, "original_label": label,
                    "scored_label": mapped, "routes": {key: value for key, value in (record.get("routes") or {}).items()
                                                       if key in ("truncated", "trunc_radius", "cofactor_ring", "coa_core", "ntp_core", "ph0")}})
        rows.append(row)
    identity_conflicts = [{"state_id": key, "identities": {identity: sorted(rids) for identity, rids in variants.items()}}
                         for key, variants in identities.items() if len(variants) > 1 and key not in state_groups]
    conflicting_ids = {rid for conflict in identity_conflicts for rids in conflict["identities"].values() for rid in rids}
    for row, record in zip(rows, records):
        if row["reaction"] in conflicting_ids:
            row["metadata_issues"].append("state_id_has_conflicting_chemical_identity")
            row["network_admitted"] = False
        if row["network_admitted"]:
            groups[(row["conditions_fingerprint"], row["settings_fingerprint"])].append(record)
    audits = []
    for (conditions_hash, settings_hash), group in groups.items():
        reaction_ids = [record.get("reaction", record.get("id")) for record in group]
        result = cycle_audit([_column(parsed_records[rid]) for rid in reaction_ids],
            [record["dG"] for record in group], reaction_ids, tolerance_kj=tolerance_kj,
            rounding_kj=[record.get("reported_rounding_kj", 0.0) for record in group],
            maximum_component_reactions=maximum_component_reactions,
            maximum_cycles_listed=maximum_cycles_listed)
        audits.append({"conditions_fingerprint": conditions_hash, "settings_fingerprint": settings_hash,
                       "reaction_ids": reaction_ids, "cycles": result})
    context = [{"state_id": key[2], "conditions_fingerprint": key[0], "settings_fingerprint": key[1],
                "representations": dict(variants)} for key, variants in representations.items() if len(variants) > 1]
    statuses = Counter(row["prediction_status"] for row in rows)
    summary = {"n_panel": len(records), "prediction_status_counts": dict(statuses),
               "n_estimates_available": statuses["estimated"], "n_network_admitted": sum(row["network_admitted"] for row in rows),
               "n_metadata_unverifiable": sum(bool(row["metadata_issues"]) for row in rows),
               "n_protocol_groups": len(groups), "n_original_state_identity_conflicts": len(identity_conflicts),
               "n_context_dependent_state_representations": len(context),
               "n_reactions_with_representation_changes": sum(bool(row["representation_changes"]) for row in rows),
               "n_reactions_with_unknown_representation_mapping": sum(bool(row["representation_unknown"]) for row in rows),
               "n_interval_available": sum(row["uncertainty"]["interval_available"] for row in rows),
               "n_reported_calibration_claims": sum(row["uncertainty"]["reported_coverage_calibrated"] for row in rows),
               "n_calibration_config_matches": sum(row["uncertainty"]["configuration_match"] is True for row in rows),
               "n_calibration_config_unknown": sum(row["uncertainty"]["configuration_match"] is None for row in rows),
               "n_calibration_target_exposed": sum(row["uncertainty"]["calibration_target_exposure"] is True for row in rows),
               "n_independent_cycles_audited": sum(audit["cycles"]["n_independent_cycles_audited"] for audit in audits),
               "n_cycle_basis_violations": sum(audit["cycles"]["n_cycle_basis_violations"] for audit in audits),
               "species_covariance_supplied": False, "external_coverage_established": False}
    return {"summary": summary, "records": rows, "protocol_groups": audits,
            "explicit_state_groups": state_groups,
            "state_identity_conflicts": identity_conflicts, "context_dependent_representations": context,
            "limitations": ["Exact charge/stereo/state identities; no automatic pseudoisomer or compartment pooling.",
                "Cycles across unequal conditions/settings are not scored together; absence of tested cycles does not establish consistency.",
                "Reported scalar intervals cannot identify shared-state covariance or guarantee novel-chemistry coverage.",
                "No projection correction, energy fitting, empirical uncertainty refitting or accuracy claim is performed."]}
