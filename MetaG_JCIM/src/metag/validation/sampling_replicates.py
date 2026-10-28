"""Audit explicitly expected independent-seed predictions without fitting energies.

A passing label describes only the observed, compatible, complete replica set.
It establishes neither all-basin convergence nor an uncertainty confidence level.
No seed is selected, averaged, or discarded to repair a failed replica set.
"""
from collections import Counter, defaultdict
from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math

from metag.validation.network_consistency import condition_fingerprint, condition_issues, state_identity

R_KJ = 8.31446261815324e-3
PASS = "within_observed_replicate_tolerance"
_SUCCESS = {"estimated", "complete", "completed", "success"}
_HASH_FIELDS = ("checkpoint_sha256", "references_sha256", "inference_sha256")
_SOFTWARE = ("fairchem-core", "torch", "rdkit", "ase", "numpy")
_ROUTE_DIAGNOSTICS = {"warnings", "species_diagnostics", "hydration_diagnostics",
                      "ring_chain_diagnostics", "sampling_uncertainty"}
_SCOPE = ("Observed independent-seed sensitivity only; fixed declared chemistry, conditions and producer. "
          "No confidence interval, all-basin convergence, experimental accuracy, or calibrated coverage claim. "
          "Recorded point estimates retain their reported rounding; input producer hashes are checked for "
          "completeness/agreement, not reverified against external binaries.")


def _finite(value):
    try:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value))
    except OverflowError:
        return False


def _seed(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _replicate_key(value):
    if isinstance(value, str) and value.strip():
        return ("replicate_id", value)
    if _seed(value):
        return ("seed_offset", value)
    raise ValueError("Expected replicate identities must be nonempty strings or nonnegative integer seed offsets")


def _species_signature(species):
    if not isinstance(species, Mapping) or not species:
        raise ValueError("missing species mapping")
    signature = []
    for label, entry in species.items():
        if (not isinstance(label, str) or not label or not isinstance(entry, (list, tuple))
                or len(entry) != 3 or not _finite(entry[0]) or entry[0] == 0):
            raise ValueError("invalid species label/coefficient/entry")
        coefficient, charge, smiles = entry
        signature.append([label, float(coefficient), state_identity(smiles, charge)])
    # Preserve direction, extent, spectators, labels, isotopes, charge and stereo.
    return sorted(signature)


def _producer_issues(config):
    producer = config.get("producer")
    if not isinstance(producer, Mapping):
        return ["missing_producer_provenance"]
    issues = []
    if type(producer.get("schema")) is not int or producer.get("schema") not in (1, 2):
        issues.append("unknown_producer_schema")
    if not _sha(producer.get("source_sha256")):
        issues.append("missing_producer_source_hash")
    if producer.get("schema") == 2:
        if producer.get("source_scope") != "raw_species_v1":
            issues.append("unknown_species_source_scope")
        if not _sha(config.get("estimator_source_sha256")):
            issues.append("missing_full_estimator_source_hash")
    if producer.get("cacheable") is not True:
        issues.append("producer_not_identified_as_cacheable")
    model = producer.get("model")
    if not isinstance(model, Mapping) or any(not _sha(model.get(key)) for key in _HASH_FIELDS):
        issues.append("incomplete_model_identity")
    software = producer.get("software")
    if (not isinstance(software, Mapping) or any(not isinstance(software.get(key), str)
            or not software[key].strip() or software[key].strip().lower() in {"unknown", "unavailable", "none"}
            for key in _SOFTWARE)):
        issues.append("incomplete_backend_software_identity")
    if not _sha(producer.get("xtb_sha256")):
        issues.append("missing_xtb_identity")
    additional = producer.get("additional_executable_sha256", {})
    if not isinstance(additional, Mapping) or any(not _sha(v) for v in additional.values()):
        issues.append("incomplete_additional_executable_identity")
    if config.get("solv_model") == "cpcmx" and (not isinstance(additional, Mapping) or not _sha(additional.get("xtbcpx"))):
        issues.append("missing_cpcmx_executable_identity")
    return issues



def _sampling_provenance(result, config):
    """Verify seed offsets of noncancelling raw estimates and hydration states.

    Explicit clusters use a SMILES-derived deterministic RNG, so different
    conformer offsets do not establish independent sampling of those terms.
    Exact signed cancellation of a repeated species/method needs no new sample.
    """
    issues, hydrated = [], {}
    seed = config.get("conformer_seed_offset")
    conv = config.get("conv")
    window = (isinstance(conv, (list, tuple)) and len(conv) == 3
              and isinstance(conv[2], int) and not isinstance(conv[2], bool) and conv[2] > 0)
    if not window:
        issues.append("unverified_seed_window")
    routes = result.get("routes", {})
    species = result.get("species_scored", {})
    if not isinstance(routes, Mapping) or not isinstance(species, Mapping):
        return issues, hydrated
    diagnostics = routes.get("species_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return issues + ["missing_species_sampling_provenance"], hydrated
    totals, groups = {}, {}
    for label, entry in species.items():
        try:
            coefficient, charge, smiles = entry
            diagnostic = diagnostics.get(label, {})
            method = diagnostic.get("method") if isinstance(diagnostic, Mapping) else None
            key = (method, state_identity(smiles, charge))
            totals[key] = totals.get(key, Fraction(0)) + Fraction(str(coefficient))
            groups[label] = key
        except (ValueError, TypeError, OverflowError):
            issues.append("invalid_species_sampling_identity: " + str(label))
    required = {label for label, key in groups.items() if totals[key] != 0}
    def implicit_check(diagnostic, label):
        sampling = diagnostic.get("sampling", {}) if isinstance(diagnostic, Mapping) else {}
        if (not isinstance(diagnostic, Mapping) or diagnostic.get("method") != "implicit"
                or not isinstance(sampling, Mapping) or sampling.get("status") != "available"
                or not _seed(sampling.get("seed_offset")) or sampling["seed_offset"] != seed):
            issues.append("unverified_implicit_seed: " + label)
    water_identity = state_identity("O", 0)
    for label in required:
        method, identity = groups[label]
        if method == "liquid_water_reference" and identity == water_identity:
            continue
        if method == "implicit":
            implicit_check(diagnostics.get(label), label)
        elif method == "explicit":
            issues.append("explicit_cluster_sampling_not_varied_by_seed_offset: " + label)
        else:
            issues.append("unverified_sampling_method: " + label)
    hydration = routes.get("hydration_diagnostics", {})
    if not isinstance(hydration, Mapping):
        issues.append("invalid_hydration_diagnostics")
    else:
        for label, diagnostic in hydration.items():
            if label not in species or not isinstance(diagnostic, Mapping):
                issues.append("invalid_hydration_parent: " + str(label)); continue
            states = diagnostic.get("states")
            if diagnostic.get("status") != "complete" or not isinstance(states, list) or not states:
                issues.append("unverified_hydration_state_set: " + label); continue
            identities = []
            for index, state in enumerate(states):
                tag = f"{label}/hydrate{index}"
                if not isinstance(state, Mapping) or state.get("status") != "estimated":
                    issues.append("incomplete_hydration_state: " + tag); continue
                nwater = state.get("n_water")
                if not _seed(nwater) or nwater == 0:
                    issues.append("invalid_hydration_water_count: " + tag); continue
                try:
                    identities.append([nwater, state_identity(state.get("smiles"), species[label][1])])
                except (ValueError, TypeError, OverflowError):
                    issues.append("invalid_hydration_state_identity: " + tag)
                if label in required:
                    implicit_check(state.get("species"), tag)
            hydrated[label] = sorted(identities)
    ring_chain = routes.get("ring_chain", [])
    if not isinstance(ring_chain, (list, tuple)):
        issues.append("invalid_ring_chain_provenance")
    elif any(label in required for label in ring_chain):
        issues.append("unverified_ring_chain_sampling")
    return issues, hydrated


def _scientific_signature(result):
    """Validate required metadata and produce independently inspectable hashes."""
    issues, signature = [], {}
    for key in ("original_species", "species_scored"):
        try:
            signature[key] = _species_signature(result.get(key))
        except (ValueError, TypeError, OverflowError) as exc:
            issues.append(key + ": " + str(exc))
    conditions = result.get("conditions")
    issues.extend(condition_issues(conditions))
    if isinstance(conditions, Mapping):
        if "T_K" in conditions and "temperature_K" in conditions and conditions["T_K"] != conditions["temperature_K"]:
            issues.append("contradictory_temperature_fields")
        try:
            signature["conditions"] = condition_fingerprint(conditions)
        except (ValueError, TypeError):
            issues.append("invalid_conditions_JSON")
    quantity, units = result.get("quantity"), result.get("energy_units")
    if quantity != "standard_transformed_reaction_gibbs":
        issues.append("unsupported_or_missing_quantity")
    if units != "kJ/mol":
        issues.append("unsupported_or_missing_energy_units")
    signature.update(quantity=quantity, energy_units=units,
                     reported_rounding_kj=result.get("reported_rounding_kj"))
    resolution = result.get("reported_rounding_kj")
    if resolution is not None and (not _finite(resolution) or resolution < 0):
        issues.append("invalid_reported_rounding")
    config = result.get("config")
    if not isinstance(config, Mapping) or not config:
        issues.append("missing_estimator_configuration")
    else:
        issues.extend(_producer_issues(config))
        sampling_issues, hydration_states = _sampling_provenance(result, config)
        issues.extend(sampling_issues)
        signature["hydration_state_identities"] = hydration_states
        if any(not isinstance(config.get(key), str) or not config[key].strip()
               for key in ("model", "physics", "solv_model")):
            issues.append("missing_model_physics_or_solvent_configuration")
        if not _seed(config.get("conformer_seed_offset")):
            issues.append("missing_or_invalid_config_seed_offset")
        # No recursive exclusions or guessed equivalences: only this one knob varies.
        signature["configuration_except_seed"] = {key: value for key, value in config.items()
                                                  if key != "conformer_seed_offset"}
        if "conditions" in config:
            try:
                declared = json.loads(config["conditions"]) if isinstance(config["conditions"], str) else config["conditions"]
                if condition_fingerprint(declared) != signature.get("conditions"):
                    issues.append("result_conditions_disagree_with_configuration")
            except (ValueError, TypeError):
                issues.append("invalid_configuration_conditions")
    routes = result.get("routes")
    if not isinstance(routes, Mapping) or not routes:
        issues.append("missing_routing_provenance")
    else:
        required_routes = ("truncated", "ph0", "cofactor_ring", "coa_core", "ntp_core", "rearrangement_reroute")
        if routes.get("input_balanced") is not True or any(type(routes.get(key)) is not bool for key in required_routes):
            issues.append("incomplete_routing_decisions")
        signature["routing_decisions"] = {key: value for key, value in routes.items() if key not in _ROUTE_DIAGNOSTICS}
        methods = routes.get("species_diagnostics", {})
        if isinstance(methods, Mapping):
            signature["species_methods"] = {key: value.get("method") for key, value in methods.items()
                                            if isinstance(value, Mapping)}
        else:
            issues.append("invalid_species_diagnostics")
    stages = result.get("stages")
    if not isinstance(stages, Mapping) or any(not _finite(stages.get(key)) for key in ("proton", "pka_transform")):
        issues.append("missing_deterministic_proton_or_pka_stage")
    else:
        signature["deterministic_stages"] = {key: stages[key] for key in ("proton", "pka_transform")}
    signature["explicit"] = result.get("explicit")
    hashes = {}
    for key, value in signature.items():
        try:
            hashes[key] = _hash(value)
        except (ValueError, TypeError, OverflowError):
            issues.append("nonfinite_or_invalid_metadata: " + key)
    return hashes, issues


def _record(row, index):
    if not isinstance(row, Mapping):
        raise ValueError("Every replica record must be a JSON object")
    rid = row.get("reaction_id")
    if "replicate_id" in row:
        value = row["replicate_id"]
        key = ("replicate_id", value) if isinstance(value, str) and value.strip() else None
    else:
        value = row.get("seed_offset")
        key = ("seed_offset", value) if _seed(value) else None
    record = {"input_index": index, "reaction_id": rid, "replicate_id": row.get("replicate_id"),
              "seed_offset": row.get("seed_offset"), "status": row.get("status"),
              "metadata_issues": [], "dG_kj": None, "U_samp_kj": None}
    result = row.get("result")
    routes = result.get("routes", {}) if isinstance(result, Mapping) else {}
    failed = (not isinstance(row.get("status"), str) or row.get("status") not in _SUCCESS or not isinstance(result, Mapping)
              or not _finite(result.get("dG")) or bool(result.get("suspect"))
              or (isinstance(routes, Mapping) and (bool(routes.get("errors"))
                  or routes.get("input_balanced") is False or bool(routes.get("incomplete_hydration"))
                  or bool(routes.get("incomplete_ring_chain")))))
    record["estimate_available"] = not failed
    record["failure_details"] = {
        "envelope": deepcopy({k: v for k, v in row.items() if k not in {"result"}}),
        "result_suspect": result.get("suspect") if isinstance(result, Mapping) else None,
        "point_energy_missing_or_nonfinite": not isinstance(result, Mapping) or not _finite(result.get("dG")),
        "routing_errors": deepcopy(routes.get("errors")) if isinstance(routes, Mapping) else None,
    } if failed else None
    if not failed:
        record["dG_kj"] = float(result["dG"])
        sigma = result.get("U_samp")
        if sigma is not None and (not _finite(sigma) or sigma < 0):
            record["metadata_issues"].append("invalid_U_samp")
        else:
            record["U_samp_kj"] = sigma
        record["metadata_fingerprints"], issues = _scientific_signature(result)
        record["metadata_issues"].extend(issues)
        config = result.get("config")
        seed = config.get("conformer_seed_offset") if isinstance(config, Mapping) else None
        record["effective_seed_offset"] = seed if _seed(seed) else None
        conv = config.get("conv") if isinstance(config, Mapping) else None
        if (_seed(seed) and isinstance(conv, (list, tuple)) and len(conv) == 3
                and isinstance(conv[2], int) and not isinstance(conv[2], bool) and conv[2] > 0):
            record["declared_seed_window"] = [seed + 1, seed + conv[2]]
        if "seed_offset" in row and (not _seed(row["seed_offset"]) or seed != row["seed_offset"]):
            record["metadata_issues"].append("envelope_seed_disagrees_with_configuration")
        conditions = result.get("conditions")
        record["temperature_K"] = conditions.get("temperature_K", conditions.get("T_K")) if isinstance(conditions, Mapping) else None
        record["reported_rounding_kj"] = result.get("reported_rounding_kj")
    elif _seed(row.get("seed_offset")):
        record["effective_seed_offset"] = row["seed_offset"]
    return rid, key, record


def _metrics(records, fold):
    values = [r["dG_kj"] for r in records]
    span = max(values)-min(values)
    rt = R_KJ*float(records[0]["temperature_K"])
    tolerance = rt*math.log(fold) if fold is not None else None
    if not math.isfinite(span) or not math.isfinite(rt) or rt <= 0 or (tolerance is not None and not math.isfinite(tolerance)):
        raise ValueError("nonfinite_thermodynamic_scaling")
    scaled = span/rt
    if not math.isfinite(scaled):
        raise ValueError("nonfinite_thermodynamic_scaling")
    ratio = math.exp(scaled) if scaled < math.log(float.fromhex('0x1.fffffffffffffp+1023')) else None
    pairs = []
    for i, left in enumerate(records):
        for right in records[i+1:]:
            delta = abs(left["dG_kj"]-right["dG_kj"])
            sigmas = (left["U_samp_kj"], right["U_samp_kj"])
            denominator = math.hypot(*sigmas) if all(s is not None for s in sigmas) else None
            if denominator is not None and not math.isfinite(denominator):
                denominator, relative, status = None, None, "sampling_sigma_scale_overflow"
            elif denominator is None:
                relative, status = None, "sampling_sigma_unavailable"
            elif denominator == 0:
                relative, status = (0., "both_difference_and_sigma_zero") if delta == 0 else (None, "nonzero_difference_with_zero_reported_sigma")
            else:
                relative, status = delta/denominator, "descriptive_only"
                if not math.isfinite(relative): relative, status = None, "ratio_overflow"
            pairs.append({"input_indices": [left["input_index"], right["input_index"]],
                          "absolute_difference_kj": delta, "quadrature_U_samp_kj": denominator,
                          "difference_over_quadrature_U_samp": relative, "status": status})
    return {"point_range_kj": span, "RT_kj_per_mol": rt, "range_over_RT": scaled,
            "equivalent_equilibrium_constant_ratio": ratio, "equilibrium_ratio_overflow": ratio is None,
            "log10_equilibrium_constant_ratio": scaled/math.log(10),
            "maximum_equilibrium_fold_change": fold, "tolerance_kj": tolerance,
            "pairwise_sampling_comparisons": pairs,
            "sampling_comparison_scope": "Descriptive differences versus supplied trajectory-derived U_samp; quadrature is a reference scale, not a calibrated hypothesis test or confidence interval"}


def audit_sampling_replicates(records, expected_replicates, *, maximum_equilibrium_fold_change=None):
    """Audit explicit envelopes against reaction -> expected replica identities.

    Expected integers refer to ``seed_offset``; strings refer to ``replicate_id``.
    Each envelope needs ``reaction_id``, explicit identity, ``status`` and ``result``.
    Effective seed offsets must be distinct even when replica labels differ.
    Declare the tolerance explicitly; omitting it yields descriptive results only.
    """
    if not isinstance(expected_replicates, Mapping) or not expected_replicates:
        raise ValueError("An explicit nonempty expected-replicate manifest is required")
    fold = maximum_equilibrium_fold_change
    if fold is not None and (not _finite(fold) or fold < 1):
        raise ValueError("maximum_equilibrium_fold_change must be finite and at least 1")
    expected = {}
    for rid, values in expected_replicates.items():
        if not isinstance(rid, str) or not rid or not isinstance(values, (list, tuple)) or not values:
            raise ValueError("Expected manifest needs nonempty reaction IDs and replica lists")
        keys = [_replicate_key(v) for v in values]
        if len(set(keys)) != len(keys): raise ValueError("Duplicate expected replicate identities")
        expected[rid] = keys
    grouped, unassigned = defaultdict(list), []
    for index, row in enumerate(records):
        rid, key, record = _record(row, index)
        if not isinstance(rid, str) or rid not in expected or key is None:
            record["identity_issue"] = "unidentified_or_unexpected_reaction_or_replica"
            unassigned.append(record)
        else:
            grouped[rid].append((key, record))
    reactions = {}
    for rid, wanted in expected.items():
        entries = grouped[rid];counts = Counter(key for key, _ in entries)
        observed = [row for _, row in entries]
        missing = [key for key in wanted if key not in counts]
        unexpected = [key for key in counts if key not in wanted]
        duplicate = [key for key, n in counts.items() if n > 1]
        seeds = Counter(row.get("effective_seed_offset") for row in observed if row.get("effective_seed_offset") is not None)
        duplicate_seeds = [seed for seed, n in seeds.items() if n > 1]
        overlaps = []
        for i, left in enumerate(observed):
            for right in observed[i+1:]:
                a, b = left.get("declared_seed_window"), right.get("declared_seed_window")
                if a and b and max(a[0], b[0]) <= min(a[1], b[1]):
                    overlaps.append({"input_indices": [left["input_index"], right["input_index"]],
                                     "shared_seed_window": [max(a[0], b[0]), min(a[1], b[1])]})
        out = {"reaction_id": rid, "expected_replicates": deepcopy(list(expected_replicates[rid])),
               "records": observed, "missing_replicates": [v for _, v in missing],
               "unexpected_replicates": [v for _, v in unexpected],
               "duplicate_replicates": [v for _, v in duplicate], "duplicate_seed_offsets": duplicate_seeds,
               "overlapping_seed_windows": overlaps,
               "comparable": False, "within_observed_replicate_tolerance": False, "metrics": None}
        if duplicate or duplicate_seeds or unexpected:
            out["status"] = "replicate_identity_conflict"
        elif overlaps:
            out["status"] = "overlapping_sampling_seed_windows"
        elif missing or any(not row["estimate_available"] for row in observed):
            out["status"] = "incomplete_replicate_set"
        elif len(wanted) < 2:
            out["status"] = "insufficient_replicates"
        elif any(row["metadata_issues"] for row in observed):
            out["status"] = "unverifiable_replicate_set"
        else:
            axes = set().union(*(row["metadata_fingerprints"] for row in observed))
            mismatch = sorted(axis for axis in axes if len({row["metadata_fingerprints"].get(axis) for row in observed}) != 1)
            out["incompatible_metadata_axes"] = mismatch
            if mismatch:
                out["status"] = "incompatible_replicates"
            else:
                try:
                    metrics = _metrics(observed, fold)
                    out.update(comparable=True, metrics=metrics)
                    within = fold is not None and metrics["point_range_kj"] <= metrics["tolerance_kj"]
                    out.update(within_observed_replicate_tolerance=within,
                               status=PASS if within else ("exceeds_observed_replicate_tolerance" if fold is not None else "descriptive_only"))
                except ValueError as exc:
                    out.update(status="unverifiable_replicate_set", numeric_issue=str(exc))
        reactions[rid] = out
    statuses = Counter(r["status"] for r in reactions.values())
    return {"schema": "metag_sampling_replicates_v1", "scope": _SCOPE,
            "maximum_equilibrium_fold_change": fold, "reactions": reactions,
            "unassigned_records": unassigned,
            "summary": {"reactions_expected": len(expected), "replicas_expected": sum(map(len, expected.values())),
                        "records_supplied": sum(len(g) for g in grouped.values())+len(unassigned),
                        "unassigned_records": len(unassigned), "status_counts": dict(statuses),
                        "comparable_reactions": sum(r["comparable"] for r in reactions.values()),
                        "all_within_observed_replicate_tolerance": not unassigned and statuses.get(PASS, 0) == len(expected)}}
