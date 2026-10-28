"""State-function and covariance constraints, independent of experimental labels."""
from copy import deepcopy

import numpy as np
import pytest

from metag.validation.network_consistency import (
    audit_network, coherent_covariance, cycle_audit, protocol_fingerprint, condition_fingerprint,
    state_identity, uncertainty_provenance,
)

A, B, C = "CCO", "CC=O", "CC(=O)O"
CONDITIONS = {"temperature_K": 298.15, "pH": 7., "ionic_strength_M": 0., "pMg": None,
              "solute_standard_state_M": 1., "water_standard_state": "pure_liquid"}


def reaction(rid, start, end, energy, *, extent=1.):
    species = {"start": [-extent, 0, start], "end": [extent, 0, end]}
    return {"reaction": rid, "original_species": species, "species_scored": deepcopy(species),
            "dG": energy * extent, "conditions": deepcopy(CONDITIONS), "config": {"physics": "test", "truncate": False},
            "energy_units": "kJ/mol", "quantity": "standard_transformed_reaction_gibbs"}


def network():
    return [reaction("ab", A, B, 3), reaction("bc", B, C, 5), reaction("ac", A, C, 8)]


def test_direct_cycle_closure_retains_reversal_extent_and_fractional_coefficients():
    records = [reaction("ab", A, B, 3, extent=.5), reaction("cb", C, B, -5, extent=2), reaction("ac", A, C, 8)]
    report = audit_network(records)
    assert report["summary"]["n_independent_cycles_audited"] == 1
    assert report["summary"]["n_cycle_basis_violations"] == 0
    assert abs(report["protocol_groups"][0]["cycles"]["worst_cycles"][0]["closure_kj"]) < 1e-12


def test_inconsistency_is_detected_without_projection_or_energy_mutation():
    records = network()
    records[-1]["dG"] += 12
    saved = deepcopy(records)
    report = audit_network(records)
    assert records == saved
    assert report["summary"]["n_cycle_basis_violations"] == 1
    assert abs(report["protocol_groups"][0]["cycles"]["worst_cycles"][0]["closure_kj"]) == pytest.approx(12)


def test_unknown_failure_and_protocol_denominators_are_not_erased():
    records = network()
    records += [reaction("failed", A, B, 3), reaction("unknown", B, C, 5), reaction("other_pH", A, C, 8)]
    records[-3].update(dG=None, status="budget_exceeded", reason="retained timeout")
    records[-2].pop("config")
    records[-1]["conditions"]["pH"] = 8
    report = audit_network(records)
    assert report["summary"]["n_panel"] == 6
    assert report["summary"]["n_estimates_available"] == 5
    assert report["summary"]["n_network_admitted"] == 4
    assert report["summary"]["n_protocol_groups"] == 2
    assert report["summary"]["n_independent_cycles_audited"] == 1
    assert report["records"][3]["failure_reason"] == "retained timeout"


def test_core_substitution_context_is_detected_with_explicit_original_mapping():
    records = network()
    records[0]["species_scored"]["core"] = records[0]["species_scored"].pop("start")
    records[0]["species_scored"]["core"][2] = "CO"
    records[0]["representation_map"] = {"start": "core"}
    report = audit_network(records)
    assert report["summary"]["n_context_dependent_state_representations"] == 1
    assert report["summary"]["n_reactions_with_representation_changes"] == 1
    assert report["context_dependent_representations"][0]["state_id"] == state_identity(A, 0)
    records[0].pop("representation_map")
    report = audit_network(records)
    assert report["records"][0]["representation_unknown"] == ["start"]


def test_stereo_charge_isotope_and_explicit_compartment_ids_remain_distinct():
    assert state_identity("C/C=C/C", 0) != state_identity("C/C=C\\C", 0)
    assert state_identity("CC(=O)[O-]", -1) != state_identity(C, 0)
    assert state_identity("[13CH3]CO", 0) != state_identity(A, 0)
    assert state_identity("[CH3:1][CH2:2][OH:3]", 0) == state_identity(A, 0)
    with pytest.raises(ValueError, match="charge differs"):
        state_identity("CC(=O)[O-]", 0)
    records = network()
    records[-1]["state_ids"] = {"start": "A_other_compartment"}
    records[-1]["state_identity_provenance"] = {"source": "explicit compartment declaration"}
    assert audit_network(records)["summary"]["n_independent_cycles_audited"] == 0


def test_conflicting_shared_state_ids_are_quarantined_not_merged():
    records = network()
    records[0]["state_ids"] = {"start": "pool"}
    records[1]["state_ids"] = {"start": "pool"}
    report = audit_network(records)
    assert report["summary"]["n_original_state_identity_conflicts"] == 1
    assert report["summary"]["n_network_admitted"] == 1


def test_pseudoisomer_pooling_requires_explicit_members_and_matched_provenance():
    records = [reaction("acid", C, A, 2), reaction("ion", A, "CC(=O)[O-]", -2)]
    records[1]["original_species"]["end"][1] = -1
    records[1]["species_scored"]["end"][1] = -1
    records[0]["state_ids"] = {"start": "acid_pool"}
    records[1]["state_ids"] = {"end": "acid_pool"}
    groups = {"acid_pool": {"member_identities": [state_identity(C, 0), state_identity("CC(=O)[O-]", -1)],
        "conditions": deepcopy(CONDITIONS), "source": "independently specified acid equilibrium",
        "reference_state_id": "neutral_acid", "reservoir_convention": "buffered proton at fixed pH; no Mg states"}}
    assert audit_network(records)["summary"]["n_network_admitted"] == 0
    result = audit_network(records, state_groups=groups)
    assert result["summary"]["n_network_admitted"] == 2
    assert result["summary"]["n_independent_cycles_audited"] == 1
    groups["acid_pool"]["conditions"]["pH"] = 8
    assert audit_network(records, state_groups=groups)["summary"]["n_network_admitted"] == 0
    with pytest.raises(ValueError, match="Pseudoisomer"):
        audit_network(records, state_groups={"acid_pool": {"member_identities": groups["acid_pool"]["member_identities"]}})


def test_correlated_species_uncertainty_cancels_cycles_and_retains_shared_error():
    matrix = np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]])
    provenance = {"source": "supplied validation model, not fitted test errors", "scope": "three shared species states", "units": "(kJ/mol)^2"}
    settings = {"physics": "test"}
    provenance.update(conditions_fingerprint=condition_fingerprint(CONDITIONS), settings_fingerprint=protocol_fingerprint(settings))
    result = coherent_covariance(matrix, np.diag([1, 4, 9]), species_keys=["A", "B", "C"],
                                 covariance_species_keys=["A", "B", "C"], provenance=provenance,
                                 conditions=CONDITIONS, settings=settings)
    assert result["reaction_covariance"][0][1] == -4
    assert result["max_abs_cycle_covariance"] < 1e-12
    assert not result["coverage_guaranteed"]
    # Independent reaction perturbations would destroy this required covariance.
    cycle = np.asarray(result["cycle_coefficients"])[0]
    assert cycle @ np.diag(np.diag(result["reaction_covariance"])) @ cycle > 0
    with pytest.raises(ValueError, match="key order"):
        coherent_covariance(matrix, np.eye(3), species_keys=["A", "B", "C"],
                            covariance_species_keys=["B", "A", "C"], provenance=provenance,
                            conditions=CONDITIONS, settings=settings)
    with pytest.raises(ValueError, match="match the network protocol"):
        coherent_covariance(matrix, np.eye(3), species_keys=["A", "B", "C"],
                            covariance_species_keys=["A", "B", "C"], provenance=provenance,
                            conditions={**CONDITIONS, "pH": 8}, settings=settings)


def test_interval_absence_and_internal_calibration_do_not_establish_external_coverage():
    row = reaction("ab", A, B, 3)
    row.update(ci95=[-10, 10], sigma_pred=3, coverage_calibrated=True,
               calibration_basis="TECRDB internal nested grouped CV")
    artifact = {"config": deepcopy(row["config"]), "per_reaction": {"ab": {}}}
    info = uncertainty_provenance(row, artifact)
    assert info["configuration_match"] and info["calibration_target_exposure"]
    assert info["interval_available"] and not info["external_coverage_established_by_audit"]
    row["config"]["truncate"] = 0  # bool/numeric equality cannot hide a configuration change.
    assert not uncertainty_provenance(row, artifact)["configuration_match"]
    row.pop("ci95")
    assert not uncertainty_provenance(row, artifact)["interval_available"]
    assert protocol_fingerprint(False) != protocol_fingerprint(0)
    artifact["per_reaction"] = [{"rid": "ab"}, {"rid": "bc"}]
    assert uncertainty_provenance(row, artifact)["calibration_target_exposure"]


def test_equal_numeric_conditions_share_protocol_but_explicit_state_ids_need_provenance():
    records = network()
    records[-1]["conditions"]["pH"] = 7  # same physical declaration as 7.0
    assert audit_network(records)["summary"]["n_independent_cycles_audited"] == 1
    records[-1]["state_ids"] = {"start": "unexplained_pool"}
    report = audit_network(records)
    assert report["summary"]["n_network_admitted"] == 2
    assert "explicit_state_ID_provenance_unavailable" in report["records"][-1]["metadata_issues"]


def test_cycle_budget_and_rounding_uncertainty_are_explicit():
    columns = [{"a": -1, "b": 1}, {"b": -1, "c": 1}, {"a": -1, "c": 1}]
    budget = cycle_audit(columns, [3, 5, 8], ["ab", "bc", "ac"], maximum_component_reactions=2)
    assert budget["status"] == "incomplete" and budget["n_independent_cycles_audited"] == 0
    rounded = cycle_audit(columns, [3.0, 5.0, 8.1], ["ab", "bc", "ac"], rounding_kj=[.1, .1, .1])
    assert rounded["worst_cycles"][0]["rounding_bound_kj"] == pytest.approx(.15)
    assert rounded["n_cycle_basis_violations"] == 0


def test_incomplete_or_invalid_conventions_remain_counted():
    records = network()
    records[0]["conditions"]["ionic_strength_M"] = False
    records[1].pop("quantity")
    records[2]["config"]["bad"] = float("nan")
    result = audit_network(records, calibration_artifact={"config": {"physics": "test"}})
    assert result["summary"]["n_panel"] == 3
    assert result["summary"]["n_network_admitted"] == 0


def test_partial_or_routing_failures_cannot_hide_behind_finite_point_estimates():
    records = network()
    records[0]["routes"] = {"incomplete_hydration": ["unresolved_state"]}
    records[1]["routes"] = {"errors": ["routing transformation failed"]}
    records[2]["routes"] = {"warnings": ["large molecule diagnostic"]}
    report = audit_network(records)
    assert report["summary"]["n_network_admitted"] == 1
    assert report["summary"]["prediction_status_counts"] == {"withheld": 2, "estimated": 1}
    assert report["records"][0]["failure_diagnostics"]["incomplete_hydration"] == ["unresolved_state"]
