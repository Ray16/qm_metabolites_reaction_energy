import numpy as np
import pytest

from metag.assembly import reaction_covariance
from metag.validation.evaluation import evaluate_predictions, selective_error_curve
from metag.tools.evaluate import evaluate_panel
from metag.validation.evaluation_conditions import condition_issues


def complete_conditions():
    return {"temperature_K": 298.15, "pH": 7, "ionic_strength_M": 0, "pMg": 14,
            "solute_standard_state_M": 1, "water_standard_state": "pure_liquid",
            "energy_units": "kJ/mol", "quantity": "standard_transformed_reaction_gibbs"}


def test_failures_and_unresolved_references_remain_visible():
    refs = {str(i): {"eligible": i != 4, "dG": 0} for i in range(5)}
    preds = {"0": {"dG": 1, "sigma_pred": 2}, "1": {"dG": 25, "sigma_pred": 2},
             "2": {"dG": None}, "3": {"dG": 1, "suspect": "unresolved"}, "4": {"dG": 5}}
    result = evaluate_predictions(refs, preds)
    assert result["summary"]["n_panel"] == 5
    assert result["summary"]["n_unresolved_references"] == 1
    assert result["summary"]["evaluation_coverage"] == 0.5
    assert result["summary"]["mae_kj"] == 13
    assert result["summary"]["n_severe_errors"] == 1
    curve = selective_error_curve(result)
    assert len(curve) == 1  # uncertainty ties cannot be broken using error
    assert curve[0]["n_accepted"] == 2
    assert curve[0]["coverage_of_eligible"] == 0.5


def test_condition_mismatch_is_not_counted_as_a_valid_error():
    refs = {"r": {"eligible": True, "dG": 0, "conditions": {"I": 0}}}
    result = evaluate_predictions(refs, {"r": {"dG": 1, "conditions": {"I": 0.25}}})
    assert result["summary"]["n_evaluated"] == 0
    assert result["summary"]["prediction_status_counts"]["condition_mismatch"] == 1


def test_empirical_interval_inclusion_does_not_claim_calibration():
    result = evaluate_predictions({"r": {"eligible": True, "dG": 1}},
                                  {"r": {"dG": 2, "ci95": [0, 4]}})
    assert result["summary"]["empirical_interval_coverage"] == 1
    assert not result["records"][0]["interval_calibration_claimed"]


def test_missing_nonfinite_and_unknown_predictions():
    refs = {"r": {"eligible": True, "dG": 1}}
    assert evaluate_predictions(refs, {})["summary"]["prediction_status_counts"]["missing_prediction"] == 1
    assert evaluate_predictions(refs, {"r": {"dG": float("nan")}})["summary"]["n_evaluated"] == 0
    with pytest.raises(ValueError):
        evaluate_predictions(refs, {"other": {"dG": 2}})


def test_failure_status_cannot_be_hidden_by_a_finite_partial_energy():
    refs = {"r": {"eligible": True, "dG": 1}}
    result = evaluate_predictions(refs, {"r": {"dG": 1, "status": "budget_exceeded"}})
    assert result["summary"]["n_evaluated"] == 0
    assert result["summary"]["prediction_status_counts"]["withheld_prediction"] == 1


def test_network_covariance_cancels_cycle_and_preserves_shared_species_error():
    # A -> B, B -> C, C -> A; every coherent energy realization closes.
    stoich = np.array([[-1, 0, 1], [1, -1, 0], [0, 1, -1]])
    result = reaction_covariance(stoich, np.diag([1, 4, 9]))
    assert result[0, 1] == -4
    assert result[0, 0] == 5
    np.testing.assert_allclose(result @ np.ones(3), 0)


@pytest.mark.parametrize("covariance", [[[1, 2], [0, 1]], [[1, 2], [2, 1]], [[1, 0], [0, float("nan")]]])
def test_invalid_covariance_is_rejected(covariance):
    with pytest.raises(ValueError):
        reaction_covariance(np.eye(2), covariance)


def test_tiny_negative_variance_is_not_tolerated_by_an_absolute_unit_floor():
    with pytest.raises(ValueError, match="positive semidefinite"):
        reaction_covariance([[1]], [[-5e-11]])


def test_small_relative_negative_mode_is_clipped_and_overflow_rejected():
    result = reaction_covariance(np.eye(2), [[1, 0], [0, -5e-11]])
    assert np.linalg.eigvalsh(result).min() >= 0
    with pytest.raises(ValueError, match="overflowed"):
        reaction_covariance([[1e308]], [[1]])


def test_large_finite_covariance_is_not_destroyed_by_symmetrization():
    assert reaction_covariance([[1]], [[1e308]])[0, 0] == pytest.approx(1e308)


def test_common_set_is_separate_from_each_methods_coverage():
    conditions = complete_conditions()
    refs = {r: {"eligible": True, "dG": 0, "conditions": conditions, "source": "measured"} for r in ("a", "b")}
    good = {"dG": 1, "conditions": conditions}
    panel = {"references": refs, "methods": {
        "one": {"provenance": {"regime": "development"}, "predictions": {"a": good, "b": good}},
        "two": {"provenance": {"regime": "retrospective"}, "predictions": {"a": good}}}}
    report = evaluate_panel(panel)
    assert report["common_evaluated_reactions"] == ["a"]
    assert report["methods"]["one"]["summary"]["evaluation_coverage"] == 1
    assert report["methods"]["two"]["summary"]["evaluation_coverage"] == 0.5


def test_evaluation_cli_requires_reference_provenance():
    with pytest.raises(ValueError, match="source provenance"):
        evaluate_panel({"references": {"r": {"eligible": True, "dG": 1}}, "methods": {}})


def test_native_conditions_can_be_retained_outside_a_declared_comparison():
    conditions = complete_conditions()
    refs = {"r": {"eligible": True, "dG": 0, "source": "measurement", "conditions": conditions}}
    panel = {"references": refs, "comparison_methods": ["matched"], "methods": {
        "matched": {"provenance": {"regime": "development"}, "predictions": {"r": {"dG": 1, "conditions": conditions}}},
        "native": {"provenance": {"regime": "native"}, "predictions": {"r": {"dG": 1, "conditions": {**conditions, "ionic_strength_M": .25}}}}}}
    report = evaluate_panel(panel)
    assert report["common_evaluated_reactions"] == ["r"]
    assert report["methods"]["native"]["summary"]["n_evaluated"] == 0
    assert list(report["common_set_metrics"]) == ["matched"]


def test_incomplete_identical_conditions_are_not_publication_eligible():
    panel = {"references": {"r": {"eligible": True, "dG": 0, "source": "measured",
                                    "conditions": {"temperature_K": 298.15}}}, "methods": {}}
    with pytest.raises(ValueError, match="incomplete conditions"):
        evaluate_panel(panel)


def test_structured_failure_and_warning_evidence_survive_evaluation():
    pred = {"dG": None, "suspect": "minimum validation failed", "routes": {
        "warnings": ["failed hydrate"], "species_diagnostics": {"hydrate": {"status": "unresolved"}}}}
    result = evaluate_predictions({"r": {"eligible": True, "dG": 1}}, {"r": pred})
    row = result["records"][0]
    assert row["failure_reason"] == "minimum validation failed"
    assert row["warnings"] == ["failed hydrate"]
    assert row["diagnostics"]["species_diagnostics"]["hydrate"]["status"] == "unresolved"


def test_warning_alone_does_not_silently_change_prediction_acceptance():
    pred = {"dG": 2, "routes": {"warnings": ["large molecule"]}}
    result = evaluate_predictions({"r": {"eligible": True, "dG": 1}}, {"r": pred})
    assert result["summary"]["n_evaluated"] == 1
    assert result["records"][0]["warnings"] == ["large molecule"]


@pytest.mark.parametrize("field,value", [("temperature_K", 0), ("ionic_strength_M", -1),
                                       ("energy_units", "kcal/mol"), ("quantity", "reaction_enthalpy")])
def test_wrong_quantity_units_and_invalid_conditions_are_flagged(field, value):
    assert condition_issues({**complete_conditions(), field: value})


def test_boolean_condition_cannot_enter_metrics_through_python_numeric_equality():
    conditions = complete_conditions()
    pred = {"dG": 1, "conditions": {**conditions, "ionic_strength_M": False}}
    panel = {"references": {"r": {"eligible": True, "dG": 1, "source": "measured", "conditions": conditions}},
             "methods": {"method": {"provenance": {"regime": "test"}, "predictions": {"r": pred}}}}
    report = evaluate_panel(panel)
    assert report["methods"]["method"]["summary"]["n_evaluated"] == 0
    assert report["common_evaluated_reactions"] == []
    assert "r" in report["methods"]["method"]["condition_schema_issues"]
    assert "accepted" not in pred  # auditing must not mutate the original prediction
