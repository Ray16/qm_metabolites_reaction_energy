"""Reference mapping/reservoir invariants; synthetic mutations test refusal paths."""
import json
import math
from dataclasses import replace

import pytest

from metag.energetics.microstates import R_KJ
from metag.energetics.protonation import macroscopic_transform
from metag.energetics.state_cycles import acid_hydration_cycle
from metag.energetics.state_registry import (
    Anchor, Conditions, Registry, Source, State, Transition, exchanges, resolve,
)


NATIVE = Conditions(298.15, "9:1 H2O:D2O", .15, pH=3, water_activity=1,
                    activity_scale="conditional_concentration", recipe="0.15 M solute; unit-water reference convention",
                    standard_state="conditional_concentration_ratios_unit_water_reference")
SOURCE = replace(NATIVE, pH=None)


def reference(id, **kwargs):
    return Source(id, "https://doi.org/10.1002/jps.24539", SOURCE,
                  location="Table1 parameter set a", prior_development_exposure=True,
                  reference_scope="microscopic", **kwargs)


def cycle(*, audit_pka=None):
    states = (State("HA_oxo", smiles="CC(=O)C(=O)O"),
              State("A_oxo", smiles="CC(=O)C(=O)[O-]", proton_count=-1, charge=-1),
              State("HA_hydrate", smiles="CC(O)(O)C(=O)O", water_count=1),
              State("A_hydrate", smiles="CC(O)(O)C(=O)[O-]", proton_count=-1, water_count=1, charge=-1))
    edges = (Transition("pka_oxo", "HA_oxo", "A_oxo", "pKa", 1.79, reference("pka_oxo", uncertainty=.06, uncertainty_kind="SE pKa units")),
             Transition("hydrate_acid", "HA_oxo", "HA_hydrate", "log10K", math.log10(2.353), reference("Kh_acid")),
             Transition("hydrate_ion", "A_oxo", "A_hydrate", "log10K", math.log10(.087), reference("Kh_ion", uncertainty=.002, uncertainty_kind="SE K units")))
    if audit_pka is not None:
        edges += (Transition("pka_hydrate", "HA_hydrate", "A_hydrate", "pKa", audit_pka,
                    reference("pKa_hydrate", determination="derived", dependencies=("pka_oxo", "Kh_acid", "Kh_ion"))),)
    return Registry("lopalco_pyruvate_diagnostic", states, edges,
                    "Previously exposed property data; conditional coupled states, not blind external validation")


MODEL = ("pka_oxo", "hydrate_acid", "hydrate_ion")


def run(registry=None, conditions=NATIVE, **kwargs):
    return resolve(registry or cycle(), conditions, reference_state="HA_oxo",
                   model_transition_ids=MODEL, **kwargs)


def test_coupled_registry_reproduces_existing_verified_four_state_utility():
    got = run()
    expected = acid_hydration_cycle(pka_oxo=1.79, kh_acid=2.353, kh_ion=.087, ph=3)
    assert got["status"] == "accepted"
    assert got["relative_only"]
    assert got["ensemble"]["populations"] == pytest.approx(expected["populations"])
    assert got["ensemble"]["G"] == pytest.approx(expected["G"])
    assert "sigma" not in got["ensemble"]
    assert got["transition_audit"][0]["source"]["prior_development_exposure"] is True


def test_proton_and_water_reservoir_signs_and_nonunit_water_activity():
    # Mathematical activity-equilibrium variant, explicitly not a relabelling
    # of the experimental concentration references.
    registry = cycle()
    activity = replace(NATIVE, activity_scale="dimensionless_activity", standard_state="1_M_solutes_unit_water")
    edges = tuple(replace(edge, source=Source(edge.source.id, "unit-test://activity-reservoir",
                 replace(activity, pH=None), reference_scope="microscopic")) for edge in registry.transitions)
    low_water = run(replace(registry, transitions=edges), conditions=replace(activity, water_activity=.5))
    p = low_water["ensemble"]["populations"]
    assert p["A_oxo"] / p["HA_oxo"] == pytest.approx(10 ** (3 - 1.79))
    assert p["HA_hydrate"] / p["HA_oxo"] == pytest.approx(2.353 * .5)
    assert p["A_hydrate"] / p["A_oxo"] == pytest.approx(.087 * .5)
    assert exchanges(cycle().states[0], cycle().states[3]) == {"proton": -1, "water": 1, "magnesium": 0}


def test_independent_cycle_disagreement_is_retained_without_averaging():
    registry = cycle(audit_pka=4.5)  # Deliberately inconsistent test mutation.
    result = run(registry)
    assert result["status"] == "no_estimate"
    assert result["ensemble"] is None
    assert "inconsistent_reference_cycle:pka_hydrate" in result["issues"]
    assert result["diagnostic_selected_ensemble"]["populations"] == run()["ensemble"]["populations"]
    audit = result["transition_audit"][-1]
    assert audit["cycle_residual_kj"] != 0
    assert audit["source"]["determination"] == "derived"
    assert audit["source"]["dependencies"] == ("pka_oxo", "Kh_acid", "Kh_ion")


def test_exactly_implied_fourth_edge_closes_without_changing_populations():
    implied = 1.79 + math.log10(2.353 / .087)
    assert run(cycle(audit_pka=implied))["ensemble"] == run()["ensemble"]


@pytest.mark.parametrize("change,issue", [
    ({"temperature_K": 298.0}, "condition_mismatch:temperature_K"),
    ({"solvent": "H2O"}, "condition_mismatch:solvent"),
    ({"ionic_strength_M": 0}, "condition_mismatch:ionic_strength_M"),
    ({"ionic_strength_M": None}, "missing_condition:ionic_strength_M"),
    ({"activity_scale": "dimensionless_activity"}, "condition_mismatch:activity_scale"),
    ({"recipe": "0.2 M solute; unit-water reference convention"}, "condition_mismatch:recipe"),
    ({"recipe": "unknown mixture"}, "missing_condition:recipe"),
    ({"pH": None}, "missing_reservoir:pH"),
    ({"water_activity": None}, "missing_reservoir:water_activity"),
    ({"water_activity": .5}, "condition_mismatch:water_activity"),
    ({"standard_state": None}, "missing_condition:standard_state"),
])
def test_incompatible_and_missing_conditions_return_no_estimate(change, issue):
    result = run(conditions=replace(NATIVE, **change))
    assert result["status"] == "no_estimate"
    assert result["ensemble"] is None
    assert any(issue in item for item in result["issues"])


def test_missing_transition_parameter_and_unreachable_states_are_not_dropped():
    registry = cycle()
    registry = replace(registry, transitions=registry.transitions[:-1] + (replace(registry.transitions[-1], value=None),))
    got = run(registry)
    assert "hydrate_ion:missing_parameter" in got["issues"]
    assert "unresolved_state:A_hydrate" in got["issues"]
    assert got["ensemble"] is None


def test_one_absolute_anchor_plus_measured_gaps_does_not_add_computed_state_gap():
    anchor = Anchor("HA_oxo", -400, NATIVE, "microscopic", "fixed matched-state calculation")
    got = run(anchor=anchor)
    assert got["state_energies_kj"]["HA_oxo"] == -400
    assert got["state_energies_kj"]["HA_hydrate"] == pytest.approx(-400 - R_KJ * 298.15 * math.log(2.353))
    assert got["ensemble"]["G"] == pytest.approx(run()["ensemble"]["G"] - 400)
    assert not got["relative_only"]
    with pytest.raises(ValueError, match="double count"):
        replace(anchor, includes_population_correction=True)


def test_calculated_and_measured_duplicate_gap_cannot_be_selected_together():
    registry = cycle()
    computed = Transition("computed_hydration", "HA_oxo", "HA_hydrate", "standard_gap_kj", 3,
                Source("qm", "calculation.json#matched-gap", SOURCE, determination="calculated", reference_scope="microscopic"),
                reservoir_convention="G(hydrate)-G(carbonyl)-mu_water_standard")
    registry = replace(registry, transitions=registry.transitions + (computed,))
    result = resolve(registry, NATIVE, reference_state="HA_oxo",
                     model_transition_ids=MODEL + ("computed_hydration",))
    assert result["status"] == "no_estimate"
    assert "duplicate_model_gap:computed_hydration" in result["issues"]


def test_computed_exchange_gap_requires_standard_reservoir_bookkeeping():
    registry = cycle()
    edges = list(registry.transitions)
    edges[1] = replace(edges[1], kind="standard_gap_kj", value=0,
                       source=Source("qm", "calculation.json", SOURCE, determination="calculated", reference_scope="microscopic"))
    got = run(replace(registry, transitions=tuple(edges)))
    assert "hydrate_acid:missing_standard_reservoir_convention" in got["issues"]


def test_conditional_population_gap_is_not_transformed_twice():
    registry = cycle()
    delta = R_KJ * 298.15 * math.log(10) * (1.79 - 3)
    conditional = replace(registry.transitions[0], kind="conditional_gap_kj", value=delta,
                          source=replace(registry.transitions[0].source, conditions=NATIVE))
    registry = replace(registry, transitions=(conditional,) + registry.transitions[1:])
    assert run(registry)["ensemble"]["populations"] == pytest.approx(run()["ensemble"]["populations"])
    assert run(registry, conditions=replace(NATIVE, pH=4))["ensemble"] is None


def test_macrostate_ladder_agrees_with_existing_sequential_transform():
    conditions = Conditions(298.15, "H2O", 0, pH=2, standard_state="1_M_solutes_unit_water")
    # Mathematical ladder; these constants are not package property defaults.
    source = Source("test_ladder", "unit-test://sequential-ladder", replace(conditions, pH=None), reference_scope="macroscopic")
    states = tuple(State(f"n{i}", scope="macroscopic", members=(f"micro_{i}",), proton_count=-i, charge=-i) for i in range(3))
    edges = tuple(Transition(f"e{i}", f"n{i}", f"n{i+1}", "pKa", pka, source, scope="macroscopic") for i, pka in enumerate([1, 4]))
    got = resolve(Registry("test", states, edges), conditions, reference_state="n0", model_transition_ids=("e0", "e1"))
    expected = macroscopic_transform([1, 4], 2, R_KJ * 298.15 * math.log(10), reference_deprotonations=0)
    assert list(got["ensemble"]["populations"].values()) == pytest.approx(expected["populations"])
    assert got["ensemble"]["G"] == pytest.approx(expected["G_transform"])
    with pytest.raises(ValueError, match="mapping"):
        Registry("bad", states, (replace(edges[0], scope="microscopic"),))
    with pytest.raises(ValueError, match="overlap"):
        Registry("bad", (State("micro_0", smiles="C"), states[0]), ())
    with pytest.raises(ValueError, match="double count"):
        replace(states[0], degeneracy=2)


def test_free_magnesium_binding_activity_and_missing_reservoir():
    conditions = Conditions(298.15, "H2O", .1, pMg_free=3, standard_state="1_M_solutes_unit_water")
    source = Source("test_Mg", "unit-test://Mg-binding", replace(conditions, pMg_free=None), reference_scope="macroscopic")
    states = (State("unbound", scope="macroscopic", members=("L",)),
              State("bound", scope="macroscopic", members=("LMg",), magnesium_count=1, charge=2))
    edge = Transition("binding", "unbound", "bound", "log10K", 4, source, scope="macroscopic")
    registry = Registry("Mg-test", states, (edge,))
    result = resolve(registry, conditions, reference_state="unbound", model_transition_ids=("binding",))
    p = result["ensemble"]["populations"]
    assert p["bound"] / p["unbound"] == pytest.approx(10)
    missing = resolve(registry, replace(conditions, pMg_free=None), reference_state="unbound", model_transition_ids=("binding",))
    assert "binding:missing_reservoir:pMg_free" in missing["issues"]


def test_structures_validate_reservoir_counts_and_preserve_pn_quarantine():
    registry = cycle()
    bad = replace(registry.states[-1], smiles="CC(=O)C(=O)[O-]")
    got = run(replace(registry, states=registry.states[:-1] + (bad,)))
    assert "A_hydrate:reservoir_composition_mismatch" in got["issues"]
    state = State("PN", smiles="NP(=O)(O)O")
    quarantine = resolve(Registry("PN", (state,), ()), NATIVE, reference_state="PN", model_transition_ids=())
    assert "PN:phosphoramidate_quarantine" in quarantine["issues"]
    assert quarantine["ensemble"] is None


def test_equivalent_smiles_cannot_duplicate_a_chemical_state_partition():
    states = (State("a", smiles="CC(=O)C(=O)O"), State("alias", smiles="OC(=O)C(C)=O"))
    source = reference("identity_alias")
    edge = Transition("alias_edge", "a", "alias", "conditional_gap_kj", 0, source)
    got = resolve(Registry("duplicate", states, (edge,)), NATIVE,
                  reference_state="a", model_transition_ids=("alias_edge",))
    assert any("duplicate_microscopic_identity" in issue for issue in got["issues"])
    assert got["ensemble"] is None


def test_empirical_isotope_condition_records_do_not_claim_scorer_support():
    result = run()
    assert "empirical_solvent_record_does_not_establish_production_scorer_support" in result["reference_quality_warnings"]
    assert "conditional_concentration_model_not_standard_activity_label" in result["reference_quality_warnings"]


def test_isotope_exchange_requires_a_specific_reservoir_mapping():
    states = (State("ordinary", smiles="C"), State("deuterated", smiles="[2H]C"))
    edge = Transition("exchange", "ordinary", "deuterated", "conditional_gap_kj", 0,
                      reference("test-isotope"))
    got = resolve(Registry("isotope_exchange", states, (edge,)), NATIVE,
                  reference_state="ordinary", model_transition_ids=("exchange",))
    assert "deuterated:isotope_reservoir_mapping_required" in got["issues"]
    assert got["ensemble"] is None


def test_registry_provenance_roundtrip_and_anchor_macrostate_mismatch():
    registry = Registry.from_dict(json.loads(json.dumps(cycle().to_dict())))
    assert run(registry)["ensemble"] == run()["ensemble"]
    bad = Anchor("HA_oxo", 0, NATIVE, "macroscopic", "test://wrong-ensemble")
    assert "anchor_reference_mapping_mismatch" in run(anchor=bad)["issues"]


def test_mutable_constructor_lists_cannot_change_an_existing_reference_model():
    registry = cycle()
    states, transitions = list(registry.states), list(registry.transitions)
    copied = Registry("immutable", states, transitions)
    states.clear()
    transitions.clear()
    assert len(copied.states) == 4
    assert len(copied.transitions) == 3
    assert run(copied)["status"] == "accepted"


def test_source_scope_cannot_be_relabelled_or_left_unmapped():
    registry = cycle()
    for scope, issue in (("unspecified", "missing_reference_scope"), ("macroscopic", "source_reference_scope_mismatch")):
        altered = replace(registry.transitions[0], source=replace(registry.transitions[0].source, reference_scope=scope))
        got = run(replace(registry, transitions=(altered,) + registry.transitions[1:]))
        assert any(issue in item for item in got["issues"])
        assert got["ensemble"] is None


def test_condition_id_is_canonical_and_discriminates_solvent_and_free_mg():
    assert Conditions(298, "H2O", 0).fingerprint() == Conditions(298.0, "H2O", 0.0).fingerprint()
    assert NATIVE.fingerprint() != replace(NATIVE, solvent="D2O").fingerprint()
    assert NATIVE.fingerprint() != replace(NATIVE, pMg_free=3).fingerprint()
    assert run()["condition_id"] == NATIVE.fingerprint()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_invalid_values_rejected(value):
    with pytest.raises(ValueError):
        Conditions(value, "H2O")
    with pytest.raises(ValueError):
        replace(cycle().transitions[0], value=value)


def test_pka_scope_and_exchange_mapping_and_uncertainty_provenance():
    registry = cycle()
    with pytest.raises(ValueError, match="release exactly one"):
        Registry("bad", registry.states, (replace(registry.transitions[1], kind="pKa"),))
    with pytest.raises(ValueError, match="charge change"):
        Registry("bad", (registry.states[0], replace(registry.states[1], charge=0)), registry.transitions[:1])
    with pytest.raises(ValueError, match="dependencies"):
        reference("missing", determination="derived")
    with pytest.raises(ValueError, match="uncertainty convention"):
        reference("missing", uncertainty=.1)


def test_printed_rounding_bound_allows_dependent_reference_without_changing_model():
    # Bound is specified from displayed precision, not the observed residual:
    # +/-0.005 pKa and +/-0.0005 K. Errors are not treated as independent SDs.
    half_pka = .005
    log_bounds = [max(abs(math.log10((k - .0005) / k)), abs(math.log10((k + .0005) / k)))
                  for k in (2.353, .087)]
    tolerance = R_KJ * NATIVE.temperature_K * math.log(10) * (2 * half_pka + sum(log_bounds))
    registry = cycle(audit_pka=3.23)
    strict, rounded = run(registry), run(registry, cycle_tolerance_kj=tolerance)
    assert strict["status"] == "no_estimate"
    assert rounded["status"] == "accepted"
    assert rounded["state_energies_kj"] == strict["state_energies_kj"]
    assert rounded["ensemble"] == strict["diagnostic_selected_ensemble"]
    assert rounded["transition_audit"][-1]["source"]["determination"] == "derived"
    assert rounded["transition_audit"][-1]["cycle_residual_kj"] == strict["transition_audit"][-1]["cycle_residual_kj"]
    assert run(cycle(audit_pka=4.5), cycle_tolerance_kj=tolerance)["status"] == "no_estimate"


@pytest.mark.parametrize("field,value", [("solvent", "unknown"), ("solvent", "unspecified isotope fraction"),
                                        ("standard_state", "not reported"), ("recipe", "unknown mixture")])
def test_matching_unknown_text_does_not_admit_even_a_single_state_model(field, value):
    conditions = replace(NATIVE, **{field: value})
    got = resolve(Registry("single", (State("methane", smiles="C"),), ()), conditions,
                  reference_state="methane", model_transition_ids=())
    assert got["status"] == "no_estimate"
    assert f"missing_condition:{field}" in got["issues"]


def test_ionic_strength_cannot_be_removed_from_source_matching():
    with pytest.raises(ValueError, match="ionic strength"):
        replace(cycle().transitions[0], condition_axes=("temperature_K", "solvent", "activity_scale", "standard_state"))


def test_unknown_source_provenance_cannot_be_admitted():
    registry = cycle()
    edge = replace(registry.transitions[0], source=replace(registry.transitions[0].source, locator="unknown primary source"))
    got = run(replace(registry, transitions=(edge,) + registry.transitions[1:]))
    assert "pka_oxo:missing_source_provenance" in got["issues"]
    assert got["ensemble"] is None


def test_unknown_standard_reservoir_convention_is_not_a_mapping():
    registry = cycle()
    edge = replace(registry.transitions[1], kind="standard_gap_kj", value=0,
                   source=replace(registry.transitions[1].source, determination="calculated"),
                   reservoir_convention="unknown chemical potential")
    got = run(replace(registry, transitions=(registry.transitions[0], edge, registry.transitions[2])))
    assert "hydrate_acid:missing_standard_reservoir_convention" in got["issues"]
    assert got["ensemble"] is None


@pytest.mark.parametrize("bad", [True, "0.15", float("nan"), float("inf"), 10 ** 1000])
def test_invalid_condition_numeric_types_and_overflow_are_refused(bad):
    with pytest.raises(ValueError):
        replace(NATIVE, ionic_strength_M=bad)


def test_metadata_booleans_and_reference_record_types_cannot_suppress_warnings():
    with pytest.raises(ValueError, match="exposure"):
        replace(reference("bad_exposure"), prior_development_exposure=1)
    with pytest.raises(ValueError, match="population-correction"):
        Anchor("HA_oxo", 0, NATIVE, "microscopic", "test://unknown-transform", includes_population_correction=None)
    with pytest.raises(ValueError, match="Conditions record"):
        replace(reference("bad_conditions"), conditions={"temperature_K": 298.15})
    with pytest.raises(ValueError, match="member id"):
        State("unknown_member", scope="macroscopic", members=(None,))
    with pytest.raises(ValueError, match="Anchor record"):
        run(anchor={"energy_kj": 0})
    with pytest.raises(ValueError, match="uncertainty_kind"):
        replace(reference("typed_uncertainty", uncertainty=.1, uncertainty_kind="SE"), uncertainty_kind=True)
    with pytest.raises(ValueError, match="finite"):
        State("unrepresentable_proton_count", scope="macroscopic", members=("member",), proton_count=10 ** 1000)


def test_nonfinite_converted_gap_returns_a_diagnostic_refusal():
    registry = cycle()
    huge = replace(registry.transitions[0], value=1e308)
    got = run(replace(registry, transitions=(huge,) + registry.transitions[1:]))
    assert "pka_oxo:nonfinite_converted_gap" in got["issues"]
    assert got["ensemble"] is None
    json.dumps(got, allow_nan=False)


def test_finite_energy_that_overflows_partition_weight_is_not_admitted():
    conditions = replace(NATIVE, temperature_K=.001)
    registry = Registry("single", (State("methane", smiles="C"),), ())
    anchor = Anchor("methane", -1e308, conditions, "microscopic", "unit-test://extreme-energy")
    got = resolve(registry, conditions, reference_state="methane", model_transition_ids=(), anchor=anchor)
    assert "unrepresentable_partition_weight:methane" in got["issues"]
    assert got["ensemble"] is None
    json.dumps(got, allow_nan=False)
