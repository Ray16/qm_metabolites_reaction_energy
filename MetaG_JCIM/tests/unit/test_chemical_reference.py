"""Thermodynamic identities independent of any benchmark-fitted number."""
from dataclasses import replace
import math

import pytest

from metag.validation.chemical_reference import (
    AcidLadder, PoolTerm, R_KJ, convert_chemical_reference, molal_to_molar_shift,
)


def ladder(standard="molar_1M"):
    return AcidLadder("acid", ("AH", "A-"), 0, (4.3,), 298.15,
                      standard, "synthetic exact acid identity", "proton-count macrostates")


def args():
    model = ladder()
    return dict(chemical_dg_kj=R_KJ * 298.15 * math.log(10) * 4.3,
                chemical_standard_state="molar_1M", temperature_K=298.15,
                target_temperature_K=298.15, target_pH=7.0,
                terms=[PoolTerm(-1, model, 0, "AH"), PoolTerm(1, model, 1, "A-")],
                proton_coefficient=1, water_coefficient=0, density_kg_L=0.99704702,
                source="synthetic dissociation G = RT ln(10) pKa",
                identity_provenance="AH -> A- + H+ is atom/charge-balanced")


@pytest.mark.parametrize("ph", [-5.0, 4.3, 7.0, 20.0])
@pytest.mark.parametrize("standard", ["molar_1M", "molal_1mol_per_kg"])
def test_acid_dissociation_closes_after_pooling(ph, standard):
    a = args()
    model = ladder(standard)
    a.update(target_pH=ph, chemical_standard_state=standard,
             terms=[PoolTerm(-1, model, 0, "AH"), PoolTerm(1, model, 1, "A-")])
    result = convert_chemical_reference(**a)
    assert result["transformed_dg_kj"] == pytest.approx(0.0, abs=2e-12)


def test_standard_shift_sign_water_exclusion_and_density_identity():
    rho, t = .99704702, 298.15
    assert molal_to_molar_shift(t, 1, rho) > 0
    assert molal_to_molar_shift(t, 1, rho) == pytest.approx(.007331, abs=1e-6)
    # The same physical chemical potential must be invariant under c=rho*m.
    m = .013
    shift = molal_to_molar_shift(t, 1, rho)
    assert shift + R_KJ*t*math.log(rho*m) == pytest.approx(R_KJ*t*math.log(m))
    a = args()
    a["chemical_standard_state"] = "molal_1mol_per_kg"
    a["water_coefficient"] = -20
    result = convert_chemical_reference(**a)
    assert result["delta_n_aqueous"] == 1  # explicit H+ counts, liquid water does not
    assert result["standard_state_shift_kj"] == pytest.approx(shift)


def test_reverse_fractional_scale_and_inert_pool():
    model = AcidLadder("inert", ("X",), 0, (), 298.15, "molar_1M", "test", "one state")
    a = args()
    a.update(chemical_dg_kj=3.2, terms=[PoolTerm(1, model, 0, "X")], proton_coefficient=0)
    baseline = convert_chemical_reference(**a)
    a.update(chemical_dg_kj=-1.6, terms=[PoolTerm(-.5, model, 0, "X")])
    assert convert_chemical_reference(**a)["transformed_dg_kj"] == pytest.approx(-.5*baseline["transformed_dg_kj"])


def test_sensitivities_against_finite_difference_and_shared_pka_cancellation():
    a = args()
    result = convert_chemical_reference(**a)
    for key in ("target_pH", "density_kg_L", "chemical_dg_kj"):
        plus, minus = dict(a), dict(a)
        plus[key] += 1e-5
        minus[key] -= 1e-5
        difference = (convert_chemical_reference(**plus)["transformed_dg_kj"] -
                      convert_chemical_reference(**minus)["transformed_dg_kj"]) / 2e-5
        assert result["parameter_jacobian"][key] == pytest.approx(difference, abs=1e-8)
    # Perturb both pool copies together: one shared constant, not two independent errors.
    modified = replace(ladder(), pkas=(4.3 + 1e-5,))
    b = dict(a, terms=[PoolTerm(-1, modified, 0, "AH"), PoolTerm(1, modified, 1, "A-")])
    difference = (convert_chemical_reference(**b)["transformed_dg_kj"] - result["transformed_dg_kj"]) / 1e-5
    assert result["parameter_jacobian"]["pKa:acid:0"] == pytest.approx(difference, abs=1e-8)
    # Gchem=RTln10*pKa has equal and opposite dependence: covariance closes this identity.
    assert result["parameter_jacobian"]["pKa:acid:0"] + R_KJ*298.15*math.log(10) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("field,value", [
    ("target_temperature_K", 303.15), ("target_ionic_strength", .25),
    ("target_free_magnesium_activity", .001), ("water_activity", .8),
    ("chemical_standard_state", "unknown"), ("proton_coefficient", 0),
    ("density_kg_L", 0), ("target_pH", float("nan")),
])
def test_unsupported_or_invalid_contract_rejected(field, value):
    a = args()
    a[field] = value
    with pytest.raises(ValueError):
        convert_chemical_reference(**a)


def test_wrong_reference_state_or_inconsistent_pool_rejected():
    with pytest.raises(ValueError, match="reference identity"):
        PoolTerm(1, ladder(), 0, "A-")
    a = args()
    other = replace(ladder(), pkas=(4.4,))
    a["terms"] = [PoolTerm(-1, ladder(), 0, "AH"), PoolTerm(1, other, 1, "A-")]
    with pytest.raises(ValueError, match="conflicting"):
        convert_chemical_reference(**a)


@pytest.mark.parametrize("density", [.99704702, 1.0])
def test_molal_density_jacobian_includes_pka_scale_change(density):
    a = args()
    model = ladder("molal_1mol_per_kg")
    a.update(chemical_standard_state="molal_1mol_per_kg", density_kg_L=density,
             terms=[PoolTerm(-1, model, 0, "AH"), PoolTerm(1, model, 1, "A-")])
    result = convert_chemical_reference(**a)
    # ChemicalG and acidpKa standards shift together; the pooledidentity stayszero.
    assert result["parameter_jacobian"]["density_kg_L"] == pytest.approx(0.0, abs=1e-12)
    values = [convert_chemical_reference(**dict(a, density_kg_L=density + d))["transformed_dg_kj"]
              for d in (-1e-5, 1e-5)]
    assert (values[1] - values[0]) / 2e-5 == pytest.approx(0.0, abs=1e-8)
