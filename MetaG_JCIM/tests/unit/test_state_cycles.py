import math

import pytest

from metag.energetics.state_cycles import acid_hydration_cycle
from metag.energetics.microstates import R_KJ


def model(ph, **kwargs):
    return acid_hydration_cycle(pka_oxo=2, kh_acid=10, kh_ion=0.1, ph=ph, **kwargs)


def test_both_experimental_limits_and_transition_are_recovered():
    assert model(-100)["apparent_hydration_constant"] == pytest.approx(10)
    assert model(100)["apparent_hydration_constant"] == pytest.approx(0.1)
    assert model(2)["apparent_hydration_constant"] == pytest.approx(5.05)


def test_pka_and_hydration_paths_close_and_population_ratios_agree():
    result = model(3, measured_pka_hydrate=4)
    p = result["populations"]
    assert p["HA_hydrate"] / p["HA_oxo"] == pytest.approx(10)
    assert p["A_hydrate"] / p["A_oxo"] == pytest.approx(0.1)
    assert p["A_hydrate"] / p["HA_hydrate"] == pytest.approx(10 ** (3 - 4))
    assert sum(p.values()) == pytest.approx(1)
    assert result["cycle_residual_kj"] == pytest.approx(0)


def test_inconsistent_fourth_measurement_remains_visible_without_refitting():
    result = model(3, measured_pka_hydrate=5)
    assert result["cycle_residual_kj"] == pytest.approx(R_KJ * 298.15 * math.log(10))
    assert result["populations"] == model(3)["populations"]


def test_hydration_correction_does_not_double_count_oxo_protonation():
    result = model(2)
    oxo_only = -R_KJ * 298.15 * math.log(2)
    assert result["G"] - oxo_only == pytest.approx(result["hydration_mixture_correction_kj"])


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_equilibrium_constants_rejected(bad):
    with pytest.raises(ValueError):
        acid_hydration_cycle(pka_oxo=2, kh_acid=bad, kh_ion=1, ph=7)
