import math

import pytest

from metag.energetics.microstates import R_KJ, combine, relative_free_energies_from_populations


def test_single_state_is_identity():
    result = combine([{"name": "keto", "G": -123.4, "sigma": 2.0}])
    assert result["G"] == pytest.approx(-123.4)
    assert result["sigma"] == pytest.approx(2.0)
    assert result["populations"] == pytest.approx({"keto": 1.0})


def test_equal_states_gain_mixing_free_energy():
    temperature = 298.15
    result = combine([
        {"name": "alpha", "G": 10.0},
        {"name": "beta", "G": 10.0},
    ], temperature)
    assert result["G"] == pytest.approx(10.0 - R_KJ * temperature * math.log(2.0))
    assert result["populations"] == pytest.approx({"alpha": 0.5, "beta": 0.5})


def test_degeneracy_is_equivalent_to_repeated_states():
    degenerate = combine([{"name": "state", "G": 5.0, "degeneracy": 3}])
    repeated = combine([
        {"name": "a", "G": 5.0},
        {"name": "b", "G": 5.0},
        {"name": "c", "G": 5.0},
    ])
    assert degenerate["G"] == pytest.approx(repeated["G"])


def test_population_offsets_recover_input_ratio():
    offsets = relative_free_energies_from_populations({"alpha": 0.36, "beta": 0.64})
    result = combine([
        {"name": "alpha", "G": 0.0, "offset": offsets["alpha"]},
        {"name": "beta", "G": 0.0, "offset": offsets["beta"]},
    ])
    assert result["populations"] == pytest.approx({"alpha": 0.36, "beta": 0.64})


def test_uncertainty_uses_thermodynamic_population_weights():
    result = combine([
        {"name": "major", "G": 0.0, "sigma": 2.0},
        {"name": "minor", "G": 100.0, "sigma": 50.0},
    ])
    assert result["populations"]["major"] > 0.999
    assert result["sigma"] == pytest.approx(2.0, rel=1e-6)


@pytest.mark.parametrize("populations", [{}, {"a": 0.0}, {"a": -1.0}, {"a": math.inf}])
def test_invalid_populations_are_rejected(populations):
    with pytest.raises(ValueError):
        relative_free_energies_from_populations(populations)


def test_invalid_state_is_rejected():
    with pytest.raises(ValueError):
        combine([{"name": "bad", "G": 0.0, "degeneracy": 0}])
