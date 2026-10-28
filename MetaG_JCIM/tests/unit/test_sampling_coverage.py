"""Coverage diagnostics must distinguish truncation from evaluated support."""
from metag.energetics.thermal_ensemble import _coverage_info


def test_thermal_cap_does_not_hide_unevaluated_eligible_minimum():
    info = _coverage_info([0.] * 11, range(10), 15., 10)
    assert info['thermal_cap_reached']
    assert info['n_within_vertical_window'] == 11
    assert info['n_reference_thermal_approximation'] == 1
    assert info['n_outside_vertical_window'] == 0


def test_window_exclusion_is_distinct_from_cap():
    info = _coverage_info([0., 16.], [0], 15., 10)
    assert not info['thermal_cap_reached']
    assert info['n_outside_vertical_window'] == 1
    assert info['n_reference_thermal_approximation'] == 1


def test_validated_override_counts_as_corrected_outside_window():
    info = _coverage_info([0., 16.], [0, 1], 15., 10)
    assert info['n_reference_thermal_approximation'] == 0
    assert info['n_outside_vertical_window'] == 1
