"""Sensitivities must reproduce perturbations without changing point energies."""
import math

import pytest

from metag.energetics.protonation import reaction_transform, reaction_transform_diagnostics


@pytest.mark.parametrize("kind", ["acid", "base"])
@pytest.mark.parametrize("side", ["react", "prod"])
@pytest.mark.parametrize("pka", [4.3, 7., 9.5])
def test_gradients_match_central_differences(kind, side, pka):
    ph, rtln, mult, h = 7., 5.708, 2.5, 1e-5
    site = [side, pka, kind, mult]
    report = reaction_transform_diagnostics([site], ph, rtln)
    dpka = (reaction_transform([[side, pka+h, kind, mult]], ph, rtln)
            - reaction_transform([[side, pka-h, kind, mult]], ph, rtln)) / (2*h)
    dph = (reaction_transform([site], ph+h, rtln)
           - reaction_transform([site], ph-h, rtln)) / (2*h)
    assert report["terms"][0]["dG_d_pKa_kj_per_unit"] == pytest.approx(dpka, abs=1e-8)
    assert report["dG_d_pH_kj_per_unit"] == pytest.approx(dph, abs=1e-8)
    assert report["correction_kj"] == reaction_transform([site], ph, rtln)


def test_same_pool_on_both_sides_has_zero_ph_derivative():
    sites = [["react", 4.3, "acid", 2], ["prod", 4.3, "acid", 2]]
    report = reaction_transform_diagnostics(sites, 7., 5.708)
    assert report["correction_kj"] == report["dG_d_pH_kj_per_unit"] == 0
    # Equal values alone cannot collapse separate pKa parameters into one.
    assert len(report["terms"]) == 2


@pytest.mark.parametrize("pka", [-10000., 10000.])
def test_extreme_site_populations_remain_finite(pka):
    report = reaction_transform_diagnostics([["prod", pka]], 7., 5.708)
    assert 0 <= report["terms"][0]["ionized_site_fraction"] <= 1
    assert math.isfinite(report["correction_kj"])


def test_empty_model_reports_zero_sensitivity():
    report = reaction_transform_diagnostics([], 7., 5.708)
    assert report["correction_kj"] == report["dG_d_pH_kj_per_unit"] == 0


@pytest.mark.parametrize("pka", [True, float("nan"), float("inf")])
def test_invalid_parameters_cannot_become_diagnostics(pka):
    with pytest.raises(ValueError):
        reaction_transform_diagnostics([["prod", pka]], 7., 5.708)
