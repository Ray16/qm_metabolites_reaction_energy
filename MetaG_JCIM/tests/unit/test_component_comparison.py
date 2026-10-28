from copy import deepcopy

import pytest

from metag.validation.components import compare_geometry_pair, EH_TO_KJ


def pair():
    rows = []
    for i, energy in enumerate((-100, -100.01)):
        rows.append({"xyz_sha256": str(i), "charge": -1, "spin_multiplicity": 1,
                     "functional": "wb97m-v", "basis": "def2-tzvpd", "grid_level": 3,
                     "nlc_grid_level": 3, "nlc_enabled": True,
                     "gas_converged": True, "smd_converged": True,
                     "gas_Eh": energy, "smd_Eh": energy - 0.1, "smd_cds_Eh": 0.001})
    return rows


def test_common_engine_offset_cancels_but_relative_change_is_checked():
    baseline, check = pair(), pair()
    for row in check:
        row["gas_Eh"] += 0.003
        row["smd_Eh"] += 0.003
    assert compare_geometry_pair(baseline, check)["full_pair_agreement"]
    check[1]["smd_Eh"] += 1 / EH_TO_KJ
    report = compare_geometry_pair(baseline, check)
    assert report["components"]["gas"]["passed"]
    assert not report["components"]["solvation"]["passed"]


def test_incomplete_solvent_check_does_not_discard_converged_gas_or_claim_full_validation():
    baseline, check = pair(), pair()
    check[1].pop("smd_Eh")
    report = compare_geometry_pair(baseline, check)
    assert report["components"]["gas"]["passed"]
    assert report["components"]["solvation"]["status"] == "incomplete"
    assert not report["full_pair_agreement"]


def test_missing_solvent_cds_is_not_a_complete_solution_energy_check():
    check = pair()
    check[1].pop("smd_cds_Eh")
    result = compare_geometry_pair(pair(), check)
    assert result["components"]["solution"]["status"] == "incomplete"


@pytest.mark.parametrize("field,value", [("xyz_sha256", "different"), ("charge", 0), ("grid_level", 4)])
def test_coordinate_and_protocol_mismatches_prevent_comparison(field, value):
    check = deepcopy(pair())
    check[0][field] = value
    result = compare_geometry_pair(pair(), check)
    assert not result["matched_inputs"]
    assert not result["components"]
