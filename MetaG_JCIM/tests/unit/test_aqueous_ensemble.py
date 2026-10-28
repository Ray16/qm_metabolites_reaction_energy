"""Admission and matched-set tests; synthetic states require no QM backend."""
from copy import deepcopy
import math
import pytest

from metag.energetics.aqueous_ensemble import assemble_aqueous_ensemble, compare_common_sampled_models


def model(name="target"):
    return {"model_id":name,"species_id":"anion","charge":-1,"spin_multiplicity":1,
            "condition_id":"water-1M-298.15","electronic_model_id":"DFT",
            "solvation_model_id":"SMD","thermal_model_id":"destination-RRHO",
            "force_threshold_eV_A":.03}


def minimum(key="a", geometry="a", energy=0.):
    m=model(); g=geometry*64
    return {"start_id":key,**{k:m[k] for k in ("model_id","species_id","charge","spin_multiplicity","condition_id")},
            "source_geometry_sha256":g,"destination_geometry_sha256":g,
            "electronic_geometry_sha256":g,"solvation_geometry_sha256":g,
            "electronic_model_id":"DFT","solvation_model_id":"SMD",
            "electronic_kj":energy,"solvation_kj":-5.,"thermal_kj":1.,
            "thermal_origin":"destination","thermal_geometry_sha256":g,
            "thermal_model_id":"destination-RRHO","thermal_temperature_K":298.15,
            "thermal_provenance_sha256":"f"*64,"identity_preserved":True,
            "scf_converged":True,"status":"complete","force_converged":True,"fmax_eV_A":.02,
            "minimum_validation":{"status":"passed","kind":"destination_hessian","geometry_sha256":g,
                                  "model_id":"target","condition_id":m["condition_id"],"provenance_sha256":"e"*64}}


def test_complete_documented_destination_set_has_conditional_sampled_G_only():
    result=assemble_aqueous_ensemble([minimum(),minimum("b","b",2.)],["a","b"],model())
    assert result["status"]=="eligible_conditional_sampled_ensemble"
    expected=-4.-8.314462618e-3*298.15*math.log1p(math.exp(-2./(8.314462618e-3*298.15)))
    assert result["conditional_sampled_aqueous_G_kj"]==pytest.approx(expected)
    assert result["aqueous_deltaG_kj"] is None and result["validated_CI_kj"] is None


@pytest.mark.parametrize("reason",["borrowed_gas","no_entropy","no_hessian","force_only","wrong_thermal_geometry",
                                   "wrong_temperature","wrong_electronic_geometry","wrong_solvent","wrong_state","failed_scf"])
def test_incoherent_or_partial_thermochemistry_cannot_be_admitted(reason):
    row=minimum()
    if reason=="borrowed_gas": row["thermal_origin"]="source_gas"
    elif reason=="no_entropy": row["thermal_kj"]=None
    elif reason in ("no_hessian","force_only"): row["minimum_validation"]={"status":"not_computed"}
    elif reason=="wrong_thermal_geometry": row["thermal_geometry_sha256"]="c"*64
    elif reason=="wrong_temperature": row["thermal_temperature_K"]=296.
    elif reason=="wrong_electronic_geometry": row["electronic_geometry_sha256"]="c"*64
    elif reason=="wrong_solvent": row["solvation_model_id"]="ALPB"
    elif reason=="wrong_state": row["charge"]=0
    else: row["scf_converged"]=False
    result=assemble_aqueous_ensemble([row],["a"],model())
    assert result["status"]=="diagnostic_only"
    assert result["conditional_sampled_aqueous_G_kj"] is None
    assert result["validated_CI_kj"] is None


def test_missing_failed_and_out_of_budget_starts_remain_in_fixed_denominator():
    failed=minimum("b","b");failed["status"]="budget_exceeded"
    result=assemble_aqueous_ensemble([minimum(),failed],["a","b","c"],model())
    assert result["missing_start_ids"]==["c"] and result["incomplete_start_ids"]==["b"]
    assert len(result["expected_start_ids"])==3 and result["conditional_sampled_aqueous_G_kj"] is None
    assert result["selected_fraction_in_complete_expected_destination_potential_set"] is None
    assert result["unobserved_destination_mass"] is None


def test_exact_duplicate_destination_geometry_has_no_artificial_RTln2():
    first=minimum();second=minimum("b")
    result=assemble_aqueous_ensemble([first,second],["a","b"],model(),selected_starts=["b"])
    assert len(result["basins"])==1
    assert result["conditional_sampled_aqueous_G_kj"]==-4.
    assert result["selected_fraction_in_calculated_destination_thermal_set"]==1.


def test_merged_destination_basin_with_different_source_gas_thermal_donors_is_conflict():
    first=minimum();second=minimum("b")
    first["thermal_origin"]=second["thermal_origin"]="source_gas"
    second["thermal_kj"]=3.
    result=assemble_aqueous_ensemble([first,second],["a","b"],model())
    assert result["basin_conflicts"][0]["thermal_donor_conflict"]
    assert result["thermal_diagnostic"] is None
    assert not result["complete_common_sampled_set"]


def test_different_geometries_are_not_merged_by_an_unsupported_basin_name():
    first=minimum();second=minimum("b","b")
    first["basin_id"]=second["basin_id"]="same"
    result=assemble_aqueous_ensemble([first,second],["a","b"],model())
    assert len(result["merge_errors"])==2 and result["conditional_sampled_aqueous_G_kj"] is None


def test_proven_basin_still_blocks_disagreeing_potentials_or_thermal_donors():
    first=minimum();second=minimum("b","b",2.)
    first["basin_id"]=second["basin_id"]="same"
    proof={"same":{"model_id":"target","condition_id":model()["condition_id"],"provenance_sha256":"d"*64}}
    result=assemble_aqueous_ensemble([first,second],["a","b"],model(),basin_equivalence=proof)
    assert result["basin_conflicts"][0]["potential_conflict"]


def test_gas_sampling_uncertainty_is_preserved_without_destination_CI():
    gas={"origin":"gas_only","spread_kj":.001,"scope":"observed trajectory spread, not omitted-basin bound"}
    result=assemble_aqueous_ensemble([minimum()],["a"],model(),source_sampling=gas)
    assert result["uncertainty"]["source_sampling"]==gas
    assert result["uncertainty"]["destination_sampling"] is None
    assert result["uncertainty"]["validated_CI_kj"] is None


def comparative_rows():
    a=[minimum(),minimum("b","b",3.),minimum("c","c",6.)]
    b=deepcopy(a)
    for row in b: row["model_id"]="alternative"; row["solvation_kj"]=-10.
    ma,mb=model(),model("alternative")
    return {"target":a,"alternative":b},{"target":ma,"alternative":mb}


def test_matched_whole_and_selected_sets_expose_destination_coverage():
    rows,models=comparative_rows()
    result=compare_common_sampled_models(rows,["a","b","c"],models,selected_starts=["a"])
    assert result["complete_common_sampled_set"]
    a,b=(result["models"][name] for name in ["target","alternative"])
    assert b["whole_expected_set_diagnostic"]["diagnostic_logsum_kj"]-a["whole_expected_set_diagnostic"]["diagnostic_logsum_kj"]==pytest.approx(-5.)
    assert 0<a["selected_fraction_in_whole_expected_set"]<1
    assert result["aqueous_deltaG_kj"] is None


def test_no_silent_intersection_when_one_model_lacks_a_start():
    rows,models=comparative_rows();rows["alternative"].pop()
    result=compare_common_sampled_models(rows,["a","b","c"],models,selected_starts=["a"])
    assert result["common_available_start_ids"]==["a","b"]
    assert not result["complete_common_sampled_set"]
    assert all(v["whole_expected_set_diagnostic"] is None for v in result["models"].values())


@pytest.mark.parametrize("field,value",[("source_geometry_sha256","d"*64),("destination_geometry_sha256","d"*64),
                                        ("thermal_kj",2.),("thermal_origin","source_gas")])
def test_mismatched_geometry_or_thermal_donor_blocks_matched_set(field,value):
    rows,models=comparative_rows();rows["alternative"][0][field]=value
    result=compare_common_sampled_models(rows,["a","b","c"],models,selected_starts=["a"])
    assert not result["complete_common_sampled_set"]


def test_comparison_deduplicates_same_destination_geometry_once():
    rows,models=comparative_rows()
    for name in rows: rows[name]=[rows[name][0],dict(rows[name][0],start_id="b")]
    result=compare_common_sampled_models(rows,["a","b"],models,selected_starts=["b"])
    assert result["complete_common_sampled_set"]
    assert result["models"]["target"]["whole_expected_set_diagnostic"]["diagnostic_logsum_kj"]==-4.
    assert result["models"]["target"]["selected_fraction_in_whole_expected_set"]==1.


@pytest.mark.parametrize("value",[float("nan"),float("inf"),True])
def test_nonfinite_or_boolean_energies_do_not_enter_partition(value):
    row=minimum();row["electronic_kj"]=value
    result=assemble_aqueous_ensemble([row],["a"],model())
    assert result["potential_diagnostic"] is None and result["conditional_sampled_aqueous_G_kj"] is None


def test_duplicate_or_unexpected_starts_are_input_errors():
    with pytest.raises(ValueError,match="Duplicate"):
        assemble_aqueous_ensemble([minimum(),minimum()],["a"],model())
    with pytest.raises(ValueError,match="Unexpected"):
        assemble_aqueous_ensemble([minimum("b")],["a"],model())


def test_failed_start_with_no_geometry_or_terms_remains_a_failure_record():
    rows,models=comparative_rows()
    rows["alternative"][1]={"start_id":"b","status":"budget_exceeded"}
    result=compare_common_sampled_models(rows,["a","b","c"],models)
    assert not result["complete_common_sampled_set"]
    assert "b" in result["models"]["alternative"]["missing_or_invalid_start_ids"]
    assert result["models"]["target"]["whole_expected_set_diagnostic"] is None


def test_finite_components_whose_sum_overflows_are_not_thermodynamic_results():
    row=minimum();row["electronic_kj"]=1e308;row["solvation_kj"]=1e308
    result=assemble_aqueous_ensemble([row],["a"],model())
    assert result["potential_diagnostic"] is None and result["conditional_sampled_aqueous_G_kj"] is None


def test_missing_thermal_model_cannot_be_matched_as_None_equals_None():
    row=minimum();spec=model()
    row["thermal_model_id"]=spec["thermal_model_id"]=None
    result=assemble_aqueous_ensemble([row],["a"],spec)
    assert result["conditional_sampled_aqueous_G_kj"] is None and result["thermal_diagnostic"] is None


def test_changed_thermal_provenance_is_not_an_exact_common_donor():
    rows,models=comparative_rows();rows["alternative"][0]["thermal_provenance_sha256"]="d"*64
    result=compare_common_sampled_models(rows,["a","b","c"],models)
    assert result["thermal_donor_mismatch_start_ids"]==["a"]
    assert not result["complete_common_sampled_set"]


@pytest.mark.parametrize("force",[-.001,True,float("nan"),float("inf")])
def test_force_must_be_finite_nonnegative_and_nonboolean(force):
    row=minimum();row["fmax_eV_A"]=force
    result=assemble_aqueous_ensemble([row],["a"],model())
    assert not result["start_audit"]["a"]["force_passed"]
    assert result["conditional_sampled_aqueous_G_kj"] is None


@pytest.mark.parametrize("field,value",[("charge",False),("charge",-1.),("charge",float("nan")),
                                        ("spin_multiplicity",True),("spin_multiplicity",1.),
                                        ("spin_multiplicity",0),("spin_multiplicity",float("inf"))])
def test_record_charge_spin_types_and_values_must_match_the_integer_model(field,value):
    row=minimum();spec=model()
    if field=="charge" and value is False:
        spec["charge"]=0  # False==0 must not silently satisfy the state match.
    row[field]=value
    result=assemble_aqueous_ensemble([row],["a"],spec)
    assert not result["start_audit"]["a"]["coherent_components"]
    assert result["conditional_sampled_aqueous_G_kj"] is None


@pytest.mark.parametrize("field,value",[("charge",False),("charge",-1.),("spin_multiplicity",True),("spin_multiplicity",0)])
def test_model_charge_spin_remain_explicitly_integer(field,value):
    spec=model();spec[field]=value
    with pytest.raises(ValueError):
        assemble_aqueous_ensemble([minimum()],["a"],spec)


@pytest.mark.parametrize("temperature",[True,0.,-1.,float("nan"),float("inf"),"1.0"])
def test_record_thermal_temperature_is_positive_finite_nonboolean_number(temperature):
    row=minimum();row["thermal_temperature_K"]=temperature
    result=assemble_aqueous_ensemble([row],["a"],model(),temperature_K=1.)
    assert not result["start_audit"]["a"]["destination_thermal_passed"]
    assert result["thermal_diagnostic"] is None and result["conditional_sampled_aqueous_G_kj"] is None


@pytest.mark.parametrize("basin",[True,0,[],""])
def test_invalid_basin_ids_are_visible_merge_failures_not_implicit_geometry_groups(basin):
    row=minimum();row["basin_id"]=basin
    result=assemble_aqueous_ensemble([row],["a"],model())
    assert result["merge_errors"]==[{"start_id":"a","reason":"invalid_basin_id"}]
    assert result["conditional_sampled_aqueous_G_kj"] is None


def test_malformed_basin_proof_is_visible_and_proof_map_type_is_checked():
    row=minimum();row["basin_id"]="declared"
    result=assemble_aqueous_ensemble([row],["a"],model(),basin_equivalence={"declared":None})
    assert result["merge_errors"][0]["reason"]=="unproved_basin_equivalence"
    with pytest.raises(ValueError,match="mapping"):
        assemble_aqueous_ensemble([row],["a"],model(),basin_equivalence=["declared"])


def test_returned_nested_metadata_does_not_alias_caller_model_or_diagnostics():
    row=minimum();spec=model();spec["metadata"]={"tags":["original"]}
    sampling={"notes":[{"status":"observed"}]}
    original_row=deepcopy(row);original_spec=deepcopy(spec);original_sampling=deepcopy(sampling)
    result=assemble_aqueous_ensemble([row],["a"],spec,source_sampling=sampling)
    result["model"]["metadata"]["tags"].append("changed")
    result["uncertainty"]["source_sampling"]["notes"][0]["status"]="changed"
    result["basins"][next(iter(result["basins"]))]["start_ids"].append("changed")
    assert row==original_row and spec==original_spec and sampling==original_sampling


def test_nonfinite_uncertainty_diagnostics_are_flagged_missing_without_mutating_source():
    import json
    sampling={"spread_kj":float("nan"),"trajectory":[.1,float("inf")]}
    result=assemble_aqueous_ensemble([minimum()],["a"],model(),source_sampling=sampling)
    assert math.isnan(sampling["spread_kj"]) and math.isinf(sampling["trajectory"][1])
    assert result["uncertainty"]["source_sampling"]=={"spread_kj":None,"trajectory":[.1,None]}
    assert len(result["uncertainty"]["source_invalid_diagnostics"])==2
    assert result["validated_CI_kj"] is None
    json.dumps(result,allow_nan=False)
