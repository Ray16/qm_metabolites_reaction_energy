import pytest

from metag.tools.audit_network import checkpoint_records
from metag.validation.network_consistency import audit_network


def test_adapter_keeps_missing_failed_and_original_state_inputs():
    species={"a": [-1, 0, "CCO"], "b": [1, 0, "CCO"]}
    reactions={key: {"species":species} for key in ["good", "failed", "missing"]}
    checkpoint={"results": {
        "good": {"status":"estimated", "result":{"reaction":"good", "dG":1,
            "energy_units":"kJ/mol", "quantity":"standard_transformed_reaction_gibbs",
            "conditions":{"T_K":298.15,"pH":7,"ionic_strength_M":0,"pMg":None,
                          "water_activity":1,"standard_state":"1M solutes; liquid water"},
            "config":{"physics":"test"}, "species_scored":species}},
        "failed": {"status":"budget_exceeded", "result":None, "error":"fixed budget"}}}
    records=checkpoint_records(checkpoint,reactions)
    assert len(records)==3
    report=audit_network(records)
    assert report["summary"]["n_panel"]==3
    assert report["summary"]["n_estimates_available"]==1
    failures={row['reaction']:row for row in report['records']}
    assert failures['failed']['failure_reason']=='fixed budget'
    assert failures['missing']['prediction_status']!='estimated'
    assert records[0]['original_species']==species


def test_unknown_or_duplicate_checkpoint_reactions_are_rejected():
    with pytest.raises(ValueError,match='unique'):
        checkpoint_records([{'reaction':'a'},{'reaction':'a'}])
    with pytest.raises(ValueError,match='outside'):
        checkpoint_records([{'reaction':'unknown'}],{'a':{'species':{}}})


def test_adapter_does_not_invent_missing_historical_quantities():
    records=checkpoint_records([{'reaction':'a','dG':1}],{'a':{'species':{'a':[1,0,'CCO']}}})
    assert 'conditions' not in records[0] and 'quantity' not in records[0]
    assert audit_network(records)['summary']['n_network_admitted']==0


def test_single_native_record_is_supported_and_nonreaction_payload_is_refused():
    assert checkpoint_records({'reaction':'a','dG':1})[0]['reaction']=='a'
    with pytest.raises(ValueError,match='reaction records'):
        checkpoint_records({'candidate':'keto04','status':'relaxation_converged','energies':{}})
