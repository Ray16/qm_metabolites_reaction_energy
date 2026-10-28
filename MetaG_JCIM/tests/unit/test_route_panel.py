"""Panel representations are checked before QM, with failures retained."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from metag.config import flag
from metag.routing.policy import RoutingPolicy
from metag.routing.production_policy import validate_production_policy
from metag.tools.audit_routes import audit_route_panel, main
from metag.validation.routing import audit_routing_consistency


def identity():
    return {'species':{'left':[-1,0,'CCO'], 'right':[1,0,'OCC']}, 'n_Hplus':0}


def test_full_molecule_preset_disables_all_structural_reduction(monkeypatch):
    monkeypatch.setenv('METAG_ROUTING_MODE', 'full_molecule')
    for key in ('COFACTOR_RING','COA_CORE','NTP_CORE','AUTO_TRUNCATE'):
        monkeypatch.setenv(key, '1')
    policy = RoutingPolicy.from_runtime(flag, __import__('os').environ)
    assert not any((policy.cofactor_ring,policy.coa_core,policy.ntp_core,policy.auto_truncate))
    assert policy.ph0_auto and policy.mode == 'full_molecule'
    validate_production_policy(policy)
    with pytest.raises(ValueError, match='cannot enable structural'):
        validate_production_policy(replace(policy, ntp_core=True))


def test_unknown_routing_mode_fails_before_work():
    with pytest.raises(ValueError, match='METAG_ROUTING_MODE'):
        RoutingPolicy.from_runtime(flag, {'METAG_ROUTING_MODE':'typo'})


def test_empty_missing_and_ambiguous_panel_cannot_pass():
    assert not audit_route_panel({})['panel_gate']['passed']
    original={'r':identity()}
    assert not audit_routing_consistency(original, {})['consistent_known_representations']
    scored=dict(identity()['species'])
    scored['left_t']=scored['left']
    report=audit_routing_consistency(original, {'r':{'species_scored':scored}})
    assert not report['consistent_known_representations']
    assert report['unmapped_uses'][0]['name']=='left'


def test_atom_maps_do_not_hide_shared_state_conflict():
    original={'r1':{'species':{'a':[-1,0,'[CH3:1][CH2:2][OH:3]']}},
              'r2':{'species':{'a':[1,0,'CCO']}}}
    outputs={'r1':{'species_scored':{'a':[-1,0,'CCO']}},
             'r2':{'species_scored':{'a_t':[1,0,'CO']}}}
    report=audit_routing_consistency(original,outputs)
    assert report['n_context_dependent_species']==1
    assert not report['consistent_known_representations']


def test_wrong_extent_mapping_is_unknown():
    report=audit_routing_consistency({'r':identity()},
        {'r':{'species_scored':{'left':[-2,0,'CCO'],'right':[1,0,'CCO']}}})
    assert report['unmapped_uses'][0]['name']=='left'


def test_gate_retains_unresolved_stereo_even_with_consistent_representations():
    rx={'species':{'a':[-1,0,'CC(O)C(=O)O'], 'b':[1,0,'CC(O)C(=O)O']}, 'n_Hplus':0}
    report=audit_route_panel({'unresolved':rx})
    assert report['representation_consistency']['consistent_known_representations']
    assert not report['panel_gate']['passed']
    assert report['panel_gate']['blocked_reactions']['unresolved']=='blocked_unresolved_stereochemistry'


def test_panel_cli_writes_failed_gate_before_nonzero_exit(tmp_path):
    source=tmp_path/'input.json';output=tmp_path/'audit.json'
    rx={'species':{'a':[-1,0,'CC(O)C(=O)O'], 'b':[1,0,'CC(O)C(=O)O']}}
    source.write_text(json.dumps({'unresolved':rx}))
    with pytest.raises(SystemExit) as exc:
        main([str(source), '-o',str(output), '--require-consistent'])
    assert exc.value.code==1
    assert not json.loads(output.read_text())['panel_gate']['passed']


def test_full_mode_scoring_config_and_preflight_share_policy(monkeypatch):
    monkeypatch.setenv('METAG_ROUTING_MODE','full_molecule')
    from metag.pipeline import effective_config
    report=audit_route_panel({'identity':identity()})
    assert report['panel_gate']['passed']
    assert report['config']['routing']==effective_config()['routing_policy']
    assert report['reactions']['identity']['routes']['routing_mode']=='full_molecule'


def test_pinned_sugar_cycle_full_mode_removes_context_representations(monkeypatch):
    source=Path(__file__).parents[1]/'regression/fixtures/sugar_routing_cycle.json'
    panel=json.loads(source.read_text())
    monkeypatch.setenv('METAG_ROUTING_MODE','reaction')
    default=audit_route_panel(panel)
    assert default['representation_consistency']['n_context_dependent_species']>0
    monkeypatch.setenv('METAG_ROUTING_MODE','full_molecule')
    full=audit_route_panel(panel)
    assert full['representation_consistency']['consistent_known_representations']
    # Historical inputs omit anomeric identities; fixing routes cannot cure that.
    assert not full['panel_gate']['passed']
    assert any(status=='blocked_unresolved_stereochemistry'
               for status in full['panel_gate']['blocked_reactions'].values())


def test_batch_preflight_refuses_before_model_loading(tmp_path):
    from unittest.mock import Mock
    from metag.cli import run_cli
    model=Mock()
    rx={'species':{'a':[-1,0,'CC(O)C(=O)O'], 'b':[1,0,'CC(O)C(=O)O']}}
    with pytest.raises(SystemExit, match='preflight failed'):
        run_cli(reactions={'unresolved':rx}, model_name='test', load_model=model,
                score=Mock(), no_estimate=Mock(), out=str(tmp_path),
                argv=['--require-consistent-panel'])
    model.assert_not_called()
    report=json.loads((tmp_path/'route_preflight_all.json').read_text())
    assert not report['panel_gate']['passed']


def test_public_cli_failed_gate_imports_no_QM_runtime(tmp_path):
    import os
    import subprocess
    import sys
    input_path=tmp_path/'input.json'
    rx={'species':{'a':[-1,0,'CC(O)C(=O)O'], 'b':[1,0,'CC(O)C(=O)O']}}
    input_path.write_text(json.dumps({'unresolved':rx}))
    code="""
import sys
from metag.cli import main
try:
    main(['--require-consistent-panel'])
except SystemExit as exc:
    assert 'preflight failed' in str(exc)
else:
    raise AssertionError('failed panel was admitted')
assert 'metag.pipeline' not in sys.modules
assert 'torch' not in sys.modules and 'fairchem' not in sys.modules
"""
    process=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,
                           env={**os.environ,'PYTHONPATH':str(Path(__file__).parents[2]/'src'),
                                'RXN_FILE':str(input_path),'METAG_OUT':str(tmp_path/'out')},timeout=20)
    assert process.returncode==0,process.stderr
