"""Failed seed batches are not evidence that a molecular ensemble converged."""
from collections import Counter
import numpy as np
import pytest
from metag import pipeline as P
from test_accuracy_review import methane, species_backend


@pytest.fixture
def sampled(monkeypatch,species_backend):
    _,atoms=methane()
    state={'kinds':[],'batch':0,'solvation_calls':Counter()}
    monkeypatch.setenv('CONFORMER_SEED_OFFSET','0')
    monkeypatch.setattr(P,'CONV_HITS',2)
    monkeypatch.setattr(P,'CONV_TOL',2.5)
    monkeypatch.setattr(P,'_THERMAL_ENSEMBLE',True)
    monkeypatch.setattr(P,'XTB_WORKERS',1)
    def pool(smi,q,seed,pool,spin):
        state['batch']+=1
        a=atoms.copy();a.info['test_batch']=state['batch'];return [a]
    def kind(batch=None):return state['kinds'][(state['batch'] if batch is None else batch)-1]
    def relax(pu,ats,**kwargs):
        e=(-50. if kind()=='nonfinite_solv_high' else -100.)/P.EV2KJ
        if kind()=='nonfinite_gas':e=float('nan')
        return ats,np.full(len(ats),e),np.full(len(ats),kind()!='unconverged',dtype=bool)
    def solv(*args):
        state['solvation_calls'][state['batch']]+=1
        if kind() in ('nonfinite_solv_high','nonfinite_solv_low'):return float('nan')
        if kind()=='nonfinite_then_retry' and state['solvation_calls'][state['batch']]==1:return float('inf')
        return -10.
    monkeypatch.setattr(P,'pool_confs',pool)
    monkeypatch.setattr(P,'batched_fire',relax)
    monkeypatch.setattr(P,'dgsolv',solv)
    monkeypatch.setattr(P,'same_connectivity',lambda a,*_:kind(a.info.get('test_batch',state['batch']))!='rearranged')
    monkeypatch.setattr(P,'same_stereochemistry',lambda a,*_:kind(a.info.get('test_batch',state['batch']))!='stereo_changed')
    def run(kinds):
        state['kinds']=kinds;monkeypatch.setattr(P,'CONV_MAX',len(kinds))
        diagnostics={};warnings=[]
        result=P.implicit_G(None,0,'C',(),1,1,lambda *_:None,'methane',warnings,diagnostics)
        return result,diagnostics,warnings
    return run,state,species_backend


@pytest.mark.parametrize('empty_kind',['unconverged','rearranged','stereo_changed','nonfinite_gas','nonfinite_solv_high'])
def test_empty_batches_cannot_manufacture_plateau_or_successful_trajectory(sampled,empty_kind):
    run,state,_=sampled
    (g,sigma),info,_=run(['valid']+[empty_kind]*4)
    assert state['batch']==5
    assert g==pytest.approx(-105.) and np.isfinite(sigma)
    assert info['sampling']['successful_batch_indices']==[1]
    assert info['sampling']['ensemble_trajectory_kj']==[-105.]
    assert info['sampling']['consecutive_plateau_hits']==0
    assert info['sampling']['stop_reason']=='batch_cap_reached'


def test_empty_batch_resets_preceding_plateau_hit(sampled):
    run,state,_=sampled
    (_, _),info,_=run(['valid','valid','unconverged','valid','valid'])
    assert state['batch']==5
    assert info['sampling']['successful_batch_indices']==[1,2,4,5]
    assert info['sampling']['stop_reason']=='plateau_criterion_met'


def test_rediscovering_valid_duplicates_remains_legitimate_search_evidence(sampled):
    run,state,_=sampled
    (g,_),info,_=run(['valid']*5)
    assert state['batch']==3 and g==pytest.approx(-105.)
    assert info['sampling']['successful_batch_indices']==[1,2,3]
    assert info['thermal']['ensemble']['n_minima']==1
    assert info['sampling']['stop_reason']=='plateau_criterion_met'


def test_nonfinite_primary_solvation_gets_one_retry_and_can_recover(sampled):
    run,state,_=sampled
    (g,_),info,_=run(['nonfinite_then_retry']*3)
    assert g==pytest.approx(-105.)
    assert state['solvation_calls']=={1:2,2:2,3:2}
    assert info['sampling']['successful_batch_indices']==[1,2,3]


def test_persistent_nonfinite_low_energy_solvation_fails_without_caching(sampled):
    run,state,writes=sampled
    result,_,warnings=run(['valid','nonfinite_solv_low','nonfinite_solv_low'])
    assert result==(None,None) and not writes
    assert state['solvation_calls']=={1:1,2:2,3:2}
    assert any('solvation failed' in message and 'ensemble incomplete' in message for message in warnings)


def test_nonfinite_auxiliary_solvation_is_missing_not_a_cached_population(monkeypatch,sampled):
    run,_,writes=sampled
    auxiliary='cosmo' if P.SOLV_MODEL!='cosmo' else 'alpb'
    monkeypatch.setattr(P,'SOLV_ALSO',[auxiliary])
    monkeypatch.setattr(P,'dgsolv',lambda sym,pos,q,model,mult:float('nan') if model==auxiliary else -10.)
    (g,_),_,warnings=run(['valid']*3)
    assert g==pytest.approx(-105.)
    assert len(writes)==1 and writes[0][0][3]['solv']==P.SOLV_MODEL
    assert any('auxiliary' in message and 'solvation failed' in message for message in warnings)
