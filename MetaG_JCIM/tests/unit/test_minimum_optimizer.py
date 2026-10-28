"""No GPU: endpoint bookkeeping, budgets, and state identity govern acceptance."""
import numpy as np
import pytest
import json
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from metag.energetics import minimum_optimizer as mo
from metag.energetics.conformers import UniqueMinima, bond_graph
from metag import pipeline as p


def helium():
    return Atoms('He',positions=[[1.,0,0]],info={'charge':0,'spin':1}), Chem.MolFromSmiles('[He]')


def test_real_cpu_lbfgs_preserves_input_and_rechecks_endpoint(monkeypatch):
    seen=[]
    def harmonic(model,at):
        seen.append(at.copy())
        return .5*np.square(at.positions).sum(),-at.positions
    monkeypatch.setattr(mo,'_energy_forces',harmonic)
    at,mol=helium();initial=at.positions.copy()
    endpoint,info=mo.recover_force_minimum(None,at,mol,set(),steps=30)
    assert info['force_accepted'] and info['model_evaluations']==len(seen)
    assert info['energy_eV']==pytest.approx(.5*np.square(endpoint.positions).sum())
    assert info['max_force_eV_A']<=.01
    assert all(x.info=={'charge':0,'spin':1} for x in seen)
    np.testing.assert_array_equal(at.positions,initial)


def test_false_optimizer_convergence_cannot_override_forces(monkeypatch):
    class Fake:
        nsteps=1
        def __init__(self,*a,**k):pass
        def run(self,**k):return True
    monkeypatch.setattr(mo,'LBFGS',Fake)
    monkeypatch.setattr(mo,'_energy_forces',lambda pu,at:(1.,np.ones((len(at),3))))
    at,mol=helium();_,info=mo.recover_force_minimum(None,at,mol,set())
    assert info['converged'] and not info['force_accepted']


def test_identity_change_rejected_even_with_zero_endpoint_force(monkeypatch):
    class Fake:
        nsteps=1
        def __init__(self,at,**k):self.at=at
        def run(self,**k):self.at.positions[1,0]=5.;return True
    monkeypatch.setattr(mo,'LBFGS',Fake)
    monkeypatch.setattr(mo,'_energy_forces',lambda pu,at:(0.,np.zeros((len(at),3))))
    mol=Chem.MolFromSmiles('[H][H]');at=Atoms('HH',positions=[[0,0,0],[.74,0,0]],info={'charge':0,'spin':1})
    _,info=mo.recover_force_minimum(None,at,mol,bond_graph(mol))
    assert info['converged'] and not info['identity_preserved'] and not info['force_accepted']


def test_nonfinite_model_output_is_not_an_endpoint(monkeypatch):
    monkeypatch.setattr(mo,'_energy_forces',lambda pu,at:(float('nan'),np.zeros((len(at),3))))
    at,mol=helium();_,info=mo.recover_force_minimum(None,at,mol,set())
    assert info['status']=='exception' and not info['force_accepted']


def test_nonfinite_start_retains_json_safe_failure_without_model(monkeypatch):
    def forbidden(*a):raise AssertionError('invalid input reached the model')
    monkeypatch.setattr(mo,'_energy_forces',forbidden)
    at,mol=helium();at.positions[0,0]=float('nan')
    _,info=mo.recover_force_minimum(None,at,mol,set())
    assert info['status']=='rejected_initial_identity' and info['model_evaluations']==0
    assert info['positions_angstrom'] is None and not info['positions_finite']
    json.dumps(info,allow_nan=False)


def test_stereochemical_inversion_rejected_with_connectivity_preserved(monkeypatch):
    class Fake:
        nsteps=1
        def __init__(self,at,**k):self.at=at
        def run(self,**k):self.at.positions[:,0]*=-1;return True
    monkeypatch.setattr(mo,'LBFGS',Fake)
    monkeypatch.setattr(mo,'_energy_forces',lambda pu,at:(0.,np.zeros((len(at),3))))
    mol=Chem.AddHs(Chem.MolFromSmiles('C[C@H](O)C(=O)O'))
    AllChem.EmbedMolecule(mol,randomSeed=17)
    at=Atoms([a.GetSymbol() for a in mol.GetAtoms()],positions=mol.GetConformer().GetPositions(),info={'charge':0,'spin':1})
    _,info=mo.recover_force_minimum(None,at,mol,bond_graph(mol))
    assert info['converged'] and not info['identity_preserved'] and not info['force_accepted']


def test_requested_steps_are_capped_and_exhaustion_not_success(monkeypatch):
    monkeypatch.setattr(mo,'MAX_STEPS',1)
    monkeypatch.setattr(mo,'_energy_forces',lambda pu,at:(.5*np.square(at.positions).sum(),-at.positions))
    at,mol=helium();_,info=mo.recover_force_minimum(None,at,mol,set(),steps=5000)
    assert info['step_budget']==1 and info['steps']==1
    assert not info['converged'] and not info['force_accepted']


def test_time_budget_fails_without_model_evaluation(monkeypatch):
    values=iter([0.,301.,302.]);monkeypatch.setattr(mo.time,'monotonic',lambda:next(values))
    at,mol=helium();_,info=mo.recover_force_minimum(None,at,mol,set())
    assert info['status']=='budget_exit' and info['model_evaluations']==0


def ethanol():
    mol=Chem.AddHs(Chem.MolFromSmiles('CCO'));AllChem.EmbedMolecule(mol,randomSeed=17)
    at=Atoms([a.GetSymbol() for a in mol.GetAtoms()],positions=mol.GetConformer().GetPositions())
    minima=UniqueMinima(template=mol);minima.add(at,-100.,-120.)
    return mol,at,minima


def setup_pipeline(monkeypatch, *, fire_identity=True, fire_converged=False, fallback_imag=False):
    mol,at,minima=ethanol();calls=[]
    monkeypatch.setattr(p,'THERMAL_RELAX_STEPS',400);monkeypatch.setattr(p,'THERMAL_RETRY_STEPS',1200)
    def fire(pu,ats,**kw):
        x=ats[0].copy();x.positions+=9
        return [x],np.array([-1.]),np.array([fire_converged])
    monkeypatch.setattr(p,'batched_fire',fire)
    monkeypatch.setattr(p,'same_connectivity',lambda *a:fire_identity)
    monkeypatch.setattr(p,'same_stereochemistry',lambda *a:True)
    def recover(pu,start,template,graph,**kw):
        calls.append(start.copy());x=start.copy();x.positions+=2
        return x,{'force_accepted':True,'steps':5,'model_evaluations':7,'elapsed_seconds':.1,'positions_angstrom':x.positions.tolist()}
    monkeypatch.setattr(mo,'recover_force_minimum',recover)
    def thermal(pu,syms,pos,q,**kw):
        recovered=np.allclose(pos,at.positions+2)
        good=recovered and not fallback_imag
        return (7. if good else None),{'n_imag':0 if good else 1,'max_imag_cm':0. if good else 100.,'_imag_vecs':[]}
    monkeypatch.setattr(p,'uma_gibbs_corr',thermal)
    return mol,at,minima,calls


def test_final_retry_uses_same_pre_fire_start_and_strict_hessian(monkeypatch):
    mol,at,u,calls=setup_pipeline(monkeypatch)
    g,info,res=p._thermal_at_minimum(None,u,(-100.,at.symbols,at.positions),mol,bond_graph(mol),0,1,'x',lambda *_:None,optimization_steps=1200)
    assert g==7. and res['kind']=='retightened' and info['optimizer']=='ASE_LBFGS'
    np.testing.assert_array_equal(calls[0].positions,at.positions)
    assert res['optimization_trials'][-1]['strict_thermal_accepted']


@pytest.mark.parametrize('steps,identity,converged',[(400,True,False),(1200,False,False),(1200,True,True)])
def test_fallback_not_triggered_outside_narrow_failure(monkeypatch,steps,identity,converged):
    mol,at,u,calls=setup_pipeline(monkeypatch,fire_identity=identity,fire_converged=converged)
    p._thermal_at_minimum(None,u,(-100.,at.symbols,at.positions),mol,bond_graph(mol),0,1,'x',lambda *_:None,optimization_steps=steps)
    assert not calls


def test_fallback_imaginary_mode_cannot_be_admitted(monkeypatch):
    mol,at,u,calls=setup_pipeline(monkeypatch,fallback_imag=True)
    g,_,res=p._thermal_at_minimum(None,u,(-100.,at.symbols,at.positions),mol,bond_graph(mol),0,1,'x',lambda *_:None,optimization_steps=1200)
    assert g is None and calls and not res['optimization_trials'][-1]['strict_thermal_accepted']


def test_validated_ensemble_reassembles_endpoint_and_retains_trials(monkeypatch):
    mol,at,u,calls=setup_pipeline(monkeypatch)
    evaluated=[]
    def electronic(pu,ats):
        evaluated.append(ats[0].positions.copy());return np.array([-2.])
    monkeypatch.setattr(p,'batched_energies',electronic)
    monkeypatch.setattr(p,'dgsolv',lambda *a,**k:-3.)
    result=p._validated_thermal_minima(None,u,{},mol,bond_graph(mol),0,1,'x',lambda *_:None)
    assert result is not None
    new,_,thermal,metadata=result
    np.testing.assert_allclose(evaluated[0],at.positions+2)
    assert new.E==pytest.approx([-2*p.EV2KJ]) and new.G==pytest.approx([-2*p.EV2KJ-3])
    assert thermal==[7.] and metadata['optimizer_trials'][-1]['candidate_index']==0
    assert metadata['optimizer_trials'][-1]['strict_thermal_accepted']
