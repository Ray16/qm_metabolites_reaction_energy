"""Regressions from the fixed severe-error audit; no fitted reaction parameters."""
import pytest
from metag import pipeline as p


def test_missing_estimate_retains_auditable_original_species():
    species={'a':[-1,0,'CCO'],'b':[1,0,'CCO']}
    result=p._no_estimate('case',{'species':species},{},'failed',lambda *_:None)
    assert result['original_species']==species
    assert result['energy_units']=='kJ/mol'
    assert result['quantity']=='standard_transformed_reaction_gibbs'
    assert not result['ci_info']['coverage_calibrated']


def test_suspect_result_cannot_keep_a_calibrated_interval_claim(monkeypatch):
    def species(*args):
        energies, sigmas=args[3],args[4]
        energies.update(a=0,b=1000);sigmas.update(a=0.1,b=0.1)
        args[1]['n_Hplus']=1  # inject a late charge leak after input/routing validation
    monkeypatch.setattr(p,'_species_free_energies',species)
    from metag import uncertainty
    monkeypatch.setattr(uncertainty,'prediction_interval',lambda *a,**k:(900,1100,1000,{'coverage_calibrated':True,'externally_calibrated':True}))
    result=p.score_reaction(None,{'species':{'a':[-1,0,'C'],'b':[1,0,'C']}},log=lambda *_:None)
    assert result['dG'] is None and result['ci95']==[None,None]
    assert not result['ci_info']['coverage_calibrated']


def test_radius_comparison_requires_distinct_canonical_models(monkeypatch):
    def score(*args,**kwargs):
        full=not kwargs.get('allow_truncate',True)
        radius=kwargs.get('trunc_radius',2)
        return dict(routes={'truncated':not full,'cofactor_ring':True},dG_raw=10.,U_samp=.1,suspect=None,
                    species_scored={'renamed':[-1,0,'OCC' if radius==2 else 'CCO'],'b':[1,0,'CC=O']})
    monkeypatch.setattr(p,'score_reaction',score)
    r=p._score_trunc_validated(None,{},[],1,1,lambda *_:None,'test',2)
    assert not r['routes']['truncated']
    assert not r['trunc_validation']['distinct_radius_models']
    assert r['trunc_validation']['remaining_core_routes']==['cofactor_ring']


def test_distinct_stable_radius_models_can_be_accepted(monkeypatch):
    def score(*args,**kwargs):
        r=kwargs.get('trunc_radius',2)
        return dict(routes={'truncated':True},dG_raw=10.+.1*r,U_samp=.1,suspect=None,
                    species_scored={'a':[-1,0,'CCO' if r==2 else 'CCCO'],'b':[1,0,'CC=O' if r==2 else 'CCC=O']})
    monkeypatch.setattr(p,'score_reaction',score)
    r=p._score_trunc_validated(None,{},[],1,1,lambda *_:None,'test',2)
    assert r['trunc_validation']['verdict']=='accepted'
    assert r['trunc_validation']['distinct_radius_models']


def test_seed_offset_is_in_reproducibility_config(monkeypatch):
    monkeypatch.setenv('CONFORMER_SEED_OFFSET','1000')
    assert p.effective_config()['conformer_seed_offset']==1000


def test_strict_thermal_result_reused_only_for_exact_geometry(monkeypatch):
    import numpy as np
    from ase import Atoms
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from metag.energetics.thermal_ensemble import ThermalTrack
    mol=Chem.AddHs(Chem.MolFromSmiles('CO')); AllChem.EmbedMolecule(mol,randomSeed=17)
    at=Atoms([a.GetSymbol() for a in mol.GetAtoms()],positions=mol.GetConformer().GetPositions())
    u=p.UniqueMinima(template=mol); u.add(at,-100.,-110.)
    calls=[]
    def correction(*args,**kwargs):
        calls.append(kwargs)
        return 5.,dict(n_imag=0,max_imag_cm=0.,_imag_vecs=[])
    track=ThermalTrack(None,mol,0,1,correction=correction,combine=p.boltz,window=15,maximum=10)
    assert track.update(u)==pytest.approx(-105.)
    def forbidden(*args,**kwargs):raise AssertionError('exact validated geometry should reuse strict Hessian')
    monkeypatch.setattr(p,'_thermal_at_minimum',forbidden)
    out=p._validated_thermal_minima(None,u,{},mol,None,0,1,'test',print,track=track)
    assert out[2]==[5.] and out[3]['n_strict_results_reused']==1
    row=out[3]['validated_minima'][0]
    np.testing.assert_array_equal(row['geometry']['positions_angstrom'],
                                  out[0].ref[0].GetConformer().GetPositions())
    assert row['geometry']['symbols']==[a.GetSymbol() for a in mol.GetAtoms()]
    assert row['electronic_kj']+row['solvation_kj']+row['thermal_kj']==row['total_kj']
    assert len(calls)==1
    # A tiny coordinate change must invalidate the strict evidence.
    x=np.array(u.ref[0].GetConformer().GetAtomPosition(0)); x[0]+=1e-6
    u.ref[0].GetConformer().SetAtomPosition(0,x)
    with pytest.raises(AssertionError,match='exact validated'):
        p._validated_thermal_minima(None,u,{},mol,None,0,1,'test',print,track=track)


def test_provisional_soft_modes_cannot_supply_strict_validation():
    from types import SimpleNamespace
    import numpy as np
    from metag.energetics.thermal_ensemble import ThermalTrack
    calls=[]
    def correction(*args,**kw):
        calls.append(kw.get('imag_as_soft',False))
        return 4.,dict(n_imag=0 if calls[-1] else 1,max_imag_cm=100.)
    mol=SimpleNamespace(GetAtoms=lambda:[])
    u=SimpleNamespace(G=[-10.],ref=[SimpleNamespace(GetConformer=lambda:SimpleNamespace(GetPositions=lambda:np.zeros((0,3))))])
    tr=ThermalTrack(None,mol,0,1,correction=correction,combine=p.boltz,window=15,maximum=10)
    tr.update(u)
    assert calls==[False,True]
    assert tr.validated=={}


def test_failed_species_keeps_thermal_diagnostics(monkeypatch):
    for flag in ('AUTO_TRUNCATE','COFACTOR_RING','PH0_AUTO','ZWITTERION_PH0','NEUTRAL_QM','TRUNC_VALIDATE'):
        monkeypatch.setenv(flag,'0')
    def fail(pu,q,smi,*args,**kw):
        kw['diagnostics']['thermal']={'status':'incomplete','n_sampled':3}
        return None,None
    monkeypatch.setattr(p,'implicit_G',fail)
    result=p.score_reaction(None,{'species':{'a':[-1,0,'CCO'],'b':[1,0,'CC=O'],'h2':[1,0,'[HH]']}},log=lambda *_:None)
    assert result['dG'] is None
    assert 'minimum validation failed' in result['suspect']
    assert result['routes']['species_diagnostics']['a']['thermal']['n_sampled']==3
    assert 'a' in result['routes']['failed_species']
    assert result['ci95']==[None,None]


def test_thermal_resolution_budget_reaches_optimizer(monkeypatch):
    import numpy as np
    seen=[]
    monkeypatch.setattr(p,'THERMAL_RELAX_STEPS',1200)
    monkeypatch.setattr(p,'batched_energies',lambda *a,**k:np.array([-1.]))
    def relax(pu,ats,**kw):
        seen.append(kw['steps']);return ats,np.array([-1.,-1.]),np.array([False,False])
    monkeypatch.setattr(p,'batched_fire',relax)
    monkeypatch.setattr(p,'same_connectivity',lambda *a:True)
    verdict,_=p._follow_imaginary_mode(None,['H'],np.zeros((1,3)),np.zeros((1,3)),0,1,set(),'test')
    assert seen==[1200] and verdict=='inconclusive'


def test_validation_budget_failure_is_distinct_from_saddle(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(p,'THERMAL_VALIDATION_MAX',1)
    info={}
    assert p._validated_thermal_minima(None,SimpleNamespace(G=[0.,1.]),{},None,None,0,1,'test',print,
                                       diagnostics=info) is None
    assert info['reason']=='candidate validation budget exceeded'
    assert info['n_sampled']==2


def test_unresolved_candidate_retried_without_dropping_ensemble(monkeypatch):
    from ase import Atoms
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol=Chem.AddHs(Chem.MolFromSmiles('CO'));AllChem.EmbedMolecule(mol,randomSeed=17)
    at=Atoms([a.GetSymbol() for a in mol.GetAtoms()],positions=mol.GetConformer().GetPositions())
    u=p.UniqueMinima(template=mol);u.add(at,-100.,-110.)
    seen=[]
    monkeypatch.setattr(p,'THERMAL_RELAX_STEPS',400)
    monkeypatch.setattr(p,'THERMAL_RETRY_STEPS',1200)
    def resolve(*args,**kwargs):
        steps=kwargs.get('optimization_steps',400);seen.append(steps)
        return (5. if steps==1200 else None),{},dict(index=0,kind='as_is',pos=at.positions,rejected=[])
    monkeypatch.setattr(p,'_thermal_at_minimum',resolve)
    out=p._validated_thermal_minima(None,u,{},mol,None,0,1,'test',lambda *_:None)
    assert seen==[400,1200]
    assert out[0].G==[-110.] and out[2]==[5.]
    assert out[3]['n_candidates_retried']==1


@pytest.mark.parametrize('second,accepted',[('artefact',True),('inconclusive',False)])
def test_strongest_artefact_does_not_validate_other_negative_modes(monkeypatch,second,accepted):
    import numpy as np
    from ase import Atoms
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol=Chem.AddHs(Chem.MolFromSmiles('CO'));AllChem.EmbedMolecule(mol,randomSeed=17)
    at=Atoms([a.GetSymbol() for a in mol.GetAtoms()],positions=mol.GetConformer().GetPositions())
    u=p.UniqueMinima(template=mol);u.add(at,-100.,-110.)
    soft_calls=[];modes=[np.zeros_like(at.positions),np.ones_like(at.positions)]
    def correction(*args,**kwargs):
        soft=kwargs.get('imag_as_soft',False)
        if soft:soft_calls.append(True)
        return 5.,dict(n_imag=0 if soft else 2,max_imag_cm=100.,_imag_vecs=list(modes))
    verdicts=iter(['artefact',second]); probes=[]
    def follow(*args,**kwargs):
        probes.append(True);return next(verdicts),None
    monkeypatch.setattr(p,'uma_gibbs_corr',correction)
    monkeypatch.setattr(p,'_follow_imaginary_mode',follow)
    monkeypatch.setattr(p,'batched_fire',lambda pu,ats,**kw:(ats,np.array([-1.]),np.array([True])))
    monkeypatch.setattr(p,'same_connectivity',lambda *a:True)
    monkeypatch.setattr(p,'same_stereochemistry',lambda *a:True)
    g,_,_=p._thermal_at_minimum(None,u,(-100.,at.symbols,at.positions),mol,None,0,1,'test',lambda *_:None)
    assert len(probes)==2
    assert (g is not None)==accepted
    assert bool(soft_calls)==accepted
