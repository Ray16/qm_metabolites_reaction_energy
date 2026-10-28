"""The public workflow evaluates complete solution pools, never reference-only gauges."""
import math
import pytest
from rdkit import Chem
from metag.routing.sugar_states import enumerate_sugar_states, canon
from metag.state_pools import KT_KJ

GLC='OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O'
FRU='OC[C@H]1OC(O)(CO)[C@@H](O)[C@@H]1O'
G6P='O=P([O-])([O-])OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O'
F6P='O=P([O-])([O-])OC[C@H]1OC(O)(CO)[C@@H](O)[C@@H]1O'
NANA='CC(=O)N[C@H]1[C@H]([C@H](O)[C@H](O)CO)OC(O)(C(=O)[O-])C[C@@H]1O'

def rx(left=GLC,right=FRU,charge=0,extent=1.):
    return {'species':{'left':[-extent,charge,left],'right':[extent,charge,right]},
            'n_Hplus':0,'species_state_pools':{name:{'kind':'reducing_sugar_equilibrium',
                'source':'Pinned source defines the equilibrating reducing-sugar parent'}
                for name in ('left','right')}}

def partition(values):
    low=min(values)
    return low-KT_KJ*math.log(math.fsum(math.exp(-(v-low)/KT_KJ) for v in values))

@pytest.fixture
def pipeline(monkeypatch):
    p=pytest.importorskip('metag.pipeline')
    from metag.config import FLAG_DEFAULTS
    for name,value in FLAG_DEFAULTS.items():monkeypatch.setenv(name,str(int(value)))
    for name in ['COFACTOR_RING','COA_CORE','NTP_CORE','AUTO_TRUNCATE','PH0_AUTO',
                 'PH0_BASES','ZWITTERION_PH0','NEUTRAL_QM','ALDEHYDE_HYDRATION',
                 'RING_CHAIN','RING_PROTON_ENSEMBLE','TRUNC_VALIDATE','SOURCE_PKA_REGISTRY']:
        monkeypatch.setenv(name,'0')
    monkeypatch.setenv('STD_STATE_1M','0')
    values={};calls=[]
    for parent,offset in [(GLC,0.),(FRU,5.)]:
        for i,state in enumerate(enumerate_sugar_states(parent).states):values[state['smiles']]=offset+i
    def implicit(pu,q,smi,*args,diagnostics=None,**kwargs):
        calls.append(canon(Chem.MolFromSmiles(smi)))
        if diagnostics is not None:diagnostics.update(method='implicit',source='controlled_test')
        return values.get(canon(Chem.MolFromSmiles(smi))),.1
    monkeypatch.setattr(p,'implicit_G',implicit)
    monkeypatch.setattr(p,'water_ref_G',lambda *a,**k:0.)
    return p,values,calls


def test_public_scoring_uses_every_required_state_and_retains_original_input(pipeline):
    p,values,calls=pipeline
    original=rx();result=p.score_reaction(None,original,log=lambda _:None)
    assert result['dG']==5. and len(calls)==10
    assert original['species']['left'][2]==GLC
    assert result['original_species']['left'][2]==GLC
    assert len(result['routes']['sugar_ensembles']['left']['states'])==5
    assert len(result['routes']['sugar_ensembles']['right']['states'])==5
    assert set(result['routes']['stereochemistry']['covered_by_declared_sugar_pools'])=={'left','right'}
    assert result['config']['caller_species_state_pools_sha256']
    assert result['sigma_pred'] is None and result['ci95']==[None,None]


def test_undeclared_source_remains_unresolved_without_species_work(pipeline):
    p,values,calls=pipeline;reaction=rx();reaction.pop('species_state_pools')
    result=p.score_reaction(None,reaction,log=lambda _:None)
    assert result['dG'] is None and not calls


def test_declared_pool_scoring_reverses_and_scales(pipeline):
    p,values,calls=pipeline
    results=[p.score_reaction(None,rx(extent=e),log=lambda _:None)['dG'] for e in [1.,-1.,2.,.5]]
    assert results==[5.,-5.,10.,2.5]


def test_two_drawings_of_one_pool_cancel_before_sampling_quadrature(pipeline):
    p,values,calls=pipeline;states=enumerate_sugar_states(GLC).states
    reaction=rx(states[1]['smiles'],states[4]['smiles'])
    result=p.score_reaction(None,reaction,log=lambda _:None)
    assert result['dG']==0. and result['U_samp']==0.
    assert result['routes']['sampling_uncertainty']['n_sampling_groups']==1
    assert len(calls)==5


def test_failed_required_alternative_refuses_reference_only_rescue(pipeline):
    p,values,calls=pipeline;values[enumerate_sugar_states(GLC).open_smiles]=None
    result=p.score_reaction(None,rx(),log=lambda _:None)
    assert result['dG'] is None
    assert 'incomplete declared sugar ensemble' in result['routes']['sugar_pool_error']


def test_cpu_preflight_and_scoring_agree_on_joint_model_refusal(pipeline,monkeypatch):
    p,values,calls=pipeline;monkeypatch.setenv('RING_CHAIN','1')
    from metag.tools.preflight import audit_reactions
    preflight=audit_reactions({'x':rx()})
    assert preflight['reactions']['x']['status']=='blocked_sugar_ensemble'
    result=p.score_reaction(None,rx(),log=lambda _:None)
    assert result['dG'] is None and not calls


def test_full_declared_pool_disables_generic_truncation(pipeline,monkeypatch):
    p,values,calls=pipeline;monkeypatch.setenv('AUTO_TRUNCATE','1')
    result=p.score_reaction(None,rx(),log=lambda _:None)
    assert result['dG']==5. and not result['routes']['truncated']
    assert result['routes']['generic_truncation_disabled_for_declared_sugar_pools']


def test_carbonyl_hydrate_and_all_ring_states_enter_one_complete_partition(pipeline,monkeypatch):
    p,values,calls=pipeline
    from metag.routing.aldehyde_hydration import enumerate_hydration_states
    sugar=enumerate_sugar_states(GLC);opened=sugar.open_smiles
    hydrated=enumerate_hydration_states(opened)
    assert hydrated.complete and len(hydrated.states)==1
    values[canon(Chem.MolFromSmiles(hydrated.states[0][1]))]=-8.
    monkeypatch.setenv('ALDEHYDE_HYDRATION','1');monkeypatch.setenv('CARBONYL_HYDRATION_ALL','1')
    monkeypatch.setenv('HYDRATION_CAL','0')
    result=p.score_reaction(None,rx(opened,sugar.states[4]['smiles']),log=lambda _:None)
    assert result['dG']==0. and result['U_samp']==0.
    actual=result['routes']['sugar_ensembles']['left']['pooled_transformed_G_kj']
    expected=partition([values[s['smiles']] for s in sugar.states]+[-8.])
    assert actual==pytest.approx(expected,abs=1e-3)


def test_pka_factor_gauge_is_owned_and_counted_once_for_charged_sugars(pipeline,monkeypatch):
    p,values,calls=pipeline
    from metag.routing.pka_transform import _neutralize_v2
    for name in ['PH0_AUTO','PH0_BASES']:monkeypatch.setenv(name,'1')
    for parent,offset in [(G6P,0.),(F6P,5.)]:
        neutral,*_=_neutralize_v2(parent)
        for i,s in enumerate(enumerate_sugar_states(neutral,0).states):values[s['smiles']]=offset+i
    result=p.score_reaction(None,rx(G6P,F6P,-2),log=lambda _:None)
    assert result['dG'] is not None
    plans=result['routes']['sugar_state_plan'];pools=result['routes']['sugar_ensembles']
    expected=pools['right']['pooled_transformed_G_kj']-pools['left']['pooled_transformed_G_kj']
    assert result['dG']==pytest.approx(expected,abs=.06)
    assert all(set(plan['states'])==set(plan['state_factors_kj']) for plan in plans.values())


def test_different_open_and_ring_acid_classes_still_describe_one_pool(pipeline,monkeypatch):
    p,values,calls=pipeline
    from metag.routing.pka_transform import _neutralize_v2
    for name in ['PH0_AUTO','PH0_BASES']:monkeypatch.setenv(name,'1')
    source=enumerate_sugar_states(NANA,-1);neutral,*_=_neutralize_v2(NANA)
    for i,state in enumerate(enumerate_sugar_states(neutral,0).states):values[state['smiles']]=float(i)
    result=p.score_reaction(None,rx(source.open_smiles,NANA,-1),log=lambda _:None)
    assert result['dG']==0. and result['U_samp']==0.
    left=result['routes']['sugar_ensembles']['left'];right=result['routes']['sugar_ensembles']['right']
    assert left['reference_owned_factor_kj']!=right['reference_owned_factor_kj']
    assert left['state_set_sha256']==right['state_set_sha256']


def test_acid_only_route_preserves_the_configured_protonated_amine_reference(pipeline,monkeypatch):
    p,values,calls=pipeline
    from metag.routing.pka_transform import _neutralize
    from metag.state_pools import prepare_sugar_pools
    from metag.routing.sugar_states import prepare_declared_sugar_inputs
    from metag.routing.planner import route_reaction
    source='[NH3+][C@H]1C(O)O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@@H]1O'
    fixed='[NH3+][C@H]1[C@@H](OP(=O)([O-])[O-])O[C@H](CO)[C@@H](O)[C@@H]1O'
    reaction={'species':{'fixed':[-1,-1,fixed],'pool':[1,-1,source]},
              'species_state_pools':{'pool':{'kind':'reducing_sugar_equilibrium','source':'Pinned parent'}}}
    monkeypatch.setenv('PH0_AUTO','1');monkeypatch.setenv('PH0_BASES','1')
    routed,routes,_=route_reaction(reaction,log=lambda _:None)
    assert routed['species']['pool'][1]==1
    plan=prepare_sugar_pools(routed,routes)['pool']
    assert plan['reference_model']=='acid_only' and plan['charge']==1
    assert all(Chem.GetFormalCharge(Chem.MolFromSmiles(s))==1 for s in plan['states'])
    for i,state in enumerate(plan['states']):values[state]=float(i)
    values[canon(Chem.MolFromSmiles(routed['species']['fixed'][2]))]=0.
    result=p.score_reaction(None,reaction,log=lambda _:None)
    assert result['dG'] is not None


def test_flat_pka_terms_cannot_be_reused_as_owned_state_factors(pipeline):
    p,values,calls=pipeline;reaction=rx();reaction['pka_sites']=[['react',4.75]]
    result=p.score_reaction(None,reaction,log=lambda _:None)
    assert result['dG'] is None and not calls
    assert 'Flat caller' in result['routes']['sugar_pool_error']
