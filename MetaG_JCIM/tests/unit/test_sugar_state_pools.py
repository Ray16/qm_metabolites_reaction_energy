import pytest
from rdkit import Chem
from metag.routing.sugar_states import enumerate_sugar_states, prepare_declared_sugar_inputs

OPEN={
 'glucose':'O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO',
 'fructose':'OCC(=O)[C@@H](O)[C@H](O)[C@H](O)CO',
 'ribose':'O=C[C@H](O)[C@H](O)[C@H](O)CO'}
@pytest.mark.parametrize('smiles',OPEN.values())
def test_open_and_each_defined_or_unspecified_ring_give_the_same_complete_pool(smiles):
 reference=enumerate_sugar_states(smiles)
 assert reference.complete and len(reference.states)==5
 assert sorted(s['ring_size'] for s in reference.states if s['kind']=='ring')==[5,5,6,6]
 for state in reference.states:
  actual=enumerate_sugar_states(state['smiles']);assert actual.complete
  assert actual.state_set_sha256==reference.state_set_sha256
  if state['kind']=='ring':
   m=Chem.MolFromSmiles(state['smiles'])
   hit=m.GetSubstructMatches(Chem.MolFromSmarts('[C;X4;R]([O;X2;R])[O;X2;H1]'))[0]
   m.GetAtomWithIdx(hit[0]).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
   unknown=enumerate_sugar_states(Chem.MolToSmiles(m))
   assert unknown.complete and unknown.state_set_sha256==reference.state_set_sha256

def test_epimers_and_enantiomers_stay_separate():
 smi=OPEN['glucose'];reference=enumerate_sugar_states(smi)
 assert enumerate_sugar_states(smi.replace('[C@H]','[C@@H]',1)).state_set_sha256!=reference.state_set_sha256
 mirror=smi.replace('@@','QQ').replace('@','@@').replace('QQ','@')
 assert enumerate_sugar_states(mirror).state_set_sha256!=reference.state_set_sha256

@pytest.mark.parametrize('smiles',['CO[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O','O=P(O)(O)O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O'])
def test_blocked_glycosides_not_mutarotated(smiles):assert not enumerate_sugar_states(smiles).applicable

def test_undefined_nonanomeric_configuration_refused():
 r=enumerate_sugar_states('O=CC(O)[C@@H](O)[C@H](O)[C@H](O)CO')
 assert r.applicable and not r.complete and 'nonanomeric' in r.reason

def test_cap_and_charge_mismatch_never_supply_partial_pool():
 assert not enumerate_sugar_states(OPEN['glucose'],max_states=4).complete
 with pytest.raises(ValueError,match='charge'):enumerate_sugar_states(OPEN['glucose'],-1)


def test_reference_drawing_is_a_gauge_and_caller_input_is_preserved():
    original={'species':{'glucose':[-1,0,'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O']},
              'species_state_pools':{'glucose':{'kind':'reducing_sugar_equilibrium','source':'Source defines an equilibrating D-glucose parent'}}}
    initial=original['species']['glucose'][2]
    explicit,plans=prepare_declared_sugar_inputs(original)
    assert original['species']['glucose'][2]==initial
    assert explicit['species']['glucose'][2]!=initial
    plan=plans['glucose'];assert len(plan['source_states'])==5
    assert plan['reference_smiles']==explicit['species']['glucose'][2]
    assert enumerate_sugar_states(plan['reference_smiles']).state_set_sha256==plan['source_state_set_sha256']
    assert not __import__('metag.validation.stereochemistry',fromlist=['inspect_stereochemistry']).inspect_stereochemistry(plan['reference_smiles'])['issues']


def test_undeclared_ambiguous_input_is_not_silently_assigned():
    original={'species':{'glucose':[-1,0,'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O']}}
    explicit,plans=prepare_declared_sugar_inputs(original)
    assert explicit is original and not plans


@pytest.mark.parametrize('declaration',[
    {'kind':'reducing_sugar_equilibrium','source':''},
    {'kind':'reducing_sugar_equilibrium','source':'Source','offset_kj':2.},
    {'kind':'other','source':'Source'},
    True,
])
def test_pool_metadata_never_accepts_unowned_energy_adjustments(declaration):
    rx={'species':{'glucose':[-1,0,OPEN['glucose']]},'species_state_pools':{'glucose':declaration}}
    with pytest.raises(ValueError):prepare_declared_sugar_inputs(rx)


def test_extra_unknown_nonanomeric_centre_cannot_be_masked_by_pool_metadata():
    rx={'species':{'glucose':[-1,0,'O=CC(O)[C@@H](O)[C@H](O)[C@H](O)CO']},
        'species_state_pools':{'glucose':{'kind':'reducing_sugar_equilibrium','source':'Source'}}}
    with pytest.raises(ValueError,match='nonanomeric'):prepare_declared_sugar_inputs(rx)


def _pool_plan(reference, factors=None):
    return {'charge':0,'reference':reference,'states':['A','B'],
            'state_factors_kj':factors or {'A':-1.,'B':-4.},
            'policy':'test-source-defined','source_state_set_sha256':'test-identity'}


def test_pool_assembles_owned_state_factors_without_double_counting_reference():
    from metag.state_pools import apply_sugar_pools,KT_KJ
    import math
    plan=_pool_plan('A')
    values,sigmas,diagnostics=apply_sugar_pools(
        {'x':plan},{'x':[-1,0,'A']},{'x':2.},{'x':.1},
        evaluate_state=lambda *args:(3.,.2,{'complete':True}))
    exact=-KT_KJ*math.log(math.exp(-1./KT_KJ)+math.exp(1./KT_KJ))
    assert values['x']+plan['state_factors_kj']['A']==pytest.approx(exact)
    assert sum(r['conditional_population'] for r in diagnostics['x']['states'])==pytest.approx(1.)
    assert diagnostics['x']['mixture_shift_kj']<0


def test_same_physical_pool_has_the_same_transformed_energy_from_either_gauge():
    from metag.state_pools import apply_sugar_pools
    calls=[]
    def unexpected(*args):calls.append(args);raise AssertionError('Both reference states already available')
    values,sigmas,diagnostics=apply_sugar_pools(
        {'x':_pool_plan('A'),'y':_pool_plan('B')},
        {'x':[-1,0,'A'],'y':[1,0,'B']},{'x':2.,'y':3.},{'x':.1,'y':.2},evaluate_state=unexpected)
    assert not calls
    assert values['x']-1.==pytest.approx(values['y']-4.)
    assert diagnostics['x']['state_set_sha256']==diagnostics['y']['state_set_sha256']
    assert sigmas['x']==sigmas['y']


@pytest.mark.parametrize('failed',[None,float('nan'),float('inf'),True])
def test_missing_required_sugar_state_refuses_the_whole_pool(failed):
    from metag.state_pools import apply_sugar_pools
    with pytest.raises(ValueError,match='Incomplete sugar'):
        apply_sugar_pools({'x':_pool_plan('A')},{'x':[-1,0,'A']},{'x':2.},{'x':.1},
                          evaluate_state=lambda *args:(failed,.2,{}))


def test_repeated_molecular_state_is_evaluated_once_without_extra_degeneracy():
    from metag.state_pools import apply_sugar_pools
    calls=[]
    def evaluate(*args):calls.append(args);return 3.,.2,{}
    plans={'x':_pool_plan('A'),'y':_pool_plan('A')}
    values,_,diagnostics=apply_sugar_pools(plans,{'x':[-1,0,'A'],'y':[1,0,'A']},
        {'x':2.,'y':2.},{'x':.1,'y':.1},evaluate_state=evaluate)
    assert len(calls)==1 and values['x']==values['y']
    assert len(diagnostics['x']['states'])==2


def test_atom_identifiers_and_input_order_do_not_change_physical_pool_identity():
    original=enumerate_sugar_states(OPEN['glucose'])
    mol=Chem.MolFromSmiles(OPEN['glucose'])
    for atom in mol.GetAtoms():atom.SetAtomMapNum(atom.GetIdx()+1)
    reversed_mol=Chem.RenumberAtoms(mol,list(reversed(range(mol.GetNumAtoms()))))
    alternate=enumerate_sugar_states(Chem.MolToSmiles(reversed_mol,canonical=False))
    assert alternate.complete and alternate.state_set_sha256==original.state_set_sha256
