"""Chemical-state identity must not be silently replaced by an isomer ensemble."""
import json
import pytest
from rdkit import Chem
from metag.validation.stereochemistry import inspect_stereochemistry, unresolved_species
from metag.tools.preflight import audit_reactions

ISOCITRATE = 'O=C([O-])CC(C(=O)[O-])C(O)C(=O)[O-]'
CITRATE = 'O=C([O-])CC(O)(CC(=O)[O-])C(=O)[O-]'


@pytest.mark.parametrize('smiles,count', [(ISOCITRATE,2),('CC(O)C(=O)O',1),
                                          ('CC=CC',1),('CC=N',1),('C[N+](CC)(CCC)CCCC',1)])
def test_missing_atom_or_bond_configuration_is_not_assigned(smiles,count):
    row=inspect_stereochemistry(smiles)
    assert row['issues']==['unspecified_stereochemistry']
    assert len(row['unresolved'])==count
    assert Chem.MolToSmiles(Chem.MolFromSmiles(smiles))==row['canonical_isomeric_smiles']
    json.dumps(row,allow_nan=False)


@pytest.mark.parametrize('smiles', [CITRATE,'CCO','CC(O)C','C=CC','CC(C)=CC','C1=CCCCC1',
                                  'N=C(N)N','CC(=O)N','C[C@H](O)C(=O)O','C/C=C/C',
                                  'C/C=C\\C','C[C@H](O)[C@@H](O)C',
                                  'O=C([O-])C[C@H](C(=O)[O-])[C@H](O)C(=O)[O-]'])
def test_symmetric_and_declared_states_do_not_raise_false_missing_stereo(smiles):
    assert not inspect_stereochemistry(smiles)['issues']


def test_atom_maps_do_not_create_physical_stereochemistry():
    assert not inspect_stereochemistry('[CH3:1]C(O)[CH3:2]')['issues']


def test_relative_enhanced_stereo_does_not_define_one_absolute_species():
    row=inspect_stereochemistry('C[C@H](F)Cl |&1:1|')
    assert row['issues']==['stereoisomer_pool_not_defined']
    assert row['relative_stereo_groups'][0]['atom_indices']==[1]


@pytest.mark.parametrize('smiles',[None,'','not a smiles'])
def test_invalid_input_rejected(smiles):
    with pytest.raises(ValueError):inspect_stereochemistry(smiles)


def reaction():
    return {'species':{'citrate':[-1,-3,CITRATE],'isocitrate':[1,-3,ISOCITRATE]},'n_Hplus':0}


def test_preflight_and_score_refuse_undeclared_stereo_before_qm(monkeypatch):
    from metag import pipeline as p
    def no_qm(*args,**kwargs):raise AssertionError('energy/cache lookup should not run')
    monkeypatch.setattr(p,'_species_free_energies',no_qm)
    result=p.score_reaction(None,reaction(),log=lambda *_:None)
    assert result['dG'] is None and 'stereochemical identity' in result['suspect']
    assert result['routes']['stereochemistry']['input']['isocitrate']['unresolved']
    assert result['config']['stereochemistry_policy']
    report=audit_reactions({'r':reaction()})
    assert report['reactions']['r']['status']=='blocked_unresolved_stereochemistry'


def test_route_cannot_introduce_an_undeclared_isomer(monkeypatch):
    from metag import pipeline as p
    rx={'species':{'left':[-1,0,'C[C@H](O)C(=O)O'],'right':[1,0,'C[C@@H](O)C(=O)O']}}
    def route(reaction,**kwargs):
        return {'species':{'left':[-1,0,'CC(O)C(=O)O'],'right':[1,0,'CC(O)C(=O)O']}}, {
            'errors':[],'warnings':[],'input_balanced':True,'truncated':False},False
    monkeypatch.setattr(p,'route_reaction',route)
    result=p.score_reaction(None,rx,log=lambda *_:None)
    assert result['dG'] is None
    assert result['routes']['stereochemistry']['input']=={}
    assert set(result['routes']['stereochemistry']['routed'])=={'left','right'}


def test_known_amp_input_and_routed_species_remain_declared():
    # Real successful10c result; source reference in analysis is checked separately.
    smiles='Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O'
    assert not inspect_stereochemistry(smiles)['issues']


@pytest.mark.parametrize('smiles', ['COP(=O)([O-])OC',
    'COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])O',
    'CO[P+]([O-])([O-])OC'])
def test_equivalent_nonbridging_phosphate_oxygens_do_not_make_P_chiral(smiles):
    row=inspect_stereochemistry(smiles)
    assert not row['issues']
    assert row['canonical_isomeric_smiles']==Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize('smiles', ['COP(=O)([18O-])OCC','COP(=S)([O-])OCC'])
def test_genuinely_distinct_phosphorus_substituents_remain_stereogenic(smiles):
    assert inspect_stereochemistry(smiles)['unresolved']


@pytest.mark.parametrize('smiles', [
    'COP(=O)(O)OCC', 'COP(=O)(O)OP(=O)(O)O',
    'COP(=O)(O)OP(=O)(O)OP(=O)(O)O',
    'Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)([O-])[O-])[C@@H](O)[C@H]1O',
])
def test_phosphate_acid_prototropy_does_not_require_P_configuration(smiles):
    row = inspect_stereochemistry(smiles)
    assert not row['issues']
    assert row['phosphate_prototropy_equivalent_for_perception']
    assert row['canonical_isomeric_smiles'] == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize('smiles', [
    'COP(=O)([18OH])OCC', 'COP(=O)(O[2H])OCC',
    'COP(=O)(OCC)OCCC', 'COP(=S)(O)OCC',
])
def test_phosphate_exception_preserves_distinguishable_ligands(smiles):
    assert inspect_stereochemistry(smiles)['unresolved']


def test_phosphate_does_not_hide_missing_carbon_configuration():
    row = inspect_stereochemistry('CC(O)COP(=O)(O)OCC')
    assert len(row['unresolved']) == 1
    assert Chem.MolFromSmiles('CC(O)COP(=O)(O)OCC').GetAtomWithIdx(
        row['unresolved'][0]['index']).GetAtomicNum() == 6
