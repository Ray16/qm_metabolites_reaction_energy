import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from metag.tools import build_reference as ref
from metag.validation.reference_transform import parse_equation, Transformer


def valid_row(**kwargs):
    return dict(K_prime='10', p_h='7', p_mg='3', temperature='298.15', ionic_strength='0.1', **kwargs)


@pytest.mark.parametrize('flag', ['todo_check_in_primary_literature', 'misannotated_value_type', 'wrong_value'])
def test_unresolved_constants_are_excluded_regardless_of_reaction(flag):
    assert ref.measurement_exclusion(valid_row(**{flag:'1'})) == flag


def test_unknown_magnesium_cannot_be_inferred_from_recipe_silence_or_total():
    row = valid_row(); row['p_mg'] = ''
    assert ref.measurement_exclusion(row, {'Cofactor':'-'}) == 'Mg_condition_unspecified'
    assert ref.measurement_exclusion(row, {'Cofactor':'magnesium acetate 50 mM'}) == 'Mg_present_free_activity_unknown'
    row['p_mg'] = '3'
    assert ref.measurement_exclusion(row, {'Cofactor':'magnesium acetate 50 mM'}) is None


def test_fractional_coefficients_and_unrecognized_equations():
    assert parse_equation('0.5 kegg:C00001 + kegg:C00001 = 2 kegg:C00002')[0] == Counter({'C00001':-1.5, 'C00002':2})
    assert parse_equation('n kegg:C00001 = kegg:C00002')[1] is False


def test_rebuild_keeps_unlabelled_structures_and_applies_orientation(monkeypatch, tmp_path):
    source, db = tmp_path/'source', tmp_path/'db'
    source.mkdir(); (db/'Biochemistry/Aliases').mkdir(parents=True)
    (db/'Biochemistry/Aliases/Unique_ModelSEED_Compound_Aliases.txt').write_text('fixture')
    monkeypatch.setattr(ref.B, 'kegg_to_modelseed', lambda _: {'C00001':'a','C00002':'b'})
    monkeypatch.setattr(ref.B, 'load_modelseed_reactions', lambda _: {'reverse':{'a':1,'b':-1}, 'none':{'c':1}})
    row = valid_row(); row.update(_source_row=2, _reference_metadata={}, _reference_comments={})
    bad = dict(row, _source_row=3, todo_check_in_primary_literature='1')
    monkeypatch.setattr(ref, 'opentecr_rows', lambda _: ([(row,'C00001 = C00002','test'), (bad,'C00001 = C00002','test')], {}))
    calls=[]
    class Transform:
        def reaction(self, rid): return rid
        def correction(self, *args): calls.append(args); return 2.
    inputs={'reverse':{'species':{},'exp':[99],'exp_old':'stale'}, 'none':{'species':{'unchanged':[]},'exp':[88]}}
    output, report=ref.rebuild(inputs, source, db, transformer=Transform())
    assert len(calls)==1
    assert output['reverse']['exp'][0] == round(ref.B.R_KJ*298.15*__import__('math').log(10)+2.,2)
    assert 'exp_old' not in output['reverse']
    assert output['none']['exp']==[] and output['none']['species']==inputs['none']['species']
    assert report['measurements'][1]['exclusion_reason']=='todo_check_in_primary_literature'
    assert inputs['reverse']['exp']==[99]
    assert output['reverse']['exp_source_rows']==[2]


def test_condition_transform_never_calls_fitted_prediction():
    class Quantity:
        def __init__(self,value): self.value=value
        def m_as(self,unit): return self.value
    def forbidden(*args,**kwargs): raise AssertionError('fitted reaction energy accessed')
    rx=SimpleNamespace(transform=lambda **kw: Quantity(float(kw['p_h'])-float(kw['p_mg'])))
    obj=Transformer.__new__(Transformer)
    obj.cc=SimpleNamespace(standard_dg_prime=forbidden)
    obj.Q_=lambda value:value
    obj._memo={};obj.reaction=lambda rid:rx
    assert obj.correction('r', 8., .1, 3., 298.15)==-12.


def test_ambiguous_compound_alias_requires_explicit_legacy_policy(tmp_path):
    from metag.validation.reference_transform import kegg_to_modelseed
    directory = tmp_path/'Biochemistry/Aliases'
    directory.mkdir(parents=True)
    (directory/'Unique_ModelSEED_Compound_Aliases.txt').write_text(
        'Source\tExternal ID\tModelSEED ID\nKEGG\tC00001\tcpdA\n'
        'KEGG\tC00001\tcpdB\nKEGG\tC00002\tcpdC\nKEGG\tC00002\tcpdC\n')
    assert kegg_to_modelseed(tmp_path) == {'C00002':'cpdC'}
    assert kegg_to_modelseed(tmp_path, ambiguity='historical_first')['C00001'] == 'cpdA'


@pytest.mark.parametrize('case', ['reaction_ambiguity', 'temperature', 'valid'])
def test_reference_gates_precede_condition_transform(monkeypatch, tmp_path, case):
    source, db = tmp_path/'source', tmp_path/'db'
    source.mkdir(); (db/'Biochemistry/Aliases').mkdir(parents=True)
    (db/'Biochemistry/Aliases/Unique_ModelSEED_Compound_Aliases.txt').write_text('fixture')
    monkeypatch.setattr(ref.B, 'kegg_to_modelseed', lambda *a, **kw: {'C00001':'a','C00002':'b'})
    reactions = {'first': {'a':-1,'b':1}}
    if case == 'reaction_ambiguity':
        reactions['second'] = {'a':-1,'b':1}
    monkeypatch.setattr(ref.B, 'load_modelseed_reactions', lambda _: reactions)
    row = valid_row()
    row.update(_source_row=2, _reference_metadata={}, _reference_comments={})
    if case == 'temperature':
        row['temperature'] = '310.15'
    monkeypatch.setattr(ref, 'opentecr_rows', lambda _: ([(row,'C00001 = C00002','test')], {}))
    calls = []
    transform = SimpleNamespace(reaction=lambda rid: rid,
                                correction=lambda *args: calls.append(args) or 0.)
    inputs = {key:{'species':{}, 'exp':[99], 'explicit':['a']} for key in reactions}
    output, report = ref.rebuild(inputs, source, db, transformer=transform, temperature_K=298.15)
    assert all(item['explicit'] == ['a'] for item in output.values())
    assert 'reference_conditions.py' in report['code_sha256']
    if case == 'valid':
        assert len(calls) == 1 and output['first']['exp']
    else:
        assert not calls and all(not item['exp'] for item in output.values())
        reason = 'ambiguous_reaction_mapping' if case == 'reaction_ambiguity' else 'temperature_mismatch'
        assert report['summary']['excluded'] == {reason:1}
        assert all(item['exp_exclusions'] == {reason:1} for item in output.values())
    if case == 'reaction_ambiguity':
        legacy, legacy_report = ref.rebuild(inputs, source, db, transformer=transform,
                                             mapping_policy='historical_first', temperature_K=298.15)
        assert legacy['first']['exp'] and not legacy['second']['exp']
        assert legacy_report['reference_id'] != report['reference_id']


def test_outside_panel_candidate_does_not_hide_mapping_ambiguity(monkeypatch, tmp_path):
    # Covered by the same candidate set even if its alphabetically first ID is outside the panel.
    source, db = tmp_path/'source', tmp_path/'db'
    source.mkdir(); (db/'Biochemistry/Aliases').mkdir(parents=True)
    (db/'Biochemistry/Aliases/Unique_ModelSEED_Compound_Aliases.txt').write_text('fixture')
    monkeypatch.setattr(ref.B, 'kegg_to_modelseed', lambda _: {'C00001':'a','C00002':'b'})
    monkeypatch.setattr(ref.B, 'load_modelseed_reactions', lambda _: {
        'aaa_outside':{'a':-1,'b':1}, 'zzz_inside':{'a':-1,'b':1}})
    row = dict(valid_row(), _source_row=2, _reference_metadata={}, _reference_comments={})
    monkeypatch.setattr(ref, 'opentecr_rows', lambda _: ([(row,'C00001 = C00002','test')], {}))
    output, report = ref.rebuild({'zzz_inside':{'species':{}}}, source, db,
                                transformer=SimpleNamespace())
    assert output['zzz_inside']['exp_exclusions'] == {'ambiguous_reaction_mapping':1}
    assert len(report['measurements']) == 1
