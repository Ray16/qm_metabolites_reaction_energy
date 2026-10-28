import importlib.util
from pathlib import Path
import pytest
from metag.validation.reference_conditions import audit_conditions, magnesium_exclusion, table_key


def test_numeric_coordinates_join():
    a = dict(zip(('part', 'page', 'col l/r', 'table from top'), ('2', '584', '2', '1')))
    b = {k: v+'.0' for k, v in a.items()}
    assert table_key(a) == table_key(b)


@pytest.mark.parametrize('recipe', ['magnesium acetate (0.050 mol dm-3)', 'MgCl2 50 mM', 'MgSO4'])
def test_total_magnesium_is_not_free_magnesium(recipe):
    assert magnesium_exclusion({'p_mg': ''}, {'Cofactor': recipe})
    assert magnesium_exclusion({'p_mg': '3'}, {'Cofactor': recipe}) is None


@pytest.mark.parametrize('recipe', ['protein 5 mg/ml', 'without magnesium', 'no added Mg', 'magnesium-free'])
def test_no_false_magnesium_recipe_detection(recipe):
    assert magnesium_exclusion({'p_mg': ''}, {'Cofactor': recipe}) is None


def test_missing_conditions_and_kinetic_derivation_preserved():
    row = {'temperature': '310.15', 'p_mg': '', 'ionic_strength': '', 'todo_check_in_primary_literature': '1'}
    comment = {'comment': 'This approximate result was calculated from kinetic data.'}
    result = audit_conditions(row, {'method': 'kinetic'}, comment)
    assert {'temperature_differs_from_prediction', 'ionic_strength_missing',
            'magnesium_condition_unspecified', 'curation_todo_check_in_primary_literature',
            'kinetic_derived_equilibrium_or_method_requires_review'} <= set(result['issues'])
    assert result['raw_row'] == row
    assert result['table_comments'] == comment
    assert result['pMg'] is None


def test_archived_builder_usable_calls_generic_guard():
    path = Path(__file__).resolve().parents[2] / 'artifacts/benchmark/build_opentecr_standard.py'
    spec = importlib.util.spec_from_file_location('reference_builder_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = {key: '' for key in ('solvent_reaction', 'wrong_value', 'duplicate_table',
                              'chemical_reference_reaction', 'virtual_entry', 'type of Kprime')}
    row.update(K_prime='29', p_h='7.25', p_mg='', temperature='298.15')
    assert module.usable(row, {'Cofactor': 'magnesium acetate (0.050 mol dm-3)'})
    row['p_mg'] = '3'
    assert module.usable(row, {'Cofactor': 'magnesium acetate (0.050 mol dm-3)'}) is None
    row['misannotated_value_type'] = '1'
    assert module.usable(row) == 'misannotated_value_type'


@pytest.mark.parametrize('recipe,pmg', [('c(MgCl2) = 0.001 M',3),
    ('c(MgCl2)=2 mM',2.7),('c(MgSO4) = 0.01 mol dm-3',2)])
def test_numeric_pmg_does_not_certify_free_activity_of_added_salt(recipe,pmg):
    from metag.validation.reference_conditions import magnesium_total_activity_ambiguity
    row={'p_mg':str(pmg),'additional data':recipe}
    evidence=magnesium_total_activity_ambiguity(row)
    assert evidence and evidence[0]['reported_pMg']==pmg
    report=audit_conditions(row)
    assert 'pMg_matches_total_magnesium_recipe' in report['issues']
    assert report['exclusion']


def test_distinct_free_value_or_source_backed_basis_not_rejected():
    from metag.validation.reference_conditions import magnesium_total_activity_ambiguity
    row={'p_mg':'3.5','additional data':'c(MgCl2) = 0.001 M'}
    assert not magnesium_total_activity_ambiguity(row)
    row.update(p_mg='3',p_mg_basis='free_activity')
    assert magnesium_total_activity_ambiguity(row)  # empty evidence cannot certify it
    row['p_mg_source']='explicit synthetic measurement provenance'
    assert not magnesium_total_activity_ambiguity(row)


def test_reference_builder_withholds_ambiguous_total_mg_before_transform():
    from metag.tools.build_reference import measurement_exclusion
    row={'K_prime':'21.3','p_h':'7.5','temperature':'298.15','p_mg':'3',
         'ionic_strength':'0.1','additional data':'c(MgCl2) = 0.001 M'}
    assert measurement_exclusion(row)=='pMg_matches_total_magnesium_recipe'
