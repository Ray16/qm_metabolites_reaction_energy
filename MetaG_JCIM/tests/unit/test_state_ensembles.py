"""Component controls retain all declared states and never manufacture coverage."""
import copy
import math
import hashlib
import json

import pytest
from rdkit import Chem

from metag.energetics.provenance import ordered_geometry_record
from metag.energetics.strict_smd import HARTREE_KJ, request_record
from metag.validation.solvent_ensemble import PRODUCTION_KT_KJ as KT
from metag.validation.state_ensembles import component_factorial, compare_state_ensembles, compare_balanced_components
from metag.validation.state_ensembles import solvation_range


def fixture(smiles='Oc1ccccn1', *, origin=0., shift=0., replacement=(1., -2.), solvents=(2., -3.)):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    charge = Chem.GetFormalCharge(molecule)
    canonical = Chem.MolToSmiles(Chem.RemoveHs(molecule))
    minima, records = [], {}
    for index, old in enumerate((0., 4.)):
        geometry = ordered_geometry_record(symbols, [[j * .4, float(index) + shift, 0.] for j in range(len(symbols))], charge)
        minima.append({'geometry': geometry, 'reference_smiles': canonical,
                       'electronic_kj': origin + old, 'solvation_kj': 0., 'thermal_kj': 0.,
                       'total_kj': origin + old})
        gas = (origin + old + replacement[index]) / HARTREE_KJ
        solvent = solvents[index]
        request = request_record(symbols, geometry['positions_angstrom'], charge, 'def2-tzvpd', xc='wb97m-v')
        records[geometry['sha256']] = {
            'status': 'complete', 'request': request,
            'backend': {'cds_total_convention_verified': True, 'producer_sha256': 'test'},
            'gas_Eh': gas, 'smd_Eh': gas + solvent / HARTREE_KJ,
            'gas_converged': True, 'smd_converged': True, 'smd_cds_Eh': .005, 'dgsolv': solvent}
    cache = {'method': 'implicit', 'smi': smiles, 'q': charge, 'settings': {'producer': 'same'},
             'G': origin - KT * math.log1p(math.exp(-4. / KT)),
             'meta': {'thermal': {'ensemble': {'validated_minima': minima, 'n_reference_thermal_approximation': 0}}}}
    return cache, records


def partition(values):
    low = min(values)
    return low - KT * math.log(sum(math.exp(-(g - low) / KT) for g in values))


@pytest.mark.parametrize('energies', [(-1000.,1000.), (1000.,-1000.), (0.,0.)])
def test_solvation_envelope_survives_arbitrary_electronic_reweighting(energies):
    cache, _ = fixture()
    rows = cache['meta']['thermal']['ensemble']['validated_minima']
    for row, electronic, solvent in zip(rows, energies, (-5.,-2.)):
        row.update(electronic_kj=electronic, solvation_kj=solvent, total_kj=electronic+solvent)
    cache['G'] = partition([r['total_kj'] for r in rows])
    unchanged = copy.deepcopy(cache)
    report = solvation_range(cache, reference_kj=1.)
    assert report['solvent_range_kj'] == [-5.,-2.]
    assert -5. <= report['conditional_transfer_kj'] <= -2.
    assert report['reference_outside_range']
    assert report['minimum_central_discrepancy_under_reweighting_kj'] == 3.
    assert report['uncertainty'] is None and cache == unchanged
    # The opposite electronic ordering approaches the opposite solvent endpoint;
    # neither endpoint can reach the reference by changing populations.
    if energies[0] < energies[1]:
        assert report['conditional_transfer_kj'] == pytest.approx(-5.)
    elif energies[0] > energies[1]:
        assert report['conditional_transfer_kj'] == pytest.approx(-2.)


def test_constant_solvation_has_no_electronic_zero_or_population_dependence():
    cache, _ = fixture(origin=1e6)
    rows = cache['meta']['thermal']['ensemble']['validated_minima']
    for row in rows:
        row['solvation_kj'] = -3.
        row['total_kj'] -= 3.
    cache['G'] -= 3.
    report = solvation_range(cache, reference_kj=-3.)
    assert report['conditional_transfer_kj'] == pytest.approx(-3.)
    assert report['minimum_central_discrepancy_under_reweighting_kj'] == 0.
    assert not report['reference_outside_range']


def test_transfer_envelope_refuses_ions_incomplete_basis_and_invalid_reference():
    charged, _ = fixture('[NH4+]')
    with pytest.raises(ValueError, match='Neutral'):
        solvation_range(charged)
    cache, _ = fixture()
    for bad in (float('nan'),float('inf'),True,'1.2'):
        with pytest.raises(ValueError, match='finite'):
            solvation_range(cache,reference_kj=bad)
    cache['meta']['thermal']['ensemble']['n_reference_thermal_approximation'] = 1
    with pytest.raises(ValueError, match='validated'):
        solvation_range(cache)


def test_factorial_reweights_each_arm_and_does_not_add_cds_twice():
    cache, records = fixture()
    report = component_factorial(cache, records)
    expected = {'original': [0., 4.], 'electronic_replaced': [1., 2.],
                'solvent_replaced': [2., 1.], 'both_replaced': [3., -1.]}
    for arm, energies in expected.items():
        assert report['arms'][arm]['G_kj'] == pytest.approx(partition(energies))
        weights = list(report['arms'][arm]['populations'].values())
        assert sum(weights) == pytest.approx(1.)
        assert weights[0] / weights[1] == pytest.approx(math.exp((energies[1] - energies[0]) / KT))
    assert report['uncertainty'] is None
    # The component interaction need not vanish after method-specific reweighting.
    arms = report['arms']
    assert abs(arms['both_replaced']['G_kj'] - arms['electronic_replaced']['G_kj']
               - arms['solvent_replaced']['G_kj'] + arms['original']['G_kj']) > .01


def test_selected_minimum_retains_parent_and_never_changes_its_cache():
    cache, records = fixture()
    before = copy.deepcopy(cache)
    key = next(iter(records))
    report = component_factorial(cache, {key: records[key]}, geometry_hashes=[key])
    assert report['sampling_scope'] == 'selected_recorded_minima'
    assert report['parent_n_minima'] == 2 and report['n_evaluated_minima'] == 1
    assert report['parent_G_kj'] == before['G']
    assert report['arms']['original']['G_kj'] == 0.
    assert cache == before


@pytest.mark.parametrize('damage', ['missing', 'failed', 'geometry', 'reference', 'charge', 'formula',
                                   'component', 'parent', 'extra', 'duplicate', 'unknown_selection', 'settings'])
def test_incompatible_sources_and_partial_replacements_are_refused(damage):
    cache, records = fixture()
    keys = list(records)
    selection = None
    minimum = cache['meta']['thermal']['ensemble']['validated_minima'][0]
    if damage == 'missing':
        records.pop(keys[0])
    elif damage == 'failed':
        records[keys[0]]['smd_converged'] = False
    elif damage == 'geometry':
        minimum['geometry']['positions_angstrom'][0][0] += .1
    elif damage == 'reference':
        minimum['reference_smiles'] = 'O=c1cccc[nH]1'
    elif damage == 'charge':
        cache['q'] = 1
    elif damage == 'formula':
        cache['smi'] = minimum['reference_smiles'] = 'CCO'
    elif damage == 'component':
        # Even an unevaluated parent conformer must reconstruct correctly.
        cache['meta']['thermal']['ensemble']['validated_minima'][1]['thermal_kj'] += 1.
        selection = [keys[0]]
        records.pop(keys[1])
    elif damage == 'parent':
        cache['G'] += 1.
    elif damage == 'extra':
        records['unrelated'] = copy.deepcopy(records[keys[0]])
    elif damage == 'duplicate':
        selection = [keys[0], keys[0]]
    elif damage == 'unknown_selection':
        selection = ['unknown']
    elif damage == 'settings':
        records[keys[0]]['request']['basis'] = 'def2-svp'
    with pytest.raises(ValueError):
        component_factorial(cache, records, geometry_hashes=selection)


def test_fixed_charge_states_have_recomputed_populations_in_every_arm():
    first, r1 = fixture()
    second, r2 = fixture('O=c1cccc[nH]1', origin=-3., shift=2., replacement=(5., 3.))
    report = compare_state_ensembles({'enol': first, 'keto': second}, {'enol': r1, 'keto': r2}, reference_state='enol')
    assert report['pools']['original']['relative_to_reference_kj'] == pytest.approx(-KT * math.log1p(math.exp(3 / KT)))
    assert report['pools']['original']['populations']['keto'] > .5
    assert report['pools']['electronic_replaced']['populations']['keto'] < .5
    assert report['uncertainty'] is None


def test_state_gaps_and_population_shifts_are_invariant_to_shared_energy_origin():
    reports = []
    for origin in (0., -3_000_000.):
        first, r1 = fixture(origin=origin)
        second, r2 = fixture('O=c1cccc[nH]1', origin=origin - 3., shift=2.)
        reports.append(compare_state_ensembles({'a': first, 'b': second}, {'a': r1, 'b': r2}, reference_state='a'))
    for arm in reports[0]['pools']:
        assert reports[0]['pools'][arm]['relative_to_reference_kj'] == pytest.approx(reports[1]['pools'][arm]['relative_to_reference_kj'], abs=1e-8)
        assert reports[0]['pools'][arm]['populations'] == pytest.approx(reports[1]['pools'][arm]['populations'], abs=1e-8)


@pytest.mark.parametrize('left,right', [
    ('CCO', 'COC'),
    ('C[C@H](O)C(=O)O', 'C[C@@H](O)C(=O)O'),
    ('CC(O)C(=O)O', 'C[C@@H](O)C(=O)O'),
    ('CC(=O)O', 'CC(=O)[O-]'),
    ('C[N+](=O)[O-]', 'C[N+]([O-])=O'),
])
def test_reservoir_changes_epimers_unknown_stereo_and_duplicate_states_are_refused(left, right):
    a, r1 = fixture(left)
    b, r2 = fixture(right, shift=2.)
    with pytest.raises(ValueError):
        compare_state_ensembles({'a': a, 'b': b}, {'a': r1, 'b': r2}, reference_state='a')


@pytest.mark.parametrize('damage', ['missing_state', 'source_settings', 'backend', 'functional'])
def test_one_declared_state_cannot_disappear_or_use_a_different_producer(damage):
    a, r1 = fixture()
    b, r2 = fixture('O=c1cccc[nH]1', shift=2.)
    replacements = {'a': r1, 'b': r2}
    if damage == 'missing_state':
        replacements.pop('b')
    elif damage == 'source_settings':
        b['settings']['producer'] = 'different'
    elif damage == 'backend':
        for record in r2.values():
            record['backend']['producer_sha256'] = 'different'
    else:
        for record in r2.values():
            record['request']['functional'] = 'PBE0'
    with pytest.raises(ValueError):
        compare_state_ensembles({'a': a, 'b': b}, replacements, reference_state='a')


def file_plan(directory):
    states = {}
    for index, (name, smiles) in enumerate((('enol', 'Oc1ccccn1'), ('keto', 'O=c1cccc[nH]1'))):
        cache, records = fixture(smiles, shift=float(index) * 2)
        def write(filename, value):
            data = json.dumps(value).encode()
            (directory / filename).write_bytes(data)
            return {'path': filename, 'sha256': hashlib.sha256(data).hexdigest()}
        states[name] = {'cache': write(name + '.json', cache), 'records': {}}
        for j, (key, record) in enumerate(records.items()):
            wrapper = {'status': 'estimated', 'result': {'record': record}} if index else record
            states[name]['records'][key] = write(name + str(j) + '.json', wrapper)
    path = directory / 'plan.json'
    path.write_text(json.dumps({'reference_state': 'enol', 'states': states}))
    return path


def test_file_comparison_hashes_all_sources_and_refuses_overwrite(tmp_path):
    from metag.tools.compare_states import compare_from_plan
    path = file_plan(tmp_path)
    output = tmp_path / 'comparison.json'
    report = compare_from_plan(path, output)
    assert len(report['source_files_sha256']) == 6
    assert report == json.loads(output.read_text())
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        compare_from_plan(path, output)
    assert output.read_bytes() == before


@pytest.mark.parametrize('damage', ['source_changed', 'failed_record', 'mixed_selection'])
def test_file_comparison_cannot_publish_tampered_or_incomplete_analysis(tmp_path, damage):
    from metag.tools.compare_states import compare_from_plan
    path = file_plan(tmp_path)
    plan = json.loads(path.read_text())
    if damage == 'source_changed':
        (tmp_path / 'enol.json').write_text('{}')
    elif damage == 'failed_record':
        record_path = tmp_path / 'enol0.json'
        data = json.dumps({'status': 'failed'}).encode()
        record_path.write_bytes(data)
        entry = next(iter(plan['states']['enol']['records'].values()))
        entry['sha256'] = hashlib.sha256(data).hexdigest()
    else:
        plan['states']['enol']['geometry_hashes'] = [next(iter(plan['states']['enol']['records']))]
    path.write_text(json.dumps(plan))
    output = tmp_path / 'withheld.json'
    with pytest.raises(ValueError):
        compare_from_plan(path, output)
    assert not output.exists()


def balanced_exchange():
    # CH3OH + C2H6 -> C2H5OH + CH4: atoms, charge and molecule count balance.
    smiles = {'methanol': 'CO', 'ethane': 'CC', 'ethanol': 'CCO', 'methane': 'C'}
    coefficients = dict(zip(smiles, (-1., -1., 1., 1.)))
    caches, records = {}, {}
    for index, (name, smi) in enumerate(smiles.items()):
        caches[name], records[name] = fixture(smi, origin=index * 5., shift=index * 2.,
                                              solvents=(float(index), float(index + 1)))
    return caches, records, coefficients


def test_balanced_exchange_cancels_elemental_energy_zeros_and_scales_and_reverses():
    caches, records, coefficients = balanced_exchange()
    base = compare_balanced_components(caches, records, coefficients)
    shifted = copy.deepcopy(records)
    for name, species_records in shifted.items():
        for record in species_records.values():
            correction = sum({'H': 100., 'C': -1000., 'O': 300.}[s] for s in record['request']['symbols']) / HARTREE_KJ
            record['gas_Eh'] += correction
            record['smd_Eh'] += correction
    shifted_report = compare_balanced_components(caches, shifted, coefficients)
    reverse = compare_balanced_components(caches, records, {name: -2 * c for name, c in coefficients.items()})
    for arm in base['contrasts']:
        value = base['contrasts'][arm]['balanced_exchange_kj']
        assert shifted_report['contrasts'][arm]['balanced_exchange_kj'] == pytest.approx(value, abs=1e-8)
        assert reverse['contrasts'][arm]['balanced_exchange_kj'] == pytest.approx(-2 * value)
    assert base['sampling_scope'] == 'all_recorded_minima' and base['uncertainty'] is None


@pytest.mark.parametrize('damage', ['unbalanced', 'missing_species', 'missing_conformer', 'mixed_backend', 'mixed_source', 'bad_coefficient'])
def test_balanced_exchange_cannot_hide_reservoirs_or_partial_replacements(damage):
    caches, records, coefficients = balanced_exchange()
    if damage == 'unbalanced':
        coefficients['methanol'] = -2.
    elif damage == 'missing_species':
        records.pop('methane')
    elif damage == 'missing_conformer':
        records['methanol'].pop(next(iter(records['methanol'])))
    elif damage == 'mixed_backend':
        for record in records['methanol'].values():
            record['backend']['producer_sha256'] = 'other'
    elif damage == 'mixed_source':
        caches['methanol']['settings']['producer'] = 'other'
    else:
        coefficients['methanol'] = True
    with pytest.raises(ValueError):
        compare_balanced_components(caches, records, coefficients)


def test_balanced_file_plan_reproduces_component_exchange(tmp_path):
    from metag.tools.compare_states import compare_from_plan
    path = file_plan(tmp_path)
    plan = json.loads(path.read_text())
    plan.pop('reference_state')
    plan['stoichiometry'] = {'enol': -1, 'keto': 1}
    path.write_text(json.dumps(plan))
    report = compare_from_plan(path, tmp_path / 'exchange.json')
    assert report['sampling_scope'] == 'all_recorded_minima' and report['uncertainty'] is None
    assert set(report['contrasts']) == {'original', 'electronic_replaced', 'solvent_replaced', 'both_replaced'}


def test_hashed_transfer_plan_preserves_complete_basis_and_refuses_tampering(tmp_path):
    from metag.tools.compare_states import compare_from_plan
    cache, _ = fixture()
    raw = json.dumps(cache).encode()
    cache_path = tmp_path / 'neutral.json'
    cache_path.write_bytes(raw)
    plan = {'mode': 'solvation_range', 'states': {
        'neutral': {'cache': {'path': cache_path.name, 'sha256': hashlib.sha256(raw).hexdigest()},
                    'reference_kj': 5.}}}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    report = compare_from_plan(path, tmp_path / 'range.json')
    assert report['status'] == 'complete' and report['uncertainty'] is None
    assert report['states']['neutral']['n_recorded_minima'] == 2
    assert report['states']['neutral']['minimum_central_discrepancy_under_reweighting_kj'] == 5.
    assert report['source_files_sha256'] == {str(cache_path): hashlib.sha256(raw).hexdigest()}
    cache_path.write_text('{}')
    with pytest.raises(ValueError, match='Source file changed'):
        compare_from_plan(path, tmp_path / 'tampered.json')
    assert not (tmp_path / 'tampered.json').exists()
