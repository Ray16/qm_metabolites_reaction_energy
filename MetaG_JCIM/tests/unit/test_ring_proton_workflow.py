"""Structural state generation and end-to-end pooling keep all required states."""
import math
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from metag.routing.ring_protons import enumerate_ring_protons
from metag.state_pools import KT_KJ, prepare_ring_pools

THF = 'CNC(=O)c1ccc(NC[C@H]2CNc3[nH]c(N)nc(=O)c3N2)cc1'
THF_ALT = 'CNC(=O)c1ccc(NC[C@H]2CNc3nc(N)[nH]c(=O)c3N2)cc1'
DHF = 'CNC(=O)c1ccc(NCC2=Nc3c(nc(N)[nH]c3=O)NC2)cc1'
DHF_ALT = 'CNC(=O)c1ccc(NCC2=Nc3c([nH]c(N)nc3=O)NC2)cc1'


def canon(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize('left,right', [(THF, THF_ALT), (DHF, DHF_ALT)])
def test_both_drawings_generate_the_same_complete_structural_state_set(left, right):
    a, b = enumerate_ring_protons(left, 0), enumerate_ring_protons(right, 0)
    assert a.complete and b.complete and a.states == b.states
    assert a.identity == b.identity
    assert set(a.states) == {canon(left), canon(right)}
    formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(left))
    assert all(rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(s)) == formula for s in a.states)


def test_atom_order_and_maps_do_not_choose_the_proton_pool():
    mol = Chem.MolFromSmiles(THF)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    shuffled = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    assert enumerate_ring_protons(Chem.MolToSmiles(shuffled, canonical=False), 0).identity == enumerate_ring_protons(THF, 0).identity


def test_cap_is_reported_as_incomplete_and_cannot_form_a_partition():
    result = enumerate_ring_protons(THF, 0, maximum=1)
    assert not result.complete and 'cap exceeded' in result.reason


@pytest.mark.parametrize('smiles,charge', [('CCO', 0), ('CC(=O)[O-]', -1), ('N', 0), ('O=c1cc[nH]cc1', 0)])
def test_unrelated_structures_do_not_acquire_ring_proton_states(smiles, charge):
    result = enumerate_ring_protons(smiles, charge)
    assert result.complete and not result.applicable and len(result.states) == 1


def test_flat_unowned_terms_and_explicit_clusters_cannot_be_mixed_with_pools():
    rx = {'species': {'a': [-1, 0, THF]}, 'pka_sites': [['react', 4.75]]}
    with pytest.raises(ValueError, match='no owned'):
        prepare_ring_pools(rx, {})
    with pytest.raises(ValueError, match='implicit state'):
        prepare_ring_pools(dict(rx, pka_sites=[]), {}, explicit=True)


@pytest.fixture
def pipeline(monkeypatch):
    p = pytest.importorskip('metag.pipeline')
    from metag.config import FLAG_DEFAULTS
    for flag in ('COFACTOR_RING', 'COA_CORE', 'NTP_CORE', 'AUTO_TRUNCATE', 'PH0_AUTO',
                 'PH0_BASES', 'ZWITTERION_PH0', 'NEUTRAL_QM', 'ANCHOR_CORRECT',
                 'ALDEHYDE_HYDRATION', 'RING_CHAIN', 'TRUNC_VALIDATE', 'SOURCE_PKA_REGISTRY', 'SMD_SOLV'):
        monkeypatch.setenv(flag, '0')
    monkeypatch.setenv('RING_PROTON_ENSEMBLE', '1')
    monkeypatch.setenv('STD_STATE_1M', '0')
    values = {canon(THF): 0., canon(THF_ALT): -36.4848442254588,
              canon(DHF): 5., canon(DHF_ALT): 60.40671337675303, '[H][H]': -10.}
    calls = []
    def implicit(pu, charge, smiles, *args, diagnostics=None, **kwargs):
        calls.append(canon(smiles))
        if diagnostics is not None:
            diagnostics.update(method='implicit', source='test_backend')
        return values.get(canon(smiles)), 1.
    monkeypatch.setattr(p, 'implicit_G', implicit)
    monkeypatch.setattr(p, 'water_ref_G', lambda *args, **kwargs: 0.)
    return p, values, calls


def reaction(scale=1.):
    return {'species': {'THF': [-scale, 0, THF], 'DHF': [scale, 0, DHF],
                        'hydrogen': [scale, 0, '[H][H]']}, 'n_Hplus': 0}


def test_public_pipeline_incorporates_the_complete_two_sided_state_shift(pipeline, monkeypatch):
    p, values, calls = pipeline
    on = p.score_reaction(None, reaction(), log=lambda _: None, allow_truncate=False)
    monkeypatch.setenv('RING_PROTON_ENSEMBLE', '0')
    off = p.score_reaction(None, reaction(), log=lambda _: None, allow_truncate=False)
    assert on['dG'] - off['dG'] == pytest.approx(36.484845, abs=.1)
    assert set(on['routes']['ring_proton_ensembles']) == {'THF', 'DHF'}
    assert all(len(row['states']) == 2 for row in on['routes']['ring_proton_ensembles'].values())
    assert on['sigma_pred'] is None and on['ci95'] == [None, None]
    assert on['config']['ring_proton_ensemble']


def test_pool_scoring_is_reversible_and_extensive(pipeline):
    p, values, calls = pipeline
    scores = [p.score_reaction(None, reaction(scale), log=lambda _: None, allow_truncate=False)
              for scale in (1., -1., 2., .5)]
    assert scores[1]['dG'] == -scores[0]['dG']
    assert scores[2]['dG'] == pytest.approx(2 * scores[0]['dG'], abs=.1)
    assert scores[3]['dG'] == pytest.approx(.5 * scores[0]['dG'], abs=.1)


def test_different_drawings_of_the_same_pool_cancel_energy_and_sampling(pipeline):
    p, values, calls = pipeline
    rx = {'species': {'left': [-1, 0, THF], 'right': [1, 0, THF_ALT]}}
    result = p.score_reaction(None, rx, log=lambda _: None, allow_truncate=False)
    assert result['dG'] == 0. and result['U_samp'] == 0.
    assert result['routes']['sampling_uncertainty']['n_sampling_groups'] == 1
    assert len(calls) == 2  # both supplied reference states reused; no third calculation


def test_missing_alternative_fails_closed_without_a_reference_only_rescue(pipeline):
    p, values, calls = pipeline
    values.pop(canon(THF_ALT))
    result = p.score_reaction(None, reaction(), log=lambda _: None, allow_truncate=False)
    assert result['dG'] is None and result['ci95'] == [None, None]
    assert 'incomplete ring-proton' in result['suspect']


def test_joint_hydration_and_ring_protons_combine_all_four_raw_states(pipeline, monkeypatch):
    p, values, calls = pipeline
    from metag.routing.aldehyde_hydration import enumerate_hydration_states
    source = 'C[C@H](O)C(=O)C1=Nc2c([nH]c(N)nc2=O)NC1'
    ring = enumerate_ring_protons(source, 0)
    assert len(ring.states) == 2
    values.clear()
    raw = []
    for index, state in enumerate(ring.states):
        hydration = enumerate_hydration_states(state)
        assert hydration.complete and len(hydration.states) == 1
        values[canon(state)] = float(index)
        values[canon(hydration.states[0][1])] = -8. - 2 * index
        raw.extend([float(index), -8. - 2 * index])
    monkeypatch.setenv('ALDEHYDE_HYDRATION', '1')
    monkeypatch.setenv('CARBONYL_HYDRATION_ALL', '1')
    monkeypatch.setenv('HYDRATION_CAL', '0')
    rx = {'species': {'left': [-1, 0, source], 'right': [1, 0, source]}}
    result = p.score_reaction(None, rx, log=lambda _: None, allow_truncate=False)
    assert result['dG'] == 0. and result['U_samp'] == 0.
    expected = min(raw) - KT_KJ * math.log(sum(math.exp(-(g - min(raw)) / KT_KJ) for g in raw))
    assert result['routes']['ring_proton_ensembles']['left']['pooled_G_kj'] == pytest.approx(expected, abs=1e-3)


def test_unsupported_joint_configuration_is_refused_before_species_work(pipeline, monkeypatch):
    p, values, calls = pipeline
    monkeypatch.setenv('ALDEHYDE_HYDRATION', '1')
    monkeypatch.setenv('CARBONYL_HYDRATION_ALL', '0')
    result = p.score_reaction(None, reaction(), log=lambda _: None, allow_truncate=False)
    assert result['dG'] is None and not calls
