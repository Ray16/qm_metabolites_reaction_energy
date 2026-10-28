"""Audited ladders are selected by owned exact full-reference identity."""
import copy
import pytest

from metag.conditions import CONDITIONS, RT_LN10
from metag.energetics.protonation import reaction_transform
from metag.protonation_registry import prepare_reaction_protonation, resolve_protonation_models, registry_identity
from metag.routing.planner import route_reaction
from metag.tools.preflight import audit_reactions

CITRATE = 'O=C(O)CC(O)(CC(=O)O)C(=O)O'


def reaction():
    return {'species': {'Citrate': [-1, -3, 'O=C([O-])CC(O)(CC(=O)[O-])C(=O)[O-]'],
                        'Acetate': [1, -1, 'CC(=O)[O-]'],
                        'Oxaloacetate': [1, -2, 'O=C([O-])CC(=O)C(=O)[O-]']}}


def routed(rx):
    return route_reaction(rx, allow_truncate=False, log=lambda _: None)


def prepare(rx, enabled=True):
    scored, routes, _ = routed(rx)
    return prepare_reaction_protonation(rx, scored['species'], scored.get('pka_sites', []),
        routes.get('protonation_sites_by_species'), CONDITIONS, enabled=enabled, rt_ln10=RT_LN10), scored, routes


def test_registered_exact_source_replaces_owned_terms_with_the_validated_transform():
    (sites, coupled, diagnostic, registry, references), scored, routes = prepare(reaction())
    before = reaction_transform(scored['pka_sites'], CONDITIONS['pH'], RT_LN10)
    after = reaction_transform(sites, CONDITIONS['pH'], RT_LN10) + coupled
    assert after - before == pytest.approx(-5.118878243380429, abs=1e-10)
    assert registry['automatic_species'] == ['Citrate']
    assert diagnostic['species']['Citrate']['model']['source']['artifact_sha256']
    assert len(registry_identity()) == 64


def test_disabled_registry_preserves_all_independent_terms():
    (sites, coupled, diagnostic, registry, references), scored, routes = prepare(reaction(), False)
    assert sites == scored['pka_sites'] and coupled == 0. and diagnostic is None
    assert not registry['enabled']


def test_exact_source_is_not_transferred_from_a_different_full_molecule_to_its_core():
    rx = {'species': {'large': [-1, 0, 'O=C(O)CC(O)(CC(=O)O)C(=O)OC']}}
    models, report = resolve_protonation_models(rx, {'large_t': [-1, 0, CITRATE]},
                                               {'large_t': [['react', 4.75, 'acid', 1]] * 3}, enabled=True)
    assert not models
    assert 'not the original full' in report['unapplied_matches']['large_t']


def test_explicit_caller_model_is_preserved_and_not_replaced():
    (sites, coupled, diagnostic, registry, references), scored, routes = prepare(reaction())
    model = copy.deepcopy(diagnostic['species']['Citrate']['model'])
    model['pKas'][0] += .01
    rx = dict(reaction(), species_protonation_models={'Citrate': model})
    (remaining, changed, checked, registry, _), *_ = prepare(rx)
    assert checked['species']['Citrate']['model']['pKas'] == model['pKas']
    assert not registry['automatic_species'] and registry['caller_species'] == ['Citrate']
    assert changed != coupled


def test_caller_flat_terms_are_never_silently_replaced():
    rx = {'species': {'acid': [-1, 0, CITRATE]}, 'pka_sites': [['react', 4.75, 'acid', 3]]}
    models, report = resolve_protonation_models(rx, rx['species'], None, enabled=True)
    assert not models
    assert 'caller flat terms' in report['unapplied_matches']['acid']


def test_a_disabled_neutral_reference_route_does_not_acquire_an_automatic_transform(monkeypatch):
    for flag in ('PH0_AUTO', 'ZWITTERION_PH0', 'NEUTRAL_QM'):
        monkeypatch.setenv(flag, '0')
    neutral = {'species': {'acid': [-1, 0, CITRATE], 'same': [1, 0, CITRATE]}}
    (sites, coupled, diagnostics, report, _), *_ = prepare(neutral)
    assert not sites and coupled == 0 and diagnostics is None
    assert report['unapplied_matches']


def test_opposite_enantiomer_is_not_a_registered_reference():
    rx = {'species': {'D': [-1, 0, 'O=C(O)C[C@@H](O)C(=O)O']}}
    models, report = resolve_protonation_models(rx, rx['species'], {'D': []}, enabled=True)
    assert not models and not report['automatic_species']


def test_identical_registered_states_cancel_and_scale_independently_of_labels():
    for scale in (1., .5, 2.):
        rx = {'species': {'left': [-scale, -3, 'O=C([O-])CC(O)(CC(=O)[O-])C(=O)[O-]'],
                          'right': [scale, -3, 'O=C([O-])CC(O)(CC(=O)[O-])C(=O)[O-]']}}
        (sites, coupled, diagnostics, report, _), *_ = prepare(rx)
        assert not sites and coupled == pytest.approx(0., abs=1e-10)
        assert set(report['automatic_species']) == {'left', 'right'}


def test_CPU_preflight_and_scoring_share_the_registered_models(monkeypatch):
    monkeypatch.setenv('SOURCE_PKA_REGISTRY', '1')
    report = audit_reactions({'any_label': reaction()})
    row = report['reactions']['any_label']
    assert row['routes']['source_protonation_registry']['automatic_species'] == ['Citrate']
    assert row['routes']['coupled_protonation_models']['species']['Citrate']['model']['pKas'] == [3.12927, 4.76227, 6.39727]


def test_source_temperature_mismatch_cannot_be_admitted():
    rx = reaction()
    scored, routes, _ = routed(rx)
    with pytest.raises(ValueError, match='condition mismatch'):
        prepare_reaction_protonation(rx, scored['species'], scored['pka_sites'],
            routes['protonation_sites_by_species'], dict(CONDITIONS, T_K=300.),
            enabled=True, rt_ln10=RT_LN10)


def test_changed_asset_is_refused_before_model_application():
    rx = reaction()
    scored, routes, _ = routed(rx)
    with pytest.raises(ValueError, match='changed after job configuration'):
        prepare_reaction_protonation(rx, scored['species'], scored['pka_sites'],
            routes['protonation_sites_by_species'], CONDITIONS, enabled=True, rt_ln10=RT_LN10,
            expected_registry_sha256='0' * 64)
