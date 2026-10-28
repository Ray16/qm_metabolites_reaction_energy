"""Calibration bookkeeping cannot turn scarce or unseen data into coverage."""
import copy
import math
import json
import numpy as np
import pytest

from metag.tools.fit_uq import _q, _matrix, _folds
from metag import uq_features as features, uncertainty
from metag.validation.uncertainty_contract import prediction_uncertainty


def test_finite_sample_quantile_does_not_clamp_to_the_maximum_for_too_few_scores():
    assert _q(np.arange(19), .95) == 18.
    with pytest.raises(ValueError, match='Insufficient calibration'):
        _q(np.arange(18), .95)
    assert _q(np.arange(99), .95) == 94.


@pytest.mark.parametrize('scores,level', [([], .95), ([1., float('nan')], .5),
    ([1., float('inf')], .5), ([-1., 2.], .5), ([[1., 2.]], .5), ([True], .5),
    ([1., 2.], True), ([1., 2.], 1.), ([1., 2.], 0.)])
def test_invalid_calibration_quantiles_are_rejected(scores, level):
    with pytest.raises(ValueError):
        _q(scores, level)


def test_a_feature_unobserved_in_all_training_rows_has_finite_imputation():
    rows = {'a': {'x': [None] + [1.] * (len(features.FEATURES) - 1)},
            'b': {'x': [None] + [3.] * (len(features.FEATURES) - 1)}}
    matrix, median = _matrix(rows, ['a', 'b'])
    assert np.isfinite(matrix).all() and np.isfinite(median).all()
    assert matrix[:, 0].tolist() == [0., 0.]
    assert median[1] == 2.


def test_CV_cannot_silently_create_empty_group_folds():
    with pytest.raises(ValueError, match='independent groups'):
        _folds(['a', 'b'], {'a': 'same', 'b': 'same'}, 5, np.random.default_rng(0))


def raw_features():
    return {name: 0. for name in features.FEATURES}


def source_config():
    return {'model': 'test', 'physics': 'frozen', 'conditions': {'T_K': 298.15},
            'estimator_source_sha256': 'a' * 64}


def synthetic_artifact(monkeypatch):
    artifact = copy.deepcopy(uncertainty._uq_artifact())
    artifact.update(config=source_config(), train_min=[-100.] * len(features.FEATURES),
                    train_max=[100.] * len(features.FEATURES), missingness_patterns=[0])
    monkeypatch.setattr(uncertainty, '_UQ_ART', artifact)
    return artifact


def test_internal_coverage_is_never_labelled_external_validation(monkeypatch):
    synthetic_artifact(monkeypatch)
    _, _, _, info = uncertainty.feature_interval(raw_features(), 0., config=source_config())
    assert info['coverage_calibrated']
    assert not info['externally_calibrated'] and not info['formal_coverage_guarantee']
    assert not info['external_validation_available']
    monkeypatch.setattr(uncertainty, 'CALIB_CONFIG', source_config())
    species = {'a': [-1, 0, 'CCO'], 'b': [1, 0, 'COC']}
    *_, info = uncertainty.prediction_interval('isomerase', ['CCO', 'COC'], 0., species=species,
                                               config=uncertainty.CALIB_CONFIG)
    assert info['coverage_calibrated'] and not info['externally_calibrated']


def test_unseen_missing_feature_pattern_withholds_a_calibration_claim(monkeypatch):
    synthetic_artifact(monkeypatch)
    raw = raw_features()
    raw['U_samp'] = None
    _, _, _, info = uncertainty.feature_interval(raw, 0., config=source_config())
    assert info['unsupported_missingness'] and not info['coverage_calibrated']


def test_a_feature_never_observed_in_training_cannot_establish_in_domain_coverage(monkeypatch):
    artifact = synthetic_artifact(monkeypatch)
    artifact['unobserved_features'] = ['U_samp']
    _, _, _, info = uncertainty.feature_interval(raw_features(), 0., config=source_config())
    assert not info['coverage_calibrated'] and 'U_samp' in info['extrapolated_features']


def test_legacy_external_alias_is_not_republished_as_an_external_claim():
    info = {'coverage_calibrated': True, 'level_calibrated': True, 'level': 95,
            'config_mismatch': [], 'ood_flags': [], 'externally_calibrated': True}
    result = prediction_uncertainty(2., -5., 5., 0., info)
    assert result['ci95'] == [-5., 5.]
    assert not result['ci_info']['externally_calibrated']


def test_training_and_inference_share_primitive_features_and_residual_units(tmp_path):
    from metag.tools.fit_uq import records_to_rows
    from metag.uq_features import primitive_reaction_features, transform
    reactions = {}
    expected = None
    for rid, scale in [('unit', 1.), ('half', .5), ('double', 2.)]:
        species = {'a': [-scale, 0, 'CCO'], 'b': [scale, 0, 'COC']}
        reactions[rid] = {'species': species}
        row = {'reaction': rid, 'dG': 10 * scale, 'exp': [6 * scale],
               'U_samp': scale, 'species_scored': species, 'stages': {'pka_transform': 2 * scale},
               'routes': {}, 'config': {'test': 'same'}}
        (tmp_path / (rid + '.json')).write_text(json.dumps(row))
        raw, extent = primitive_reaction_features(species, species, {}, row['stages'], scale)
        assert extent == scale
        if expected is None:
            expected = transform(raw)
        assert transform(raw) == pytest.approx(expected)
    rows = records_to_rows(str(tmp_path), reactions)
    assert all(row['x'] == pytest.approx(expected) and row['err'] == 4. for row in rows.values())


def test_class_calibration_normalizes_energies_spread_and_anchor_extent():
    from metag.tools.calibrate import _normalize
    records = []
    for rid, scale in [('unit', 1.), ('double', 2.), ('half', .5)]:
        records.append({'rid': rid, 'class': 'test', 'dG': 10 * scale, 'dG_raw': 11 * scale,
                        'exp': 6 * scale, 'U_samp': scale,
                        'original_species': {'a': [-scale, 0, 'CCO'], 'b': [scale, 0, 'COC']},
                        'anchor': {'offset': scale, 'extent': scale, 'direction': 1}})
    rows = _normalize(records)
    for key in ('dG', 'dG_raw', 'exp', 'U', 'off', 'ext'):
        assert rows[0][key] == rows[1][key] == rows[2][key]


@pytest.mark.parametrize('kind', ['duplicate', 'ambiguous_reference', 'negative_spread'])
def test_calibration_cannot_silently_choose_a_duplicate_or_ambiguous_target(kind):
    from metag.tools.calibrate import _normalize
    row = {'rid': 'r', 'class': 'test', 'dG': 10., 'exp': [6.], 'U_samp': 1.}
    rows = [row]
    if kind == 'duplicate':
        rows.append(copy.deepcopy(row))
    elif kind == 'ambiguous_reference':
        row['exp'] = [6., 7.]
    else:
        row['U_samp'] = -1.
    with pytest.raises(ValueError):
        _normalize(rows)


@pytest.mark.parametrize('field,value', [('U_samp', False), ('dG_raw', float('inf')),
    ('anchor.offset', float('nan')), ('anchor.extent', 0.), ('anchor.direction', .5),
    ('anchor.direction', True)])
def test_invalid_calibration_metadata_cannot_enter_a_fold(field, value):
    from metag.tools.calibrate import _normalize
    row = {'rid': 'r', 'class': 'test', 'dG': 1., 'exp': 2.}
    if field.startswith('anchor.'):
        row['anchor'] = {field.split('.')[1]: value}
    else:
        row[field] = value
    with pytest.raises(ValueError):
        _normalize([row])


def test_class_calibration_tests_every_row_and_never_splits_a_group():
    from metag.tools.calibrate import calibrate
    records = [(f'r{i}', 'test', float(i)) for i in range(17)]
    groups = {f'r{i}': f'family{i // 3}' for i in range(17)}
    report = calibrate(records, groups=groups)
    assert report['n_cv_heldout'] == len(records)
    assert set(report['cv_group_assignments'].values()) == set(range(5))
    assert report['n_cv_groups'] == 6
    with pytest.raises(ValueError, match='independent groups'):
        calibrate(records, groups={rid: 'same' for rid, *_ in records})
    with pytest.raises(ValueError, match='declared group'):
        calibrate(records, groups={})


def test_equal_empty_configs_cannot_establish_interval_applicability(monkeypatch):
    art = synthetic_artifact(monkeypatch)
    art['config'] = {}
    *_, info = uncertainty.feature_interval(raw_features(), 0., config={})
    assert not info['coverage_calibrated'] and info['config_mismatch']
    monkeypatch.setattr(uncertainty, 'CALIB_CONFIG', {'model': 'old'})
    assert uncertainty.calibration_mismatch({'model': 'old'})


def test_atomic_calibration_write_rejects_nonfinite_data_without_destroying_previous(tmp_path):
    from metag.tools.calibrate import write_artifact
    path = tmp_path / 'calibration.json'
    write_artifact({'version': 'previous'}, path)
    with pytest.raises(ValueError):
        write_artifact({'sigma': float('nan')}, path)
    assert json.loads(path.read_text()) == {'version': 'previous'}


def test_class_snapshot_cannot_mix_versions_or_accept_negative_scales(tmp_path):
    path = tmp_path / 'calibration.json'
    previous = {'classes': {'test': {'sigma': 3., 'q95abs': 8.}},
                'default_sigma': 4., 'interval_sigma_mult': 2., 'config': source_config()}
    path.write_text(json.dumps(previous))
    snapshot, digest, error = uncertainty._class_snapshot(path)
    path.write_text(json.dumps({**previous, 'default_sigma': -1.}))
    assert snapshot == previous and digest and error is None
    invalid, digest, error = uncertainty._class_snapshot(path)
    assert invalid == {} and digest is None and error


def test_undefined_ranking_metrics_do_not_abort_a_valid_uncertainty_assessment():
    pytest.importorskip('sklearn')
    from metag.tools.fit_uq import nested_cv
    rng = np.random.default_rng(45)
    rows = {f'r{i}': {'x': rng.normal(size=len(features.FEATURES)).tolist(), 'err': 2.} for i in range(80)}
    report = nested_cv(rows, {rid: rid for rid in rows}, repeats=1)
    assert report['auroc_err_gt20'] is None and report['spearman'] is None
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match='at least one repeat'):
        nested_cv(rows, {rid: rid for rid in rows}, repeats=0)
