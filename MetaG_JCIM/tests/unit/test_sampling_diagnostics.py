import numpy as np
from metag import pipeline as P
from metag.energetics.sampling import sampling_diagnostics
from test_accuracy_review import species_backend  # shared mocked QM fixture; no real model


def test_fresh_trajectory_and_cache_roundtrip(monkeypatch, species_backend):
    monkeypatch.setattr(P, '_THERMAL_ENSEMBLE', True)
    diagnostic = {}
    result = P.implicit_G(None, 0, 'C', (), 1, 1, lambda *_: None, 'methane', diagnostics=diagnostic)
    assert result[0] == -105.
    sampling = diagnostic['sampling']
    assert sampling['stop_reason'] == 'plateau_criterion_met'
    assert sampling['successful_batch_indices'] == [1, 2, 3]
    assert sampling['ensemble_trajectory_kj'] == [-105.] * 3
    assert sampling['minimum_gas_energy_trajectory_kj'] == [-100.] * 3
    args, kwargs = species_backend[-1]
    monkeypatch.setattr(P._sc, 'get', lambda *a, **kw: (args[4], args[5], kwargs['meta']))
    cached = {}
    assert P.implicit_G(None, 0, 'C', (), 1, 1, lambda *_: None, 'methane', diagnostics=cached) == result
    assert cached['source'] == 'cache'
    assert cached['sampling'] == sampling


def test_historical_cache_does_not_claim_convergence(monkeypatch):
    monkeypatch.setattr(P._sc, 'get', lambda *a, **kw: (-5., 0., {'seeds': 3}))
    info = {}
    assert P.implicit_G(None, 0, 'C', (), 1, 1, lambda *_: None, 'methane', diagnostics=info) == (-5., 0.)
    assert info['sampling']['status'] == 'unavailable'
    assert 'stop_reason' not in info['sampling']


def test_cap_and_successful_batches_are_distinct():
    info = sampling_diagnostics([1., -2.], [1, 4], [-10., -11.], seeds=4, hits=0,
                                required_hits=2, maximum=4, tolerance=2.5)
    assert info['stop_reason'] == 'batch_cap_reached'
    assert info['batches_attempted'] == 4
    assert info['successful_batch_indices'] == [1, 4]
    assert info['last_ensemble_increment_kj'] == 3.


def test_routed_diagnostics_water_and_mock_backend(monkeypatch):
    monkeypatch.setattr(P, 'implicit_G', lambda *a, **kw: (-10., .1))
    monkeypatch.setattr(P, 'water_ref_G', lambda *a, **kw: -20.)
    rx = {'species': {'water': [-1, 0, 'O'], 'methane': [1, 0, 'C']}}
    routes, energies, sigma = {'warnings': []}, {}, {}
    P._species_free_energies(None, rx, routes, energies, sigma, 0., (), 1, 1,
                            lambda *_: None, lambda _: False, lambda _: False)
    info = routes['species_diagnostics']
    assert info['water']['sampling']['status'] == 'not_applicable'
    assert info['methane']['sampling']['status'] == 'unavailable'
    assert energies == {'water': -20., 'methane': -10.}
