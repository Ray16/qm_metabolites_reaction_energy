"""Partition-sum sensitivity tests using independent numerical derivatives."""
import math
import numpy as np
import pytest

from metag.energetics.aqueous_states import hydrate_all_sites, mixture_sampling_uncertainty
from metag.routing import aldehyde_hydration as ah


def test_unpopulated_state_does_not_inflate_sampling_uncertainty():
    sigma, info = mixture_sampling_uncertainty([0., 1000.], [.4, 1e6], ah.RT)
    assert sigma == pytest.approx(.4)
    assert info['conditional_populations'][1] < 1e-100


def test_equal_states_have_population_weighted_sensitivity():
    sigma, info = mixture_sampling_uncertainty([-100., -100.], [2., 4.], ah.RT)
    assert sigma == pytest.approx(math.sqrt(5))
    assert info['input_sensitivities'] == [.5, .5]
    shifted = mixture_sampling_uncertainty([0., 0.], [2., 4.], ah.RT)
    assert shifted == (sigma, info)


@pytest.mark.parametrize('calibrate', [False, True])
def test_production_hydration_sigma_matches_energy_finite_differences(calibrate):
    smiles = 'O=CC=O'
    states = ah.enumerate_hydration_states(smiles).states
    assert len(states) == 2
    sd = np.array([.1, .2, .3])
    def evaluate(x):
        energies, sigmas, routes = {'x':x[0]}, {'x':sd[0]}, {'warnings':[]}
        def implicit(pu, q, smi, *args, **kwargs):
            index = [state[1] for state in states].index(smi) + 1
            return x[index], sd[index]
        hydrate_all_sites(None, {}, 'x', 0, smiles, energies, sigmas, 0., (), 1, 1,
                          lambda _:None, routes, implicit=implicit,
                          water_reference=lambda _:-2., acid_transform=lambda _:0., calibrate=calibrate)
        return energies['x'], sigmas['x'], routes['hydration_diagnostics']['x']
    values = np.array([5., 2., -2.])
    energy, sigma, diagnostic = evaluate(values)
    step = 1e-5
    differences = []
    for index in range(3):
        delta = np.eye(3)[index] * step
        differences.append((evaluate(values + delta)[0] - evaluate(values - delta)[0]) / (2*step))
    expected = np.linalg.norm(np.asarray(differences) * sd)
    assert sigma == pytest.approx(expected, rel=1e-7)
    np.testing.assert_allclose(diagnostic['sampling_uncertainty']['input_sensitivities'], differences, atol=1e-8)
    assert diagnostic['carbonyl_population'] + sum(
        state['conditional_population'] for state in diagnostic['states']) == pytest.approx(1.)


def test_calibration_shared_reference_is_not_independent_noise():
    # With slope zero every effective state moves with the one carbonyl input;
    # hydrate energies and their uncertainty must have no effect at all.
    jacobian = [[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]]
    sigma, info = mixture_sampling_uncertainty([0., 2., -1.], [3., 1e6, 1e6], ah.RT, jacobian)
    assert sigma == pytest.approx(3.)
    assert info['input_sensitivities'][1:] == [0., 0.]


@pytest.mark.parametrize('energies,sigmas,rt', [
    ([], [], 2.5), ([float('nan')], [1], 2.5), ([0], [None], 2.5),
    ([0], [-1], 2.5), ([0], [1], 0), ([0], [1, 2], 2.5),
])
def test_invalid_sampling_inputs_cannot_become_finite_uncertainty(energies, sigmas, rt):
    with pytest.raises(ValueError):
        mixture_sampling_uncertainty(energies, sigmas, rt)


def test_shared_species_cancels_in_sampling_and_mean():
    from metag.assembly import assemble_reaction
    species = {'left':[-1, 0, 'CCO'], 'right':[1, 0, 'OCC']}
    result = assemble_reaction(species, {'left':-100., 'right':-100.}, {'left':5., 'right':5.},
                               n_hplus=0, proton_energy=0, pka_sites=[], ph=7, rt_ln10=5.7,
                               sampling_groups={'left':'same', 'right':'same'})
    assert result.total == result.sampling_sigma == 0.


def test_partial_shared_species_cancellation_and_independent_method():
    from metag.assembly import assemble_reaction
    species = {'left':[-.5, 0, 'CCO'], 'right':[1, 0, 'OCC']}
    kwargs = dict(n_hplus=0, proton_energy=0, pka_sites=[], ph=7, rt_ln10=5.7)
    energy, sigma = {'left':-100., 'right':-100.}, {'left':4., 'right':4.}
    shared = assemble_reaction(species, energy, sigma, **kwargs,
                               sampling_groups={'left':'same', 'right':'same'})
    independent = assemble_reaction(species, energy, sigma, **kwargs,
                                    sampling_groups={'left':'implicit', 'right':'explicit'})
    assert shared.sampling_sigma == 2.
    assert independent.sampling_sigma == pytest.approx(math.sqrt(20.))
    assert shared.total == independent.total


def test_inconsistent_repeated_state_is_rejected():
    from metag.assembly import assemble_reaction
    with pytest.raises(ValueError, match='inconsistent'):
        assemble_reaction({'left':[-1, 0, 'CCO'], 'right':[1, 0, 'OCC']},
                          {'left':-100., 'right':-90.}, {'left':5., 'right':5.},
                          n_hplus=0, proton_energy=0, pka_sites=[], ph=7, rt_ln10=5.7,
                          sampling_groups={'left':'same', 'right':'same'})


def test_production_combines_repeated_species_without_erasing_method_error(monkeypatch):
    from metag import pipeline as p
    monkeypatch.setattr(p, '_flag', lambda name: name == 'STD_STATE_1M')
    def implicit(*args, diagnostics, **kwargs):
        diagnostics['method'] = 'implicit'
        return -100., 5.
    monkeypatch.setattr(p, 'implicit_G', implicit)
    result = p.score_reaction(None, {'species': {'left':[-1, 0, 'CCO'], 'right':[1, 0, 'OCC']}},
                              log=lambda _:None)
    assert result['dG'] == result['U_samp'] == 0.
    assert result['sigma_pred'] is None
    assert result['uncertainty_diagnostic']['sigma_kj'] > 0.
    assert result['routes']['sampling_uncertainty']['n_sampling_groups'] == 1
