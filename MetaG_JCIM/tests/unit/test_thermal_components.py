"""Physical gas-thermochemistry identities and explicit state boundaries."""
import math
import numpy as np
import pytest
from ase import Atoms
from ase import units
from metag.energetics.thermal import gibbs_corr_from_hessian, EV2KJ


def calculate(monkeypatch, *, qrrho=False, **kwargs):
    monkeypatch.setenv('QRRHO', '1' if qrrho else '0')
    atoms = Atoms('OH2', positions=[[0, 0, 0], [.96, 0, 0], [-.24, .93, 0]])
    return gibbs_corr_from_hessian(atoms, np.eye(9), 0., 'nonlinear', 2,
                                   return_info=True, **kwargs)


@pytest.mark.parametrize('qrrho', [False, True])
def test_enthalpy_entropy_decomposition_closes_and_is_finite(monkeypatch, qrrho):
    g, info = calculate(monkeypatch, qrrho=qrrho, temperature_K=350.)
    c = info['thermodynamics']
    assert g == c['gibbs_correction_kj']
    assert g == pytest.approx(c['enthalpy_correction_kj'] - 350*c['entropy_kj_per_K'], abs=1e-10)
    assert c['n_internal_modes'] == 3
    assert c['zpe_kj'] > 0
    assert all(math.isfinite(v) for v in c.values() if isinstance(v, (int, float)))


def test_ideal_gas_pressure_law_changes_entropy_but_not_enthalpy(monkeypatch):
    g1, i1 = calculate(monkeypatch, temperature_K=500., pressure_Pa=101325.)
    g2, i2 = calculate(monkeypatch, temperature_K=500., pressure_Pa=202650.)
    # ASE uses its own physical constants and MetaG retains its historical
    # rounded eV->kJ/mol factor. Verify the law with those declared units.
    expected = units.kB*EV2KJ*500*math.log(2)
    assert g2-g1 == pytest.approx(expected, rel=1e-12)
    assert i2['thermodynamics']['enthalpy_correction_kj'] == i1['thermodynamics']['enthalpy_correction_kj']
    assert i2['thermodynamics']['entropy_kj_per_K'] < i1['thermodynamics']['entropy_kj_per_K']


def test_qrrho_changes_entropy_and_preserves_harmonic_enthalpy(monkeypatch):
    g0, i0 = calculate(monkeypatch)
    g1, i1 = calculate(monkeypatch, qrrho=True)
    c0, c1 = i0['thermodynamics'], i1['thermodynamics']
    assert c0['enthalpy_correction_kj'] == c1['enthalpy_correction_kj']
    assert c0['zpe_kj'] == c1['zpe_kj']
    assert c0['qrrho_entropy_shift_kj_per_K'] == 0.
    assert g1-g0 == pytest.approx(-c1['temperature_K']*c1['qrrho_entropy_shift_kj_per_K'], abs=1e-10)


@pytest.mark.parametrize('field,value', [('temperature_K',0),('temperature_K',-1),
    ('temperature_K',float('nan')),('temperature_K',True),('pressure_Pa',0),
    ('pressure_Pa',float('inf')),('pressure_Pa',False),('temperature_K','298.15')])
def test_invalid_thermodynamic_conditions_rejected(monkeypatch, field, value):
    with pytest.raises(ValueError, match=field):
        calculate(monkeypatch, **{field:value})


def test_defaults_and_explicit_conditions_agree(monkeypatch):
    from metag.energetics.thermal import T
    default, _ = calculate(monkeypatch)
    explicit, _ = calculate(monkeypatch, temperature_K=T, pressure_Pa=101325.)
    assert explicit == default
