"""Coupled state thermodynamics from independently specified equilibrium data.

These utilities do not infer constants, change production routing, or transform
constants between experimental conditions. A conditional pKa/K must apply to
the requested solvent, temperature, ionic strength and reference convention.
"""
import math

from metag.energetics.microstates import combine, R_KJ


def acid_hydration_cycle(*, pka_oxo, kh_acid, kh_ion, ph, temperature_K=298.15,
                         measured_pka_hydrate=None):
    """Resolve HAoxo, Aoxo, HAhydrate and Ahydrate in a unit-water reservoir.

    Kh_acid = [HAhydrate]/[HAoxo], Kh_ion = [Ahydrate]/[Aoxo]. The three
    independent inputs imply pKa_hydrate = pKa_oxo + log10(Kh_acid/Kh_ion).
    An optional fourth measured constant audits closure; it is not fitted or
    silently averaged into the three-parameter model. Free energies are relative
    to HAoxo at the specified pH, not absolute formation energies or offsets to
    add to independently computed state gaps.
    """
    values = (pka_oxo, kh_acid, kh_ion, ph, temperature_K)
    if any(isinstance(x, bool) or not math.isfinite(float(x)) for x in values):
        raise ValueError("State-cycle inputs must be finite numbers")
    pka_oxo, kh_acid, kh_ion, ph, temperature_K = map(float, values)
    if kh_acid <= 0 or kh_ion <= 0 or temperature_K <= 0:
        raise ValueError("Hydration constants and temperature must be positive")
    rt = R_KJ * temperature_K
    ion_gap = rt * math.log(10) * (pka_oxo - ph)
    states = [{"name": "HA_oxo", "G": 0.0},
              {"name": "A_oxo", "G": ion_gap},
              {"name": "HA_hydrate", "G": -rt * math.log(kh_acid)},
              {"name": "A_hydrate", "G": ion_gap - rt * math.log(kh_ion)}]
    model = combine(states, temperature_K)
    # Stable oxo ion fraction gives the exact mixture hydration constant,
    # including extreme pH where directly exponentiating ratios overflows.
    exponent = ph - pka_oxo
    small = 10 ** (-abs(exponent))
    ion_fraction = 1 / (1 + small) if exponent >= 0 else small / (1 + small)
    hydration_ratio = (1 - ion_fraction) * kh_acid + ion_fraction * kh_ion
    model.update(states=states, pH=ph, apparent_hydration_constant=hydration_ratio,
                 hydration_fraction=hydration_ratio / (1 + hydration_ratio),
                 hydration_mixture_correction_kj=-rt * math.log1p(hydration_ratio),
                 implied_pka_hydrate=pka_oxo + (math.log(kh_acid) - math.log(kh_ion)) / math.log(10),
                 cycle_residual_kj=None,
                 scope="conditional four-state model; equilibrium constants must match experimental conditions")
    if measured_pka_hydrate is not None:
        if isinstance(measured_pka_hydrate, bool) or not math.isfinite(float(measured_pka_hydrate)):
            raise ValueError("Measured hydrate pKa must be finite")
        model["cycle_residual_kj"] = rt * math.log(10) * (
            float(measured_pka_hydrate) - model["implied_pka_hydrate"])
    return model
