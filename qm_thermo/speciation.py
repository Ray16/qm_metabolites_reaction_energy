"""Small, explicit building blocks for pH-dependent microspecies families.

The fixed-species Alberty transform remains useful as a baseline.  These helpers
add the missing partition function when a microspecies family and an independent
pKa have been curated.  They intentionally do not infer pKas from the composite:
anion solvation makes those relative energies unreliable in this project.
"""
from __future__ import annotations

import math

R_KJ = 8.314462618e-3


def monoprotic_base_fraction(pH: float, pKa: float) -> float:
    """Fraction of A- in AH ⇌ A- + H+ at the supplied pH."""
    ratio = 10.0 ** (pH - pKa)
    return ratio / (1.0 + ratio)


def monoprotic_family_correction_kJ(
    pH: float, pKa: float, temperature_K: float = 298.15
) -> float:
    """Additive correction from an AH reference state to the AH/A- family.

    If ``G'_AH`` is the transformed energy of the protonated reference state,
    the family energy is ``G'_AH + correction``.  The relative state population
    is imposed by a curated pKa, rather than by the untrusted QM anion/neutral
    energy difference.
    """
    ratio = 10.0 ** (pH - pKa)
    return -R_KJ * temperature_K * math.log1p(ratio)
