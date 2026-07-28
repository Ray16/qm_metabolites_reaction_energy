"""Reusable assembly of aqueous compound energies from conformer records.

The electronic-energy model is deliberately outside this module.  A runner for
UMA, MACE-POLAR, ORCA, or another model produces one electronic energy per
conformer; this module combines it with solvation and thermal terms in a single,
testable place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .thermo import boltzmann_average_gibbs_kJ


@dataclass(frozen=True)
class ConformerTerms:
    """Energy terms for one conformer, all in kJ/mol."""

    electronic_kJ: float
    solvation_kJ: float
    thermal_kJ: float

    @property
    def aqueous_gibbs_kJ(self) -> float:
        return self.electronic_kJ + self.solvation_kJ + self.thermal_kJ


@dataclass(frozen=True)
class EnsembleEnergy:
    """Boltzmann-averaged aqueous free energy and conformer weights."""

    gibbs_kJ: float
    weights: tuple[float, ...]

    @property
    def effective_conformers(self) -> float:
        return 1.0 / sum(weight * weight for weight in self.weights)


def boltzmann_ensemble(
    conformers: Sequence[ConformerTerms], *, temperature_K: float
) -> EnsembleEnergy:
    """Return the ensemble free energy and normalized conformer weights."""
    if not conformers:
        raise ValueError("cannot assemble an empty conformer ensemble")
    values = [conformer.aqueous_gibbs_kJ for conformer in conformers]
    free_energy = boltzmann_average_gibbs_kJ(values, temperature_K)
    # The relative Boltzmann factors follow directly from dG; deriving weights
    # here ensures all electronic models use identical ensemble bookkeeping.
    import math

    rt = 8.314462618e-3 * temperature_K
    minimum = min(values)
    factors = [math.exp(-(value - minimum) / rt) for value in values]
    partition = sum(factors)
    return EnsembleEnergy(free_energy, tuple(value / partition for value in factors))


def extract_ensemble_energy(
    record: Mapping[str, object], *, temperature_K: float
) -> EnsembleEnergy:
    """Read a standard MLIP breakdown record and recompute its ensemble energy.

    This validates that the persisted `G_aq_kJ` agrees with the individual
    terms rather than treating generated JSON as an opaque source of truth.
    UMA records use ``E_UMA_kJ`` and MACE records use ``E_elec_kJ``.
    """
    raw_conformers = record.get("conformers")
    if not isinstance(raw_conformers, list) or not raw_conformers:
        raise ValueError("breakdown record has no conformer records")
    terms = []
    for conformer in raw_conformers:
        if not isinstance(conformer, Mapping):
            raise ValueError("invalid conformer record")
        electronic = conformer.get("E_elec_kJ", conformer.get("E_UMA_kJ"))
        if electronic is None:
            raise ValueError("conformer record has no electronic energy")
        try:
            terms.append(ConformerTerms(
                electronic_kJ=float(electronic),
                solvation_kJ=float(conformer["dGsolv_kJ"]),
                thermal_kJ=float(conformer["G_RRHO_kJ"]),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid conformer energy terms") from exc
    return boltzmann_ensemble(terms, temperature_K=temperature_K)
