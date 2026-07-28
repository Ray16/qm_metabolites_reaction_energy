"""Small, explicit building blocks for pH-dependent microspecies families.

The fixed-species Alberty transform remains useful as a baseline.  These helpers
add the missing partition function when a microspecies family and an independent
pKa have been curated.  They intentionally do not infer pKas from the composite:
anion solvation makes those relative energies unreliable in this project.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

R_KJ = 8.314462618e-3


@dataclass(frozen=True)
class ProtonationFamily:
    """A sequential acid/base family anchored to one calculated microspecies.

    ``pka_values`` are macroscopic, sequential acid dissociation constants in
    the order ``H_nA, H_(n-1)A, ..., A``.  ``reference_deprotonations`` is the
    number of protons lost by the QM structure relative to ``H_nA``.  This
    construction uses externally curated pKas to form the biochemical
    partition function; it never infers relative protonation energies from the
    present solvation model.
    """

    compound_id: str
    pka_values: tuple[float, ...]
    reference_deprotonations: int
    source: str
    citation: str
    reference_label: str = "stored_modelseed_microspecies"

    def __post_init__(self) -> None:
        if not self.pka_values:
            raise ValueError(f"{self.compound_id}: at least one pKa is required")
        if not 0 <= self.reference_deprotonations <= len(self.pka_values):
            raise ValueError(
                f"{self.compound_id}: reference_deprotonations must index a family state"
            )
        if not self.source or not self.citation:
            raise ValueError(f"{self.compound_id}: source and citation are required")

    @classmethod
    def from_dict(cls, compound_id: str, value: dict[str, Any]) -> "ProtonationFamily":
        required = {"pka_values", "reference_deprotonations", "source", "citation"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"{compound_id}: microspecies record missing {sorted(missing)}")
        return cls(
            compound_id=compound_id,
            pka_values=tuple(float(x) for x in value["pka_values"]),
            reference_deprotonations=int(value["reference_deprotonations"]),
            source=str(value["source"]),
            citation=str(value["citation"]),
            reference_label=str(value.get("reference_label", "stored_modelseed_microspecies")),
        )

    def log_relative_weights(self, pH: float) -> tuple[float, ...]:
        """Log population weights for 0..n deprotonations, relative to H_nA."""
        log10 = math.log(10.0)
        values = [0.0]
        running = 0.0
        for pka in self.pka_values:
            running += (pH - pka) * log10
            values.append(running)
        return tuple(values)

    def fractions(self, pH: float) -> tuple[float, ...]:
        """Equilibrium microspecies fractions at the supplied pH."""
        logs = self.log_relative_weights(pH)
        maximum = max(logs)
        weights = [math.exp(value - maximum) for value in logs]
        total = sum(weights)
        return tuple(value / total for value in weights)

    def correction_from_reference_kJ(self, pH: float, temperature_K: float = 298.15) -> float:
        """Legendre/Boltzmann correction to the reference transformed energy.

        If the normal scorer has formed ``G'_reference`` for the selected QM
        microspecies, the rigorously transformed family energy is
        ``G'_reference + correction``.  The reference term is subtracted from
        the log partition so this is safe to add after the existing Alberty
        transform and does not double-count its pH contribution.
        """
        logs = self.log_relative_weights(pH)
        maximum = max(logs)
        log_partition = maximum + math.log(sum(math.exp(value - maximum) for value in logs))
        return -R_KJ * temperature_K * (log_partition - logs[self.reference_deprotonations])


def families_from_dict(data: dict[str, Any]) -> dict[str, ProtonationFamily]:
    """Strictly load curated family metadata, rejecting ambiguous records."""
    return {compound_id: ProtonationFamily.from_dict(compound_id, value)
            for compound_id, value in data.items()}


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
