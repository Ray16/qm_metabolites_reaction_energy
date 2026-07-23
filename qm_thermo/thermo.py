"""Assemble a single aqueous standard Gibbs energy per compound.

Combines the per-conformer DFT results into one ensemble free energy by Boltzmann
averaging, and applies the 1 atm -> 1 M standard-state correction so the reported
G(aq) is a proper 1 mol/L aqueous standard quantity (the eQuilibrator convention).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import config
from .qm_backend import QMResult

HARTREE_TO_KJ = 2625.499639


def standard_state_correction_kJ(conditions: config.Conditions) -> float:
    """1 atm ideal-gas -> 1 mol/L standard-state shift, RT*ln(R*T/P0).

    ORCA's RRHO thermochemistry uses a 1 atm gas reference; adding this constant
    per species moves the entropy reference to 1 mol/L for solution-phase work.
    """
    # Molar volume of an ideal gas at (T, 1 atm) in L/mol = R*T/P.
    rt = conditions.R_kJ * conditions.temperature_K
    molar_volume_L = 0.0820574 * conditions.temperature_K  # R in L*atm/mol/K
    return rt * math.log(molar_volume_L)


@dataclass(frozen=True)
class CompoundEnergy:
    """Final aqueous standard Gibbs energy for one compound.

    `gibbs_kJ` is the tier-1 (r2SCAN-3c) ensemble G(aq). When a higher-level
    single point is run, `gibbs_highlevel_kJ` holds the tier-2 ensemble G(aq)
    (high-level electronic energy + tier-1 RRHO thermal correction).
    """

    cpd_id: str
    gibbs_kJ: float                 # ensemble G(aq), 1 M standard state, kJ/mol
    n_conformers: int               # conformers that survived to DFT
    min_conformer_kJ: float         # lowest single-conformer G (before averaging)
    method: str
    all_converged: bool
    gibbs_highlevel_kJ: float | None = None
    sp_method: str | None = None
    wall_seconds: float | None = None   # total ORCA wall time for this compound

    def best_gibbs_kJ(self) -> float:
        """Tier-2 energy when available, else tier-1."""
        return self.gibbs_highlevel_kJ if self.gibbs_highlevel_kJ is not None else self.gibbs_kJ

    def to_dict(self) -> dict:
        return {
            "cpd_id": self.cpd_id,
            "gibbs_kJ": self.gibbs_kJ,
            "n_conformers": self.n_conformers,
            "min_conformer_kJ": self.min_conformer_kJ,
            "method": self.method,
            "all_converged": self.all_converged,
            "gibbs_highlevel_kJ": self.gibbs_highlevel_kJ,
            "sp_method": self.sp_method,
            "wall_seconds": self.wall_seconds,
        }


def boltzmann_average_gibbs_kJ(
    gibbs_values_kJ: list[float], temperature_K: float
) -> float:
    """Ensemble free energy G = -RT ln( sum_i exp(-G_i/RT) ).

    Numerically stabilised by shifting to the minimum before exponentiating.
    """
    if not gibbs_values_kJ:
        raise ValueError("no conformer energies to average")
    rt = config.DEFAULT_CONDITIONS.R_kJ * temperature_K
    g_min = min(gibbs_values_kJ)
    z = sum(math.exp(-(g - g_min) / rt) for g in gibbs_values_kJ)
    return g_min - rt * math.log(z)


def assemble_compound_energy(
    cpd_id: str,
    conformer_results: list[QMResult],
    *,
    highlevel_gibbs_hartree: list[float] | None = None,
    sp_method: str | None = None,
    wall_seconds: float | None = None,
    conditions: config.Conditions = config.DEFAULT_CONDITIONS,
) -> CompoundEnergy:
    """Boltzmann-average conformer DFT results and apply standard-state correction.

    If `highlevel_gibbs_hartree` is given (per conformer: high-level electronic
    energy + tier-1 RRHO thermal correction), a tier-2 ensemble G is also formed.
    """
    if not conformer_results:
        raise ValueError(f"{cpd_id}: no QM results to assemble")

    def _ensemble(values_hartree: list[float]) -> float:
        kJ = [g * HARTREE_TO_KJ for g in values_hartree]
        return (
            boltzmann_average_gibbs_kJ(kJ, conditions.temperature_K)
            + standard_state_correction_kJ(conditions)
        )

    gibbs_hartree = [r.gibbs_hartree for r in conformer_results]
    ensemble_kJ = _ensemble(gibbs_hartree)

    highlevel_kJ = None
    if highlevel_gibbs_hartree is not None:
        highlevel_kJ = _ensemble(highlevel_gibbs_hartree)

    return CompoundEnergy(
        cpd_id=cpd_id,
        gibbs_kJ=ensemble_kJ,
        n_conformers=len(conformer_results),
        min_conformer_kJ=min(g * HARTREE_TO_KJ for g in gibbs_hartree),
        method=conformer_results[0].method,
        all_converged=all(r.converged for r in conformer_results),
        gibbs_highlevel_kJ=highlevel_kJ,
        sp_method=sp_method,
        wall_seconds=wall_seconds,
    )
