"""Reaction Gibbs energies and the transform to biochemical standard conditions.

From per-species absolute QM Gibbs energies we form reaction Delta_r G (elemental
references cancel because reactions are atom/charge balanced), then apply the
Alberty Legendre transform to pH 7 and an extended Debye-Huckel ionic-strength
correction to obtain Delta_r G'^o -- directly comparable to openTECR/eQuilibrator.

References: Alberty, "Thermodynamics of Biochemical Reactions" (2003);
Noor et al., eQuilibrator (2012).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import config

if TYPE_CHECKING:  # only for type hints; avoids importing rdkit at runtime
    from .structures import Metabolite

# Proton compound id in ModelSEED; excluded from the transformed stoichiometry
# because pH fixes its chemical potential (handled by the N_H terms instead).
PROTON_ID = "cpd00067"

# Extended Debye-Huckel constants at 298.15 K (Alberty).
DH_A = 0.510651      # mol^-1/2 L^1/2
DH_B = 1.6           # mol^-1/2 L^1/2


@dataclass(frozen=True)
class SpeciesInfo:
    """Per-species data needed for the biochemical transform."""

    cpd_id: str
    n_hydrogens: int     # total H atoms in the species (explicit + implicit)
    charge: int


def species_info(meta: Metabolite) -> SpeciesInfo:
    """Count hydrogens (incl. those folded into heavy atoms) and read the charge."""
    n_h = sum(
        a.GetTotalNumHs() + (1 if a.GetSymbol() == "H" else 0)
        for a in meta.mol.GetAtoms()
        if a.GetSymbol() != "H"
    )
    # meta.mol already has explicit Hs, so just count H atoms directly instead:
    n_h = sum(1 for a in meta.mol.GetAtoms() if a.GetSymbol() == "H")
    return SpeciesInfo(meta.cpd_id, n_hydrogens=n_h, charge=meta.charge)


def debye_huckel_term_kJ(
    charge: int, n_hydrogens: int, conditions: config.Conditions
) -> float:
    """Ionic-strength contribution to a species' transformed formation energy.

    -RT ln(10) * A * (z^2 - N_H) * sqrt(I) / (1 + B sqrt(I)).
    The (z^2 - N_H) grouping folds in the H+ ion's own activity correction.
    """
    I = conditions.ionic_strength_M
    if I <= 0:
        return 0.0
    rt_ln10 = conditions.R_kJ * conditions.temperature_K * math.log(10)
    sqrt_I = math.sqrt(I)
    return -rt_ln10 * DH_A * (charge ** 2 - n_hydrogens) * sqrt_I / (1 + DH_B * sqrt_I)


def hydrogen_referenced_energy_kJ(
    gibbs_kJ: float, info: SpeciesInfo, conditions: config.Conditions
) -> float:
    """Subtract the per-hydrogen proton reference so reactions are H-balanced.

    ModelSEED balances reactions with explicit H+ which we exclude; folding the
    aqueous-proton reference into each species' bound hydrogens (G - N_H * mu_H)
    restores exact cancellation of the hydrogen total-energy reference.
    """
    return (gibbs_kJ + conditions.gas_1atm_to_1M_kJ
            - info.n_hydrogens * conditions.proton_reference_kJ)


def transformed_formation_energy_kJ(
    gibbs_kJ: float, info: SpeciesInfo, conditions: config.Conditions
) -> float:
    """Standard transformed Gibbs energy of a single species at (pH, I).

    Delta_f G'^o = (G - N_H * mu_H) + N_H * RT ln(10) * pH + Debye-Huckel(z, N_H).

    The first term H-references the absolute QM G; the proton (Legendre) and
    Debye-Huckel terms then move it to biochemical standard conditions.
    """
    rt_ln10 = conditions.R_kJ * conditions.temperature_K * math.log(10)
    proton_term = info.n_hydrogens * rt_ln10 * conditions.pH
    return (
        hydrogen_referenced_energy_kJ(gibbs_kJ, info, conditions)
        + proton_term
        + debye_huckel_term_kJ(info.charge, info.n_hydrogens, conditions)
    )


@dataclass(frozen=True)
class Reaction:
    """A reaction as stoichiometry over ModelSEED compound ids (products +)."""

    rxn_id: str
    stoichiometry: dict[str, float]   # cpd_id -> coefficient (H+ excluded)

    def compounds(self) -> set[str]:
        return set(self.stoichiometry)


@dataclass(frozen=True)
class ReactionEnergy:
    rxn_id: str
    dG_standard_kJ: float       # Delta_r G^o (chemical, summed absolute QM G)
    dG_transformed_kJ: float    # Delta_r G'^o at (pH, I, T)


def reaction_dG(
    reaction: Reaction,
    compound_gibbs_kJ: dict[str, float],
    species: dict[str, SpeciesInfo],
    *,
    conditions: config.Conditions = config.DEFAULT_CONDITIONS,
) -> ReactionEnergy:
    """Compute Delta_r G^o and the transformed Delta_r G'^o for one reaction.

    Raises KeyError if any non-proton species lacks a computed energy.
    """
    dG_std = 0.0
    dG_trans = 0.0
    for cpd_id, coeff in reaction.stoichiometry.items():
        if cpd_id == PROTON_ID:
            continue
        g = compound_gibbs_kJ[cpd_id]
        info = species[cpd_id]
        dG_std += coeff * hydrogen_referenced_energy_kJ(g, info, conditions)
        dG_trans += coeff * transformed_formation_energy_kJ(g, info, conditions)
    return ReactionEnergy(reaction.rxn_id, dG_std, dG_trans)
