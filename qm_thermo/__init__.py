"""QM thermodynamics pipeline for ModelSEED.

Compute aqueous standard Gibbs energies for metabolite structures and transformed
reaction Gibbs energies (Delta_r G'^o) at the quantum level (ORCA DFT, with xtb
conformer screening), as a more accurate alternative to the group/component
contribution methods currently used by ModelSEED.

Stage modules:
    structures  -> load/validate pH-7 microspecies (RDKit)
    conformers  -> ETKDG ensemble + GFN2-xTB screening
    qm_backend  -> ORCA DFT (opt+freq+SMD) behind a pluggable interface
    thermo      -> Boltzmann-averaged ensemble G(aq), standard-state correction
    compute     -> per-compound driver + parallel batch (cached)
    reactions   -> Delta_r G and Alberty/Debye-Huckel transform to Delta_r G'^o
    benchmark   -> compare against openTECR + existing GC methods
"""

__all__ = [
    "config",
    "structures",
    "geometry",
    "conformers",
    "qm_backend",
    "thermo",
    "compute",
    "reactions",
]
"""Reusable building blocks for the thermodynamic composite workflow."""

from .composite import ConformerTerms, EnsembleEnergy, boltzmann_ensemble, extract_ensemble_energy

__all__ = ["ConformerTerms", "EnsembleEnergy", "boltzmann_ensemble", "extract_ensemble_energy"]
