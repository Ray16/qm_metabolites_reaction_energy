"""Reaction thermodynamics assembled from already-computed species energies.

No routing, model loading, empirical fitting, or result rounding belongs here.
All energies must use the same reference state and units (kJ/mol).
"""
from dataclasses import dataclass
import math

import numpy as np

from metag.energetics.protonation import reaction_transform


@dataclass(frozen=True)
class ReactionEnergy:
    species_sum: float
    proton: float
    protonation: float
    total: float
    sampling_sigma: float


def assemble_reaction(species, energies, sigmas, *, n_hplus, proton_energy,
                      pka_sites, ph, rt_ln10, sampling_groups=None, coupled_protonation=0.0,
                      sampling_offsets=None):
    """Apply stoichiometry, proton bookkeeping, and the independent-site transform.

    Distinct input estimates are independent by default. Explicit sampling
    groups identify repeated uses of the SAME computed species estimate, whose
    signed coefficients must combine before propagating sampling uncertainty.
    This does not provide covariance of systematic model errors across species.
    Known deterministic reference offsets can identify the same sampled estimate
    expressed in different protonation gauges; these change the consistency
    check only, never reaction energies or uncertainty magnitudes.
    The summation and addition order deliberately match the established
    pipeline so extracting this function does not change cancellation/rounding.
    """
    species_sum = sum(coeff * energies[name] for name, (coeff, q, smi) in species.items())
    proton = n_hplus * proton_energy
    total = species_sum + proton
    if sampling_offsets is not None:
        if (sampling_groups is None or not isinstance(sampling_offsets, dict)
                or not set(sampling_offsets) <= set(species)):
            raise ValueError('Sampling reference offsets require known species and explicit groups')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in sampling_offsets.values()):
            raise ValueError('Finite deterministic sampling reference offsets required')
    sampling_offsets = sampling_offsets or {}
    if sampling_groups is None:
        sampling_sigma = float(np.sqrt(sum((coeff * sigmas[name])**2
                                          for name, (coeff, q, smi) in species.items())))
    else:
        if set(sampling_groups) != set(species):
            raise ValueError("Sampling groups must identify every scored species")
        estimates, coefficients = {}, {}
        for name, (coeff, q, smi) in species.items():
            group = sampling_groups[name]
            energy, sigma = energies[name], sigmas[name]
            if not math.isfinite(energy) or not math.isfinite(sigma) or sigma < 0:
                raise ValueError("Shared species estimates must have finite energies and nonnegative sigmas")
            energy += sampling_offsets.get(name, 0.)
            if not math.isfinite(energy):
                raise ValueError('Sampling reference energy overflowed')
            if group in estimates:
                old_energy, old_sigma = estimates[group]
                if (not math.isclose(energy, old_energy, rel_tol=0, abs_tol=1e-6)
                        or not math.isclose(sigma, old_sigma, rel_tol=0, abs_tol=1e-12)):
                    raise ValueError("Repeated sampling identity has inconsistent energy or uncertainty")
            estimates[group] = (energy, sigma)
            coefficients[group] = coefficients.get(group, 0.) + coeff
        sampling_sigma = math.sqrt(math.fsum((coefficients[group] * estimate[1])**2
                                             for group, estimate in estimates.items()))
    protonation = reaction_transform(pka_sites, ph, rt_ln10)
    if (isinstance(coupled_protonation, bool) or not isinstance(coupled_protonation, (int, float))
            or not math.isfinite(coupled_protonation)):
        raise ValueError('Coupled protonation contribution must be finite')
    protonation += coupled_protonation
    total += protonation
    return ReactionEnergy(species_sum, proton, protonation, total, sampling_sigma)


def reaction_covariance(stoichiometry, species_covariance):
    """Propagate a supplied species covariance through S (species x reactions).

    This utility supplies no fitted covariance and does not alter deployed
    reaction intervals. State, reservoir and representation errors must be
    included in the caller's coherent species model before network use.
    """
    matrix = np.asarray(stoichiometry, dtype=float)
    covariance = np.asarray(species_covariance, dtype=float)
    if matrix.ndim != 2 or covariance.shape != (matrix.shape[0], matrix.shape[0]):
        raise ValueError("Expected S with species rows and a square species covariance")
    if not np.isfinite(matrix).all() or not np.isfinite(covariance).all():
        raise ValueError("Stoichiometry and covariance must be finite")
    scale = float(np.max(np.abs(covariance))) if covariance.size else 0.0
    if not np.allclose(covariance, covariance.T, atol=1e-10 * scale, rtol=0):
        raise ValueError("Species covariance must be symmetric")
    covariance = covariance * 0.5 + covariance.T * 0.5
    if covariance.size:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
            raise ValueError("Covariance eigenvalue decomposition exceeded finite numeric range")
        if eigenvalues.min() < -1e-10 * scale:
            raise ValueError("Species covariance must be positive semidefinite")
        if eigenvalues.min() < 0:
            covariance = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    with np.errstate(over="ignore", invalid="ignore"):
        propagated = matrix.T @ covariance @ matrix
    if not np.isfinite(propagated).all():
        raise ValueError("Propagated covariance overflowed; check coefficient/variance scales")
    return propagated
