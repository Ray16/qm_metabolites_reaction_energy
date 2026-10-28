"""MetaG -- physics-informed reaction free energies for metabolism.

A self-routing QM (UMA MLIP) pipeline that scores standard transformed reaction Gibbs energies
(ΔrG'°) from molecular structures. Uncertainty calibration is conditional on
the declared estimator and chemical domain; external validation is incomplete.

Public API
----------
Pure-logic (no GPU) -- routing, corrections, uncertainty:
    from metag import symmetry, water_count, uncertainty
    from metag.routing import anchor, aldehyde_hydration, pka_transform, cofactor_cores, truncate

QM pipeline (needs the `uma` runtime: torch + fairchem + xtb) is imported lazily:
    from metag.pipeline import score_reaction        # raises a clear error if the backend is missing

The routing/correction layer is structure-gated. Production point predictions
use no TECRDB-fitted reaction-energy offset. Benchmark-guided development and
within-database uncertainty calibration are not independent transfer validation.
"""
__version__ = "0.2.0.dev24"

__all__ = ["symmetry", "water_count", "uncertainty", "routing", "__version__"]
