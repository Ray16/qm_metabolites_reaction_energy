"""MetaG -- first-principles reaction free energies for metabolism.

A self-routing QM (UMA MLIP) pipeline that scores standard transformed reaction Gibbs energies
(ΔrG'°) from the reaction's structures, with calibrated, CV-validated uncertainty for downstream flux analysis.

Public API
----------
Pure-logic (no GPU) -- routing, corrections, uncertainty:
    from metag import symmetry, water_count, uncertainty
    from metag.routing import anchor, aldehyde_hydration, pka_transform, cofactor_cores, truncate

QM pipeline (needs the `uma` runtime: torch + fairchem + xtb) is imported lazily:
    from metag.pipeline import score_reaction        # raises a clear error if the backend is missing

The routing/correction layer is physics-based and gated on structure; the only pieces calibrated on
the experimental database (TECRDB) are the 8 anchor sub-classes (reported alongside dG_raw) and the
uncertainty layer (class-conditional σ, fully nested CV). See README.md.
"""
__version__ = "0.1.0"

__all__ = ["symmetry", "water_count", "uncertainty", "routing", "__version__"]
