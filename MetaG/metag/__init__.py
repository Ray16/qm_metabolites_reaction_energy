"""MetaG -- first-principles reaction free energies for metabolism.

A self-routing QM (UMA MLIP) pipeline that scores standard transformed reaction Gibbs energies
(ΔrG'°) from structure alone, with calibrated, CV-validated uncertainty for downstream flux analysis.

Public API
----------
Pure-logic (no GPU) -- routing, corrections, uncertainty:
    from metag import symmetry, solvation, uncertainty
    from metag.routing import anchor, aldehyde_hydration, pka_transform, cofactor_cores, truncate

QM pipeline (needs the `uma` runtime: torch + fairchem + xtb) is imported lazily:
    from metag.pipeline import score_reaction        # raises a clear error if the backend is missing

The routing/correction layer is physics-based and gated on structure; the only pieces calibrated to
the experimental database are the 3 anchor sub-classes (LOO-validated) and the uncertainty layer
(class-conditional σ, nested-CV). See README.md.
"""
__version__ = "0.1.0"

__all__ = ["symmetry", "solvation", "uncertainty", "routing", "__version__"]
