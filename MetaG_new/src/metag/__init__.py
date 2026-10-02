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

The routing/correction layer is structure-gated. Frozen release 2026-10-01c
uses no TECRDB-fitted point correction; its class-conditional uncertainty is
calibrated by grouped cross-validation within TECRDB.
"""
__version__ = "0.2.0.dev0"

__all__ = ["symmetry", "water_count", "uncertainty", "routing", "__version__"]
