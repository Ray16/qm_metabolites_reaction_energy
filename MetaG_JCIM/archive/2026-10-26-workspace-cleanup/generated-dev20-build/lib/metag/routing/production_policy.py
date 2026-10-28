"""Boundary between the transferable estimator and retained historical experiments."""
from metag.config import flag


def validate_production_policy(policy):
    """Historical offsets/lookups are inspectable, but cannot enter a production route."""
    if policy.mode not in ("reaction", "full_molecule", "compound_reference"):
        raise ValueError("Unknown routing mode: " + policy.mode)
    if policy.mode in ("full_molecule", "compound_reference") and any((policy.cofactor_ring, policy.coa_core,
                                               policy.ntp_core, policy.auto_truncate)):
        raise ValueError(policy.mode + " routing mode cannot enable structural cores or truncation")
    from metag.routing.pka_transform import _polyacid_pka_enabled
    forbidden = []
    if policy.anchor_correct:
        forbidden.append("ANCHOR_CORRECT (reaction-benchmark offsets)")
    if _polyacid_pka_enabled():
        forbidden.append("POLYACID_PKA (named-compound lookup)")
    if flag("WATER_REF_HYDROLYASE"):
        forbidden.append("WATER_REF_HYDROLYASE (reaction-family water offset)")
    if flag("SMD_SOLV"):
        forbidden.append("SMD_SOLV (legacy solute-only correction with inconsistent water treatment)")
    if forbidden:
        raise ValueError("Unsupported production configuration: " + ", ".join(forbidden)
                         + ". Disable these switches. Historical helpers remain available for "
                         "reproducing archived experiments; they are not part of the general estimator.")
