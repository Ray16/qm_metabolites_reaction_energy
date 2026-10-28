"""Immutable scientific policy for reaction routing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping


@dataclass(frozen=True)
class RoutingPolicy:
    """Effective routing switches for one route-planning operation."""

    cofactor_ring: bool
    coa_core: bool
    ntp_core: bool
    auto_truncate: bool
    route_full: bool
    trunc_anomeric_radius: bool
    trunc_fg_cuts: bool
    trunc_v2: bool
    trunc_maxanion_retry: bool
    trunc_spectator_cations: bool
    ph0_auto: bool
    ph0_isomerase: bool
    ph0_bases: bool
    neutral_qm: bool
    zwitterion_ph0: bool
    anchor_correct: bool
    trunc_radius: int
    trunc_radius_explicit: bool
    ring_chain: bool = False
    mode: str = "reaction"

    @classmethod
    def from_runtime(
        cls,
        flag: Callable[[str], bool],
        environment: Mapping[str, str],
        trunc_radius: int | None = None,
    ) -> "RoutingPolicy":
        """Resolve legacy environment switches once at the routing boundary."""
        radius_explicit = trunc_radius is not None or "TRUNC_RADIUS" in environment
        radius = int(
            trunc_radius
            if trunc_radius is not None
            else environment.get("TRUNC_RADIUS", "2")
        )
        if radius < 0:
            raise ValueError(f"truncation radius must be nonnegative, got {radius}")
        mode = environment.get("METAG_ROUTING_MODE", "reaction").strip().lower()
        if mode not in ("reaction", "full_molecule", "compound_reference"):
            raise ValueError("METAG_ROUTING_MODE must be reaction, full_molecule or compound_reference")
        policy = cls(
            cofactor_ring=flag("COFACTOR_RING"),
            coa_core=flag("COA_CORE"),
            ntp_core=flag("NTP_CORE"),
            auto_truncate=flag("AUTO_TRUNCATE"),
            route_full=flag("ROUTE_FULL"),
            trunc_anomeric_radius=flag("TRUNC_ANOMERIC_RADIUS"),
            trunc_fg_cuts=flag("TRUNC_FG_CUTS"),
            trunc_v2=flag("TRUNC_V2"),
            trunc_maxanion_retry=flag("TRUNC_MAXANION_RETRY"),
            trunc_spectator_cations=flag("TRUNC_SPECTATOR_CATIONS"),
            ph0_auto=flag("PH0_AUTO"),
            ph0_isomerase=flag("PH0_ISOMERASE"),
            ph0_bases=flag("PH0_BASES"),
            neutral_qm=flag("NEUTRAL_QM"),
            zwitterion_ph0=flag("ZWITTERION_PH0"),
            anchor_correct=flag("ANCHOR_CORRECT"),
            trunc_radius=radius,
            trunc_radius_explicit=radius_explicit,
            ring_chain=flag("RING_CHAIN"),
            mode=mode,
        )
        if mode in ("full_molecule", "compound_reference"):
            # This preset controls structural reduction only. Protonation and
            # aqueous-state models retain their explicit scientific scope.
            policy = replace(policy, cofactor_ring=False, coa_core=False,
                             ntp_core=False, auto_truncate=False)
        if mode == "compound_reference":
            policy = replace(policy, ph0_auto=True, ph0_bases=True,
                             ph0_isomerase=True, neutral_qm=True)
        return policy
