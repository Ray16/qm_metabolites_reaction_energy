"""Immutable scientific policy for reaction routing."""

from __future__ import annotations

from dataclasses import dataclass
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
        return cls(
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
        )
