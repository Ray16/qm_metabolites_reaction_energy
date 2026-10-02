"""Thermodynamic combination of resolved aqueous species states.

Conformers share a molecular graph.  Tautomers, anomers, ring forms, and other
chemical microstates do not, and must therefore be scored as separate species
before their partition functions are combined:

    G_eff = -RT ln sum_i g_i exp[-(G_i + offset_i) / RT]

``offset_i`` is reserved for independently established state free energies
(for example, offsets derived from aqueous populations).  It must not be fit
to reaction free-energy residuals.  This module intentionally does not choose
or generate states: automatic tautomer canonicalization and a global
alpha/beta choice have both failed validation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


R_KJ = 8.314462618e-3
DEFAULT_T_K = 298.15


def relative_free_energies_from_populations(
    populations: Mapping[str, float], temperature: float = DEFAULT_T_K
) -> dict[str, float]:
    """Convert normalized or unnormalized populations to relative state free energies.

    The most populated state has offset zero.  Multiplying all populations by
    a common constant therefore leaves the result unchanged.  These values
    are the *complete* relative state free energies.  They should be combined
    with one computed absolute reference state, not added to separately
    computed state free energies (which would double-count the state gap).
    """
    if not populations:
        raise ValueError("at least one microstate population is required")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    clean = {}
    for name, population in populations.items():
        p = float(population)
        if not math.isfinite(p) or p <= 0:
            raise ValueError(f"microstate {name!r} has non-positive or non-finite population")
        clean[name] = p
    reference = max(clean.values())
    rt = R_KJ * temperature
    return {name: -rt * math.log(population / reference) for name, population in clean.items()}


def combine(states: Iterable[Mapping[str, float]], temperature: float = DEFAULT_T_K) -> dict:
    """Combine resolved state free energies and return effective G, populations, and sigma.

    Each state requires ``name`` and ``G``.  Optional ``offset`` defaults to
    zero, ``degeneracy`` to one, and ``sigma`` to zero.  State errors are
    treated as independent when propagating ``sigma``.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    rows = []
    for raw in states:
        name = str(raw["name"])
        energy = float(raw["G"])
        offset = float(raw.get("offset", 0.0))
        degeneracy = float(raw.get("degeneracy", 1.0))
        sigma = float(raw.get("sigma", 0.0) or 0.0)
        if not all(math.isfinite(v) for v in (energy, offset, degeneracy, sigma)):
            raise ValueError(f"microstate {name!r} contains a non-finite value")
        if degeneracy <= 0:
            raise ValueError(f"microstate {name!r} has non-positive degeneracy")
        if sigma < 0:
            raise ValueError(f"microstate {name!r} has negative sigma")
        rows.append((name, energy + offset, degeneracy, sigma))
    if not rows:
        raise ValueError("at least one microstate is required")

    rt = R_KJ * temperature
    reduced = [math.log(g) - energy / rt for _, energy, g, _ in rows]
    pivot = max(reduced)
    partition = sum(math.exp(value - pivot) for value in reduced)
    log_partition = pivot + math.log(partition)
    populations = {
        name: math.exp(value - log_partition)
        for (name, _, _, _), value in zip(rows, reduced)
    }
    sigma = math.sqrt(sum((populations[name] * state_sigma) ** 2
                          for name, _, _, state_sigma in rows))
    return {
        "G": -rt * log_partition,
        "sigma": sigma,
        "populations": populations,
        "temperature_K": temperature,
    }
