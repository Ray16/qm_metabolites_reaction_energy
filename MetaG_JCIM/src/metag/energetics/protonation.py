"""Numerically stable independent-site protonation bookkeeping.

These terms assume independent sites relative to a specified reference
microstate. A list of macroscopic sequential pKas is not generally equivalent
to this product; coupled sites require a binding polynomial/state model.
The caller supplies RT ln(10) to keep units and temperature explicit.
"""
import math


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def normalize_site(site):
    """Validate the public [side, pKa, kind='acid', multiplicity=1] format."""
    if not isinstance(site, (list, tuple)) or not 2 <= len(site) <= 4:
        raise ValueError("pKa site must contain side, pKa, optional kind and multiplicity")
    side, pka = site[:2]
    kind = site[2] if len(site) > 2 else "acid"
    mult = site[3] if len(site) > 3 else 1.0
    if side not in ("react", "prod"):
        raise ValueError("pKa site side must be 'react' or 'prod'")
    if kind not in ("acid", "base"):
        raise ValueError("pKa site kind must be 'acid' or 'base'")
    _finite(pka, "pKa")
    if _finite(mult, "pKa site multiplicity") <= 0:
        raise ValueError("pKa site multiplicity must be positive")
    return side, pka, kind, mult


def site_free_energy(pka, kind, ph, rt_ln10):
    """G(transformed) - G(reference microstate), in the units of rt_ln10."""
    _finite(pka, "pKa")
    _finite(ph, "pH")
    if _finite(rt_ln10, "RT ln(10)") <= 0:
        raise ValueError("RT ln(10) must be positive")
    if kind not in ("acid", "base"):
        raise ValueError("pKa site kind must be 'acid' or 'base'")
    exponent = ph - pka if kind == "acid" else pka - ph
    # log10(1 + 10**exponent), without overflow or loss of small populations.
    log_partition = max(exponent, 0.0) + math.log1p(10.0 ** -abs(exponent)) / math.log(10.0)
    return -rt_ln10 * log_partition


def reaction_transform(sites, ph, rt_ln10):
    """Sum product-minus-reactant independent-site contributions."""
    terms = []
    for site in sites:
        side, pka, kind, mult = normalize_site(site)
        terms.append((1 if side == "prod" else -1) * mult * site_free_energy(pka, kind, ph, rt_ln10))
    return math.fsum(terms)


def reaction_transform_diagnostics(sites, ph, rt_ln10):
    """Analytic sensitivities of the declared independent-site correction.

    Derivatives are kJ/mol per pKa unit or per pH unit when rt_ln10 is
    kJ/mol. Each pKa is a separate parameter; equal numerical pKas do not
    establish shared experimental provenance. Populations describe this site
    model, not measured molecular populations or uncertainty bounds.
    """
    _finite(ph, "pH")
    if _finite(rt_ln10, "RT ln(10)") <= 0:
        raise ValueError("RT ln(10) must be positive")
    rows = []
    for index, site in enumerate(sites):
        side, pka, kind, mult = normalize_site(site)
        exponent = ph - pka if kind == "acid" else pka - ph
        # Evaluate the logistic on its bounded side, including extreme pKas.
        small = 10.0 ** -abs(exponent)
        population = 1.0 / (1.0 + small) if exponent >= 0 else small / (1.0 + small)
        coefficient = mult if side == "prod" else -mult
        dpka = coefficient * rt_ln10 * population * (1 if kind == "acid" else -1)
        contribution = coefficient * site_free_energy(pka, kind, ph, rt_ln10)
        if not all(math.isfinite(value) for value in (dpka, contribution)):
            raise ValueError("Protonation sensitivity outside finite numeric range")
        rows.append({"term_index": index, "side": side, "kind": kind,
                     "pKa": pka, "multiplicity": mult,
                     "ionized_site_fraction": population,
                     "contribution_kj": contribution,
                     "dG_d_pKa_kj_per_unit": dpka})
    return {"model": "independent_sites", "pH": ph, "terms": rows,
            "correction_kj": math.fsum(row["contribution_kj"] for row in rows),
            "dG_d_pH_kj_per_unit": -math.fsum(row["dG_d_pKa_kj_per_unit"] for row in rows),
            "scope": "Local sensitivity of the declared correction only; excludes QM, solvent, reservoir and missing-state errors. No pKa uncertainty or measured molecular populations inferred.",
            "reference_requirement": "Microscopic pKas must match the scored reference. Sequential macroscopic pKas require an explicit proton-count ensemble and cannot be substituted into the independent-site product."}


def macroscopic_transform(pkas, ph, rt_ln10, *, reference_deprotonations):
    """Sequential acid ladder relative to an explicitly selected proton-count macrostate.

    Weight of state n relative to the fully protonated state is
    10**(n*pH - sum(pKas[:n])). Macroscopic constants already include
    microstate multiplicity: do not add combinatorial degeneracies again.
    An intermediate reference denotes its entire proton-count ensemble, NOT a
    single calculated tautomer/geometry. The caller must establish that mapping
    and parameter provenance before using this in a molecular energy pipeline.
    See Ullmann, JPCB 107 (2003), doi:10.1021/jp026454v.
    """
    values = [_finite(p, "macroscopic pKa") for p in pkas]
    _finite(ph, "pH")
    if _finite(rt_ln10, "RT ln(10)") <= 0:
        raise ValueError("RT ln(10) must be positive")
    if (isinstance(reference_deprotonations, bool)
            or not isinstance(reference_deprotonations, int)
            or not 0 <= reference_deprotonations <= len(values)):
        raise ValueError("reference_deprotonations must identify a proton-count macrostate")
    weights = [0.0]
    for pka in values:
        weights.append(weights[-1] + ph - pka)
    pivot = max(weights)
    norm = math.fsum(10.0 ** (w - pivot) for w in weights)
    log_partition = pivot + math.log10(norm)
    populations = [10.0 ** (w - pivot) / norm for w in weights]
    return {
        "G_transform": -rt_ln10 * (log_partition - weights[reference_deprotonations]),
        "populations": populations,
        "dG_d_pKa": [rt_ln10 * (math.fsum(populations[index + 1:])
                                  - (1 if index < reference_deprotonations else 0))
                      for index in range(len(values))],
        "dG_d_pH": -rt_ln10 * (math.fsum(index * population for index, population in enumerate(populations))
                               - reference_deprotonations),
        "reference_deprotonations": reference_deprotonations,
        "reference_scope": "proton-count macrostate ensemble",
    }
