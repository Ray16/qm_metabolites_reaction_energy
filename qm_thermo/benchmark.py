"""Benchmark QM Delta_r G'^o against experiment and the existing GC methods.

Selects ModelSEED reactions fully covered by the computed compound set, evaluates
QM transformed reaction energies, joins ModelSEED-GCM / eQuilibrator / experimental
(openTECR) values, and reports error statistics + a parity plot.
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass

from . import config
from .reactions import ReactionEnergy, SpeciesInfo, reaction_dG, species_info
from .references import (
    ReferenceReaction,
    load_equilibrator_reaction_dG,
    reactions_within,
)
from .structures import Metabolite
from .thermo import CompoundEnergy


# Reactions touching these compounds live on the aqueous-electron / O2 redox
# reference that openTECR/eQuilibrator use, which is incompatible with absolute
# QM total energies: e.g. 2 H2 + O2 -> 2 H2O is genuinely ~-500 kJ/mol by QM (and
# by gas-phase thermochemistry) yet is reported near -130 on the biochemical redox
# scale. Including them both wrecks the benchmark and biases the proton-reference
# fit, so they are dropped from both unless explicitly kept. Fixing this properly
# needs a separate fitted aqueous-electron reference, analogous to the proton one.
REDOX_REFERENCE_COMPOUNDS = frozenset({
    "cpd00007",  # O2
    "cpd11640",  # H2
    "cpd00025",  # H2O2
})


def is_redox_reaction(ref: ReferenceReaction) -> bool:
    """True if the reaction touches an O2/H2/H2O2 redox-reference species."""
    return bool(ref.reaction.compounds() & REDOX_REFERENCE_COMPOUNDS)


@dataclass
class BenchmarkRow:
    rxn_id: str
    qm_dG_transformed_kJ: float
    mseed_dG_kJ: float | None
    eq_dG_kJ: float | None
    exp_dG_kJ: float | None
    exp_err_kJ: float | None


def build_benchmark(
    compound_energies: dict[str, CompoundEnergy],
    metabolites: list[Metabolite],
    *,
    conditions: config.Conditions = config.DEFAULT_CONDITIONS,
    use_highlevel: bool = True,
    exclude_redox: bool = True,
) -> list[BenchmarkRow]:
    """Compute QM reaction energies and join with reference values.

    `use_highlevel` selects the tier-2 (wB97M-V) ensemble G where available,
    falling back to tier-1 (r2SCAN-3c) per compound. `exclude_redox` drops
    O2/H2/H2O2 reactions whose redox reference is incompatible with the absolute
    QM energies (see REDOX_REFERENCE_COMPOUNDS).
    """
    gibbs = {
        cid: (ce.best_gibbs_kJ() if use_highlevel else ce.gibbs_kJ)
        for cid, ce in compound_energies.items()
    }
    species = {m.cpd_id: species_info(m) for m in metabolites}
    available = set(gibbs)

    refs = reactions_within(available)
    eq = load_equilibrator_reaction_dG()

    rows: list[BenchmarkRow] = []
    for rid, ref in sorted(refs.items()):
        if exclude_redox and is_redox_reaction(ref):
            continue
        try:
            energy = reaction_dG(ref.reaction, gibbs, species, conditions=conditions)
        except KeyError:
            continue  # missing a species energy despite filter (defensive)
        eq_rec = eq.get(rid, {})
        rows.append(
            BenchmarkRow(
                rxn_id=rid,
                qm_dG_transformed_kJ=energy.dG_transformed_kJ,
                mseed_dG_kJ=ref.mseed_dG_kJ,
                eq_dG_kJ=eq_rec.get("eq_dG_kJ"),
                exp_dG_kJ=eq_rec.get("exp_dG_kJ"),
                exp_err_kJ=eq_rec.get("exp_err_kJ"),
            )
        )
    return rows


def calibrate_proton_reference(
    compound_energies: dict[str, CompoundEnergy],
    metabolites: list[Metabolite],
    *,
    conditions: config.Conditions = config.DEFAULT_CONDITIONS,
    exclude_redox: bool = True,
) -> tuple[float, int]:
    """Least-squares fit of the aqueous-proton reference mu_H to openTECR.

    Delta_r G'^o(mu_H) = Delta_r G'^o(0) - mu_H * sum(nu_i * N_H,i). Regressing the
    residual (computed-at-mu_H=0 minus experimental) on the net bound-hydrogen
    change through the origin yields mu_H. Returns (mu_H_kJ, n_reactions_used).
    """
    import dataclasses

    base = dataclasses.replace(conditions, proton_reference_kJ=0.0)
    gibbs = {cid: ce.best_gibbs_kJ() for cid, ce in compound_energies.items()}
    species = {m.cpd_id: species_info(m) for m in metabolites}
    eq = load_equilibrator_reaction_dG()

    num = den = 0.0
    n = 0
    for rid, ref in reactions_within(set(gibbs)).items():
        if exclude_redox and is_redox_reaction(ref):
            continue
        exp = eq.get(rid, {}).get("exp_dG_kJ")
        if exp is None:
            continue
        try:
            dG0 = reaction_dG(ref.reaction, gibbs, species, conditions=base)
        except KeyError:
            continue
        net_h = sum(
            coeff * species[c].n_hydrogens
            for c, coeff in ref.reaction.stoichiometry.items()
        )
        if abs(net_h) < 1e-9:
            continue  # proton-neutral reactions carry no info about mu_H
        num += net_h * (dG0.dG_transformed_kJ - exp)
        den += net_h * net_h
        n += 1
    mu_h = num / den if den else conditions.proton_reference_kJ
    return mu_h, n


def error_stats(rows: list[BenchmarkRow]) -> dict[str, dict[str, float]]:
    """MAE/RMSE of each method vs the experimental column, where available."""
    import math

    def _stats(pairs: list[tuple[float, float]]) -> dict[str, float]:
        if not pairs:
            return {"n": 0, "MAE": float("nan"), "RMSE": float("nan")}
        errs = [pred - exp for pred, exp in pairs]
        mae = sum(abs(e) for e in errs) / len(errs)
        rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
        return {"n": len(errs), "MAE": mae, "RMSE": rmse}

    methods = {
        "QM": lambda r: r.qm_dG_transformed_kJ,
        "ModelSEED_GCM": lambda r: r.mseed_dG_kJ,
        "eQuilibrator": lambda r: r.eq_dG_kJ,
    }
    out: dict[str, dict[str, float]] = {}
    for name, getter in methods.items():
        pairs = [
            (getter(r), r.exp_dG_kJ)
            for r in rows
            if r.exp_dG_kJ is not None and getter(r) is not None
        ]
        out[name] = _stats(pairs)
    return out


def write_csv(rows: list[BenchmarkRow], path: str | None = None) -> str:
    path = path or os.path.join(config.BENCHMARK_DIR, "reaction_benchmark.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    return path


def plot_parity(rows: list[BenchmarkRow], path: str | None = None) -> str | None:
    """Parity plot of each method's Delta_r G'^o vs experiment."""
    pts = [r for r in rows if r.exp_dG_kJ is not None]
    if not pts:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = path or os.path.join(config.FIGURES_DIR, "qm_reaction_parity.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    exp = [r.exp_dG_kJ for r in pts]
    series = {
        "QM (this work)": [r.qm_dG_transformed_kJ for r in pts],
        "ModelSEED GCM": [r.mseed_dG_kJ for r in pts],
        "eQuilibrator": [r.eq_dG_kJ for r in pts],
    }
    for name, ys in series.items():
        xy = [(e, y) for e, y in zip(exp, ys) if y is not None]
        if xy:
            ax.scatter(*zip(*xy), s=20, alpha=0.7, label=name)
    lim = [min(exp) - 20, max(exp) + 20]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set_xlabel("Experimental Delta_r G'^o (openTECR), kJ/mol")
    ax.set_ylabel("Predicted Delta_r G'^o, kJ/mol")
    ax.set_title("QM vs group-contribution reaction energies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
