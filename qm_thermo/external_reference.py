"""Parameter-free external-reference corrections for the QM composite.

This layer never fits to the benchmark reactions.  Each correction cancels a
badly-solvated shared moiety (aqueous continuum solvation of multiply-charged
anions is the dominant composite error) against an *external* experimental
anchor -- a reaction that is not in the benchmark set, or a tabulated standard
reduction potential.  It is therefore a separately labelled hybrid model, kept
out of :mod:`qm_thermo.reactions` (the uncalibrated QM baseline) exactly like
:mod:`qm_thermo.reaction_correction`.

Three correction kinds are supported (see pipeline/reference_reactions.json):

* ``isodesmic`` -- for a target reaction sharing a bond-change class with an
  externally measured reference, report
  ``dG(target) = dG_QM(target) - dG_QM(reference) + dG_exp(reference)``.
  The shared species (e.g. pyrophosphate) cancel exactly; the residual must be
  a charge-balanced, low-magnitude swap for the anion-solvation error to cancel
  (this is checked and the residual charge imbalance is reported for audit).

* ``redox_couple_equalization`` -- two redox reactions that differ only by an
  electronically-spectator group (NAD vs NADP: a distal 2'-phosphate) must have
  reduction free energies equal to within their tabulated E0' difference.  The
  QM value of the accurate counterpart is transferred to the target with the
  small experimental E0' offset.  Uses only the counterpart's QM value and
  external E0' tables -- never the target's experimental value.

* ``speciation_substitution`` -- replace a compound's stored microspecies with a
  curated externally-justified structure (e.g. the methylglyoxal gem-diol
  hydrate), balancing any change in molecular formula with explicit water.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import config
from .reactions import Reaction, SpeciesInfo, reaction_dG

WATER_KEY = "h2o"


@dataclass
class CorrectionResult:
    """A single applied correction and everything needed to audit it."""

    reaction_id: str
    kind: str
    baseline_kJ: float
    corrected_kJ: float
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def shift_kJ(self) -> float:
        return self.corrected_kJ - self.baseline_kJ


def _dG(stoich, G, S, conditions):
    return reaction_dG(Reaction("_ref", stoich), G, S, conditions=conditions).dG_transformed_kJ


def _residual_charge(stoich_target, stoich_ref, S) -> int:
    """Net charge of (target - reference); nonzero means anion error will not cancel."""
    net: dict[str, float] = {}
    for c, v in stoich_target.items():
        net[c] = net.get(c, 0.0) + v
    for c, v in stoich_ref.items():
        net[c] = net.get(c, 0.0) - v
    return round(sum(coeff * S[c].charge for c, coeff in net.items() if c in S))


def apply_isodesmic(spec, reactions, G, S, conditions, *, baseline):
    """External same-bond-change reference; returns {rxn_id: CorrectionResult}."""
    out = {}
    ref_stoich = {c: float(v) for c, v in spec["reference_stoichiometry"].items()}
    exp_ref = float(spec["experimental_dG_kJ"])
    qm_ref = _dG(ref_stoich, G, S, conditions)
    reference_residual = exp_ref - qm_ref            # per-class additive shift
    for rid in spec["applies_to"]:
        if rid not in reactions:
            continue
        base = baseline[rid]
        imbalance = _residual_charge(reactions[rid], ref_stoich, S)
        out[rid] = CorrectionResult(
            reaction_id=rid, kind="isodesmic", baseline_kJ=base,
            corrected_kJ=base + reference_residual,
            provenance={
                "reference_name": spec["reference_name"],
                "experimental_dG_kJ": exp_ref,
                "qm_reference_dG_kJ": qm_ref,
                "reference_residual_kJ": reference_residual,
                "residual_reaction_net_charge": imbalance,
                "citation": spec["citation"],
                "note": spec.get("experimental_note", ""),
            },
        )
    return out


def apply_redox_equalization(spec, reactions, *, baseline):
    """Transfer the accurate counterpart's QM value with the external E0' offset."""
    target = spec["target"]
    counterpart = spec["reference_counterpart"]
    if target not in reactions or counterpart not in baseline:
        return {}
    # dG_offset(target - reference couple) = -n F (E0_target - E0_reference)
    n = 2
    offset = -n * config.DEFAULT_CONDITIONS.F_kJ_per_V * (
        float(spec["E0_target_V"]) - float(spec["E0_reference_V"]))
    offset *= float(spec["direction_sign"])
    corrected = baseline[counterpart] + offset
    return {target: CorrectionResult(
        reaction_id=target, kind="redox_couple_equalization",
        baseline_kJ=baseline[target], corrected_kJ=corrected,
        provenance={
            "couple_target": spec["couple_target"],
            "couple_reference": spec["couple_reference"],
            "counterpart_reaction": counterpart,
            "counterpart_qm_dG_kJ": baseline[counterpart],
            "E0_offset_kJ": offset,
            "citation": spec["citation"],
            "note": spec.get("note", ""),
        },
    )}


def apply_speciation_substitution(spec, reactions, G, S, conditions, micro_energies, *, baseline):
    """Swap one compound to a curated microspecies, balancing formula with water."""
    rid = spec["reaction"]
    if rid not in reactions:
        return {}
    compound = spec["compound"]
    record = micro_energies[spec["microspecies_record"]]
    G2, S2 = dict(G), dict(S)
    G2[compound] = float(record["G_aq_kJ"])
    S2[compound] = SpeciesInfo(compound, int(spec["n_hydrogens"]), int(spec["charge"]))
    stoich = dict(reactions[rid])
    water_balance = float(spec.get("water_balance", 0))
    if water_balance:
        if WATER_KEY not in micro_energies:
            raise KeyError("speciation_substitution needs a water record 'h2o' in microspecies energies")
        water = micro_energies[WATER_KEY]
        G2[WATER_KEY] = float(water["G_aq_kJ"])
        S2[WATER_KEY] = SpeciesInfo(WATER_KEY, 2, 0)
        stoich[WATER_KEY] = stoich.get(WATER_KEY, 0.0) + water_balance
    corrected = _dG(stoich, G2, S2, conditions)
    return {rid: CorrectionResult(
        reaction_id=rid, kind="speciation_substitution",
        baseline_kJ=baseline[rid], corrected_kJ=corrected,
        provenance={
            "compound": compound,
            "microspecies_record": spec["microspecies_record"],
            "water_balance": water_balance,
            "citation": spec["citation"],
            "note": spec.get("note", ""),
        },
    )}


def apply_all(references, reactions, G, S, conditions, micro_energies, *, baseline):
    """Apply every configured correction; return corrected values + provenance.

    Each reaction is corrected by at most one kind (they target disjoint
    reactions here).  Reactions with no configured correction keep the baseline.
    """
    results: dict[str, CorrectionResult] = {}
    for spec in references.get("isodesmic", []):
        results.update(apply_isodesmic(spec, reactions, G, S, conditions, baseline=baseline))
    for spec in references.get("redox_couple_equalization", []):
        results.update(apply_redox_equalization(spec, reactions, baseline=baseline))
    for spec in references.get("speciation_substitution", []):
        results.update(apply_speciation_substitution(
            spec, reactions, G, S, conditions, micro_energies, baseline=baseline))

    overlap = set()
    for spec in references.get("isodesmic", []):
        overlap |= set(spec.get("applies_to", []))
    values = dict(baseline)
    provenance = {}
    for rid, res in results.items():
        values[rid] = res.corrected_kJ
        provenance[rid] = {"kind": res.kind, "baseline_kJ": res.baseline_kJ,
                           "corrected_kJ": res.corrected_kJ, "shift_kJ": res.shift_kJ,
                           **res.provenance}
    return values, provenance
