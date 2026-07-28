#!/usr/bin/env python
"""Demonstrate pH/speciation leverage on the ten-reaction benchmark.

This is a diagnostic, not a new reported model.  It compares the current fixed
ModelSEED microspecies with two already-computed structural alternatives:

* GSH thiol (instead of ModelSEED's thiolate); its AH/A- family is weighted by
  an *external* monoprotic pKa, not by QM relative energies.
* methylglyoxal gem-diol hydrate (a structural hydration correction, included
  to avoid attributing its large effect to proton speciation).

The default GSH pKa (8.9) is the literature value used by the prior diagnostic;
pass --pka to make a sensitivity calculation.  Do not use this script as a
general pKa predictor or a calibrated benchmark score.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
from rdkit import Chem

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)

from qm_thermo import config
from qm_thermo.composite import extract_ensemble_energy
from qm_thermo.reactions import Reaction, SpeciesInfo, reaction_dG
from qm_thermo.speciation import monoprotic_base_fraction, monoprotic_family_correction_kJ

BREAKDOWN = os.path.join(THERMO, "mlip", "G_aq_macepolar_deep.json")
MICRO = os.path.join(THERMO, "mlip", "G_aq_microspecies.json")
RXN_CSV = os.path.join(HERE, "top10_reactions_stereo_significant.csv")
WATER_M = 55.5


def n_hydrogens(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return sum(atom.GetTotalNumHs() + (atom.GetSymbol() == "H") for atom in mol.GetAtoms())


def load_compound_energies(path: str) -> dict[str, float]:
    breakdown = json.load(open(path))
    temperature = config.DEFAULT_CONDITIONS.temperature_K
    energies = {}
    for compound, record in breakdown.items():
        if "conformers" in record:
            assembled = extract_ensemble_energy(record, temperature_K=temperature).gibbs_kJ
            if abs(assembled - float(record["G_aq_kJ"])) > 1e-3:
                raise ValueError(f"{compound}: inconsistent conformer breakdown")
            energies[compound] = assembled
        else:
            energies[compound] = float(record["G_aq_kJ"])
    return energies


def score(reactions, G, S, conditions):
    return {reaction_id: reaction_dG(Reaction(reaction_id, stoich), G, S,
                                     conditions=conditions[reaction_id]).dG_transformed_kJ
            for reaction_id, stoich in reactions.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pka", type=float, default=8.9,
                    help="external GSH thiol pKa used for this sensitivity test")
    ap.add_argument("--breakdown", default=BREAKDOWN)
    ap.add_argument("--out", default=os.path.join(THERMO, "results", "benchmark",
                                                    "speciation_sensitivity.json"))
    args = ap.parse_args()

    G0 = load_compound_energies(args.breakdown)
    micro = json.load(open(MICRO))
    species_raw = json.load(open(os.path.join(HERE, "species.json")))
    S0 = {compound: SpeciesInfo(compound, int(value["n_hydrogens"]), int(value["charge"]))
          for compound, value in species_raw.items()}
    reactions = json.load(open(os.path.join(HERE, "reactions.json")))
    meta = {row["modelseed_rxn"]: row for row in csv.DictReader(open(RXN_CSV))}
    expected = {reaction_id: float(meta[reaction_id]["tecrdb_dG_kJ"])
                for reaction_id in reactions}
    C = config.DEFAULT_CONDITIONS

    def conditions_at(which: str):
        return {reaction_id: config.Conditions(
            pH=(7.0 if which == "pH7" else float(meta[reaction_id][f"pH_{which}"]))
        ) for reaction_id in reactions}

    midpoint = {reaction_id: config.Conditions(
        pH=(float(meta[reaction_id]["pH_min"]) + float(meta[reaction_id]["pH_max"])) / 2.0
    ) for reaction_id in reactions}
    base7 = score(reactions, G0, S0, conditions_at("pH7"))
    base_mid = score(reactions, G0, S0, midpoint)
    base_low = score(reactions, G0, S0, conditions_at("min"))
    base_high = score(reactions, G0, S0, conditions_at("max"))

    # Structural swap: this isolates the error caused by selecting a chemically
    # inappropriate pH-7 structure, independent of the pKa partition function.
    G_gsh, S_gsh = dict(G0), dict(S0)
    thiol = micro["cpd00042_thiol"]
    G_gsh["cpd00042"] = float(thiol["G_aq_kJ"])
    S_gsh["cpd00042"] = SpeciesInfo("cpd00042", n_hydrogens(thiol["smiles"]), -1)
    gsh_swap = score(reactions, G_gsh, S_gsh, midpoint)

    G_hydrate, S_hydrate = dict(G0), dict(S0)
    hydrate, water = micro["cpd00428_hydrate"], micro["h2o"]
    G_hydrate["cpd00428"] = float(hydrate["G_aq_kJ"]) - (
        float(water["G_aq_kJ"]) + C.R_kJ * C.temperature_K * math.log(WATER_M)
    )
    S_hydrate["cpd00428"] = SpeciesInfo("cpd00428", n_hydrogens(hydrate["smiles"]) - 2, 0)
    hydrate_swap = score(reactions, G_hydrate, S_hydrate, midpoint)
    G_swap, S_swap = dict(G_gsh), dict(S_gsh)
    G_swap["cpd00428"] = G_hydrate["cpd00428"]
    S_swap["cpd00428"] = S_hydrate["cpd00428"]
    structural_swap = score(reactions, G_swap, S_swap, midpoint)

    # pKa-anchored AH/A- family: apply the family partition to the *thiol*
    # reference.  This does not trust the raw thiolate-vs-thiol QM gap.
    pka_family = {}
    fractions = {}
    for reaction_id, condition in midpoint.items():
        G_family, S_family = dict(G_swap), dict(S_swap)
        correction = monoprotic_family_correction_kJ(
            condition.pH, args.pka, condition.temperature_K)
        G_family["cpd00042"] += correction
        fractions[reaction_id] = monoprotic_base_fraction(condition.pH, args.pka)
        pka_family[reaction_id] = score(
            {reaction_id: reactions[reaction_id]}, G_family, S_family,
            {reaction_id: condition})[reaction_id]

    models = {
        "fixed_pH7": base7,
        "fixed_pH_midpoint": base_mid,
        "fixed_pH_low": base_low,
        "fixed_pH_high": base_high,
        "GSH_thiol_structural_swap": gsh_swap,
        "methylglyoxal_hydrate_structural_swap": hydrate_swap,
        "curated_structural_swap": structural_swap,
        "GSH_pKa_anchored_family": pka_family,
    }
    print("pH/speciation sensitivity; all non-baseline columns are diagnostic only")
    print(f"GSH pKa = {args.pka:.2f}; pH-midpoint thiolate fraction is shown per reaction\n")
    print(f"{'rxn':10}{'pH range':>12}{'base(mid)':>11}{'range span':>12}"
          f"{'structure':>11}{'pKa family':>12}{'A- frac':>9}")
    for reaction_id in sorted(reactions, key=lambda r: int(meta[r]["rank"])):
        span = base_high[reaction_id] - base_low[reaction_id]
        print(f"{reaction_id:10}{meta[reaction_id]['pH_min']:>5}-{meta[reaction_id]['pH_max']:<5}"
              f"{base_mid[reaction_id]:11.1f}{span:12.1f}{structural_swap[reaction_id]:11.1f}"
              f"{pka_family[reaction_id]:12.1f}{fractions[reaction_id]:9.3f}")
    for name, values in models.items():
        mae = np.mean([abs(values[reaction_id] - expected[reaction_id]) for reaction_id in reactions])
        print(f"{name:28} MAE = {mae:5.1f} kJ/mol")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"gsh_pka": args.pka, "thiolate_fraction": fractions,
               "expected_kJ": expected, "models_kJ": models}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
