#!/usr/bin/env python
"""Score the reported QM/ML composite.

The production model uses a deep conformer ensemble, measured-pH Alberty
transformation, and no fitted anion-solvation correction.  The rejected
empirical calibration is archived outside this repository's active workflow.

Run: python final_model.py [--breakdown PATH]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)
from qm_thermo import config
from qm_thermo.composite import extract_ensemble_energy
from qm_thermo.reactions import Reaction, SpeciesInfo, reaction_dG

RXN_CSV = os.path.join(HERE, "top10_reactions_stereo_significant.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--breakdown", default=os.path.join(
        THERMO, "mlip", "G_aq_macepolar_deep.json"))
    args = ap.parse_args()

    bd = json.load(open(args.breakdown))
    spec = json.load(open(os.path.join(HERE, "species.json")))
    reactions = json.load(open(os.path.join(HERE, "reactions.json")))
    meta = {r["modelseed_rxn"]: r for r in csv.DictReader(open(RXN_CSV))}
    exp = {r: float(m["tecrdb_dG_kJ"]) for r, m in meta.items() if r in reactions}

    # Reassemble from per-conformer terms when available.  This gives every
    # electronic model (UMA, MACE-POLAR, or a future substitute) one common
    # scoring path and catches corrupted/incompatible output early.
    G = {}
    for compound, record in bd.items():
        if "conformers" in record:
            assembled = extract_ensemble_energy(
                record, temperature_K=config.DEFAULT_CONDITIONS.temperature_K)
            persisted = float(record["G_aq_kJ"])
            if abs(assembled.gibbs_kJ - persisted) > 1e-3:
                raise ValueError(
                    f"{compound}: persisted G_aq_kJ disagrees with conformer terms "
                    f"({persisted:.6f} vs {assembled.gibbs_kJ:.6f} kJ/mol)"
                )
            G[compound] = assembled.gibbs_kJ
        else:
            G[compound] = float(record["G_aq_kJ"])
    S = {c: SpeciesInfo(c, n_hydrogens=int(v["n_hydrogens"]), charge=int(v["charge"]))
         for c, v in spec.items()}
    C = config.DEFAULT_CONDITIONS
    cond7 = {r: C for r in reactions}
    condX = {r: config.Conditions(pH=(float(meta[r]["pH_min"]) +
                                      float(meta[r]["pH_max"])) / 2.0)
             for r in reactions}
    models = [("pH7 base", cond7), ("+pH match", condX)]
    res = {label: {r: reaction_dG(Reaction(r, st), G, S,
                                   conditions=conditions[r]).dG_transformed_kJ
                   for r, st in reactions.items()}
           for label, conditions in models}

    labels = [label for label, _ in models]
    width = 14
    print(f"{'rxn':10}{'exp':>8}" + "".join(f"{label:>{width}}" for label in labels))
    for r in sorted(reactions, key=lambda r: -abs(res[labels[0]][r] - exp[r])):
        print(f"{r:10}{exp[r]:8.1f}" + "".join(f"{res[label][r]:{width}.1f}"
                                                  for label in labels))
    print(f"\n{'MAE':10}{'':8}" + "".join(
        f"{np.mean([abs(res[label][r] - exp[r]) for r in reactions]):{width}.1f}"
        for label in labels))
    print(f"{'signs ok':10}{'':8}" + "".join(
        f"{sum(res[label][r] * exp[r] > 0 for r in reactions):{width}d}"
        for label in labels))
    print("\nreported model = '+pH match'; no empirical anion-solvation correction is applied.")

    out_dir = os.path.join(THERMO, "results", "benchmark")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "final_model_out.json"), "w"), indent=2)
    with open(os.path.join(out_dir, "perreaction_dG.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "rxn", "name", "exp", "dGP", *labels])
        for r in sorted(reactions, key=lambda r: int(meta[r]["rank"])):
            w.writerow([meta[r]["rank"], r, meta[r]["name"], f"{exp[r]:.1f}",
                        f"{float(meta[r]['dGpredictor_modelseed_dG_kJ']):.1f}",
                        *[f"{res[label][r]:.1f}" for label in labels]])


if __name__ == "__main__":
    main()
