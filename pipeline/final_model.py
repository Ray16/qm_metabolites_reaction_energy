#!/usr/bin/env python
"""Score the reported QM/ML composite.

The production model uses a deep conformer ensemble, a pH-7 fixed-microspecies
baseline, and no fitted anion-solvation correction. A pH-midpoint column is a
fixed-microspecies sensitivity diagnostic, not a complete speciation model. The rejected
empirical calibration is archived outside this repository's active workflow.

Run: python final_model.py [--breakdown PATH] [--pH-mode fixed|families]
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
from qm_thermo.speciation import families_from_dict
from qm_thermo.reaction_correction import canonical_signature

RXN_CSV = os.path.join(HERE, "top10_reactions_stereo_significant.csv")
FAMILIES_JSON = os.path.join(HERE, "microspecies_families.json")
MICROSPECIES_G_JSON = os.path.join(THERMO, "mlip", "G_aq_microspecies.json")


def score_with_families(reactions, G, S, conditions, families):
    """Apply ν[-RT ln(Z/w_ref)] after transforming the calculated reference state."""
    values, provenance = {}, {}
    for reaction_id, stoich in reactions.items():
        condition = conditions[reaction_id]
        value = reaction_dG(Reaction(reaction_id, stoich), G, S,
                            conditions=condition).dG_transformed_kJ
        applied = []
        for compound, coeff in stoich.items():
            family = families.get(compound)
            if family is None:
                continue
            correction = family.correction_from_reference_kJ(condition.pH, condition.temperature_K)
            value += coeff * correction
            applied.append({"compound": compound, "coefficient": coeff,
                            "correction_kJ": correction, "fractions": family.fractions(condition.pH),
                            "source": family.source, "citation": family.citation})
        values[reaction_id], provenance[reaction_id] = value, applied
    return values, provenance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--breakdown", default=os.path.join(
        THERMO, "mlip", "G_aq_macepolar_deep.json"))
    ap.add_argument("--pH-mode", choices=("fixed", "families"), default="fixed",
                    help="fixed is the reported baseline; families applies curated pKa ensembles")
    ap.add_argument("--families", default=FAMILIES_JSON,
                    help="curated microspecies-family metadata")
    ap.add_argument("--write-calibration-input",
                    help="write labelled reaction rows for the separate class-calibration CLI")
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
    models = [("pH7 fixed species", cond7), ("fixed-species pH midpoint [diag]", condX)]
    res = {label: {r: reaction_dG(Reaction(r, st), G, S,
                                   conditions=conditions[r]).dG_transformed_kJ
                   for r, st in reactions.items()}
           for label, conditions in models}
    family_provenance = {}
    if args.pH_mode == "families":
        label = "pH-midpoint microspecies families [curated]"
        raw_families = json.load(open(args.families))
        families = families_from_dict(raw_families)
        # A family is anchored to an explicitly calculated reference structure
        # when available.  We therefore transform the thiol QM microspecies,
        # for example, rather than pretending the stored thiolate remains a
        # valid structure at every pH.  Missing records are fatal: silently
        # falling back would make the provenance claim false.
        G_family, S_family = dict(G), dict(S)
        micro_energies = json.load(open(MICROSPECIES_G_JSON))
        for compound, record in raw_families.items():
            energy_record = record.get("reference_energy_record")
            if energy_record is None:
                continue
            if energy_record not in micro_energies:
                raise KeyError(f"{compound}: missing microspecies energy {energy_record}")
            if "reference_charge" not in record or "reference_n_hydrogens" not in record:
                raise ValueError(f"{compound}: reference structure requires charge and hydrogen count")
            G_family[compound] = float(micro_energies[energy_record]["G_aq_kJ"])
            S_family[compound] = SpeciesInfo(compound, int(record["reference_n_hydrogens"]),
                                              int(record["reference_charge"]))
        res[label], family_provenance = score_with_families(
            reactions, G_family, S_family, condX, families)
        models.append((label, condX))

    labels = [label for label, _ in models]
    width = max(14, max(len(label) for label in labels) + 2)
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
    print("\nreported baseline = 'pH7 fixed species'. The pH-midpoint column holds "
          "microspecies fixed and is diagnostic only.")
    if args.pH_mode == "families":
        covered = sorted({x["compound"] for rows in family_provenance.values() for x in rows})
        print("curated family mode is separate from the baseline; pKa coverage: " +
              (", ".join(covered) if covered else "none"))

    out_dir = os.path.join(THERMO, "results", "benchmark")
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"models_kJ": res, "pH_mode": args.pH_mode,
               "family_provenance": family_provenance},
              open(os.path.join(out_dir, "final_model_out.json"), "w"), indent=2)
    with open(os.path.join(out_dir, "perreaction_dG.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "rxn", "name", "exp", "dGP", *labels])
        for r in sorted(reactions, key=lambda r: int(meta[r]["rank"])):
            w.writerow([meta[r]["rank"], r, meta[r]["name"], f"{exp[r]:.1f}",
                        f"{float(meta[r]['dGpredictor_modelseed_dG_kJ']):.1f}",
                        *[f"{res[label][r]:.1f}" for label in labels]])
    if args.write_calibration_input:
        classes = json.load(open(os.path.join(HERE, "reaction_classes.json")))
        # Select the last emitted prediction: pH-family mode, when requested,
        # otherwise the fixed-species diagnostic.  This writes data only; the
        # correction remains a separate explicit operation.
        selected = labels[-1]
        with open(args.write_calibration_input, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=("reaction_id", "signature", "reaction_class",
                                                     "predicted_kJ", "experimental_kJ"))
            writer.writeheader()
            for reaction_id, stoich in reactions.items():
                writer.writerow({"reaction_id": reaction_id,
                                 "signature": canonical_signature(stoich),
                                 "reaction_class": classes.get(reaction_id, "other"),
                                 "predicted_kJ": res[selected][reaction_id],
                                 "experimental_kJ": exp[reaction_id]})


if __name__ == "__main__":
    main()
