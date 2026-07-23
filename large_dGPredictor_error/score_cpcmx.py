#!/usr/bin/env python
"""Score the CPCM-X composite against the ALPB one, per solute class.

recompute_dgsolv_cpcmx.py produced ensemble_deep_cpcmx.json and the matching
G_aq_macepolar_deep_cpcmx.json months ago, but the two solvation models were
never compared on the calibration set, so "CPCM-X is better for ions" was an
assumption carried from the literature rather than a measurement here.

The comparison has to be made on the pKa set, not on the reactions: the pKa
pairs measure ONE deprotonation each, so an error there is attributable to a
single solute, whereas a reaction error is a sum over four species and can
cancel. The reaction MAEs are printed too, but they are the weaker evidence.

Prerequisite: pka_cal_mp.json and pka_cal_mp_cpcmx.json, from

  PKA_G_JSON=../uma_workflow/G_aq_pka_mp.json       PKA_OUT=pka_cal_mp.json       python analyze_pka.py
  PKA_G_JSON=../uma_workflow/G_aq_pka_mp_cpcmx.json PKA_OUT=pka_cal_mp_cpcmx.json python analyze_pka.py

Run:  /homes/rzhu/miniforge3/envs/palm/bin/python score_cpcmx.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    return {r["key"]: r for r in json.load(open(os.path.join(HERE, name)))["rows"]}


def main():
    alpb, cpcmx = load("pka_cal_mp.json"), load("pka_cal_mp_cpcmx.json")
    shared = [k for k in alpb if k in cpcmx]

    groups = {}
    for k in shared:
        groups.setdefault((alpb[k]["group"], alpb[k]["kind"]), []).append(k)

    print("per-species free-energy error from experimental pKa (kJ/mol)\n")
    print(f"{'group':11}{'kind':10}{'n':>3}{'ALPB mean':>11}{'ALPB MAE':>10}"
          f"{'CPCMX mean':>12}{'CPCMX MAE':>11}")
    for (g, kind), ks in sorted(groups.items()):
        ea = [alpb[k]["err_kJ"] for k in ks]
        ec = [cpcmx[k]["err_kJ"] for k in ks]
        print(f"{g:11}{kind:10}{len(ks):3d}{np.mean(ea):11.1f}"
              f"{np.mean(np.abs(ea)):10.1f}{np.mean(ec):12.1f}"
              f"{np.mean(np.abs(ec)):11.1f}")

    for kind in ("anionic", "cationic"):
        ks = [k for k in shared if alpb[k]["kind"] == kind]
        ea = [alpb[k]["err_kJ"] for k in ks]
        ec = [cpcmx[k]["err_kJ"] for k in ks]
        print(f"{'ALL':11}{kind:10}{len(ks):3d}{np.mean(ea):11.1f}"
              f"{np.mean(np.abs(ea)):10.1f}{np.mean(ec):12.1f}"
              f"{np.mean(np.abs(ec)):11.1f}    sd {np.std(ea):5.1f} ->"
              f" {np.std(ec):5.1f}")

    # The cationic family is the control: BH+ -> B creates no anion, so a shift
    # there is a baseline/proton-reference artefact, not ion solvation.
    da = np.mean([alpb[k]["err_kJ"] for k in shared if alpb[k]["kind"] == "anionic"]) \
        - np.mean([alpb[k]["err_kJ"] for k in shared if alpb[k]["kind"] == "cationic"])
    dc = np.mean([cpcmx[k]["err_kJ"] for k in shared if alpb[k]["kind"] == "anionic"]) \
        - np.mean([cpcmx[k]["err_kJ"] for k in shared if alpb[k]["kind"] == "cationic"])
    print(f"\nanion-specific part (anionic - cationic): "
          f"ALPB {da:+.1f}   CPCM-X {dc:+.1f} kJ/mol")
    print("scatter, not bias, is what propagates to reactions: a uniform shift "
          "cancels in\na balanced reaction, an sd of 71.8 does not.")


if __name__ == "__main__":
    main()
