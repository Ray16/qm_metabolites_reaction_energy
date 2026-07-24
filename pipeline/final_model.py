#!/usr/bin/env python
"""The reported model, and the ladder of what each layer buys.

Reported by default:

  1. pH matching        evaluate at the pH each TECRDB value was measured at
                        rather than a blanket pH 7 (bookkeeping, free)
  2. deep ensemble      605 conformers instead of 126, RMSD+energy dedup
                        (--breakdown selects which ensemble to score)

OFF by default, kept only as a diagnostic (--anion-corr):

  3. anion correction   cumulative charge ladder calibrated on experimental pKa
  4. microspecies       GSH thiolate -> thiol; methylglyoxal -> its gem-diol

Layers 3-4 are NOT part of the reported model. Scored against a calibration
built with the same gas+solvation models, the correction makes the MAE WORSE
(38.3 -> 44.3 on the ten) while improving the signs (5/10 -> 9/10). That split
is why it is retained at all: the ladder captures the DIRECTION of the anion
error but not its magnitude, which is evidence about where the error lives. It
is not an accuracy improvement and must not be reported as one. An earlier
"33.0" came from applying a CPCM-X ladder to an ALPB composite; see FINDINGS.md.

Because that pairing bug is silent and severe, --cal is REQUIRED with
--anion-corr and has no default. The calibration must come from the same gas
model and the same solvation model as --breakdown.

Run:  /homes/rzhu/miniforge3/envs/palm/bin/python final_model.py [--breakdown PATH]
      ... --anion-corr --cal pka_cal_mp.json      # diagnostic only
"""
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
sys.path.insert(0, HERE)
from qm_thermo import config                                          # noqa: E402
from qm_thermo.reactions import Reaction, SpeciesInfo, reaction_dG    # noqa: E402

RXN_CSV = os.path.join(HERE, "top10_reactions_stereo_significant.csv")
MICRO = os.path.join(THERMO, "uma_workflow", "G_aq_microspecies.json")
WATER_M = 55.5

# Anionic site types for the pKa-calibrated solvation correction; matched in
# order, first match wins per atom, anything negative and untyped falls back to
# the set-wide anionic mean.
SMARTS = [
    ("phosphate", "[$([O-][PX4]),$([S-][PX4])]"),
    ("carboxyl",  "[$([O-]C=O)]"),
    ("phenol",    "[$([O-]c)]"),
    ("thiol",     "[$([S-][CX4]),$([S-]c)]"),
]
FALLBACK = "anionic_mean"


def site_counts(smiles):
    """Count anionic sites by group."""
    mol = Chem.MolFromSmiles(smiles)
    counts, claimed = {}, set()
    for group, sma in SMARTS:
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts(sma)):
            if match[0] in claimed:
                continue
            claimed.add(match[0])
            counts[group] = counts.get(group, 0) + 1
    n_neg = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() < 0)
    if n_neg > len(claimed):
        counts[FALLBACK] = n_neg - len(claimed)
    return counts


def n_h(smiles):
    return sum(a.GetTotalNumHs() + (1 if a.GetSymbol() == "H" else 0)
               for a in Chem.MolFromSmiles(smiles).GetAtoms())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--breakdown", default=os.path.join(
        THERMO, "uma_workflow", "G_aq_ensemble_fast.json"))
    ap.add_argument("--anion-corr", action="store_true",
                    help="add the anion-correction and microspecies layers. "
                         "DIAGNOSTIC ONLY -- worsens MAE, improves signs.")
    ap.add_argument("--cal", help="pKa calibration JSON. Required with "
                                  "--anion-corr; must come from the same gas "
                                  "and solvation models as --breakdown. No "
                                  "default: a silent mismatch voids the result.")
    args = ap.parse_args()
    if args.anion_corr and not args.cal:
        ap.error("--anion-corr requires --cal (e.g. --cal pka_cal_mp.json for an "
                 "ALPB/MACE-POLAR breakdown, pka_cal_mp_cpcmx.json for CPCM-X). "
                 "There is deliberately no default.")

    bd = json.load(open(args.breakdown))
    mic = json.load(open(MICRO))
    cal = json.load(open(args.cal)) if args.cal else None
    mets = {m["id"]: m for m in json.load(open(os.path.join(
        HERE, "metabolites.json")))}
    spec = json.load(open(os.path.join(HERE, "species.json")))
    reactions = json.load(open(os.path.join(HERE, "reactions.json")))
    meta = {r["modelseed_rxn"]: r for r in csv.DictReader(open(RXN_CSV))}
    exp = {r: float(m["tecrdb_dG_kJ"]) for r, m in meta.items() if r in reactions}

    C = config.DEFAULT_CONDITIONS
    RT = C.R_kJ * C.temperature_K

    def build_ladder(cal):
        """Cumulative charge ladder.

        Each pKa pair measures the error of adding ONE charge to a species that
        may already be charged, i.e. an increment at charge index k.
        Multiplying a mixed-k group mean by the number of sites (the earlier
        form) is not a valid extrapolation and over-corrects polyanions ~4x at
        q=-4. The correct aggregate is the running sum of increments up to the
        species' anionic-site count.
        """
        inc = {int(k): v for k, v in cal.get("increment_by_charge", {}).items()}
        if not inc:
            raise SystemExit("calibration has no increment ladder; "
                             "rerun analyze_pka.py")
        kmax = max(inc)
        ladder, run = {}, 0.0
        for k in range(1, kmax + 1):
            run += inc.get(k, inc[kmax])
            ladder[k] = run
        print(f"anion-solvation correction ladder from {args.cal} "
              f"(cumulative, kJ/mol):")
        for k in sorted(ladder):
            print(f"   {k} anionic site(s): {ladder[k]:+7.1f}")
        return ladder, kmax

    if cal is not None:
        corr_ladder, kmax = build_ladder(cal)

    def anion_corr(smiles):
        if cal is None:
            return 0.0
        n = sum(site_counts(smiles).values())
        if n == 0:
            return 0.0
        if n > kmax:            # beyond calibration: hold the last value, flag it
            print(f"   [extrapolation] {n} anionic sites > calibrated {kmax}; "
                  f"holding correction at the q=-{kmax} value")
            return corr_ladder[kmax]
        return corr_ladder[n]

    G0 = {c: r["G_aq_kJ"] for c, r in bd.items()}
    S0 = {c: SpeciesInfo(c, n_hydrogens=int(v["n_hydrogens"]), charge=int(v["charge"]))
          for c, v in spec.items()}
    cond7 = {r: C for r in reactions}
    condX = {r: config.Conditions(pH=(float(meta[r]["pH_min"]) +
                                      float(meta[r]["pH_max"])) / 2.0)
             for r in reactions}

    # ---- corrected species maps ----
    Gc = {c: g - anion_corr(mets[c]["smiles"]) for c, g in G0.items()}

    th = mic["cpd00042_thiol"]
    hyd, wat = mic["cpd00428_hydrate"], mic["h2o"]
    mu_w = wat["G_aq_kJ"] + RT * math.log(WATER_M)

    def with_microspecies(G, S, corrected):
        G, S = dict(G), dict(S)
        g_th = th["G_aq_kJ"] - (anion_corr(th["smiles"]) if corrected else 0.0)
        G["cpd00042"] = g_th
        S["cpd00042"] = SpeciesInfo("cpd00042", n_hydrogens=n_h(th["smiles"]),
                                    charge=-1)
        g_pool = hyd["G_aq_kJ"] - mu_w                       # neutral: no anion corr
        G["cpd00428"] = g_pool
        S["cpd00428"] = SpeciesInfo("cpd00428", n_hydrogens=n_h(hyd["smiles"]) - 2,
                                    charge=0)
        return G, S

    # The reported model stops at +pH match. The two further layers are only
    # built when --anion-corr is given, and are labelled as diagnostics so no
    # one lifts a number out of them by accident.
    ladder = [
        ("pH7 base",   G0, S0, cond7),
        ("+pH match",  G0, S0, condX),
    ]
    if cal is not None:
        Gm, Sm = with_microspecies(Gc, S0, True)
        ladder += [
            ("+anion cal [diag]", Gc, S0, condX),
            ("+species [diag]",   Gm, Sm, condX),
        ]
    res = {}
    for label, G, S, cd in ladder:
        res[label] = {r: reaction_dG(Reaction(r, st), G, S,
                                     conditions=cd[r]).dG_transformed_kJ
                      for r, st in reactions.items()}

    labels = [l for l, *_ in ladder]
    W = max(12, max(len(l) for l in labels) + 2)
    print(f"{'rxn':10}{'exp':>8}" + "".join(f"{l:>{W}}" for l in labels))
    for r in sorted(reactions, key=lambda r: -abs(res[labels[0]][r] - exp[r])):
        print(f"{r:10}{exp[r]:8.1f}" + "".join(f"{res[l][r]:{W}.1f}" for l in labels))
    print(f"\n{'MAE':10}{'':8}" + "".join(
        f"{np.mean([abs(res[l][r] - exp[r]) for r in reactions]):{W}.1f}"
        for l in labels))
    print(f"{'signs ok':10}{'':8}" + "".join(
        f"{sum(1 for r in reactions if res[l][r] * exp[r] > 0):{W}d}"
        for l in labels))
    print(f"\nreported model = '+pH match'."
          + (" Columns marked [diag] are NOT the reported model: the anion "
             "ladder\nworsens MAE and improves signs; see FINDINGS.md."
             if cal is not None else
             " Anion-correction layers omitted (pass --anion-corr --cal PATH)."))

    # ---- self-consistency: does the microspecies choice still matter? ----
    print("\nself-consistency (GSH thiolate vs thiol, redox reactions):")
    variants = [(False, G0)] + ([(True, Gc)] if cal is not None else [])
    for corrected, Gbase in variants:
        Ga, Sa = Gbase, S0
        Gb, Sb = with_microspecies(Gbase, S0, corrected)
        d = [abs(reaction_dG(Reaction(r, reactions[r]), Ga, Sa,
                             conditions=condX[r]).dG_transformed_kJ
                 - reaction_dG(Reaction(r, reactions[r]), Gb, Sb,
                               conditions=condX[r]).dG_transformed_kJ)
             for r in ("rxn00086", "rxn00070")]
        tag = "with anion correction " if corrected else "without correction    "
        print(f"   {tag} |thiolate - thiol| = {np.mean(d):5.1f} kJ/mol")

    out = os.path.join(HERE, "final_model_out.json")
    json.dump({l: res[l] for l in labels}, open(out, "w"), indent=2)

    csv_out = os.path.join(HERE, "perreaction_dG.csv")
    with open(csv_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "rxn", "cls", "name", "exp", "dGP", *labels])
        cls_by_rank = {1: "redox", 2: "redox", 3: "redox", 4: "redox",
                       5: "glycosyl", 6: "glycosyl", 7: "glyoxalase",
                       8: "glycosyl", 9: "nucleotidyl", 10: "nucleotidyl"}
        for r in sorted(reactions, key=lambda r: int(meta[r]["rank"])):
            w.writerow([meta[r]["rank"], r, cls_by_rank[int(meta[r]["rank"])],
                        meta[r]["name"], f"{exp[r]:.1f}",
                        f"{float(meta[r]['dGpredictor_modelseed_dG_kJ']):.1f}",
                        *[f"{res[l][r]:.1f}" for l in labels]])
    print(f"\nwrote {out}\nwrote {csv_out}")


if __name__ == "__main__":
    main()
