#!/usr/bin/env python
"""Does Mg2+ speciation bookkeeping reduce the QM error on Mg-buffered reactions?

RESULT: no. DIAGNOSTIC ONLY -- this term is NEVER added to the reported composite
(final_model.py). On the 14 Mg-buffered reactions the net per-reaction correction
is +0.3 kJ/mol mean (range -4.2 .. +3.2) and MAE moves 35.5 -> 36.4. Mg binds
phosphates on BOTH sides of these transfer reactions, so it cancels to <=4 kJ/mol
-- far below the ~35 kJ/mol continuum-solvation error. The "pMg-reported
reactions have higher error" pattern is confounding, not causal: TECRDB buffers
Mg precisely for the polyphosphate reactions where the anion-solvation error is
worst. See FINDINGS.md, "Unmodelled experimental variables".

The composite computes Delta_r G'^o at the measured pH and ionic strength but
with NO magnesium, i.e. at pMg = infinity. Many TECRDB nucleotide/phosphate
measurements are made in an Mg buffer (pMg 2-4, i.e. [Mg2+] ~ 0.1-4 mM), where
Mg2+ binds the phosphate groups and shifts the *measured* Delta_r G'^o by
several kJ/mol. Comparing our pMg=inf number against an Mg-buffered measurement
charges that shift to QM. This script adds it back and re-scores.

Mg binding enters exactly like the proton in the Alberty transform: a reactant
that binds Mg is stabilised by -RT ln(1 + K_Mg [Mg2+]) (one dominant 1:1 site),
so the reaction picks up

    dG_mg = sum_i  nu_i * ( -RT ln(1 + K_i [Mg2+]) )                     (kJ/mol)

added to the pMg=inf value to bring it to the reported pMg. Reactions with no
reported Mg (pMg >= 13.8) get dG_mg = 0 and are untouched.

Association constants log10 K_Mg (MgL, 25 C, I ~ 0.1-0.25 M) are from Alberty,
"Thermodynamics of Biochemical Reactions" (2003), Tables; NIST Critically
Selected Stability Constants. They carry ~0.3-0.5 log-unit (2-3 kJ/mol)
uncertainty each -- the correction is only as good as these, which is the point:
it tests whether the *bookkeeping term* explains the Mg-subset excess error.

Run:  python mg_speciation.py
"""
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RT = 8.314462618e-3 * 298.15   # kJ/mol
LN10 = math.log(10)
NO_MG_PMG = 13.8               # pMg at/above this = no added Mg (correction -> 0)

# log10 K for Mg2+ binding the DOMINANT microspecies at pH 7, by ModelSEED id.
# Grouped by structure: nucleoside triphosphate ~4.0, di-/diphosphate ~3.0-3.3,
# PRPP ~4.2, mono-/sugar-phosphate and Pi ~1.6-1.9, CoA ~1.5. Non-phosphate = 0.
LOGK_MG = {
    # nucleoside triphosphates (P-O-P-O-P), MgNTP2-
    "cpd00002": 4.0,   # ATP
    "cpd00052": 4.0,   # CTP
    "cpd00062": 4.0,   # UTP
    # nucleoside diphosphates / diphosphate diesters (P-O-P), MgNDP-
    "cpd00008": 3.0,   # ADP
    "cpd00026": 3.0,   # UDP-glucose
    "cpd00387": 3.0,   # ADP-ribose / GDP-sugar
    "cpd00256": 3.0,   # CDP-choline
    # free pyrophosphate (HP2O7^3-), MgHP2O7-
    "cpd00012": 3.3,   # PPi
    # 5-phospho-alpha-D-ribose 1-diphosphate (mono + diphosphate), strong
    "cpd00103": 4.2,   # PRPP
    # CoA / acetyl-CoA (buried diphosphate diester + 3'-phosphate), weak
    "cpd00010": 1.5,   # CoA
    "cpd00022": 1.5,   # acetyl-CoA
    # inorganic phosphate (HPO4^2-)
    "cpd00009": 1.9,   # Pi
    # nucleoside monophosphates and sugar / glyceryl monophosphates
    "cpd00018": 1.6,   # AMP
    "cpd00126": 1.6,   # GMP
    "cpd00072": 1.6,   # fructose-6-P
    "cpd00089": 1.6,   # glucose-1-P / hexose-1-P
    "cpd00101": 1.6,   # ribose-5-P
    "cpd00102": 1.6,   # glyceraldehyde-3-P / DHAP
    "cpd00169": 1.6,   # 3-phosphoglycerate
    "cpd00198": 1.6,   # sugar phosphate
    "cpd00236": 1.6,   # erythrose-4-P / sugar phosphate
    "cpd00238": 1.6,   # sedoheptulose-7-P
    "cpd00475": 1.6,   # aldose-1-phosphate
    "cpd00457": 1.0,   # phosphocholine (zwitterion, weak)
}


def species_mg_shift(cpd_id, mg_M):
    """-RT ln(1 + K[Mg2+]) for one species; 0 for non-binders."""
    k = LOGK_MG.get(cpd_id)
    if not k or mg_M <= 0:
        return 0.0
    return -RT * math.log1p((10.0 ** k) * mg_M)


def reaction_mg_correction(stoich, mg_M):
    """dG_mg = sum nu_i * (-RT ln(1 + K_i[Mg2+])), same sign convention as the
    scored reactions (products +, reactants -)."""
    return sum(coeff * species_mg_shift(cpd, mg_M) for cpd, coeff in stoich.items())


def main():
    scored = {s["r"]: s for s in json.load(
        open(os.path.join(HERE, "bench226_scored.json")))}
    meta = {r["modelseed_rxn"]: r for r in csv.DictReader(
        open(os.path.join(HERE, "bench226_meta.csv")))}
    rxns = json.load(open(os.path.join(HERE, "bench226_reactions.json")))

    # median reported pMg per TECRDB reaction string (added-Mg measurements only)
    tec = defaultdict(list)
    for t in csv.DictReader(open(os.path.join(HERE, "TECRDB.csv"))):
        try:
            pmg = float(t["p_mg"])
        except (ValueError, KeyError):
            continue
        if pmg < NO_MG_PMG:
            tec[t["reaction"]].append(pmg)

    rows = []
    for r, s in scored.items():
        m = meta.get(r)
        if not m:
            continue
        pmgs = tec.get(m.get("tecrdb_reaction", ""), [])
        if not pmgs:
            continue                       # no added Mg -> correction is 0
        pmg = float(np.median(pmgs))
        mg_M = 10.0 ** (-pmg)
        dg_mg = reaction_mg_correction(rxns.get(r, {}), mg_M)
        e, q = s["e"], s["q"]
        rows.append(dict(r=r, name=m["name"][:32], pmg=pmg, dg_mg=dg_mg,
                         e=e, q=q, err0=q - e, err1=(q + dg_mg) - e))

    rows.sort(key=lambda x: -abs(x["err0"]))
    print(f"{'rxn':10}{'pMg':>5}{'dG_mg':>8}{'exp':>8}{'QM':>8}"
          f"{'err(no Mg)':>12}{'err(+Mg)':>10}  name")
    for x in rows:
        print(f"{x['r']:10}{x['pmg']:5.1f}{x['dg_mg']:+8.1f}{x['e']:8.1f}"
              f"{x['q']:8.1f}{x['err0']:+12.1f}{x['err1']:+10.1f}  {x['name']}")

    e0 = np.mean([abs(x["err0"]) for x in rows])
    e1 = np.mean([abs(x["err1"]) for x in rows])
    print(f"\nMg subset (n={len(rows)}):  MAE without Mg = {e0:5.1f}   "
          f"with Mg = {e1:5.1f}   ({'-' if e1 < e0 else '+'}{abs(e1-e0):.1f} kJ/mol)")
    signed = np.mean([x["dg_mg"] for x in rows])
    print(f"mean Mg correction = {signed:+.1f} kJ/mol "
          f"(range {min(x['dg_mg'] for x in rows):+.1f} .. "
          f"{max(x['dg_mg'] for x in rows):+.1f})")


if __name__ == "__main__":
    main()
