#!/usr/bin/env python
"""Measure the pipeline's ion-solvation error against experimental pKa.

For AH -> A(-) + H+ the pipeline predicts
    dG_deprot = G_aq(A) + mu_H - G_aq(AH)      ->   pKa_calc = dG_deprot / RT ln10
so the residual (pKa_calc - pKa_exp) * RT ln10 is the pipeline's free-energy
error on that one acid/base pair, with no fitting involved.

Splitting the set into anionic (AH -> A-) and cationic (BH+ -> B) acids separates
the two candidate causes: a wrong proton reference shifts BOTH families equally,
whereas an anion-solvation error shifts only the anionic ones. That distinction
is not recoverable from the metabolite reactions alone.

Run:  /homes/rzhu/miniforge3/envs/palm/bin/python analyze_pka.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)
from qm_thermo import config    # noqa: E402

# The calibration is only valid for the SAME gas model and SAME solvation model
# as the reactions it will be applied to -- an ALPB ladder applied to a CPCM-X
# composite double-counts the anion error, and did, silently, for months. So the
# two paths are overridden together and the output is NAMED BY MODEL. Never
# write an unlabelled "pka_calibration.json" again: the filename is the only
# thing standing between a future run and that bug.
#
#   G_aq_pka.json        -> pka_cal_uma.json        UMA        + ALPB
#   G_aq_pka_mp.json     -> pka_cal_mp.json         MACE-POLAR + ALPB
#   G_aq_pka_mp_cpcmx.json -> pka_cal_mp_cpcmx.json MACE-POLAR + CPCM-X
G_JSON = os.environ.get("PKA_G_JSON",
                        os.path.join(THERMO, "uma_workflow", "G_aq_pka.json"))
PAIRS = os.path.join(HERE, "pka_pairs.json")
OUT = os.environ.get("PKA_OUT", os.path.join(HERE, "pka_cal_uma.json"))
if bool(os.environ.get("PKA_G_JSON")) != bool(os.environ.get("PKA_OUT")):
    raise SystemExit("set PKA_G_JSON and PKA_OUT together, or neither: an "
                     "output name that does not match its input model is how "
                     "the mismatched-calibration bug happened.")


def main():
    G = {c: r["G_aq_kJ"] for c, r in json.load(open(G_JSON)).items()}
    pairs = json.load(open(PAIRS))
    C = config.DEFAULT_CONDITIONS
    RT = C.R_kJ * C.temperature_K
    RTln10 = RT * math.log(10)
    mu_H = C.proton_reference_kJ

    # Quality screen, applied BEFORE looking at any error, so it cannot be
    # circular: a reference is usable only if both members are true minima with
    # a physically sensible thermal correction. Small, highly charged anions
    # (PO4^3-, P2O7^4-) are not bound in a continuum solvent without explicit
    # water, and they fail here as imaginary frequencies or negative G_RRHO
    # (impossible: the zero-point term alone is positive).
    ens = json.load(open(os.path.join(HERE, "pka_xtb.json")))

    # Screen on the MAGNITUDE of the imaginary mode, not its presence: a hindered
    # methyl/OH rotor sits at a few negative cm-1 and is harmless, a real saddle
    # point at hundreds. Counting them alike rejected acetic acid, which is absurd.
    IMAG_CM_TOL = float(os.environ.get("IMAG_CM_TOL", "50"))

    def bad(cpd):
        confs = ens.get(cpd)
        if not confs:
            return "missing"
        c0 = confs[0]
        imag = c0.get("imag_cm", 0.0)
        if imag < -IMAG_CM_TOL:
            return f"imag={imag:.0f}cm-1"
        if c0.get("G_RRHO_kJ", 0.0) < 0:
            return f"G_RRHO={c0['G_RRHO_kJ']:.0f}<0"
        return None

    rows, dropped = [], []
    for p in pairs:
        if p["acid"] not in G or p["base"] not in G:
            print(f"   [skip] {p['key']}: missing ensemble")
            continue
        why = bad(p["acid"]) or bad(p["base"])
        if why:
            dropped.append((p["key"], why))
            continue
        dG = G[p["base"]] + mu_H - G[p["acid"]]
        pka = dG / RTln10
        rows.append(dict(**p, pKa_calc=pka, err_pKa=pka - p["pKa_exp"],
                         err_kJ=(pka - p["pKa_exp"]) * RTln10))

    if dropped:
        print("dropped by quality screen (unconverged / not a minimum):")
        for k, why in dropped:
            print(f"   {k:14} {why}")
        print()
    print(f"{'pair':14}{'group':11}{'kind':10}{'pKa_exp':>9}{'pKa_calc':>10}"
          f"{'err(pKa)':>10}{'err(kJ)':>9}")
    for r in sorted(rows, key=lambda r: (r["kind"], r["group"])):
        print(f"{r['key']:14}{r['group']:11}{r['kind']:10}{r['pKa_exp']:9.2f}"
              f"{r['pKa_calc']:10.2f}{r['err_pKa']:10.2f}{r['err_kJ']:9.1f}")

    print()
    for kind in ("anionic", "cationic"):
        e = [r["err_kJ"] for r in rows if r["kind"] == kind]
        if e:
            print(f"{kind:9}  n={len(e):2d}  mean err = {np.mean(e):+7.1f} kJ/mol   "
                  f"sd = {np.std(e):5.1f}   MAE = {np.mean(np.abs(e)):5.1f}")
    print()
    for g in sorted({r["group"] for r in rows}):
        e = [r["err_kJ"] for r in rows if r["group"] == g]
        print(f"   {g:11} n={len(e)}  mean err = {np.mean(e):+7.1f} kJ/mol")

    # ---- charge dependence ----
    # The metabolites are polyanions, so the key question is whether the error
    # per added negative charge stays constant (=> additive per-site correction
    # is valid) or shrinks as charge accumulates (=> additive over-corrects).
    pairs_by_q = {}
    for r in rows:
        if r["kind"] == "anionic":
            pairs_by_q.setdefault(r.get("q_base", r.get("base_charge")), []).append(r)
    if any(k is not None for k in pairs_by_q):
        print("\nerror vs charge of the anion formed:")
        for q in sorted(k for k in pairs_by_q if k is not None):
            e = [r["err_kJ"] for r in pairs_by_q[q]]
            print(f"   q={q:+d}  n={len(e):2d}  mean err = {np.mean(e):+7.1f} kJ/mol"
                  f"   sd = {np.std(e):5.1f}   [{', '.join(r['key'] for r in pairs_by_q[q])}]")

    anio = [r["err_kJ"] for r in rows if r["kind"] == "anionic"]
    catio = [r["err_kJ"] for r in rows if r["kind"] == "cationic"]
    if anio and catio:
        print(f"\ninterpretation: proton-reference-only error would shift both "
              f"families alike.\n   anionic - cationic = "
              f"{np.mean(anio) - np.mean(catio):+.1f} kJ/mol  <- anion-specific part")

    # ---- increments per added charge, the quantity the correction actually needs ----
    # Each pair measures the error of adding ONE charge to a species that may
    # already be charged, i.e. an INCREMENT at charge index k = |q_base|. A
    # correction built as (group mean) x (number of sites) mixes increments from
    # different k and is not a valid extrapolation; the cumulative form below is.
    inc = {}
    for r in rows:
        if r["kind"] == "anionic" and r.get("q_base") is not None:
            inc.setdefault(abs(int(r["q_base"])), []).append(r["err_kJ"])
    inc_mean = {k: float(np.mean(v)) for k, v in inc.items()}
    if inc_mean:
        print("\nincrement per added negative charge (the correction ladder):")
        run = 0.0
        for k in sorted(inc_mean):
            run += inc_mean[k]
            print(f"   k={k}  n={len(inc[k]):2d}  increment = {inc_mean[k]:+7.1f}"
                  f"   cumulative correction for a q=-{k} species = {run:+7.1f}")
        flat = inc_mean.get(1, 0.0)
        print(f"   [additive per-site model would give {flat * max(inc_mean):+.1f} "
              f"at q=-{max(inc_mean)} vs {run:+.1f} measured]")

    json.dump(dict(rows=rows,
                   mean_anionic_kJ=float(np.mean(anio)) if anio else None,
                   mean_cationic_kJ=float(np.mean(catio)) if catio else None,
                   increment_by_charge=inc_mean),
              open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
