#!/usr/bin/env python3
"""Extract REAL experimental Delta_r G'^o (TECRDB / NIST) for the benchmark reactions.

Parsing is taken verbatim from component-contribution's authoritative reader
(training_data.py::read_tecrdb):
  columns = [URL, REF_ID, METHOD, EVAL, EC, ENZYME NAME,
             REACTION IN KEGG IDS, REACTION IN COMPOUND NAMES,
             K, K', T, I, pH, pMg]
  skip rows where K', T, or pH is empty
  dG'0 [kJ/mol] = -R * T * ln(K'),  R = 8.31e-3 kJ/(K*mol)
This dG'0 is the TRANSFORMED energy AT THE MEASURED T/pH/I/pMg -- NOT standard.

We map each ModelSEED benchmark reaction to a KEGG compound multiset, match it
(direction-aware) to TECRDB reactions, and report:
  - the near-standard subset (pH 6.5-7.5, T 296-300 K, I<=0.35 M) as the value
    comparable to QM/eQuilibrator standard conditions (T=298.15, pH=7, I=0.25)
  - the full set of measurements for transparency
No condition transform is applied; out-of-window measurements are reported
separately, never averaged into the comparable value.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import defaultdict

R = 8.31e-3  # kJ/(K*mol) -- component-contribution thermodynamic_constants.R
PROTON_KEGG = "C00080"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THERMO = os.path.join(ROOT, "thermodynamic_calc")
MSEED = os.path.join(ROOT, "ModelSEEDDatabase", "Biochemistry")
TECRDB = os.path.join(THERMO, "data", "experimental", "TECRDB.tsv")

# Near-standard condition window (compared to T=298.15, pH=7.0, I=0.25 M)
PH_LO, PH_HI = 6.5, 7.5
T_LO, T_HI = 296.0, 300.0
I_MAX = 0.35


def load_cpd_to_kegg() -> dict[str, set]:
    """ModelSEED compound id -> set of KEGG C-numbers (source == 'KEGG')."""
    m = defaultdict(set)
    path = os.path.join(MSEED, "Aliases", "Unique_ModelSEED_Compound_Aliases.txt")
    with open(path) as fh:
        next(fh)  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            seed, ext, src = parts[0], parts[1], parts[2]
            if src == "KEGG" and ext.startswith("C") and ext[1:].isdigit():
                m[seed].add(ext)
    return m


def kegg_multiset(stoich: dict, cpd2kegg: dict[str, set]):
    """ModelSEED {cpd: coeff} -> {kegg: coeff} (H+ dropped). None if any cpd
    lacks a unique KEGG id (ambiguity would make the match unsafe)."""
    out = {}
    for cpd, coeff in stoich.items():
        keggs = cpd2kegg.get(cpd, set()) - {PROTON_KEGG}
        if cpd == "cpd00067":  # proton
            continue
        if len(keggs) != 1:
            return None  # unmapped or ambiguous -> refuse to guess
        k = next(iter(keggs))
        if k == PROTON_KEGG:
            continue
        out[k] = out.get(k, 0.0) + coeff
    return {k: v for k, v in out.items() if abs(v) > 1e-9}


def parse_kegg_formula(s: str):
    """'C00001 + 2 C00002 = C00003' -> {kegg: coeff} (left negative, right
    positive), H+ dropped. Returns None on any unparseable token."""
    if "=" not in s:
        return None
    left, right = s.split("=", 1)
    out = {}
    for side, sign in ((left, -1.0), (right, +1.0)):
        for term in side.split("+"):
            term = term.strip()
            if not term:
                continue
            bits = term.split()
            if len(bits) == 1:
                coeff, cid = 1.0, bits[0]
            else:
                try:
                    coeff, cid = float(bits[0]), bits[1]
                except ValueError:
                    return None
            if cid == PROTON_KEGG:
                continue
            if not (cid.startswith("C") and cid[1:].isdigit()):
                return None
            out[cid] = out.get(cid, 0.0) + sign * coeff
    return {k: v for k, v in out.items() if abs(v) > 1e-9}


def same_reaction(a: dict, b: dict):
    """Return +1 if a==b, -1 if a==reverse(b), else 0 (exact coeff match)."""
    if a.keys() != b.keys():
        return 0
    if all(abs(a[k] - b[k]) < 1e-6 for k in a):
        return 1
    if all(abs(a[k] + b[k]) < 1e-6 for k in a):
        return -1
    return 0


def main():
    cpd2kegg = load_cpd_to_kegg()

    # Benchmark reaction ids and their ModelSEED stoichiometry (full, incl water)
    rows = list(csv.DictReader(open(os.path.join(THERMO, "results/benchmark/reaction_benchmark.csv"))))
    want = {r["rxn_id"] for r in rows}
    seed_rxn = {}
    for path in glob.glob(os.path.join(MSEED, "reaction_*.json")):
        for rec in json.load(open(path)):
            if rec["id"] in want:
                st = {}
                for s in rec["stoichiometry"]:
                    st[s["compound"]] = st.get(s["compound"], 0.0) + s["coefficient"]
                seed_rxn[rec["id"]] = st

    # KEGG signature per benchmark reaction
    seed_kegg = {}
    for rid, st in seed_rxn.items():
        km = kegg_multiset(st, cpd2kegg)
        if km:
            seed_kegg[rid] = km

    # Parse TECRDB -> list of (kegg_multiset, dG'0, T, I, pH, pMg, ref, eval, names)
    measurements = []
    skipped = 0
    with open(TECRDB) as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if not row or len(row) < 14:
                continue
            (url, ref, method, ev, ec, enz, kegg_rxn, names,
             K, Kp, T, I, pH, pMg) = row[:14]
            if Kp == "" or T == "" or pH == "":
                skipped += 1
                continue
            try:
                Tv = float(T)
                Kpv = float(Kp)
            except ValueError:
                skipped += 1
                continue
            if Kpv <= 0:
                skipped += 1
                continue
            km = parse_kegg_formula(kegg_rxn)
            if km is None:
                continue
            dG = -R * Tv * math.log(Kpv)
            Iv = float(I) if I not in ("", None) else float("nan")
            measurements.append({
                "kegg": km, "dG": dG, "T": Tv, "I": Iv,
                "pH": float(pH), "ref": ref, "eval": ev, "names": names,
            })

    # Match
    results = {}
    for rid, sig in seed_kegg.items():
        hits = []
        for meas in measurements:
            d = same_reaction(sig, meas["kegg"])
            if d != 0:
                m = dict(meas)
                m["dG"] = d * meas["dG"]  # align to our reaction direction
                hits.append(m)
        if hits:
            results[rid] = hits

    # Report
    print(f"TECRDB measurements parsed: {len(measurements)} (skipped {skipped} empty/invalid)")
    print(f"benchmark reactions: {len(want)} ({len(seed_kegg)} fully KEGG-mappable)")
    print(f"reactions with >=1 TECRDB match: {len(results)}\n")

    def stats(vals):
        n = len(vals)
        mean = sum(vals) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in vals) / n) if n > 1 else 0.0
        return n, mean, sd

    out_rows = []
    print(f"{'rxn':10s} {'n_std':>5s} {'dG_exp_std':>11s} {'sd':>5s}  "
          f"{'n_all':>5s} {'dG_all_range':>16s}  reaction")
    for rid in sorted(results):
        hits = results[rid]
        near = [h for h in hits
                if PH_LO <= h["pH"] <= PH_HI and T_LO <= h["T"] <= T_HI
                and (math.isnan(h["I"]) or h["I"] <= I_MAX)]
        allv = [h["dG"] for h in hits]
        names = hits[0]["names"]
        if near:
            n, mean, sd = stats([h["dG"] for h in near])
            std_str = f"{mean:11.1f} {sd:5.1f}"
        else:
            n, mean, sd = 0, float("nan"), float("nan")
            std_str = f"{'--':>11s} {'--':>5s}"
        rng = f"[{min(allv):6.1f},{max(allv):6.1f}]"
        print(f"{rid:10s} {n:>5d} {std_str}  {len(allv):>5d} {rng:>16s}  {names[:48]}")
        out_rows.append({
            "rxn_id": rid,
            "n_measurements_total": len(allv),
            "n_measurements_nearstd": n,
            "dG_exp_nearstd_kJ": "" if math.isnan(mean) else round(mean, 2),
            "dG_exp_nearstd_sd_kJ": "" if math.isnan(sd) else round(sd, 2),
            "dG_all_min_kJ": round(min(allv), 2),
            "dG_all_max_kJ": round(max(allv), 2),
            "reaction": names,
        })

    out_path = os.path.join(THERMO, "results", "benchmark", "experimental_dG_TECRDB.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    n_std = sum(1 for r in out_rows if r["dG_exp_nearstd_kJ"] != "")
    print(f"\nwrote {out_path}")
    print(f"  {n_std}/{len(out_rows)} matched reactions have a near-standard "
          f"(pH 6.5-7.5, T 296-300K, I<=0.35) experimental value")


if __name__ == "__main__":
    main()
