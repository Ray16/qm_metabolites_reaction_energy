"""Rebuild the benchmark experimental reference from openTECR (the community re-curation of TECRDB).

Same matched reactions (reactions_tecrdb_all.json, 367), same matching rule and same Legendre transform as
build_tecrdb_standard.py (pH 7, I = 0, no Mg2+,
per-measurement T); only the measurement source changes.

Source: openTECR "actual data" sheet (opentecr_source/sheet_actual_data.csv, exported from the openTECR Google
workbook, see opentecr_source/README.md). Rows carry
  * `id` = ...zenodo.3978439/files/TECRDB.csv#entryN  -> row N of tecrdb_source/TECRDB.csv (verified identical,
    row for row, to the component-contribution training file), which supplies the KEGG equation;
  * no `id` -> new or corrected data; the KEGG equation is recovered from the openTECR reaction name (names of
    id-carrying rows -> their KEGG equation; blank names inherit the table's reaction from "table metadata").

Rows used: apparent K' (type of Kprime not K/Kc), pH present, aqueous (not solvent_reaction), not flagged
wrong_value / duplicate_table (non-authoritative) / chemical_reference_reaction. Missing I -> imputed 0.25 M as before.

Run (eQuilibrator env):  ~/miniforge3/envs/eqapi/bin/python analysis/build_opentecr_standard.py
Writes experiments/qm_mlip_solvation/scripts/reactions_opentecr_std.json and analysis/opentecr_standard_report.json.
"""
import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict

import build_tecrdb_standard as B

HERE = B.HERE
SRC = os.path.join(HERE, "opentecr_source")
DATA = os.path.join(SRC, "sheet_actual_data.csv")
META = os.path.join(SRC, "sheet_table_metadata.csv")
ALL_INP = B.INP          # reactions_tecrdb_all.json: the 367 TECRDB-matched reactions (same start as the TECRDB builder)
STD_INP = B.OUT          # reactions_tecrdb_std.json: the TECRDB-built reference, for the comparison fields only
OUT = os.path.join(B.ROOT, "experiments", "qm_mlip_solvation", "scripts", "reactions_opentecr_std.json")
REPORT = os.path.join(HERE, "opentecr_standard_report.json")
TABLE_KEY = ("part", "page", "col l/r", "table from top")
# openTECR spellings (new rows) of reactions that exist in TECRDB under a different name; checked by hand.
ALIAS = {
    "2 glutathione(red)(ox) + NADP(aq) = glutathione(ox)(aq) + NADPH(aq)":
        "2 kegg:C00051 + kegg:C00006 = kegg:C00127 + kegg:C00005",
    "inosine(aq) + orthophosphate(aq) = hypoxanthine(aq) + alpha-D-ribose 1-phosphate(aq)":
        "kegg:C00294 + kegg:C00009 = kegg:C00262 + kegg:C00620",
    "CoA(aq) + acetate(aq) + ATP(aq) = acetyl-CoA(aq) + ADP(aq) + orthophosphate(aq)":
        "kegg:C00010 + kegg:C00033 + kegg:C00002 = kegg:C00024 + kegg:C00008 + kegg:C00009",
}


def _norm(name):
    return re.sub(r"\s+\(", "(", " ".join((name or "").split()))


def _key(row):
    return tuple(str(B._f(row[k]) if B._f(row[k]) is not None else row[k]).strip() for k in TABLE_KEY)


def opentecr_rows():
    """openTECR rows with a KEGG equation attached: [(row, kegg_equation, how)]."""
    old = list(csv.DictReader(open(B.TECRDB)))
    rows = list(csv.DictReader(open(DATA)))
    table_rxn = {_key(m): _norm(m["reaction"]) for m in csv.DictReader(open(META)) if _norm(m["reaction"])}
    name2kegg = defaultdict(set, {_norm(k): {v} for k, v in ALIAS.items()})
    kegg2ec = defaultdict(set)                                        # EC guard for name-mapped rows
    for o in old:
        kegg2ec[o["reaction"]].update(e.strip() for e in o["EC"].split("&"))
    for r in rows:
        if "#entry" in r["id"]:
            o = old[int(r["id"].split("#entry")[1]) - 1]
            name2kegg[_norm(r["reaction"])].add(o["reaction"])
            name2kegg[_norm(o["description"])].add(o["reaction"])
    out, how = [], Counter()
    for r in rows:
        name = _norm(r["reaction"]) or table_rxn.get(_key(r), "")
        if "#entry" in r["id"]:
            out.append((r, old[int(r["id"].split("#entry")[1]) - 1]["reaction"], "entry id")); how["entry id"] += 1
        elif len(name2kegg.get(name, ())) == 1:
            keq = next(iter(name2kegg[name]))
            # a blank reaction cell inherits the table's reaction; openTECR has tables whose reaction text does not
            # match their enzyme (64BOJ/GAU: EC 2.1.3.5 oxamate carbamoyltransferase labelled as the ornithine
            # reaction), so a row with an EC must share an EC component with what TECRDB records for that reaction
            ec = {e.strip() for e in (r["EC"] or "").split("&") if e.strip()}
            if ec and not ec & kegg2ec[keq]:
                how["unmapped (EC disagrees with reaction)"] += 1; continue
            out.append((r, keq, "name")); how["name"] += 1
        else:
            how["unmapped (ambiguous)" if name in name2kegg else "unmapped (name not in TECRDB)"] += 1
    return out, how


def usable(r):
    if r["solvent_reaction"]:
        return "solvent (non-aqueous) reaction"
    if r["wrong_value"]:
        return "wrong_value (superseded by an error_correction row)"
    if r["duplicate_table"] and not r["authoritative_version_of_a_duplicate_table"]:
        return "duplicate table (non-authoritative copy)"
    if r["chemical_reference_reaction"] or r["virtual_entry"]:
        return "chemical reference / virtual entry"
    if r["type of Kprime"].strip() in ("K", "Kc"):
        return "species-level K (not K')"
    Kp = B._f(r["K_prime"])
    if not Kp or Kp <= 0:
        return "no K'"
    if B._f(r["p_h"]) is None:
        return "no pH"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impute-I", type=float, default=0.25)
    a = ap.parse_args()
    R = json.load(open(ALL_INP))
    TSTD = json.load(open(STD_INP))
    kegg = B.kegg_to_modelseed(B.MSDB)
    ms = B.load_modelseed_reactions(B.MSDB)
    index = defaultdict(list)
    for rid, st in ms.items():
        index[B.signature(st, {B.PROTON, B.WATER})].append((rid, +1))
        index[B.signature({c: -v for c, v in st.items()}, {B.PROTON, B.WATER})].append((rid, -1))

    rows, how = opentecr_rows()
    tr = B.Transformer(ms)
    per, per_I0, native, excluded, stats, breakdown = (defaultdict(list), defaultdict(list), defaultdict(list),
                                                       Counter(), Counter(), Counter())
    for k in ("no K' (species K only, or none)", "no pH", "reaction not transformable in eQuilibrator"):
        excluded[k] = 0                                               # always reported (also when zero)
    cond = defaultdict(list)                                          # per used measurement, for the condition medians
    for r, keq, src in rows:
        eq, ok = B.parse_equation(keq)
        if not ok or any(k not in kegg for k in eq):
            continue
        hits = index.get(B.signature({kegg[k]: v for k, v in eq.items()}, {B.PROTON, B.WATER}), [])
        if not hits:
            continue
        rid, orient = sorted(hits)[0]
        if rid not in R:
            continue
        why = usable(r)
        if why:
            breakdown[why] += 1
            # same key as build_tecrdb_standard.py so downstream tools read both reports alike
            excluded["no K' (species K only, or none)" if why in ("no K'", "species-level K (not K')") else why] += 1
            continue
        if tr.reaction(rid) is None:
            excluded["reaction not transformable in eQuilibrator"] += 1; continue
        Kp, ph = B._f(r["K_prime"]), B._f(r["p_h"])
        T = B._f(r["temperature"]) or 298.15
        I, pmg = B._f(r["ionic_strength"]), B._f(r["p_mg"])
        stats["I imputed" if I is None else "I reported"] += 1
        stats["no Mg (pMg missing)" if pmg is None else "pMg reported"] += 1
        stats[f"source: {src}"] += 1
        if r["error_correction"]:
            stats["error_correction rows used"] += 1
        elif src == "name":
            stats["added rows used (no TECRDB id)"] += 1
        cond[("pH", rid)].append(ph); cond[("T", rid)].append(T)
        if I is not None:
            cond["I"].append(I); cond["rid_I"].append(rid)
        if pmg is not None:
            cond["rid_Mg"].append(rid)
        pmg = 14.0 if pmg is None else pmg
        dg_obs = orient * -B.R_KJ * T * math.log(Kp)
        native[rid].append(dg_obs)
        per[rid].append(dg_obs + tr.correction(rid, ph, a.impute_I if I is None else I, pmg, T))
        per_I0[rid].append(dg_obs + tr.correction(rid, ph, 0.0 if I is None else I, pmg, T))

    out, rep_rx, dropped = {}, {}, []
    for rid, rx in R.items():
        vals = per.get(rid)
        if not vals:
            dropped.append(rid); continue
        e, nat = statistics.median(vals), statistics.median(native[rid])
        out[rid] = dict(rx, exp=[round(e, 2)], exp_sd=round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
                        exp_n=len(vals), exp_native=[round(nat, 2)], exp_tecrdb_std=TSTD.get(rid, {}).get("exp"),
                        exp_conditions="openTECR; pH 7, I 0, no Mg2+ (Legendre-transformed per measurement; T as measured)")
        t = TSTD.get(rid)
        rep_rx[rid] = {"native": round(nat, 2), "standard": round(e, 2), "shift": round(e - nat, 2), "n": len(vals),
                       "standard_if_I_imputed_0": round(statistics.median(per_I0[rid]), 2),
                       "tecrdb_std": t["exp"][0] if t else None,
                       "shift_vs_tecrdb_std": round(e - t["exp"][0], 2) if t else None,
                       "n_tecrdb": t.get("exp_n") if t else None}
    sh = [abs(v["shift"]) for v in rep_rx.values()]
    s0 = [abs(v["standard"] - v["standard_if_I_imputed_0"]) for v in rep_rx.values()]
    st = [abs(v["shift_vs_tecrdb_std"]) for v in rep_rx.values() if v["shift_vs_tecrdb_std"] is not None]
    pct = lambda xs, q: sorted(xs)[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]
    # per-reaction medians, then the median / 5-95 % range over reactions (the manuscript's convention)
    rx_ph = [statistics.median(cond[("pH", rid)]) for rid in out]
    rx_T = [statistics.median(cond[("T", rid)]) for rid in out]
    rep = {"source": "openTECR snapshot 2026-10-02 (opentecr_source/README.md)",
           "conditions": B.STD, "impute_I_M": a.impute_I, "temperature": "per measurement (not transformed)",
           "mapping": dict(how), "excluded_measurements": dict(excluded), "excluded_breakdown": dict(breakdown),
           "measurement_stats": dict(stats),
           "measurement_conditions": {"basis": "per-reaction median, then median / 5-95% over reactions",
                                      "median_pH": round(statistics.median(rx_ph), 2),
                                      "pH_5_95": [round(pct(rx_ph, 0.05), 2), round(pct(rx_ph, 0.95), 2)],
                                      "median_T": round(statistics.median(rx_T), 2),
                                      "median_reported_I": round(statistics.median(cond["I"]), 3),
                                      "n_reactions_I_reported": len(set(cond["rid_I"])),
                                      "n_reactions_Mg_reported": len(set(cond["rid_Mg"]))},
           "summary": {"n_reactions": len(out), "n_dropped": len(dropped), "dropped": dropped,
                       "mean_abs_shift": round(statistics.mean(sh), 2), "median_abs_shift": round(statistics.median(sh), 2),
                       "n_shift_gt_5": sum(x > 5 for x in sh), "n_shift_gt_10": sum(x > 10 for x in sh),
                       "I_imputation_sensitivity_mean_abs": round(statistics.mean(s0), 2),
                       "I_imputation_sensitivity_max": round(max(s0), 2)},
           "vs_tecrdb_std": {"mean_abs_shift": round(statistics.mean(st), 2), "median_abs_shift": round(statistics.median(st), 2),
                             "n_changed": sum(x >= 0.005 for x in st), "n_shift_gt_1": sum(x > 1 for x in st),
                             "n_shift_gt_5": sum(x > 5 for x in st), "n_shift_gt_10": sum(x > 10 for x in st)},
           "reactions": rep_rx}
    json.dump(out, open(OUT, "w"), indent=1)
    json.dump(rep, open(REPORT, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("mapping", "excluded_measurements", "excluded_breakdown", "measurement_stats",
                                          "measurement_conditions", "summary", "vs_tecrdb_std")}, indent=1))
    print(f"wrote {OUT}\nwrote {REPORT}")


if __name__ == "__main__":
    main()
