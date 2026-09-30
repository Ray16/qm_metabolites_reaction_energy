"""Rebuild the TECRDB-367 experimental reference at MetaG's conditions (pH 7, I = 0, no Mg2+, per-measurement T).

The benchmark's `exp` was the median of -RT ln K' over each reaction's raw TECRDB measurements AT THEIR OWN
conditions (median pH 7.5, I ~0.25 M where reported, Mg2+ in some) -- not the transformed standard ΔrG'° MetaG
predicts. This script applies the Legendre transform to EVERY measurement before aggregating, exactly as
component contribution does when it trains on TECRDB:

    ΔG'°(pH 7, I 0, no Mg; T) = -RT ln K'_obs  +  [ΔG'°_eQ(7, 0, pMg 14; T) - ΔG'°_eQ(pH, I, pMg; T)]

The bracket is a difference of eQuilibrator predictions at two conditions for the same reaction; the fitted
formation energies cancel exactly, so it depends only on the species' tabulated dissociation / Mg-binding
constants and the extended Debye-Hückel term (no fitted free energy enters -> no leakage). Temperature is NOT
transformed (no ΔrH for most reactions): each measurement stays at its own T (reported as a limitation).

Matching of TECRDB rows to ModelSEED reactions is identical to the original builder (pipeline/build_tecrdb_set.py,
git 6f7f94e): KEGG -> ModelSEED compound multiset, protons and water ignored, both orientations. The script first
REPRODUCES the old `exp` (native conditions, same rows) as a check, then emits the transformed reference.

Rows used: K' present (rows that report only a species-level K are EXCLUDED -- the old builder wrongly treated
them as K'), pH present. Missing ionic strength -> imputed (--impute-I, default 0.25 M, the median reported
value; the effect of imputing 0 instead is reported); missing pMg -> no Mg2+.

Run (eQuilibrator env):  ~/miniforge3/envs/eqapi/bin/python analysis/build_tecrdb_standard.py
Writes experiments/qm_mlip_solvation/scripts/reactions_tecrdb_std.json (same reactions/structures as
reactions_tecrdb_all.json, exp replaced) and analysis/tecrdb_standard_report.json.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                        # thermodynamic_calc
MSDB = os.path.join(os.path.dirname(ROOT), "ModelSEEDDatabase")
INP = os.path.join(ROOT, "experiments", "qm_mlip_solvation", "scripts", "reactions_tecrdb_all.json")
OUT = os.path.join(ROOT, "experiments", "qm_mlip_solvation", "scripts", "reactions_tecrdb_std.json")
REPORT = os.path.join(HERE, "tecrdb_standard_report.json")
TECRDB = os.path.join(HERE, "tecrdb_source", "TECRDB.csv")
PROTON, WATER = "cpd00067", "cpd00001"
R_KJ = 8.314462618e-3
KEGG_RE = re.compile(r"C\d{5}")
STD = dict(p_h=7.0, ionic_strength=0.0, p_mg=14.0)                   # MetaG conditions (pMg 14 = no Mg2+)


# ---- matching: verbatim logic of pipeline/build_tecrdb_set.py (git 6f7f94e) ----------------------------
def kegg_to_modelseed(db):
    path = os.path.join(db, "Biochemistry", "Aliases", "Unique_ModelSEED_Compound_Aliases.txt")
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["Source"] != "KEGG":
                continue
            kegg, seed = row["External ID"].strip(), row["ModelSEED ID"].strip()
            if KEGG_RE.fullmatch(kegg) and (kegg not in out or seed < out[kegg]):
                out[kegg] = seed
    return out


def parse_equation(text):
    if "=" not in text:
        return Counter(), False
    left, right = text.split("=", 1)
    stoich = Counter()
    for side, sign in ((left, -1), (right, 1)):
        for term in side.split("+"):
            term = term.strip()
            if not term:
                continue
            m = KEGG_RE.search(term)
            if not m:
                return Counter(), False
            coeff = re.match(r"\s*(\d+)\s", term.replace("kegg:", " "))
            stoich[m.group(0)] += sign * (int(coeff.group(1)) if coeff else 1)
    return stoich, bool(stoich)


def load_modelseed_reactions(db):
    reactions = {}
    for path in sorted(glob.glob(os.path.join(db, "Biochemistry", "reaction_??.tsv"))):
        with open(path) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if row.get("is_obsolete", "0") not in ("0", "", "False"):
                    continue
                st = {}
                for part in (row.get("stoichiometry") or "").split(";"):
                    f = part.split(":")
                    if len(f) >= 2:
                        try:
                            st[f[1]] = st.get(f[1], 0.0) + float(f[0])
                        except ValueError:
                            pass
                if st:
                    reactions[row["id"]] = st
    return reactions


def signature(st, drop):
    return tuple(sorted((c, round(v, 6)) for c, v in st.items() if c not in drop and abs(v) > 1e-9))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def matched_measurements(keep_rids):
    """[(rid, orient, row)] for every TECRDB row matching a benchmark reaction (same rule as the builder)."""
    kegg = kegg_to_modelseed(MSDB)
    ms = load_modelseed_reactions(MSDB)
    index = defaultdict(list)
    for rid, st in ms.items():
        index[signature(st, {PROTON, WATER})].append((rid, +1))
        index[signature({c: -v for c, v in st.items()}, {PROTON, WATER})].append((rid, -1))
    out = []
    for row in csv.DictReader(open(TECRDB)):
        eq, ok = parse_equation(row.get("reaction") or "")
        if not ok or any(k not in kegg for k in eq):
            continue
        hits = index.get(signature({kegg[k]: v for k, v in eq.items()}, {PROTON, WATER}), [])
        if not hits:
            continue
        rid, orient = sorted(hits)[0]
        if rid in keep_rids:
            out.append((rid, orient, row))
    return out, ms


# ---- eQuilibrator transform --------------------------------------------------------------------------
class Transformer:
    def __init__(self, ms_reactions):
        from equilibrator_api import ComponentContribution, Q_
        self.cc, self.Q_, self.ms = ComponentContribution(), Q_, ms_reactions
        self._rxn, self._memo = {}, {}

    def reaction(self, rid):
        if rid not in self._rxn:
            st = self.ms[rid]
            lhs = [f"{abs(v):g} seed:{c}" for c, v in st.items() if v < 0 and c != PROTON]
            rhs = [f"{abs(v):g} seed:{c}" for c, v in st.items() if v > 0 and c != PROTON]
            r = self.cc.parse_reaction_formula(" + ".join(lhs) + " = " + " + ".join(rhs))
            self._rxn[rid] = r if (r.is_balanced() and r.can_be_transformed()) else None
        return self._rxn[rid]

    def dgp(self, rid, p_h, ionic_strength, p_mg, T):
        key = (rid, round(p_h, 3), round(ionic_strength, 4), round(p_mg, 3), round(T, 2))
        if key not in self._memo:
            cc, Q_ = self.cc, self.Q_
            cc.p_h = Q_(p_h); cc.ionic_strength = Q_(f"{ionic_strength}M")
            cc.p_mg = Q_(p_mg); cc.temperature = Q_(f"{T}K")
            self._memo[key] = float(cc.standard_dg_prime(self.reaction(rid)).value.m_as("kJ/mol"))
        return self._memo[key]

    def correction(self, rid, p_h, ionic_strength, p_mg, T):
        """ΔG'°(standard) - ΔG'°(observed conditions), same T: the Legendre-transform difference."""
        return (self.dgp(rid, STD["p_h"], STD["ionic_strength"], STD["p_mg"], T)
                - self.dgp(rid, p_h, ionic_strength, p_mg, T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impute-I", type=float, default=0.25, help="ionic strength (M) for rows reporting none")
    a = ap.parse_args()
    R = json.load(open(INP))
    meas, ms = matched_measurements(set(R))

    # 1. reproduce the old reference (native conditions, K' or K as the old builder did) -> validates matching
    old = defaultdict(list)
    for rid, orient, row in meas:
        K = _f(row.get("K_prime")) or _f(row.get("K")); T = _f(row.get("temperature")) or 298.15
        if K and K > 0:
            old[rid].append(orient * -R_KJ * T * math.log(K))
    repro = [abs(statistics.median(old[r]) - R[r]["exp"][0]) for r in R if old.get(r)]
    print(f"reproduce old exp: {len(repro)}/{len(R)} reactions, max |Δ| {max(repro):.3f} kJ")

    # 2. transform every usable measurement
    tr = Transformer(ms)
    per, per_I0, stats, excluded = defaultdict(list), defaultdict(list), Counter(), Counter()
    for rid, orient, row in meas:
        Kp, T, ph = _f(row.get("K_prime")), _f(row.get("temperature")) or 298.15, _f(row.get("p_h"))
        if not Kp or Kp <= 0:
            excluded["no K' (species K only, or none)"] += 1; continue
        if ph is None:
            excluded["no pH"] += 1; continue
        if tr.reaction(rid) is None:
            excluded["reaction not transformable in eQuilibrator"] += 1; continue
        I = _f(row.get("ionic_strength")); pmg = _f(row.get("p_mg"))
        stats["I imputed" if I is None else "I reported"] += 1
        stats["no Mg (pMg missing)" if pmg is None else "pMg reported"] += 1
        pmg = 14.0 if pmg is None else pmg
        dg_obs = orient * -R_KJ * T * math.log(Kp)
        corr = tr.correction(rid, ph, a.impute_I if I is None else I, pmg, T)
        per[rid].append({"dG_obs": dg_obs, "corr": corr, "T": T, "pH": ph, "I": I, "pMg": pmg,
                         "ref": row.get("reference")})
        per_I0[rid].append(dg_obs + tr.correction(rid, ph, 0.0 if I is None else I, pmg, T))

    # 3. aggregate (median over measurements, as before) and write
    out, rep = {}, {"conditions": STD, "impute_I_M": a.impute_I, "temperature": "per measurement (not transformed)",
                    "excluded_measurements": dict(excluded), "measurement_stats": dict(stats), "reactions": {}}
    dropped = []
    for rid, rx in R.items():
        obs = per.get(rid)
        if not obs:
            dropped.append(rid); continue
        vals = [o["dG_obs"] + o["corr"] for o in obs]
        std_exp = statistics.median(vals)
        out[rid] = dict(rx, exp=[round(std_exp, 2)], exp_sd=round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
                        exp_native=rx["exp"], exp_n=len(vals),
                        exp_conditions="pH 7, I 0, no Mg2+ (Legendre-transformed per measurement; T as measured)")
        rep["reactions"][rid] = {"native": rx["exp"][0], "standard": round(std_exp, 2),
                                 "shift": round(std_exp - rx["exp"][0], 2), "n": len(vals),
                                 "standard_if_I_imputed_0": round(statistics.median(per_I0[rid]), 2),
                                 "median_correction": round(statistics.median(o["corr"] for o in obs), 2)}
    rep["dropped_reactions"] = dropped
    sh = [abs(v["shift"]) for v in rep["reactions"].values()]
    s0 = [abs(v["standard"] - v["standard_if_I_imputed_0"]) for v in rep["reactions"].values()]
    rep["summary"] = {"n_reactions": len(out), "n_dropped": len(dropped),
                      "mean_abs_shift": round(statistics.mean(sh), 2), "median_abs_shift": round(statistics.median(sh), 2),
                      "n_shift_gt_5": sum(x > 5 for x in sh), "n_shift_gt_10": sum(x > 10 for x in sh),
                      "I_imputation_sensitivity_mean_abs": round(statistics.mean(s0), 2),
                      "I_imputation_sensitivity_max": round(max(s0), 2)}
    json.dump(out, open(OUT, "w"), indent=1)
    json.dump(rep, open(REPORT, "w"), indent=1)
    print(json.dumps(rep["summary"], indent=1)); print("excluded:", dict(excluded)); print("stats:", dict(stats))
    print(f"wrote {OUT}\nwrote {REPORT}")


if __name__ == "__main__":
    main()
