"""Score the PRODUCTION species estimator on reactions with INDEPENDENT experimental references
(gas: TRC/Gurvich/CODATA via NIST CCCBDB; hydration: FreeSolv v0.52; no TECRDB/openTECR data).

Decomposes the aqueous error into gas (UMA E + RRHO, lowest minimum) and solvation (ALPB vs FreeSolv).
Input: output of species_decomp.py.  Usage: python class_bias_analysis.py merged_species.json results.json
"""
import json, sys
T = 298.15; STD = 8.314e-3 * T * __import__("math").log(0.082057 * T)
# gas dfH (kJ/mol), S (J/mol/K) at 298.15 K, NIST CCCBDB experimental pages (source in comment)
GAS = {"water": (-241.81, 188.84),        # Ruscic 2006 / CODATA
       "ethylene": (52.40, 219.32),       # Gurvich 1989
       "ethanol": (-234.80, 281.62),      # Gurvich 1989
       "propene": (19.70, 266.73),        # TRC 1994
       "2-propanol": (-272.70, 309.20),   # TRC 1994
       "isobutene": (-17.80, 293.20),     # Cox&Pilcher 1970 / TRC
       "tBuOH": (-312.40, 326.70),        # TRC 1994
       "ammonia": (-45.94, 192.77),       # CODATA
       "methanol": (-201.00, 239.87),     # Gurvich 1989
       "methylamine": (-22.50, 242.89),   # TRC 1994
       "ethylamine": (-47.50, 283.78),    # TRC 1994
       "acetic_acid": (-432.30, 283.47),  # TRC 1994 (monomer)
       "ethyl_acetate": (-445.43, 362.75)}  # Wiberg 1991 / Stull 1969 (S uncertain: 377.0 alt.)
# FreeSolv v0.52 experimental hydration free energies (kcal/mol, 1 M -> 1 M)
HYD = {"ethylene": 1.28, "ethanol": -5.00, "propene": 1.32, "2-propanol": -4.74, "isobutene": 1.16,
       "tBuOH": -4.47, "ammonia": -4.29, "methanol": -5.10, "methylamine": -4.55, "ethylamine": -4.50,
       "acetic_acid": -6.69, "ethyl_acetate": -2.94}
G_LIQ_MINUS_GAS_WATER = -237.13 + 228.58   # CODATA dfG(l) - dfG(g)
RXN = {"ethylene hydration": {"ethylene": -1, "water": -1, "ethanol": 1},
       "propene hydration": {"propene": -1, "water": -1, "2-propanol": 1},
       "isobutene hydration": {"isobutene": -1, "water": -1, "tBuOH": 1},
       "ethylamine hydrolysis (deamination)": {"ethylamine": -1, "water": -1, "ethanol": 1, "ammonia": 1},
       "methylamine hydrolysis (deamination)": {"methylamine": -1, "water": -1, "methanol": 1, "ammonia": 1},
       "ethyl acetate hydrolysis": {"ethyl_acetate": -1, "water": -1, "acetic_acid": 1, "ethanol": 1}}

def main():
    d = json.load(open(sys.argv[1])); wref = d["_water_ref"]
    out = {}
    for name, st in RXN.items():
        if any(s not in d or "E" not in d[s] for s in st):
            continue
        g_exp = sum(n * (GAS[s][0] - T * GAS[s][1] / 1000) for s, n in st.items())
        g_pred = sum(n * (d[s]["E"] + d[s]["gcorr"]) for s, n in st.items())
        g_pred_q = sum(n * (d[s]["E"] + d[s]["gcorr_qrrho"]) for s, n in st.items())
        aq_exp = g_exp + sum(n * (STD + 4.184 * HYD[s]) if s != "water" else n * G_LIQ_MINUS_GAS_WATER
                             for s, n in st.items())
        aq_pred = sum(n * ((d[s]["G_prod"] if s != "water" else wref) + STD) for s, n in st.items())
        solv_err = sum(n * (d[s]["solv_alpb"] - 4.184 * HYD[s]) for s, n in st.items() if s != "water")
        out[name] = {k: round(v, 2) for k, v in dict(
            gas_exp=g_exp, gas_pred=g_pred, gas_err=g_pred - g_exp, gas_err_qrrho=g_pred_q - g_exp,
            aq_exp=aq_exp, aq_pred=aq_pred, aq_err=aq_pred - aq_exp, solv_err_alpb=solv_err).items()}
        print(f"{name:40s} " + " ".join(f"{k}={v:+.1f}" for k, v in out[name].items()))
    return out



# ---------------------------------------------------------------- FreeSolv group errors (ALPB)
from rdkit import Chem
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
FREESOLV_ALPB = ROOT.parent / "MetaG/analysis/review_fixes/freesolv_solv_models.json"   # per-molecule ALPB vs exp
GROUPS = {   # neutral sp3 amine N (not amide/thioamide/amidine, not aryl-bonded); NH3 handled as a compound
    "prim_amine": "[NX3;H2;!$(N[#6]=[#7,#8,#16]);!$(N-a)][CX4]",
    "sec_amine": "[NX3;H1;!$(N[#6]=[#7,#8,#16]);!$(N-a)]([CX4])[CX4]",
    "tert_amine": "[NX3;H0;!$(N[#6]=[#7,#8,#16]);!$(N-a)]([CX4])([CX4])[CX4]",
    "alcohol": "[OX2H][CX4]", "carboxylic": "[CX3](=O)[OX2H]", "amide": "[CX3](=O)[NX3]"}
PAT = {k: Chem.MolFromSmarts(v) for k, v in GROUPS.items()}

def freesolv_group_errors():
    """Mean ALPB-exp error over FreeSolv molecules whose ONLY heteroatom group is one copy of the group."""
    rows = json.load(open(FREESOLV_ALPB)); out = {}
    for g, pat in PAT.items():
        errs = []
        for r in rows:
            m = Chem.MolFromSmiles(r["smi"])
            if m is None: continue
            hits = len(m.GetSubstructMatches(pat))
            others = sum(len(m.GetSubstructMatches(p)) for k, p in PAT.items() if k != g)
            het = sum(a.GetSymbol() not in ("C", "H") for a in m.GetAtoms())
            if hits == 1 and others == 0 and het == (2 if g in ("carboxylic", "amide") else 1):
                errs.append(r["alpb"] - r["exp"])
        out[g] = {"n": len(errs), "mean": round(float(__import__("numpy").mean(errs)), 2),
                  "sd": round(float(__import__("numpy").std(errs)), 2)}
    nh3 = [r for r in rows if r["smi"] == "N"][0]
    out["NH3"] = {"n": 1, "mean": round(nh3["alpb"] - nh3["exp"], 2), "sd": None}
    return out

def amine_corr(smi, q, ge):
    """Species correction (kJ/mol) = -(FreeSolv mean ALPB error) per neutral amine group; neutral species only."""
    if q != 0: return 0.0
    if smi == "N": return -ge["NH3"]["mean"]
    m = Chem.MolFromSmiles(smi)
    return -sum(len(m.GetSubstructMatches(PAT[g])) * ge[g]["mean"] for g in ("prim_amine", "sec_amine", "tert_amine"))

def reassemble(ge):
    import statistics as st
    from collections import defaultdict
    rep = json.load(open(ROOT / "analysis/accuracy_review/report.json"))["ranked_reactions"]
    rows = []
    for r in rep:
        d = sum(c * amine_corr(s, q, ge) for c, q, s in r["species_scored"].values())
        rows.append((r["reaction"], r["class"], r["production_error"], r["production_error"] + d, d))
    cls = defaultdict(list)
    for x in rows: cls[x[1]].append(x)
    summ = {"overall": {"n_touched": sum(abs(x[4]) > 1e-6 for x in rows),
                        "MAE_before": round(st.mean(abs(x[2]) for x in rows), 3),
                        "MAE_after": round(st.mean(abs(x[3]) for x in rows), 3)}}
    for c, v in cls.items():
        t = [x for x in v if abs(x[4]) > 1e-6]
        if t:
            summ[c] = {"n": len(v), "n_touched": len(t), "bias_before": round(st.mean(x[2] for x in v), 1),
                       "bias_after": round(st.mean(x[3] for x in v), 1),
                       "MAE_before": round(st.mean(abs(x[2]) for x in v), 2),
                       "MAE_after": round(st.mean(abs(x[3]) for x in v), 2)}
    big = sorted([x for x in rows if abs(x[4]) > 1e-6], key=lambda x: -abs(x[4]))
    summ["per_reaction"] = [dict(rid=a, cls=b, err=round(e0, 1), err_new=round(e1, 1), delta=round(dd, 1))
                            for a, b, e0, e1, dd in big]
    return summ

def modelseed_touched(ge):
    import os
    os.environ.setdefault("METAG_CACHE", "/nonexistent")
    sys.path.insert(0, str(ROOT / "src"))
    from metag import pipeline as p
    g = json.load(open(ROOT / "artifacts/results/generality_inputs.json"))
    deltas = {}
    for rid, raw in sorted(g.items()):
        rx = p.route_reaction(raw, log=lambda *a: None)[0]
        d = sum(c * amine_corr(s, q, ge) for c, q, s in rx["species"].values())
        if abs(d) > 1e-6: deltas[rid] = round(d, 1)
    return {"n_reactions": len(g), "n_touched": len(deltas),
            "abs_delta_mean": round(sum(map(abs, deltas.values())) / max(1, len(deltas)), 2), "deltas": deltas}


# ---------------------------------------------------------------- ester / thioester vs Jencks et al. 1960
# Jencks, Cordes & Carriuolo, JBC 235:3608 (1960): hydrolysis of neutral esters to the FREE acid in dilute
# aqueous solution (39 C): O-ester (ethyl acetate etc.) -1.9 kcal/mol, thiol ester (ethyl thiolacetate) -4.4.
JENCKS = {"O-ester": -1.9 * 4.184, "thioester": -4.4 * 4.184}
def ester_panel(d):
    w = d["_water_ref"]; G = lambda k: d[k]["G_prod"]
    hyd = {"ethyl acetate": G("acetic_acid") + G("ethanol") - G("ethyl_acetate") - w,
           "S-ethyl thioacetate": G("acetic_acid") + G("ethanethiol") - G("S-ethyl_thioacetate") - w,
           "S-methyl thioacetate": G("acetic_acid") + G("methanethiol") - G("S-methyl_thioacetate") - w}
    out = {k: {"pred": round(v, 1), "exp_Jencks": round(JENCKS["O-ester" if k == "ethyl acetate" else "thioester"], 1),
               "err": round(v - JENCKS["O-ester" if k == "ethyl acetate" else "thioester"], 1)} for k, v in hyd.items()}
    t = hyd["ethyl acetate"] - hyd["S-ethyl thioacetate"]
    out["thio->O acyl transfer"] = {"pred": round(-t, 1), "exp_Jencks": round(JENCKS["thioester"] - JENCKS["O-ester"], 1),
                                    "err": round(-t - (JENCKS["thioester"] - JENCKS["O-ester"]), 1)}
    return out

# ---------------------------------------------------------------- QM-implied ammonium pKa (cation vs neutral route)
G_HPLUS0 = -26.3 - 1104.5          # pipeline G(H+, aq, 1 M) before the pH term
EXP_PKA = {"[NH4+]": 9.245, "CC[NH3+]": 10.65}   # NIST (Goldberg 2002) / ethylamine 10.65
def qm_pka(cache):
    import glob
    want = {"[NH4+]": ("N", 1), "CC[NH3+]": ("CCN", 1), "C[C@H]([NH3+])C(=O)O": ("C[C@H](N)C(=O)O", 1)}
    G = {}
    for f in glob.glob(str(cache) + "/*.json"):
        c = json.load(open(f)); s = c["settings"]
        if (s.get("solv") == "alpb" and s.get("physics") == "2026-10-01c" and not s.get("via")
                and "acid_hb_filter" not in s):
            G[(c["smi"], c["q"])] = c["G"]
    out = {}
    for cat, (neu, q) in want.items():
        if (cat, q) in G and (neu, 0) in G:
            pka = -(G[(cat, q)] - G[(neu, 0)] - G_HPLUS0) / (8.314e-3 * T * 2.302585)
            out[cat] = {"qm_pka": round(pka, 2), "exp": EXP_PKA.get(cat)}
    return out

if __name__ == "__main__" and len(sys.argv) > 2:
    ge = freesolv_group_errors()
    res = {"independent_panel": main(), "freesolv_alpb_group_error": ge,
           "amine_solvation_correction_TECRDB": reassemble(ge), "amine_correction_ModelSEED": modelseed_touched(ge),
           "ester_thioester_vs_Jencks": ester_panel(json.load(open(sys.argv[1]))),
           "qm_ammonium_pka": qm_pka(ROOT.parent / "MetaG/analysis/sweep_20261001/cache")}
    json.dump(res, open(sys.argv[2], "w"), indent=1)
    print(json.dumps({k: v for k, v in res["amine_solvation_correction_TECRDB"].items() if k != "per_reaction"}, indent=1))
    print(json.dumps({k: v for k, v in res["amine_correction_ModelSEED"].items() if k != "deltas"}))
