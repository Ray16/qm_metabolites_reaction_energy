"""Carbonyl hydration: UMA + xtb (ALPB / COSMO) vs the cited K_hyd set (khyd_verified.json); fits and
leave-one-out-validates the linear calibration log K_exp = a log K_calc + b used by HYDRATION_CAL.
Reads species G from the sweep cache. Writes khyd_validation.json."""
import json, glob, math, os, sys
import numpy as np
from rdkit import Chem
D = os.path.dirname(os.path.abspath(__file__))
RT = 8.314462618e-3 * 298.15; STD = RT * math.log(24.46); LN10 = math.log(10)
W = json.load(open(os.path.join(D, "water_ref_G_expt.json")))["G"]
can = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
G = {}
for f in glob.glob(os.path.join(D, "cache", "*.json")):
    r = json.load(open(f)); st = r["settings"] if isinstance(r["settings"], dict) else eval(r["settings"])
    if r.get("method") != "implicit" or str(r["q"]) != "0" or st.get("solv_relax") or st.get("acid_hb_filter") \
            or st.get("thermal_ensemble"):
        continue
    G[(can(r["smi"]), st["solv"])] = float(r["G"])
K = {k: v for k, v in json.load(open(os.path.join(D, "khyd_verified.json"))).items() if not k.startswith("_")}
rows = []
for name, (c, h, kexp, ref) in K.items():
    row = {"name": name, "logK_exp": math.log10(kexp), "ref": ref}
    for m in ("alpb", "cosmo"):
        gc, gh = G.get((can(c), m)), G.get((can(h), m))
        row[m] = None if gc is None or gh is None else -(gh - gc - W - STD) / (RT * LN10)
    rows.append(row)
ok = [r for r in rows if r["alpb"] is not None]
y = np.array([r["logK_exp"] for r in ok]); x = np.array([r["alpb"] for r in ok])
xc = np.array([r["cosmo"] for r in ok])
loo = []
for i in range(len(y)):
    m = np.ones(len(y), bool); m[i] = False
    a, b = np.polyfit(x[m], y[m], 1); loo.append(a * x[i] + b - y[i])
a, b = np.polyfit(x, y, 1)
oxo = np.array(["acid" in r["name"] for r in ok])
out = {"n": len(ok), "missing": [r["name"] for r in rows if r["alpb"] is None],
       "MAE_logK": {"cosmo": round(float(np.abs(xc - y).mean()), 2), "alpb_raw": round(float(np.abs(x - y).mean()), 2),
                    "alpb_cal_LOO": round(float(np.mean(np.abs(loo))), 2)},
       "MAE_logK_alpha_oxo_acids": {"alpb_raw": round(float(np.abs(x - y)[oxo].mean()), 2),
                                    "alpb_cal_LOO": round(float(np.abs(np.array(loo))[oxo].mean()), 2)},
       "bias_logK": {"cosmo": round(float((xc - y).mean()), 2), "alpb_raw": round(float((x - y).mean()), 2)},
       "r_alpb": round(float(np.corrcoef(x, y)[0, 1]), 3),
       "calibration": {"a": round(float(a), 3), "b_logK": round(float(b), 3), "b_kJ": round(float(-b * RT * LN10), 3)},
       "rows": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]}
# DEPLOYED calibration = fit on the APPLICATION domain: the carbonyls the pipeline hydrates. alpha-KETO acids
# (ketone carbon bonded to a carboxyl carbon) are not hydrated by the pipeline (aldehyde_hydration._KETO_ACID),
# so they are excluded from the fit; the domain is decided by the same SMARTS the pipeline uses.
sys.path.insert(0, os.path.dirname(os.path.dirname(D)))
from metag.routing import aldehyde_hydration as ah
def in_domain(row):
    c = K[row["name"]][0]
    m = Chem.MolFromSmiles(c)
    return not m.HasSubstructMatch(ah._KETO_ACID)
dom = [r for r in ok if in_domain(r)]
yd = np.array([r["logK_exp"] for r in dom]); xd = np.array([r["alpb"] for r in dom])
lood = []
for i in range(len(yd)):
    m = np.ones(len(yd), bool); m[i] = False
    a_, b_ = np.polyfit(xd[m], yd[m], 1); lood.append(a_ * xd[i] + b_ - yd[i])
ad, bd = np.polyfit(xd, yd, 1)
out["calibration_domain_fit"] = {
    "domain": "carbonyls the pipeline hydrates (aldehydes, ketones, glyoxylic acid); alpha-keto acids excluded",
    "members": [r["name"] for r in dom], "n": len(dom),
    "raw_MAE_logK": round(float(np.abs(xd - yd).mean()), 2), "raw_bias": round(float((xd - yd).mean()), 2),
    "cal_LOO_MAE_logK": round(float(np.mean(np.abs(lood))), 2),
    "cal_LOO_MAE_kJ": round(float(np.mean(np.abs(lood)) * RT * LN10), 2),
    "a": round(float(ad), 3), "b_logK": round(float(bd), 3), "b_kJ_per_event": round(float(-bd * RT * LN10), 3),
    "deployed_HYDRATION_CAL": list(ah.HYDRATION_CAL)}
json.dump(out, open(os.path.join(D, "khyd_validation.json"), "w"), indent=1)
print("DEPLOYED (application-domain) fit:", out["calibration_domain_fit"])
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
for r in out["rows"]: print("  %-20s exp %6.2f  alpb %s  cosmo %s" % (r["name"], r["logK_exp"], r["alpb"], r["cosmo"]))
