#!/usr/bin/env python
"""Phase 1 assessment: ab-initio per-species accuracy vs Du 2018 formation ΔfG.

Ab-initio gives absolute aqueous G; Du gives ΔfG (element-referenced), per charge state,
in J/mol. We charge-match, then fit atom-equivalents:

    G_aq(ai)  =  ΔfG(exp)  +  Σ_e n_e c_e  +  c_z * z  +  c0

The regression residual = per-species ab-initio error with the (systematic) element
reference removed. We report in-sample and leave-one-out MAE, overall and by |charge|.
Element counts come from the actual computed geometry (xyz).
"""
import csv, json, re, os
import numpy as np

ROOT = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS"
TC = f"{ROOT}/thermodynamic_calc"
MATCHED = f"{TC}/data/external/du2018_group_contribution/du2018_formation_matched.csv"
FORMATION = f"{ROOT}/data/organic_cpd_thermo_data.csv"
GAQ = f"{TC}/mlip/G_aq_tecrdb_full.json"
ENS = f"{TC}/pipeline/ensemble_tecrdb_full.json"

# ---- Du dG_f per (compound_id, charge), J/mol -> kJ/mol ----
duG = {}
for r in csv.DictReader(open(FORMATION)):
    if r["data type"] != "dG_f":
        continue
    cid = r["updated_compound_id"].strip() or re.sub(r"_(-?\d+)$", "", r["updated_species_id"])
    try:
        duG[(cid, int(r["charge"]))] = float(r["value"]) / 1000.0
    except ValueError:
        pass

gaq = json.load(open(GAQ))
ens = json.load(open(ENS))

def elem_counts(cpd):
    """element -> count, from the first stored conformer xyz."""
    recs = ens.get(cpd)
    if not recs:
        return None
    xyz = recs[0].get("xyz")
    if not xyz or not os.path.exists(xyz):
        return None
    out = {}
    with open(xyz) as fh:
        lines = fh.read().splitlines()
    n = int(lines[0].split()[0])
    for ln in lines[2:2 + n]:
        el = ln.split()[0]
        out[el] = out.get(el, 0) + 1
    return out

rows = []
for r in csv.DictReader(open(MATCHED)):
    if r["in_abinitio_set"] != "1" or r["has_dG_f"] != "1" or not r["matched_cpd"]:
        continue
    cpd = r["matched_cpd"]
    if cpd not in gaq:
        continue
    z = gaq[cpd]["charge"]
    key = (r["du_compound_id"], z)
    if key not in duG:
        continue                       # require exact charge-state match
    ec = elem_counts(cpd)
    if ec is None:
        continue
    rows.append(dict(cpd=cpd, name=gaq[cpd].get("name", ""), z=z,
                     G_ai=gaq[cpd]["G_aq_kJ"], dGf=duG[key], ec=ec))

print(f"charge-matched species with geometry: {len(rows)}")
elements = sorted({e for r in rows for e in r["ec"]})
print("elements:", elements)

# design matrix: element counts + charge + intercept
X = np.array([[r["ec"].get(e, 0) for e in elements] + [r["z"], 1.0] for r in rows], float)
y = np.array([r["G_ai"] - r["dGf"] for r in rows], float)   # absolute - formation

def fit_predict(Xtr, ytr, Xte):
    coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return Xte @ coef

# in-sample
resid_in = y - fit_predict(X, y, X)
# leave-one-out
resid_loo = np.empty(len(rows))
for i in range(len(rows)):
    m = np.ones(len(rows), bool); m[i] = False
    resid_loo[i] = y[i] - fit_predict(X[m], y[m], X[i:i+1])[0]

def stats(res):
    a = np.abs(res)
    return a.mean(), np.sqrt((res**2).mean()), np.median(a)

print("\n=== per-species ΔfG accuracy (atom-equivalent referenced) ===")
print(f"n = {len(rows)}, params = {X.shape[1]}")
mae, rmse, med = stats(resid_in);  print(f"in-sample : MAE {mae:6.1f}  RMSE {rmse:6.1f}  median {med:6.1f}  kJ/mol")
mae, rmse, med = stats(resid_loo); print(f"LOO-CV    : MAE {mae:6.1f}  RMSE {rmse:6.1f}  median {med:6.1f}  kJ/mol")

print("\n=== LOO MAE by |charge| ===")
az = np.array([abs(r["z"]) for r in rows])
print(f"{'|z|':>4}{'n':>5}{'MAE':>9}")
for zz in sorted(set(az)):
    m = az == zz
    print(f"{zz:>4}{m.sum():>5}{np.abs(resid_loo[m]).mean():>9.1f}")

# worst offenders
order = np.argsort(-np.abs(resid_loo))
print("\n=== worst 12 (LOO residual) ===")
for i in order[:12]:
    r = rows[i]
    print(f"  {r['name'][:26]:26s} z={r['z']:>2}  resid {resid_loo[i]:8.1f} kJ/mol")

# save residuals
with open(f"{TC}/data/external/du2018_group_contribution/assess_dGf_residuals.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["cpd", "name", "charge", "G_ai_kJ", "dGf_exp_kJ", "resid_loo_kJ"])
    for i, r in enumerate(rows):
        w.writerow([r["cpd"], r["name"], r["z"], f"{r['G_ai']:.3f}", f"{r['dGf']:.3f}", f"{resid_loo[i]:.2f}"])
print("\nwrote assess_dGf_residuals.csv")
