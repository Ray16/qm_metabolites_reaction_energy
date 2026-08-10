#!/usr/bin/env python
"""Phase 2 assessment: ab-initio RRHO entropy vs Du 2018 ΔfS.

ab-initio S is gas-phase RRHO (1 atm); Du ΔfS is aqueous, element-referenced (J/K/mol).
So Du_ΔfS(aq) = S_xtb(gas) + [S_solv - Σ n_e S_elem].  We (a) report absolute ab-initio
ΔfS via tabulated element entropies for a direct look, and (b) fit an entropy
atom-equivalent  S_xtb - Du_ΔfS = Σ n_e c_e + c_z z + c0  ; the residual is the per-species
RRHO-entropy accuracy (times 298.15 K -> the entropic part of the ΔG error).
"""
import csv, json, re, os
import numpy as np

ROOT = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS"
TC = f"{ROOT}/thermodynamic_calc"
MATCHED = f"{TC}/data/external/du2018_group_contribution/du2018_formation_matched.csv"
FORMATION = f"{ROOT}/data/organic_cpd_thermo_data.csv"
HS = f"{TC}/mlip/HS_split_tecrdb_full.json"
GAQ = f"{TC}/mlip/G_aq_tecrdb_full.json"
ENS = f"{TC}/pipeline/ensemble_tecrdb_full.json"
T = 298.15
# standard molar entropy of the element in its reference state, per ATOM (J/K/mol)
S_ELEM = {"C": 5.74, "H": 65.34, "O": 102.57, "N": 95.80, "P": 41.09, "S": 32.05}

duS = {}   # (compound_id, charge) -> dS_f  (J/K/mol)
for r in csv.DictReader(open(FORMATION)):
    if r["data type"] != "dS_f":
        continue
    cid = r["updated_compound_id"].strip() or re.sub(r"_(-?\d+)$", "", r["updated_species_id"])
    try:
        duS[(cid, int(r["charge"]))] = float(r["value"])
    except ValueError:
        pass

hs = json.load(open(HS))
gaq = json.load(open(GAQ))
ens = json.load(open(ENS))

def elem_counts(cpd):
    recs = ens.get(cpd)
    if not recs:
        return None
    xyz = recs[0].get("xyz")
    if not xyz or not os.path.exists(xyz):
        return None
    out = {}
    for ln in open(xyz).read().splitlines()[2:]:
        p = ln.split()
        if len(p) >= 4:
            out[p[0]] = out.get(p[0], 0) + 1
    return out

rows = []
for r in csv.DictReader(open(MATCHED)):
    if r["has_dS_f"] != "1" or not r["matched_cpd"]:
        continue
    cpd = r["matched_cpd"]
    if cpd not in hs or "error" in hs[cpd] or cpd not in gaq:
        continue
    z = gaq[cpd]["charge"]
    key = (r["du_compound_id"], z)
    if key not in duS:
        continue
    ec = elem_counts(cpd)
    if ec is None or any(e not in S_ELEM for e in ec):
        continue
    S_ai = hs[cpd]["S_kJ_per_K"] * 1000.0                 # J/K/mol, gas RRHO
    dfS_ai = S_ai - sum(n * S_ELEM[e] for e, n in ec.items())   # absolute ab-initio ΔfS(gas)
    rows.append(dict(cpd=cpd, name=gaq[cpd].get("name", ""), z=z, ec=ec,
                     S_ai=S_ai, dfS_ai=dfS_ai, dfS_exp=duS[key]))

print(f"entropy-matched species: {len(rows)}")
if not rows:
    raise SystemExit("no matched species yet (HS split still running?)")

elements = sorted({e for r in rows for e in r["ec"]})
X = np.array([[r["ec"].get(e, 0) for e in elements] + [r["z"], 1.0] for r in rows], float)
y = np.array([r["S_ai"] - r["dfS_exp"] for r in rows], float)     # J/K/mol

def loo(X, y):
    res = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        c, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        res[i] = y[i] - X[i] @ c
    return res

# (a) direct absolute comparison
dfS_ai = np.array([r["dfS_ai"] for r in rows])
dfS_exp = np.array([r["dfS_exp"] for r in rows])
off = dfS_ai - dfS_exp
print("\n=== absolute ΔfS: ab-initio(gas) vs Du(aq) ===")
print(f"mean signed offset {off.mean():.1f} J/K/mol (= solvation-entropy gap), sd {off.std():.1f}")
print(f"Pearson r = {np.corrcoef(dfS_ai, dfS_exp)[0,1]:.3f}")

# (b) entropy atom-equivalent (removes element ref + mean solvation-entropy)
res = loo(X, y)
S_mae = np.abs(res).mean(); S_med = np.median(np.abs(res))
print("\n=== per-species RRHO-entropy accuracy (element-referenced, LOO) ===")
print(f"n={len(rows)}  MAE {S_mae:.1f}  median {S_med:.1f}  J/K/mol")
print(f"  -> entropic part of ΔG error  T*MAE = {T*S_mae/1000:.1f}  kJ/mol (median {T*S_med/1000:.1f})")

print("\n=== worst 10 (|LOO entropy residual|, J/K/mol) ===")
for i in np.argsort(-np.abs(res))[:10]:
    r = rows[i]
    print(f"  {r['name'][:26]:26s} z={r['z']:>2}  resid {res[i]:8.1f}  (T*resid {T*res[i]/1000:+.1f} kJ)")

with open(f"{TC}/data/external/du2018_group_contribution/assess_dSf_residuals.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["cpd","name","charge","S_ai_JKmol","dfS_ai_JKmol","dfS_exp_JKmol","loo_resid_JKmol"])
    for i, r in enumerate(rows):
        w.writerow([r["cpd"], r["name"], r["z"], f"{r['S_ai']:.2f}", f"{r['dfS_ai']:.2f}",
                    f"{r['dfS_exp']:.2f}", f"{res[i]:.2f}"])
print("\nwrote assess_dSf_residuals.csv")
