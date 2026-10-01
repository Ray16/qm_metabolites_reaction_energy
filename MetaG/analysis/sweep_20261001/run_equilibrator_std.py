"""Run the GENUINE eQuilibrator (component-contribution) on the TECRDB benchmark reactions.

The five-way figure had been reading the cached `thermodynamics.eQuilibrator` field out of the
ModelSEED database JSON -- that field is stale/broken for this comparison (slope 0.20, corr 0.31
vs experiment; hydrolases parked at +97.9). This computes eQuilibrator ΔrG'° for real, from the
ModelSEED stoichiometry via the `seed:` compound namespace, at the standard TECRDB conditions
(pH 7, I = 0.25 M, pMg 3, 298.15 K). Protons are dropped (handled by the Legendre transform);
water is kept. Output cached to eq_real_tecrdb_std.json for the figure to consume.
"""
import json, glob, os
from equilibrator_api import ComponentContribution, Q_

ROOT = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS"
DB = f"{ROOT}/ModelSEEDDatabase/Biochemistry"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eq_real_tecrdb_std.json")

exp = json.load(open(f"{ROOT}/thermodynamic_calc/results/benchmark/tecrdb_full_scored.json"))["experiment_kJ"]
stoich = {}
for f in glob.glob(f"{DB}/reaction_*.json"):
    for r in json.load(open(f)):
        stoich[r["id"]] = r.get("stoichiometry")

cc = ComponentContribution()
cc.p_h = Q_(7.0); cc.ionic_strength = Q_("0M"); cc.p_mg = Q_(14.0); cc.temperature = Q_("298.15K")


def build(rid):
    l, r = [], []
    for s in stoich[rid]:
        cpd, c = s["compound"], s["coefficient"]
        if cpd == "cpd00067":            # proton -> handled by the pH transform, drop it
            continue
        (l if c < 0 else r).append(f"{abs(c)} seed:{cpd}")
    return " + ".join(l) + " = " + " + ".join(r)


res = {}
skipped = {}
for i, rid in enumerate(sorted(exp)):
    if rid not in stoich or not stoich[rid]:
        skipped[rid] = "no stoichiometry"; continue
    try:
        rxn = cc.parse_reaction_formula(build(rid))
        if not rxn.is_balanced():
            skipped[rid] = "unbalanced"; continue
        dg = cc.standard_dg_prime(rxn)
        res[rid] = {"dG_kJ": float(dg.value.m_as("kJ/mol")),
                    "sigma_kJ": float(dg.error.m_as("kJ/mol"))}
    except Exception as e:
        skipped[rid] = str(e)[:120]
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(exp)} done, {len(res)} scored, {len(skipped)} skipped", flush=True)

json.dump({"conditions": {"pH": 7.0, "I_M": 0.0, "pMg": 14.0, "T_K": 298.15},
           "predictions": res, "skipped": skipped}, open(OUT, "w"), indent=1)
print(f"\nscored {len(res)}, skipped {len(skipped)} -> {OUT}")
# quick fidelity report
import numpy as np
K = [k for k in res if k in exp]
E = np.array([exp[k] for k in K]); Q = np.array([res[k]["dG_kJ"] for k in K])
print(f"n={len(K)}  MAE={np.abs(Q-E).mean():.2f}  med={np.median(np.abs(Q-E)):.2f}  "
      f"corr={np.corrcoef(E,Q)[0,1]:.3f}  slope={np.polyfit(E,Q,1)[0]:.3f}")
if skipped:
    from collections import Counter
    print("skip reasons:", Counter(v.split(':')[0][:40] for v in skipped.values()))
