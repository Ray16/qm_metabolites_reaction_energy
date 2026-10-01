"""Run the GENUINE group-contribution method (GCM) on the TECRDB benchmark reactions.

The five-way figure had been reading the cached `thermodynamics."Group contribution"` field out of the
ModelSEED database JSON -- that field is stale/broken for this comparison (slope 0.20, corr 0.38 vs
experiment), the same corruption the cached eQuilibrator field had.

This computes a real group-contribution ΔrG'° from eQuilibrator's trained group model:
  * pure-GC formation energy of a compound  =  (its group-incidence row in params.train_G) . dG0_gc
    -- i.e. every compound is DECOMPOSED INTO GROUPS and estimated from group energies alone, with no
    reactant-contribution term (that is exactly what "group contribution" means, and what component-
    contribution falls back to for novel compounds).
  * reaction chemical ΔG_gc  =  sum_i coeff_i * ΔGf_gc(compound_i)
  * ΔrG'° (transformed) = ΔG_gc + (eQuilibrator's Legendre transform), where the transform
    ΔrG'° - ΔrG (pH 7, I = 0.25 M, pMg 3, 298.15 K) is a property of the compounds' protonation
    states, independent of whether the formation energy came from RC or GC.

Compounds absent from the 631-compound training set (no train_G row) are skipped (undecomposable),
mirroring eQuilibrator's own residual behaviour. Output cached to gc_real_tecrdb_std.json.
"""
import json, glob, os
import numpy as np
from equilibrator_api import ComponentContribution, Q_

ROOT = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS"
DB = f"{ROOT}/ModelSEEDDatabase/Biochemistry"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gc_real_tecrdb_std.json")

exp = json.load(open(f"{ROOT}/thermodynamic_calc/results/benchmark/tecrdb_full_scored.json"))["experiment_kJ"]
stoich = {}
for f in glob.glob(f"{DB}/reaction_*.json"):
    for r in json.load(open(f)):
        stoich[r["id"]] = r.get("stoichiometry")

cc = ComponentContribution()
cc.p_h = Q_(7.0); cc.ionic_strength = Q_("0M"); cc.p_mg = Q_(14.0); cc.temperature = Q_("298.15K")

# pure group-contribution formation energy per TRAINING compound: train_G row . dG0_gc
p = cc.predictor.params
G = p.train_G; dG0_gc = np.asarray(p.dG0_gc).flatten(); Gv = G.values
gcf = {cid: float(Gv[i] @ dG0_gc) for i, cid in enumerate(G.index)}

# fast, covariance-free chemical CC ΔG via the trained estimate vector (x @ mu). Verified to reproduce
# cc.predictor.standard_dg(rxn) exactly (9.08/-799.47/82.55 checks), but ~100x cheaper -- so we only pay
# for ONE covariance evaluation per reaction (standard_dg_prime, needed for the Legendre transform).
pre = cc.predictor.preprocess
mu_vec = np.asarray(pre.mu).flatten()


def chem_dg(rxn):
    x, resid = pre.decompose_reaction(rxn)
    if resid:
        return None
    return float(x @ mu_vec)


def build(rid):
    l, r = [], []
    for s in stoich[rid]:
        if s["compound"] == "cpd00067":
            continue
        (l if s["coefficient"] < 0 else r).append(f"{abs(s['coefficient'])} seed:{s['compound']}")
    return " + ".join(l) + " = " + " + ".join(r)


res, skipped = {}, {}
for i, rid in enumerate(sorted(exp)):
    if rid not in stoich or not stoich[rid]:
        skipped[rid] = "no stoichiometry"; continue
    try:
        rxn = cc.parse_reaction_formula(build(rid))
        if not rxn.is_balanced():
            skipped[rid] = "unbalanced"; continue
        gc_chem, missing = 0.0, False
        for cpd, coef in rxn.items(protons=False):
            if cpd.id not in gcf:
                skipped[rid] = f"compound {cpd.id} not decomposable (no group vector)"; missing = True; break
            gc_chem += coef * gcf[cpd.id]
        if missing:
            continue
        dg_chem = chem_dg(rxn)                                         # CC chemical (fast x@mu, exact)
        if dg_chem is None:
            skipped[rid] = "undecomposable (residual)"; continue
        dg_prime = cc.standard_dg_prime(rxn).value.m_as("kJ/mol")      # official transformed
        res[rid] = {"dG_kJ": gc_chem + (dg_prime - dg_chem)}           # GC + Legendre transform
    except Exception as e:
        skipped[rid] = str(e)[:120]
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(exp)} done, {len(res)} scored, {len(skipped)} skipped", flush=True)

json.dump({"method": "group_contribution (eQuilibrator group model, GC-only + Legendre transform)",
           "conditions": {"pH": 7.0, "I_M": 0.0, "pMg": 14.0, "T_K": 298.15},
           "predictions": res, "skipped": skipped}, open(OUT, "w"), indent=1)
K = [k for k in res if k in exp]
E = np.array([exp[k] for k in K]); P = np.array([res[k]["dG_kJ"] for k in K])
print(f"\nscored {len(res)}, skipped {len(skipped)} -> {OUT}")
print(f"n={len(K)}  MAE={np.abs(P-E).mean():.2f}  med={np.median(np.abs(P-E)):.2f}  "
      f"corr={np.corrcoef(E,P)[0,1]:.3f}  slope={np.polyfit(E,P,1)[0]:.3f}")
if skipped:
    from collections import Counter
    print("skip reasons:", Counter(v.split(':')[0][:40] for v in skipped.values()))
