"""Ground-truth-free probe of additivity-method generalization on ModelSEED: ModelSEED stores BOTH the
Jankowski group-contribution and the eQuilibrator estimate for the reactions where both exist. Where two
TECRDB-fit additivity methods DISAGREE on ModelSEED is direct evidence of where additivity is unreliable --
and if the disagreement GROWS with structural size/complexity, that is the 'accuracy degrades with novelty'
signal, measured without any experiment."""
import json, glob, re, os
import numpy as np

DB = os.environ.get("MODELSEED_DB",
    "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/ModelSEEDDatabase/Biochemistry")


def heavy_from_formula(f):
    if not f or f in ("null", "None"):
        return 0
    return sum(int(n or 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f) if el and el != "H")


# compound -> heavy-atom count
cpd_heavy = {}
for fp in glob.glob(f"{DB}/compound_*.json"):
    for c in json.load(open(fp)):
        cpd_heavy[c["id"]] = heavy_from_formula(c.get("formula"))

SENT = 1e7
rows = []      # (delta, max_heavy, n_cpd, id, name)
for fp in glob.glob(f"{DB}/reaction_*.json"):
    for r in json.load(open(fp)):
        if r.get("is_obsolete"):
            continue
        t = r.get("thermodynamics") or {}
        gc, eq = t.get("Group contribution"), t.get("eQuilibrator")
        if not (gc and eq):
            continue
        g, e = gc[0], eq[0]
        if abs(g) >= SENT or abs(e) >= SENT:
            continue
        cids = (r.get("compound_ids") or "").split(";")
        mh = max((cpd_heavy.get(c, 0) for c in cids), default=0)
        rows.append((g - e, mh, len(cids), r["id"], r.get("name", "")))

d = np.array([x[0] for x in rows]); mh = np.array([x[1] for x in rows])
print(f"reactions with BOTH GC and eQ estimates: {len(rows)}")
print(f"GC - eQ divergence:  mean|Δ| {np.mean(np.abs(d)):.1f}  median|Δ| {np.median(np.abs(d)):.1f}  "
      f"RMS {np.sqrt(np.mean(d**2)):.1f} kJ/mol")
for thr in (10, 30, 50):
    print(f"  |Δ| > {thr:2d} kJ: {100*np.mean(np.abs(d)>thr):.0f}% of reactions")
print()
print("=== does divergence GROW with molecule size? (max heavy atoms in the reaction) ===")
for lo, hi in [(0, 10), (10, 20), (20, 35), (35, 999)]:
    m = (mh >= lo) & (mh < hi)
    if m.sum():
        print(f"  max-heavy {lo:2d}-{hi:<3d}  n={m.sum():5d}  mean|Δ| {np.mean(np.abs(d[m])):5.1f} kJ/mol")
from scipy.stats import spearmanr
rho, p = spearmanr(np.abs(d), mh)
print(f"  Spearman(|Δ|, max heavy atoms) = {rho:+.2f}  (p={p:.1e})")
print()
print("=== biggest GC-vs-eQ disagreements (where additivity methods can't agree) ===")
for dd, m, n, rid, nm in sorted(rows, key=lambda x: -abs(x[0]))[:8]:
    print(f"  {rid}  Δ={dd:+7.0f}  heavy={m:3d}  {nm[:44]}")
