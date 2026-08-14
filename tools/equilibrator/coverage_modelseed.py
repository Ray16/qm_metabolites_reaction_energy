#!/usr/bin/env python3
"""eQuilibrator structure-based ModelSEED coverage (efficient).

For every complete-structure ModelSEED compound (SMILES -> InChIKey via RDKit, no
'*'), decide whether eQ's component-contribution yields a FINITE-uncertainty
formation energy.

Efficiency: the naive path (search_compound_by_inchi_key per compound) costs
~380 ms/call (unindexed SQL LIKE) -> ~3 h. Instead we bulk-load the whole
compound cache ONCE (0.2 s) into an in-memory {inchikey_block -> [internal_id]}
map, then do only cheap primary-key fetches (0.9 ms) + formation tests (~10 ms),
memoized per connectivity block. ~5 min total.

Coverage rule (matches eQ's real behaviour): covered <=> some cache compound with
this InChIKey connectivity block gives standard_dg_formation with finite mu and
sigma_inf_norm < 1e-6 (a nonzero sigma_inf = infinite-uncertainty null component =
eQ cannot estimate it).

Input : scratch/ms_inchikeys.json  {cpd_id: inchikey}   (built in gnndgf env)
Output: data/eq_coverage_modelseed.json
Env   : conda activate eqapi
"""
import os, sqlite3, json, time, numpy as np
from collections import defaultdict
from equilibrator_api import ComponentContribution

SCRATCH = "/tmp/claude-21574/-nfs-lambda-stor-01-homes-rzhu-ModelSEED-FAISS/be47355a-a611-4ec8-8e0e-44ebccd5f7fd/scratchpad"
OUT = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/tools/equilibrator/data/eq_coverage_modelseed.json"
DB = os.path.expanduser("~/.cache/equilibrator/compounds.sqlite")
THRESH = 1e-6

t0 = time.time()
cc = ComponentContribution()
IK = json.load(open(f"{SCRATCH}/ms_inchikeys.json"))

# 1) bulk-load cache once: connectivity block -> [internal ids]
rows = sqlite3.connect(DB).execute(
    "SELECT id, inchi_key FROM compounds WHERE inchi_key IS NOT NULL").fetchall()
block2ids = defaultdict(list)
for i, ik in rows:
    block2ids[ik[:14]].append(i)
print(f"[setup] CC + bulk cache ({len(rows)} compounds, {len(block2ids)} blocks) "
      f"in {time.time()-t0:.0f}s", flush=True)

def estimable(compound):
    try:
        mu, s_fin, s_inf = cc.standard_dg_formation(compound)
    except Exception:
        return False
    if mu is None:
        return False
    if s_inf is None:
        return True
    return float(np.linalg.norm(np.asarray(s_inf, float))) < THRESH

memo = {}  # block -> "covered" / "found_notEst" / "notfound"
def classify(block):
    if block in memo:
        return memo[block]
    ids = block2ids.get(block)
    if not ids:
        r = "notfound"
    else:
        r = "found_notEst"
        for i in ids[:8]:
            c = cc.ccache.get_compound_by_internal_id(i)
            if estimable(c):
                r = "covered"; break
    memo[block] = r
    return r

t = time.time()
cov = notest = notfound = 0
per = {}
for k, (cid, ik) in enumerate(IK.items()):
    r = classify(ik[:14])
    per[cid] = r
    cov += r == "covered"; notest += r == "found_notEst"; notfound += r == "notfound"
    if k % 3000 == 0:
        print(f"{k}/{len(IK)} covered={cov} notEst={notest} notfound={notfound} "
              f"({time.time()-t:.0f}s)", flush=True)

N = len(IK)
summary = dict(n_complete_structures=N, covered=cov, found_not_estimable=notest,
               not_found=notfound, pct_covered=round(100*cov/N, 1),
               unique_blocks_tested=len(memo), thresh_sigma_inf=THRESH,
               runtime_s=round(time.time()-t0))
json.dump({"summary": summary, "per_compound": per}, open(OUT, "w"))
print(json.dumps(summary, indent=2))
print(f"\neQ covers {cov}/{N} = {100*cov/N:.1f}% of complete-structure ModelSEED compounds "
      f"(not in cache {notfound}, in cache but not estimable {notest})")
print(f"total {time.time()-t0:.0f}s -> {OUT}")
