"""Priority item 6: state-function (cycle-closure) check of MetaG on TECRDB, by route and by metabolite.
Residual r_j = ΔG_j - Sᵀf̂ (part of ΔG no compound-energy assignment explains) for MetaG and, as the baseline,
for the experimental values themselves (measurements do not close exactly either). σ = 1 kJ for all (a
common unit) so the comparison is in kJ, not in sampling-noise units. Usage: cycle_closure_by_route.py RESULTS_DIR"""
import glob, json, os, statistics, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from metag.tools.cycle_closure import closure_report, compound_key, stoich_matrix
TC = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
rx = json.load(open(os.path.join(TC, "experiments/qm_mlip_solvation/scripts/reactions_tecrdb_std.json")))
R = {}
for f in glob.glob(os.path.join(sys.argv[1], "*.json")):
    r = json.load(open(f))
    if r.get("dG") is not None: R[r["reaction"]] = r
rids = sorted(R)
def recs(val):
    return [{"rid": k, "species": rx[k]["species"], "dG": val(k), "sigma": 1.0} for k in rids]
rep_m = closure_report(recs(lambda k: R[k]["dG"]), max_cycles_listed=400)
rep_e = closure_report(recs(lambda k: statistics.median(rx[k]["exp"])), max_cycles_listed=400)
res_m = {d["rid"]: d["residual"] for d in rep_m["worst_reactions"]}
res_e = {d["rid"]: d["residual"] for d in rep_e["worst_reactions"]}
def route(k):
    rt = R[k].get("routes", {}); st = R[k].get("stages", {})
    tags = [t for t in ("cofactor_ring", "ntp_core", "truncated", "ph0") if rt.get(t)]
    return "+".join(tags) or "full"
by = collections.defaultdict(list)
for k in res_m: by[route(k)].append((res_m[k], res_e.get(k, 0.0)))
out = {"MetaG": {k: rep_m[k] for k in ("n_reactions", "n_compounds", "dof", "n_in_cycles", "rms_residual", "n_cycles")},
       "experiment": {k: rep_e[k] for k in ("n_reactions", "n_compounds", "dof", "n_in_cycles", "rms_residual", "n_cycles")},
       "rms_residual_by_route": {r: {"n": len(v), "MetaG": round(float(np.sqrt(np.mean([a * a for a, _ in v]))), 2),
                                     "experiment": round(float(np.sqrt(np.mean([b * b for _, b in v]))), 2)}
                                 for r, v in sorted(by.items(), key=lambda x: -len(x[1]))},
       "worst_cycles_MetaG": (rep_m.get("worst_cycles") or [])[:15]}
# metabolites on the worst-closing reactions (excess over experimental residual)
cnt = collections.Counter()
for k in res_m:
    if abs(res_m[k]) - abs(res_e.get(k, 0.0)) > 5:
        for nm, (c, q, s) in rx[k]["species"].items():
            if s != "O": cnt[nm] += 1
out["metabolites_on_excess_nonclosure"] = cnt.most_common(15)
json.dump(out, open(os.path.join(HERE, "cycle_closure_by_route.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "worst_cycles_MetaG"}, indent=1))
