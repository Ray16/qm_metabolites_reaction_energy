"""Selection-adjusted error: nested policy-selection CV over the 2^7 benchmark-selected switches.
Policy grid evaluated with THERMAL_ENSEMBLE=0 (every variant species exists in that cache generation; the thermal estimator is orthogonal to the routing switches).
Inside each training fold the policy (switch combination) with the lowest training MAE is chosen and scored
on the held-out fold. Folds = near-duplicate groups (metag.tools.calibrate.reaction_groups), 5 folds x 20
repeats. Also reports the per-switch leave-one-out sensitivity of the final configuration.
Errors are recomputed from each grid record's dG against the benchmark reference (BENCH_REF, default the
openTECR standardized reference), not taken from the stored err (which was scored against TECRDB)."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from metag.tools.calibrate import reaction_groups
D = os.path.dirname(os.path.abspath(__file__))
FLAGS = ["PH0_ISOMERASE", "NTP_CORE", "CARBONYL_HYDRATION_ALL", "PKA_ENV", "ZWITTERION_PH0", "ARYLAMINE_NONBASIC", "ACID_HB_FILTER"]
FINAL = "1111110"
R = {os.path.basename(f)[2:-5]: json.load(open(f))["res"] for f in glob.glob(os.path.join(D, "policy_grid", "g_*.json"))}
common = sorted(set.intersection(*[set(v) for v in R.values()]))
REF = os.environ.get("BENCH_REF", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(D))), "experiments",
                                               "qm_mlip_solvation", "scripts", "reactions_opentecr_std.json"))
rx = json.load(open(REF))
E = {k: np.array([v[r]["dG"] - rx[r]["exp"][0] for r in common]) for k, v in R.items()}
keys = sorted(E)
groups = reaction_groups({r: rx[r] for r in common})
gid = np.array([groups[r] if isinstance(groups, dict) else groups[i] for i, r in enumerate(common)])
ug = np.unique(gid)
rng = np.random.default_rng(0)
out_sel, out_fix, chosen = [], [], []
for rep in range(20):
    perm = rng.permutation(ug); fold_of = {g: i % 5 for i, g in enumerate(perm)}
    f = np.array([fold_of[g] for g in gid])
    abs_sel = np.empty(len(common)); abs_fix = np.empty(len(common))
    for k in range(5):
        tr, te = f != k, f == k
        best = min(keys, key=lambda c: np.abs(E[c][tr]).mean())
        chosen.append(best)
        abs_sel[te] = np.abs(E[best][te]); abs_fix[te] = np.abs(E[FINAL][te])
    out_sel.append(abs_sel.mean()); out_fix.append(abs_fix.mean())
best_all = min(keys, key=lambda c: np.abs(E[c]).mean())
res = {"reference": os.path.basename(REF), "n_common": len(common), "n_groups": int(len(ug)), "flags": FLAGS,
       "final_policy": FINAL, "final_dev_MAE": round(float(np.abs(E[FINAL]).mean()), 3),
       "all_off_MAE": round(float(np.abs(E["0000000"]).mean()), 3),
       "best_policy_full_data": best_all, "best_full_data_MAE": round(float(np.abs(E[best_all]).mean()), 3),
       "nested_selected_MAE_mean": round(float(np.mean(out_sel)), 3), "nested_selected_MAE_sd": round(float(np.std(out_sel)), 3),
       "chosen_policy_frequency": {c: chosen.count(c) for c in sorted(set(chosen), key=chosen.count, reverse=True)[:8]},
       "sensitivity_flip_one_switch_from_final": {}}
for b, fl in enumerate(FLAGS):
    alt = FINAL[:b] + ("0" if FINAL[b] == "1" else "1") + FINAL[b + 1:]
    d = np.abs(E[alt]) - np.abs(E[FINAL])
    res["sensitivity_flip_one_switch_from_final"][fl] = {
        "final_value": int(FINAL[b]), "MAE_if_flipped": round(float(np.abs(E[alt]).mean()), 3),
        "delta_MAE": round(float(d.mean()), 3), "n_changed_gt1kJ": int((np.abs(E[alt] - E[FINAL]) > 1).sum()),
        "bootstrap95_delta": [round(float(x), 3) for x in np.percentile(
            [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)], [2.5, 97.5])]}
json.dump(res, open(os.path.join(D, "nested_policy_cv.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
