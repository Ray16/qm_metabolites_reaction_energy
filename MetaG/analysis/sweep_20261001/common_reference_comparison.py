"""Review item 9: every method on ONE common reaction set, scored against BOTH references, with the
evaluation regime labelled. MetaG predictions from a results directory (one JSON per reaction, key dG).
  reference A = openTECR standardized (PRIMARY, reported in the paper): each openTECR measurement Legendre-
                transformed to pH 7, I = 0, no Mg (MetaG's estimand); analysis/build_opentecr_standard.py
  reference A' = TECRDB standardized: same transform on the uncorrected NIST TECRDB (SI comparison only)
  reference B = native: median of -RT ln K' at the measurement conditions (what dGPredictor was trained on)
eQuilibrator / group contribution are evaluated at pH 7, I = 0, pMg 14 (*_std.json, matching A) and at their
original pH 7, I = 0.25, pMg 3 (*_real_tecrdb.json)."""
import glob, json, os, statistics, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); TC = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
metag_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "final_20261002_opentecr_calibrated")
ref = lambda name: {k: statistics.median(v["exp"]) for k, v in json.load(open(os.path.join(TC, "experiments/qm_mlip_solvation/scripts", name))).items()}
opn = ref("reactions_opentecr_std.json")
std = ref("reactions_tecrdb_std.json")
nat = json.load(open(os.path.join(TC, "results/benchmark/tecrdb_full_scored.json")))["experiment_kJ"]
pred = {}
pred["MetaG (never fit; policy selected on TECRDB -> development)"] = {json.load(open(f))["reaction"]: json.load(open(f))["dG"] for f in glob.glob(os.path.join(metag_dir, "*.json"))}
ld = lambda p: {k: v["dG_kJ"] for k, v in json.load(open(p))["predictions"].items() if v.get("dG_kJ") is not None and abs(v["dG_kJ"]) < 1e6}
pred["eQuilibrator CC (in-sample, I=0, pMg 14)"] = ld(os.path.join(HERE, "eq_real_tecrdb_std.json"))
pred["eQuilibrator CC (in-sample, I=0.25, pMg 3)"] = ld(os.path.join(HERE, "..", "eq_real_tecrdb.json"))
pred["Group contribution (in-sample, I=0, pMg 14)"] = ld(os.path.join(HERE, "gc_real_tecrdb_std.json"))
pred["dGPredictor original (in-sample, published weights)"] = {k: v["dG_kJ"] for k, v in json.load(open(os.path.join(TC, "results/eq/dgpredictor_full.json"))).items() if v.get("dG_kJ") is not None}
pred["dGPredictor retrained (held-out, family-grouped CV)"] = {k: v["dgp_retrained_heldout"] for k, v in json.load(open(os.path.join(TC, "results/eq/dgp_retrained_heldout.json"))).items()}
pred["dGPredictor original vocab (held-out, family-grouped CV)"] = {k: v["dgp_heldout"] + nat[k] for k, v in json.load(open(os.path.join(TC, "experiments/qm_mlip_solvation/artifacts/heldout_dgp_vs_uma.json"))).items() if k in nat}
common = sorted(set(opn) & set(std) & set(nat) & set.intersection(*[set(p) for p in pred.values()]))
out = {"n_common": len(common), "primary_reference": "openTECR standardized (reactions_opentecr_std.json)", "methods": {}}
for name, p in pred.items():
    e_opn = np.array([p[k] - opn[k] for k in common])
    e_std = np.array([p[k] - std[k] for k in common]); e_nat = np.array([p[k] - nat[k] for k in common])
    out["methods"][name] = {"MAE_vs_openTECR": round(float(np.abs(e_opn).mean()), 2), "median_vs_openTECR": round(float(np.median(np.abs(e_opn))), 2),
                            "MAE_vs_TECRDB_standardized": round(float(np.abs(e_std).mean()), 2), "median_vs_TECRDB_standardized": round(float(np.median(np.abs(e_std))), 2),
                            "MAE_vs_native": round(float(np.abs(e_nat).mean()), 2), "median_vs_native": round(float(np.median(np.abs(e_nat))), 2)}
json.dump(out, open(os.path.join(HERE, "common_reference_comparison.json"), "w"), indent=1)
print("common reaction set n =", len(common))
print("%-62s %10s %10s %10s" % ("method (regime)", "openTECR", "TECRDB std", "native"))
for n, v in out["methods"].items():
    print("%-62s %10.2f %10.2f %10.2f" % (n, v["MAE_vs_openTECR"], v["MAE_vs_TECRDB_standardized"], v["MAE_vs_native"]))
