"""Merge stale + fresh UMA TECRDB errors into the CURRENT-pipeline set.

The shipped calibration per_reaction err was frozen before this session's fixes, but it is still CURRENT
for every reaction NOT touched by those fixes (verified: unaffected reactions re-score identically). Only
the 29 thioester/CoA reactions changed (truncation guard + adenylylate anchor; O2 count in TECRDB = 0).
So: current_err = re-scored err for the 29 affected, stale calibration err for the rest.

Writes analysis/uma_tecrdb_current.json {rid: err}, and reports how much the aggregate moved.
"""
import json, glob, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
cal = {r["rid"]: r["err"] for r in json.load(open(os.path.join(HERE, "..", "metag", "data", "sigma_class_calibrated.json")))["per_reaction"]}
fresh = {}
for f in glob.glob(os.path.join(HERE, "tecrdb_rescore_results", "*.json")):
    r = json.load(open(f))
    if "error" not in r and r.get("err") is not None:
        fresh[r["reaction"]] = r["err"]
current = dict(cal); current.update(fresh)
json.dump(current, open(os.path.join(HERE, "uma_tecrdb_current.json"), "w"), indent=1)

# aggregate change on the common evaluable set (need GC/eQ later; here just UMA vs its own stale)
stale_vals = np.array([abs(cal[k]) for k in cal])
curr_vals = np.array([abs(current[k]) for k in current])
print(f"merged {len(current)} reactions ({len(fresh)} freshly re-scored)")
print(f"UMA |err| MAE:  stale {stale_vals.mean():.2f}  ->  current {curr_vals.mean():.2f}")
print("\nchanged reactions (|Δerr| > 1):")
for k in sorted(fresh):
    if k in cal and abs(fresh[k] - cal[k]) > 1.0:
        print(f"  {k}: {cal[k]:+.1f} -> {fresh[k]:+.1f}  (Δ {fresh[k]-cal[k]:+.1f})")
