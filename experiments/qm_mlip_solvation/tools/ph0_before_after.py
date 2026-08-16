"""Before/after table for the pH-0 AUTO correction on the phosphate/NTP class.
Compares the BASELINE implicit-anion ΔG (artifacts/ph0_val/baseline_<rid>.json) against the
pH-0 AUTO ΔG (artifacts/ph0_val/ph0_<rid>.json) vs experiment. Prints the MAE before/after and
per-reaction improvement. No GPU."""
import json, os, glob

HERE = os.path.dirname(__file__)
VAL = os.path.join(HERE, "..", "artifacts", "ph0_val")


def load(path):
    try:
        d = json.load(open(path))
        r = d[0] if isinstance(d, list) else d
        return r.get("dG"), (r.get("exp") or [None])[0]
    except Exception:
        return None, None


rids = sorted({os.path.basename(f).split("_", 1)[1][:-5]
               for f in glob.glob(os.path.join(VAL, "ph0_*.json"))})
print(f"{'rxn':10s} {'exp':>7s} {'baseline':>9s} {'errB':>7s} {'pH0':>8s} {'errA':>7s} {'Δimprove':>9s}")
eB, eA = [], []
for rid in rids:
    dgB, exp = load(os.path.join(VAL, f"baseline_{rid}.json"))
    dgA, expA = load(os.path.join(VAL, f"ph0_{rid}.json"))
    exp = exp if exp is not None else expA
    if dgA is None or exp is None:
        print(f"{rid:10s}  (incomplete)"); continue
    errA = dgA - exp
    eA.append(abs(errA))
    if dgB is not None:
        errB = dgB - exp
        eB.append(abs(errB))
        imp = abs(errB) - abs(errA)
        print(f"{rid:10s} {exp:7.1f} {dgB:9.1f} {errB:+7.1f} {dgA:8.1f} {errA:+7.1f} {imp:+9.1f}")
    else:
        print(f"{rid:10s} {exp:7.1f} {'--':>9s} {'--':>7s} {dgA:8.1f} {errA:+7.1f} {'--':>9s}")

if eB:
    print(f"\nMAE baseline (implicit-anion): {sum(eB)/len(eB):.1f} kJ/mol  (n={len(eB)})")
if eA:
    print(f"MAE pH-0 AUTO:                 {sum(eA)/len(eA):.1f} kJ/mol  (n={len(eA)})")
