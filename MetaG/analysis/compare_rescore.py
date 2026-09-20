"""Compare a fresh rescore dir against the current cached results (analysis/tecrdb_rescore_current).
Reports per-reaction before/after err, sigma, z=|err|/sigma, and a verdict (fixed / improved /
regressed / unchanged). Success criterion for the reliability work: fewer confidently-wrong (z>2)
and no new catastrophic (|err|>40) errors, no control regression.

    python analysis/compare_rescore.py analysis/rescore_A [ctrl_id,ctrl_id,...]
"""
import json, glob, os, sys

CUR = "analysis/tecrdb_rescore_current"


def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(f))
        if r.get("err") is None:
            continue
        r["z"] = abs(r["err"]) / max(r.get("sigma_pred", 1) or 1, 1e-6)
        out[r["reaction"]] = r
    return out


def main():
    new_dir = sys.argv[1]
    controls = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else set()
    cur, new = load(CUR), load(new_dir)
    ids = [rid for rid in new if rid in cur]
    print(f"{'rxn':10} {'err0':>7} {'err1':>7} {'d|err|':>7} {'z0':>4} {'z1':>4} {'verdict':10} note")
    n_cw0 = n_cw1 = 0
    for rid in sorted(ids, key=lambda r: -abs(new[r]["err"])):
        a, b = cur[rid], new[rid]
        d = abs(b["err"]) - abs(a["err"])
        cw0, cw1 = a["z"] > 2, b["z"] > 2
        n_cw0 += cw0; n_cw1 += cw1
        if abs(a["err"] - b["err"]) < 0.5:
            v = "unchanged"
        elif d < -3:
            v = "FIXED" if abs(b["err"]) < 20 else "improved"
        elif d > 3:
            v = "REGRESSED"
        else:
            v = "~same"
        tag = " [CTRL]" if rid in controls else ""
        print(f"{rid:10} {a['err']:+7.1f} {b['err']:+7.1f} {d:+7.1f} {a['z']:4.1f} {b['z']:4.1f} "
              f"{v:10} {b.get('note','')[6:46]}{tag}")
    print(f"\nconfidently-wrong (z>2) in this set: before={n_cw0}  after={n_cw1}")
    cat0 = sum(1 for r in ids if abs(cur[r]['err']) > 40)
    cat1 = sum(1 for r in ids if abs(new[r]['err']) > 40)
    print(f"catastrophic (|err|>40)     : before={cat0}  after={cat1}")
    regr = [r for r in ids if r in controls and abs(new[r]['err']) - abs(cur[r]['err']) > 3]
    print(f"control regressions (>3 kJ) : {regr if regr else 'NONE'}")


if __name__ == "__main__":
    main()
