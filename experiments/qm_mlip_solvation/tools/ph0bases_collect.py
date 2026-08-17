"""Collect the PH0_BASES sweep: new err (logs/ph0bases_sweep) vs pre-fix baseline (routed logs).
Reports per-reaction before/after and the aggregate MAE over the amine-change class + outliers.
"""
import os, re, json, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
pfa = _load("pfa", "ph0_final_analysis.py"); meh = _load("meh", "make_error_histogram.py"); d = pfa.d
SWEEP = os.path.join(HERE, "..", "logs", "ph0bases_sweep")

def new_err(rid):
    p = os.path.join(SWEEP, f"{rid}.log")
    if not os.path.exists(p): return None
    m = re.findall(r"err \[([+-]?\d+\.\d+)\]", open(p, errors="ignore").read())
    return float(m[-1]) if m else None

def main():
    ids = json.load(open(os.path.join(HERE, "..", "scripts", "ph0bases_sweep_ids.json")))
    rows = []
    for rid in ids:
        ne = new_err(rid)
        old = meh.uma_dG(rid)
        oe = (old - d[rid]["exp"][0]) if (old is not None and abs(old) < 200) else None
        rows.append((rid, oe, ne, d[rid]["note"][12:45]))
    done = [r for r in rows if r[2] is not None]
    print(f"done {len(done)}/{len(ids)}")
    both = [r for r in done if r[1] is not None]
    if both:
        oldmae = np.mean([abs(r[1]) for r in both]); newmae = np.mean([abs(r[2]) for r in both])
        print(f"\naggregate over {len(both)} amine-change reactions: MAE {oldmae:.1f} -> {newmae:.1f}")
        improved = sum(1 for r in both if abs(r[2]) < abs(r[1]) - 2)
        worse = sum(1 for r in both if abs(r[2]) > abs(r[1]) + 2)
        print(f"  improved {improved}, worse {worse}, ~same {len(both)-improved-worse}")
    print(f"\n{'rid':10s} {'before':>7s} {'after':>7s}  note")
    for rid, oe, ne, note in sorted(done, key=lambda r: (abs(r[2]) if r[2] else 0), reverse=True):
        ob = f"{oe:+7.1f}" if oe is not None else "   n/a "
        print(f"{rid:10s} {ob} {ne:+7.1f}  {note}")

if __name__ == "__main__":
    main()
