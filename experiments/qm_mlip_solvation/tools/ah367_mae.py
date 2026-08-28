"""Updated full-367 MAE from the aldehyde-hydration correction. For each aldehyde reaction, the pure
hydration delta_dG = Σ coeff*(logged per-species hydration shift) is applied to that reaction's baseline
error (isolates hydration from resampling noise). Non-aldehyde reactions keep their baseline. Reactions
whose ON log isn't finished yet are left at baseline."""
import sys, json, re, os
import numpy as np
sys.path.insert(0, "scripts")

D = json.load(open("scripts/reactions_tecrdb_all.json"))
B = json.load(open("../gnn_dgf/artifacts/per_rxn_benchmark.json"))
ald = set(json.load(open("artifacts/aldehyde_rids.json")))


def delta(rid):
    f = f"logs/ah367_on/{rid}.log"
    if not os.path.exists(f) or "ΔG =" not in open(f).read():
        return None, []
    txt = open(f).read()
    d = 0.0; hits = []
    for m in re.finditer(r"\[hydration: (\S+) carbonyl.*?shift ([+-][0-9.]+)\]", txt):
        nm, sh = m.group(1), float(m.group(2))
        coeff = None
        for sp, (c, q, s) in D[rid]["species"].items():
            if nm == sp or nm == sp + "_t" or nm.replace("_t", "") == sp:
                coeff = c; break
        if coeff is None:
            for sp, (c, q, s) in D[rid]["species"].items():
                if nm[:8] == sp[:8]:
                    coeff = c; break
        if coeff is not None:
            d += coeff * sh; hits.append((nm, coeff, sh))
    return d, hits


old, new, moved, pending = [], [], [], []
for rid, v in B.items():
    if v.get("uma_qm") is None:
        continue
    eo = v["uma_qm_err"]
    en = eo
    if rid in ald:
        dd, hits = delta(rid)
        if dd is None:
            pending.append(rid)
        else:
            en = eo + dd
            if abs(dd) > 0.5:
                moved.append((rid, eo, en, dd, hits))
    old.append(abs(eo)); new.append(abs(en))

print(f"aldehyde reactions: {len(ald)}   ON-complete: {len(ald)-len(pending)}   pending(left at baseline): {len(pending)}")
print(f"FULL-{len(old)} MAE:  baseline {np.mean(old):.3f}  ->  hydration-corrected {np.mean(new):.3f}  "
      f"(Δ {np.mean(old)-np.mean(new):+.3f})")
print(f"median |err|: {np.median(old):.2f} -> {np.median(new):.2f}   |err|>20: {int((np.array(old)>20).sum())} -> {int((np.array(new)>20).sum())}")
print(f"\nreactions moved >0.5 kJ ({len(moved)}), by magnitude:")
for rid, eo, en, dd, hits in sorted(moved, key=lambda x: -abs(x[3])):
    tag = D[rid].get("note", "")[:34]
    better = "better" if abs(en) < abs(eo) - 0.3 else ("worse" if abs(en) > abs(eo) + 0.3 else "flat")
    print(f"  {rid}: err {eo:+6.1f} -> {en:+6.1f}  (ΔdG {dd:+5.1f}) {better:6s} {tag}")
if pending:
    print(f"\npending (rerun to finalize): {pending}")
