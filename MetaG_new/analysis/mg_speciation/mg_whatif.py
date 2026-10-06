"""Sensitivity (NOT a correction): (a) for reactions whose measurements carry NO pMg, how far would the reference
move if the assay had in fact contained free Mg2+ at pMg 3 (typical kinase assay; Alberty 2003 uses pMg 3 as the
physiological reference)?  (b) the effect of excluding rows outside the extended Debye-Hueckel validity range
(c(Mg2+) > 0.1 M, i.e. pMg < 1, I_Mg alone >= 0.3 M; Alberty: extended DH valid to I ~ 0.35 M) -- a validity
criterion fixed a priori, reported for every affected reaction regardless of the direction of the change.
Run: ~/miniforge3/envs/eqapi/bin/python mg_whatif.py"""
import json, os, statistics as st, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
TC = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(TC, "MetaG", "analysis"))
import build_tecrdb_standard as B
M = json.load(open(os.path.join(HERE, "per_measurement.json")))
RX = json.load(open(os.path.join(HERE, "per_reaction.json")))
tr = B.Transformer(B.load_modelseed_reactions(B.MSDB))
by = defaultdict(list)
for m in M:
    by[m["rid"]].append(m)
a = {}
for rid, v in RX.items():
    if v["n_pMg"] == 0 and v["mgprone"]:
        ms = by[rid]
        base = st.median(m["dg_obs"] + m["corr_total"] for m in ms)
        alt = st.median(m["dg_obs"] + tr.correction(rid, m["pH"], 0.25 if m["I"] is None else m["I"], 3.0, m["T"]) for m in ms)
        a[rid] = dict(ref_shift_if_pMg3=round(alt - base, 2), err=v["err"], note=v["note"][:60])
b = {}
for rid, ms in by.items():
    bad = [m for m in ms if m["pMg"] is not None and m["pMg"] < 1]
    if bad:
        keep = [m["dg_obs"] + m["corr_total"] for m in ms if not (m["pMg"] is not None and m["pMg"] < 1)]
        base = st.median(m["dg_obs"] + m["corr_total"] for m in ms)
        b[rid] = dict(n_excluded=len(bad), n=len(ms), ref_shift=round(st.median(keep) - base, 2) if keep else None, err=RX[rid]["err"])
json.dump(dict(no_pMg_whatif_pMg3=a, exclude_pMg_lt_1=b), open(os.path.join(HERE, "whatif.json"), "w"), indent=1)
for r, v in sorted(a.items(), key=lambda kv: -abs(kv[1]["err"]))[:20]:
    print(r, v)
s = [abs(v["ref_shift_if_pMg3"]) for v in a.values()]
print("n", len(s), "median|shift|", st.median(s), "max", max(s), "n>5", sum(x > 5 for x in s))
print(json.dumps(b, indent=0))
