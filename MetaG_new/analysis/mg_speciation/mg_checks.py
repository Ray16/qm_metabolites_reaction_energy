"""Follow-up checks (CPU): (1) within-reaction consistency of Mg-transformed vs Mg-free measurements (tests the
eQuilibrator Mg constants on the data itself, no QM involved); (2) class-stratified residual bias Mg vs non-Mg;
(3) effect on the reference median of Mg-containing rows that carry no p_mg (untransformed), upper bound;
(4) mixed-solvent (cosolvent) rows that the builder keeps as aqueous.
Run: /homes/rzhu/miniforge3/envs/uma/bin/python mg_checks.py"""
import json, os, re, statistics as st, random
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, "per_measurement.json")))
RX = json.load(open(os.path.join(HERE, "per_reaction.json")))
by = defaultdict(list)
for m in M:
    by[m["rid"]].append(m)
out = {}
# (1)
cons = []
for rid, ms in by.items():
    a = [m["dg_obs"] + m["corr_total"] for m in ms if m["pMg"] is not None and m["pMg"] < 6]
    b = [m["dg_obs"] + m["corr_total"] for m in ms if m["pMg"] is None and not m["mg_text_additional"]]
    if len(a) >= 2 and len(b) >= 2:
        an = [m["dg_obs"] + m["corr_noMg"] for m in ms if m["pMg"] is not None and m["pMg"] < 6]
        cons.append(dict(rid=rid, n_mg=len(a), n_free=len(b), d_transformed=st.median(a) - st.median(b),
                         d_if_Mg_ignored=st.median(an) - st.median(b)))
out["within_reaction_consistency"] = cons
out["consistency_summary"] = dict(n=len(cons), median_abs_d_transformed=st.median(abs(c["d_transformed"]) for c in cons),
                                  median_abs_d_if_Mg_ignored=st.median(abs(c["d_if_Mg_ignored"]) for c in cons)) if cons else None
# (2) class-stratified: within classes having both Mg and non-Mg reactions, mean (err_Mg - err_noMg), weighted by min n
cls = defaultdict(lambda: ([], []))
for rid, v in RX.items():
    if v["err"] is None: continue
    cls[v["cls"]][0 if v["n_pMg"] else 1].append(v["err"])
strata = {c: (len(a), len(b), st.mean(a) - st.mean(b)) for c, (a, b) in cls.items() if a and b}
w = {c: 1 / (1 / s[0] + 1 / s[1]) for c, s in strata.items()}
def strat_diff(cl):
    ww = {c: 1 / (1 / len(a) + 1 / len(b)) for c, (a, b) in cl.items() if a and b}
    return sum(ww[c] * (st.mean(cl[c][0]) - st.mean(cl[c][1])) for c in ww) / sum(ww.values())
obs = strat_diff(cls)
rnd = random.Random(0); k = 0; N = 10000                              # permutation within class
for _ in range(N):
    pc = {}
    for c, (a, b) in cls.items():
        pool = a + b; rnd.shuffle(pool); pc[c] = (pool[:len(a)], pool[len(a):])
    k += abs(strat_diff(pc)) >= abs(obs)
out["class_stratified_bias_diff"] = dict(value=obs, perm_p=(k + 1) / (N + 1), strata={c: dict(n_mg=s[0], n_free=s[1], diff=round(s[2], 2)) for c, s in strata.items()})
# (3) untransformed Mg, upper bound effect on reference medians
unt = {}
for rid, ms in by.items():
    if any(m.get("untransformed_mg_upper") is not None for m in ms):
        base = st.median(m["dg_obs"] + m["corr_total"] for m in ms)
        upd = st.median(m["dg_obs"] + m["corr_total"] + (m.get("untransformed_mg_upper") or 0) for m in ms)
        unt[rid] = dict(n_rows=sum(m.get("untransformed_mg_upper") is not None for m in ms), n=len(ms),
                        ref_shift_upper=round(upd - base, 2), err=RX[rid]["err"])
out["untransformed_Mg_ref_shift_upper_bound"] = unt
# (4) cosolvent rows
cos = {}
for rid, ms in by.items():
    c = [m for m in ms if "cosolvent" in m["additional"]]
    if c:
        base = st.median(m["dg_obs"] + m["corr_total"] for m in ms)
        rest = [m["dg_obs"] + m["corr_total"] for m in ms if "cosolvent" not in m["additional"]]
        cos[rid] = dict(n_cosolvent=len(c), n=len(ms), ref=round(base, 2),
                        ref_aqueous_only=round(st.median(rest), 2) if rest else None, err=RX[rid]["err"])
out["cosolvent_rows"] = cos
json.dump(out, open(os.path.join(HERE, "checks.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "within_reaction_consistency"}, indent=1))
for c in sorted(cons, key=lambda c: -abs(c["d_transformed"])):
    print(c["rid"], c["n_mg"], c["n_free"], round(c["d_transformed"], 2), round(c["d_if_Mg_ignored"], 2), RX[c["rid"]]["note"][:40])
