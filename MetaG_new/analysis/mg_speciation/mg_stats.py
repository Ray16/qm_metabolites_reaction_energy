"""Per-reaction Mg effect on the benchmark reference + residual comparison Mg vs non-Mg (CPU only).
Run: /homes/rzhu/miniforge3/envs/uma/bin/python mg_stats.py   (after mg_decompose.py)"""
import json, os, statistics as st, random
HERE = os.path.dirname(os.path.abspath(__file__))
TC = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
M = json.load(open(os.path.join(HERE, "per_measurement.json")))
REF = json.load(open(os.path.join(TC, "MetaG_new", "src", "metag", "data", "reactions_opentecr_std.json")))
REP = {r["reaction"]: r for r in json.load(open(os.path.join(TC, "MetaG_new", "analysis", "accuracy_review", "report.json")))["ranked_reactions"]}
by = {}
for m in M:
    by.setdefault(m["rid"], []).append(m)
rx = {}
for rid, ms in by.items():
    std = st.median(m["dg_obs"] + m["corr_total"] for m in ms)
    std_noMg = st.median(m["dg_obs"] + m["corr_noMg"] for m in ms)
    pm = [m for m in ms if m["pMg"] is not None]
    unt = [m for m in ms if m["pMg"] is None and m.get("untransformed_mg_upper") is not None]
    txt = [m for m in ms if m["pMg"] is None and (m["mg_text_additional"])]
    e = REP.get(rid, {})
    rx[rid] = dict(n=len(ms), n_pMg=len(pm), frac_pMg=len(pm) / len(ms),
                   pMg_median=st.median(m["pMg"] for m in pm) if pm else None,
                   mg_part_median=st.median(m["mg_part"] for m in pm) if pm else 0.0,
                   ref=std, ref_check=REF[rid]["exp"][0], ref_if_Mg_ignored=std_noMg, mg_shift_of_ref=std - std_noMg,
                   n_untransformed_mg=len(unt), n_mg_text_no_pMg=len(txt),
                   untransformed_upper_median=st.median(m["untransformed_mg_upper"] for m in unt) if unt else None,
                   err=e.get("production_error"), cls=e.get("class"), note=e.get("note", "")[:70],
                   mgprone="Mg-prone" in e.get("note", ""))
json.dump(rx, open(os.path.join(HERE, "per_reaction.json"), "w"), indent=1)
assert all(abs(v["ref"] - v["ref_check"]) < 0.02 for v in rx.values()), "builder not reproduced"


def boot(a, b, f, n=10000, seed=0):
    rnd = random.Random(seed); d = []
    for _ in range(n):
        d.append(f([rnd.choice(a) for _ in a]) - f([rnd.choice(b) for _ in b]))
    d.sort(); return d[int(.025 * n)], d[int(.975 * n)]


def perm(a, b, f, n=10000, seed=1):
    rnd = random.Random(seed); obs = abs(f(a) - f(b)); pool = a + b; k = 0
    for _ in range(n):
        rnd.shuffle(pool); k += abs(f(pool[:len(a)]) - f(pool[len(a):])) >= obs
    return (k + 1) / (n + 1)

sc = {r: v for r, v in rx.items() if v["err"] is not None}
mae = lambda x: st.mean(abs(e) for e in x); mean = st.mean
mg = [v["err"] for v in sc.values() if v["n_pMg"] > 0]; no = [v["err"] for v in sc.values() if v["n_pMg"] == 0]
res = dict(n_scored=len(sc), n_Mg=len(mg), n_noMg=len(no),
           MAE_Mg=mae(mg), MAE_noMg=mae(no), dMAE_CI95=boot(mg, no, mae), dMAE_perm_p=perm(mg, no, mae),
           bias_Mg=mean(mg), bias_noMg=mean(no), dbias_CI95=boot(mg, no, mean), dbias_perm_p=perm(mg, no, mean),
           median_abs_Mg=st.median(abs(e) for e in mg), median_abs_noMg=st.median(abs(e) for e in no))
# would dropping the Mg transform make Mg reactions better or worse?  (diagnostic only -- NOT a selection rule)
mgv = [v for v in sc.values() if v["n_pMg"] > 0]
res["MAE_Mg_if_ref_ignored_Mg"] = mae([v["err"] + v["mg_shift_of_ref"] for v in mgv])
res["mg_shift_of_ref_abs_median"] = st.median(abs(v["mg_shift_of_ref"]) for v in mgv)
res["mg_shift_of_ref_abs_max"] = max(abs(v["mg_shift_of_ref"]) for v in mgv)
res["n_mg_shift_gt5"] = sum(abs(v["mg_shift_of_ref"]) > 5 for v in mgv)
# among Mg-prone-flagged (NTP/PPi) reactions only: measured with Mg vs without
fl = [v for v in sc.values() if v["mgprone"]]
a = [v["err"] for v in fl if v["n_pMg"] > 0]; b = [v["err"] for v in fl if v["n_pMg"] == 0]
res["mgprone"] = dict(n_with_pMg=len(a), n_without=len(b), MAE_with=mae(a) if a else None, MAE_without=mae(b) if b else None,
                      bias_with=mean(a) if a else None, bias_without=mean(b) if b else None,
                      dMAE_CI95=boot(a, b, mae) if a and b else None, dMAE_perm_p=perm(a, b, mae) if a and b else None)
# correlation residual vs applied Mg shift
xs = [v["mg_shift_of_ref"] for v in mgv]; ys = [v["err"] for v in mgv]
mx, my = mean(xs), mean(ys)
res["pearson_err_vs_mgshift"] = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (
    (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** .5)
res["untransformed_Mg_reactions"] = {r: (v["n_untransformed_mg"], v["n"], round(v["untransformed_upper_median"], 2), v["err"])
                                     for r, v in rx.items() if v["n_untransformed_mg"]}
json.dump(res, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
print("\nrid n n_pMg pMg_med mgShiftRef err class note")
for r, v in sorted(rx.items(), key=lambda kv: -abs(kv[1]["mg_shift_of_ref"])):
    if v["n_pMg"] or v["mgprone"]:
        print(r, v["n"], v["n_pMg"], v["pMg_median"], round(v["mg_shift_of_ref"], 2), v["err"], v["cls"], v["note"][:55])
