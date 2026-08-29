"""Analyse the representative ModelSEED sample: UMA (MetaG) vs eQuilibrator vs GC, two arms.

CONSENSUS arm (|GC-eQ|<8): the incumbents agree -> their mean is our best proxy for truth. |UMA-consensus|
  is a generalization test on TYPICAL ModelSEED reactions (the closest thing to accuracy we can get with
  no experiment). This is the key validation the curated divergent set could not give.
DIVERGENCE arm (|GC-eQ|>=8): the incumbents disagree -> adjudication: which does UMA back, per EC class.

Outputs: analysis/rep_summary.tsv + figures/rep_consensus_validation.png + figures/rep_adjudication.png
"""
import json, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
                     "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
                     "figure.dpi": 300, "savefig.dpi": 300})
HERE = os.path.dirname(os.path.abspath(__file__))
INP = json.load(open(os.path.join(HERE, "representative_inputs.json")))
CONSENSUS = 8.0

rows = []
for f in sorted(glob.glob(os.path.join(HERE, "representative_results", "*.json"))):
    r = json.load(open(f))
    if "error" in r:
        continue
    rid = r["reaction"]
    ec = INP.get(rid, {}).get("note", "").split("EC=")[-1].strip() if rid in INP else ""
    gc, eq, uma = r["gc"], r["eq"], r["dG"]
    rows.append(dict(rid=rid, ec1=ec.split(".")[0] if ec else "?", uma=uma, gc=gc, eq=eq,
                     cons=(gc + eq) / 2.0, gc_eq=gc - eq,
                     arm="consensus" if abs(gc - eq) < CONSENSUS else "diverge"))

con = [x for x in rows if x["arm"] == "consensus"]
div = [x for x in rows if x["arm"] == "diverge"]
EC_NAME = {"1": "Oxidoreductase", "2": "Transferase", "3": "Hydrolase", "4": "Lyase",
           "5": "Isomerase", "6": "Ligase"}

# ---- CONSENSUS: |UMA - consensus| (validation) ----
def stats(xs):
    a = np.abs(xs)
    return (np.mean(a), np.median(a), np.sqrt(np.mean(a**2))) if len(a) else (float("nan"),)*3
print(f"n={len(rows)} scored ({len(con)} consensus, {len(div)} divergence)\n")
if con:
    d_uma = np.array([x["uma"] - x["cons"] for x in con])
    mae, med, rmse = stats(d_uma)
    print("CONSENSUS ARM -- |UMA - incumbent-consensus| (proxy accuracy on typical reactions):")
    print(f"  MAE={mae:.1f}  median={med:.1f}  RMSE={rmse:.1f} kJ/mol  (n={len(con)})")
    within = {k: int(np.mean(np.abs(d_uma) <= k) * 100) for k in (10, 20, 30)}
    print(f"  within 10 kJ: {within[10]}%   20 kJ: {within[20]}%   30 kJ: {within[30]}%")
    for c in sorted(EC_NAME):
        sub = [x["uma"] - x["cons"] for x in con if x["ec1"] == c]
        if sub:
            print(f"    EC{c} {EC_NAME[c]:15s} MAE={np.mean(np.abs(sub)):5.1f} (n={len(sub)})")

# ---- DIVERGENCE: which incumbent does UMA back ----
if div:
    ngc = sum(1 for x in div if abs(x["uma"] - x["gc"]) < abs(x["uma"] - x["eq"]))
    print(f"\nDIVERGENCE ARM -- UMA adjudication (n={len(div)}):")
    print(f"  closer to GC: {ngc}/{len(div)}   closer to eQ: {len(div)-ngc}/{len(div)}")
    for c in sorted(EC_NAME):
        sub = [x for x in div if x["ec1"] == c]
        if sub:
            g = sum(1 for x in sub if abs(x["uma"]-x["gc"]) < abs(x["uma"]-x["eq"]))
            print(f"    EC{c} {EC_NAME[c]:15s} GC:{g} eQ:{len(sub)-g}")

# ---- table ----
with open(os.path.join(HERE, "rep_summary.tsv"), "w") as fh:
    fh.write("rxn\tEC\tarm\tUMA\tGC\teQ\tconsensus\tUMA-cons\tUMA-GC\tUMA-eQ\n")
    for x in sorted(rows, key=lambda z: (z["arm"], z["ec1"])):
        fh.write(f"{x['rid']}\t{x['ec1']}\t{x['arm']}\t{x['uma']:+.1f}\t{x['gc']:+.1f}\t{x['eq']:+.1f}\t"
                 f"{x['cons']:+.1f}\t{x['uma']-x['cons']:+.1f}\t{x['uma']-x['gc']:+.1f}\t{x['uma']-x['eq']:+.1f}\n")

# ---- figure 1: consensus validation scatter (UMA vs consensus) ----
if con:
    fig, ax = plt.subplots(figsize=(8, 8))
    xs = [x["cons"] for x in con]; ys = [x["uma"] for x in con]
    lim = [min(xs+ys)-20, max(xs+ys)+20]
    ax.plot(lim, lim, "--", color="black", lw=1.2, alpha=0.6)
    cols = {"1": "#0072B2", "2": "#D55E00", "3": "#009E73", "4": "#CC79A7", "5": "#E69F00", "6": "#56B4E9"}
    for c in sorted(EC_NAME):
        p = [x for x in con if x["ec1"] == c]
        if p:
            ax.scatter([x["cons"] for x in p], [x["uma"] for x in p], s=80, color=cols[c],
                       label=f"EC{c}", edgecolor="black", linewidth=0.4)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("incumbent consensus ΔrG′°  (kJ/mol)")
    ax.set_ylabel("UMA ΔrG′°  (kJ/mol)")
    ax.legend(loc="upper left", ncol=2, framealpha=0.95)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "figures", "rep_consensus_validation.png"), bbox_inches="tight")
    print("\nwrote figures/rep_consensus_validation.png")
print("wrote analysis/rep_summary.tsv")
