"""Three-way UMA vs eQuilibrator vs GC comparison on the representative ModelSEED sample.

NO experiment exists for these -> the honest analysis has two learnable parts:
  (1) CONSENSUS arm (|GC-eQ|<8): where the incumbents AGREE, their mean is a truth-proxy -> |UMA-consensus|
      is an accuracy estimate on TYPICAL reactions. (Caveat: GC & eQ share a redox blind spot, so a few
      consensus reactions are agree-but-both-wrong; those inflate the MEAN -> report the MEDIAN.)
  (2) DIVERGENCE arm (|GC-eQ|>=8): adjudication -- which incumbent does UMA back, per EC class.
Excludes transport reactions / numerical blow-ups (|dG|>1e4 kJ -- not chemistry).
Outputs: rep_summary.tsv + figures/rep_consensus_validation.png + figures/rep_disagreement.png
"""
import json, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
                     "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 15,
                     "figure.dpi": 300, "savefig.dpi": 300})
HERE = os.path.dirname(os.path.abspath(__file__))
INP = json.load(open(os.path.join(HERE, "representative_inputs.json")))
CONSENSUS = 8.0
BLOWUP = 1e4
EC_NAME = {"1": "Oxidoreductase", "2": "Transferase", "3": "Hydrolase", "4": "Lyase",
           "5": "Isomerase", "6": "Ligase"}
COLS = {"1": "#0072B2", "2": "#D55E00", "3": "#009E73", "4": "#CC79A7", "5": "#E69F00", "6": "#56B4E9"}

rows, dropped = [], []
for f in sorted(glob.glob(os.path.join(HERE, "representative_results", "*.json"))):
    r = json.load(open(f))
    if "error" in r:
        continue
    if abs(r["dG"]) > BLOWUP:                              # transport / numerical failure -> not chemistry
        dropped.append(r["reaction"]); continue
    ec = INP.get(r["reaction"], {}).get("note", "").split("EC=")[-1].strip() if r["reaction"] in INP else ""
    rows.append(dict(rid=r["reaction"], ec1=ec.split(".")[0] if ec else "?", uma=r["dG"], gc=r["gc"],
                     eq=r["eq"], cons=(r["gc"] + r["eq"]) / 2, gc_eq=r["gc"] - r["eq"],
                     arm="consensus" if abs(r["gc"] - r["eq"]) < CONSENSUS else "diverge"))
con = [x for x in rows if x["arm"] == "consensus"]
div = [x for x in rows if x["arm"] == "diverge"]

# ---- stats ----
print(f"n={len(rows)} scored ({len(con)} consensus, {len(div)} divergence); dropped {len(dropped)} transport/blowup {dropped}\n")
if con:
    a = np.abs([x["uma"] - x["cons"] for x in con])
    print(f"CONSENSUS ARM  |UMA - consensus|  median={np.median(a):.1f}  MAE={a.mean():.1f} kJ  (n={len(con)})")
    print(f"  within 10 kJ: {int((a<=10).mean()*100)}%   20 kJ: {int((a<=20).mean()*100)}%   30 kJ: {int((a<=30).mean()*100)}%")
if div:
    ngc = sum(1 for x in div if abs(x["uma"]-x["gc"]) < abs(x["uma"]-x["eq"]))
    print(f"DIVERGENCE ARM  UMA backs GC: {ngc}/{len(div)}   eQ: {len(div)-ngc}/{len(div)}")

with open(os.path.join(HERE, "rep_summary.tsv"), "w") as fh:
    fh.write("rxn\tEC\tarm\tUMA\tGC\teQ\tconsensus\tUMA-cons\n")
    for x in sorted(rows, key=lambda z: (z["arm"], z["ec1"])):
        fh.write(f"{x['rid']}\t{x['ec1']}\t{x['arm']}\t{x['uma']:+.1f}\t{x['gc']:+.1f}\t{x['eq']:+.1f}\t{x['cons']:+.1f}\t{x['uma']-x['cons']:+.1f}\n")

# redox reactions share the GC/eQ blind spot -> consensus is NOT truth there (validated on rxn00054 by DFT).
# Split the consensus arm: non-redox = real validation (consensus~truth); redox = called out separately.
def _is_redox(rid):
    v = INP.get(rid, {})
    ec = v.get("note", "").split("EC=")[-1].strip() if "EC=" in v.get("note", "") else ""
    if ec.startswith("1."):
        return True
    smis = " ".join(s[2] for s in v.get("species", {}).values())
    return "O=O" in smis or "OO" in smis
for x in con:
    x["redox"] = _is_redox(x["rid"])
nonredox = [x for x in con if not x["redox"]]
redox = [x for x in con if x["redox"]]

# NOTE: the consensus-vs-UMA scatter was RETIRED (deleted) -- it put the GC/eQ consensus on an axis as if
# it were ground truth, which it is not (GC & eQ share blind spots). Agreement with the incumbents is not
# validation. Real accuracy lives in analysis/reference_scoreboard.py (error vs experiment/DFT). The stats
# above (non-redox median, redox divergence) are kept only as DESCRIPTIVE agreement, not a truth claim.

# ---- FIG: pairwise disagreement distribution (DESCRIPTIVE: how far apart are the three methods) ----
if rows:
    fig, ax = plt.subplots(figsize=(8, 6))
    data = [np.abs([x["uma"]-x["gc"] for x in rows]),
            np.abs([x["uma"]-x["eq"] for x in rows]),
            np.abs([x["gc"]-x["eq"] for x in rows])]
    labels = ["|UMA − GC|", "|UMA − eQ|", "|GC − eQ|"]
    bp = ax.boxplot(data, vert=True, showfliers=False, widths=0.6, patch_artist=True,
                    medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], ["#009E73", "#D55E00", "#0072B2"]):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    for i, d in enumerate(data, 1):
        ax.scatter(np.random.normal(i, 0.05, len(d)), d, s=18, color="0.3", alpha=0.5, zorder=3)
        ax.text(i, np.median(d), f" {np.median(d):.0f}", va="center", ha="left", fontsize=15)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(labels)
    ax.set_ylabel("pairwise |ΔΔrG′°|  (kJ/mol)")
    ax.set_ylim(0, min(200, max(np.percentile(np.concatenate(data), 95), 60)))
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "figures", "rep_disagreement.png"), bbox_inches="tight")
    print("wrote figures/rep_disagreement.png (y clipped to 95th pctile for readability)")
