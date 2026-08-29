"""Three-way comparison of UMA (MetaG) vs eQuilibrator vs Group-Contribution on the scored ModelSEED
reactions. NO experimental ground truth exists for these (0/4545 TECRDB substrate-level matches), so this
is an ADJUDICATION landscape, not an accuracy ranking: where the two incumbents disagree, which one does
the first-principles method back? Plus the anchor-validated reactions where we DO know UMA is right.

Outputs: analysis/three_way_comparison.tsv + figures/three_way_comparison.png
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- project figure conventions: 18pt floor, dpi 300, tight bbox, no gray footnotes ---
plt.rcParams.update({"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
                     "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
                     "figure.dpi": 300, "savefig.dpi": 300})

HERE = os.path.dirname(os.path.abspath(__file__))
INP = json.load(open(os.path.join(HERE, "combined_inputs.json")))
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "combined_results", "*.json"))):
    r = json.load(open(f))
    if "error" in r:
        continue
    rid = r["reaction"]
    ec = ""
    note = INP.get(rid, {}).get("note", "")
    if "EC=" in note:
        ec = note.split("EC=")[-1].strip()
    rows.append(dict(rid=rid, ec=ec, uma=r["dG"], gc=r["gc"], eq=r["eq"],
                     u_gc=r["dG"] - r["gc"], u_eq=r["dG"] - r["eq"], gc_eq=r["gc"] - r["eq"]))

# anchor-validated / UMA-validated-correct reactions (independent evidence UMA is right)
VALIDATED = {"rxn00024", "rxn07602", "rxn03854", "rxn04034",       # redox/epoxide: real thermochemistry
             "rxn00226", "rxn00418"}                                # adenylylate: ligase-cycle anchor

# ---- table ----
rows.sort(key=lambda x: abs(x["gc_eq"]), reverse=True)
tsv = os.path.join(HERE, "three_way_comparison.tsv")
with open(tsv, "w") as fh:
    fh.write("rxn\tEC\tUMA\tGC\teQ\tUMA-GC\tUMA-eQ\tGC-eQ\tcloser\tvalidated\n")
    for x in rows:
        closer = "GC" if abs(x["u_gc"]) < abs(x["u_eq"]) else "eQ"
        fh.write(f"{x['rid']}\t{x['ec']}\t{x['uma']:+.1f}\t{x['gc']:+.1f}\t{x['eq']:+.1f}\t"
                 f"{x['u_gc']:+.1f}\t{x['u_eq']:+.1f}\t{x['gc_eq']:+.1f}\t{closer}\t"
                 f"{'yes' if x['rid'] in VALIDATED else ''}\n")

# ---- summary ----
n = len(rows)
closer_gc = sum(1 for x in rows if abs(x["u_gc"]) < abs(x["u_eq"]))
med_ugc = np.median([abs(x["u_gc"]) for x in rows])
med_ueq = np.median([abs(x["u_eq"]) for x in rows])
med_gceq = np.median([abs(x["gc_eq"]) for x in rows])
print(f"n={n} reactions (no experimental ground truth; adjudication landscape)")
print(f"  UMA closer to GC: {closer_gc}/{n}   closer to eQ: {n-closer_gc}/{n}")
print(f"  median |UMA-GC|={med_ugc:.1f}  |UMA-eQ|={med_ueq:.1f}  |GC-eQ|={med_gceq:.1f} kJ/mol")
print(f"  (the incumbents disagree by a median {med_gceq:.0f} kJ -- at least one is wrong on each)")

# ---- figure: per-reaction dumbbell, sorted by |GC-eQ| ----
rows_plot = sorted(rows, key=lambda x: x["gc_eq"])
y = np.arange(len(rows_plot))
fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows_plot) + 2))
for i, x in enumerate(rows_plot):
    ax.plot([x["gc"], x["eq"]], [i, i], "-", color="#bbbbbb", lw=2, zorder=1)
ax.scatter([x["gc"] for x in rows_plot], y, s=90, color="#0072B2", label="Group Contribution", zorder=3)
ax.scatter([x["eq"] for x in rows_plot], y, s=90, color="#D55E00", label="eQuilibrator", zorder=3)
ax.scatter([x["uma"] for x in rows_plot], y, s=140, marker="D", color="#009E73",
           edgecolor="black", linewidth=0.8, label="UMA (MetaG)", zorder=4)
# mark the anchor/thermochemistry-validated reactions
for i, x in enumerate(rows_plot):
    if x["rid"] in VALIDATED:
        ax.scatter([x["uma"]], [i], s=300, facecolors="none", edgecolors="#009E73", linewidth=2.2, zorder=2)
ax.set_yticks(y)
ax.set_yticklabels([f"{x['rid']} {x['ec']}" for x in rows_plot])
ax.set_xlabel("ΔrG′°  (kJ/mol)")
ax.axvline(0, color="black", lw=0.8, alpha=0.4)
ax.legend(loc="lower left", framealpha=0.95)
ax.margins(y=0.01)
fig.tight_layout()
out = os.path.join(HERE, "..", "figures", "three_way_comparison.png")
fig.savefig(out, bbox_inches="tight")
print(f"\nwrote {tsv}\nwrote {os.path.abspath(out)}  (circled UMA = independently validated correct)")
