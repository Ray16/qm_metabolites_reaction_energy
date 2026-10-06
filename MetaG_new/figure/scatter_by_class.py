"""Parity scatter (MetaG vs openTECR) coloured by reaction class.

Writes two figures next to this script:
  scatter_by_family.png      one scatter, 17 classes grouped into 9 chemical families + other
  scatter_by_class_grid.png  one small panel per class (class highlighted, others grey)
Reads the same frozen artifacts as analysis/figures/make_results_figure.py (panel a).
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
RESULTS = os.environ.get(
    "METAG_RESULTS", os.path.join(PKG, "artifacts", "results", "metag_opentecr_calibrated"))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 18, "axes.titlesize": 18,
    "axes.labelsize": 18, "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 18, "figure.dpi": 300, "savefig.dpi": 300,
})

# class -> chemical family (Okabe-Ito colours; grey for the unclassified bucket)
FAMILIES = [
    ("NAD(P)/flavin redox", "#0072B2", ["NAD(P)-redox(other)", "flavin/FAD-redox"]),
    ("Reductive amination", "#56B4E9", ["reductive-amination-DH"]),
    ("Phosphoryl transfer", "#D55E00", ["kinase/phosphotransfer", "phosphatase", "carboxyP"]),
    ("Phosphagen (P-N)", "#CC0000", ["phosphagen(P-N/Mg)"]),
    ("CoA thioester", "#E69F00", ["CoA-thioester"]),
    ("Glycosyl/PRT/nucleoside", "#CC79A7", ["glycosyl/PRT/nucleoside"]),
    ("Lyase (hydratase, aldolase, NH3-lyase)", "#009E73",
     ["hydratase", "aldolase", "ammonia-lyase"]),
    ("Group transfer / hydrolysis", "#F0E442",
     ["transaminase", "carbamoyltransfer", "amide/amidine-hydrolysis"]),
    ("Isomerase/mutase", "#000000", ["isomerase/mutase"]),
    ("Other", "#BBBBBB", ["other/clean"]),
]
CLASS_LABEL = {"NAD(P)-redox(other)": "NAD(P)-redox", "glycosyl/PRT/nucleoside": "glycosyl/PRT",
               "amide/amidine-hydrolysis": "amide hydrolysis", "other/clean": "other"}

records = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RESULTS, "*.json")))]
exp = np.array([r["exp"] for r in records])
pred = np.array([r["dG"] for r in records])
cls = np.array([r["class"] for r in records])
known = {c for _, _, cs in FAMILIES for c in cs}
assert set(cls) <= known, f"unmapped classes: {set(cls) - known}"
lo, hi = min(exp.min(), pred.min()) - 5, max(exp.max(), pred.max()) + 5


def frame(ax):
    ax.plot([lo, hi], [lo, hi], color="#333333", linestyle="--", linewidth=1, zorder=0)
    ax.set(xlim=(lo, hi), ylim=(lo, hi))
    ax.set_aspect("equal")


# ---- 1. single scatter, coloured by family --------------------------------
fig, ax = plt.subplots(figsize=(17, 10))
frame(ax)
# draw "Other" first (underneath), then families from largest to smallest so small ones stay visible
order = [FAMILIES[-1]] + sorted(FAMILIES[:-1], key=lambda f: -np.isin(cls, f[2]).sum())
for name, color, members in order:
    m = np.isin(cls, members)
    err = np.abs(pred[m] - exp[m])
    edge = "#666666" if color in ("#F0E442", "#BBBBBB") else "none"
    ax.scatter(exp[m], pred[m], s=70, color=color, alpha=0.85, edgecolor=edge, linewidth=0.6,
               label=f"{name} (n={m.sum()}, MAE {err.mean():.1f})")
ax.set_xlabel(r"openTECR $\Delta_rG'^\circ$, standardized (kJ mol$^{-1}$)")
ax.set_ylabel(r"MetaG $\Delta_rG'^\circ$ (kJ mol$^{-1}$)")
ax.text(0.04, 0.96, f"MAE {np.mean(np.abs(pred-exp)):.2f}\nmedian "
        f"{np.median(np.abs(pred-exp)):.2f}\nn={len(records)}", transform=ax.transAxes, va="top")
handles, labels = ax.get_legend_handles_labels()
fam_order = [f[0] for f in FAMILIES]
pairs = sorted(zip(handles, labels), key=lambda hl: fam_order.index(hl[1].split(" (n=")[0]))
ax.legend(*zip(*pairs), loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
          markerscale=1.4, handletextpad=0.3)
fig.savefig(os.path.join(HERE, "scatter_by_family.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ---- 2. small multiples, one panel per class ------------------------------
classes = sorted(set(cls), key=lambda c: -np.sum(cls == c))
color_of = {c: col for _, col, cs in FAMILIES for c in cs}
ncol = 5
nrow = int(np.ceil(len(classes) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.6 * nrow), sharex=True, sharey=True)
for ax, c in zip(axes.flat, classes):
    frame(ax)
    m = cls == c
    ax.scatter(exp[~m], pred[~m], s=14, color="#DDDDDD", linewidth=0)
    col = color_of[c] if color_of[c] not in ("#BBBBBB", "#F0E442") else "#555555"
    ax.scatter(exp[m], pred[m], s=45, color=col, alpha=0.9, linewidth=0)
    err = pred[m] - exp[m]
    ax.set_title(f"{CLASS_LABEL.get(c, c)}\nn={m.sum()}, MAE {np.abs(err).mean():.1f}, "
                 f"bias {err.mean():+.1f}", fontsize=18)
for ax in axes.flat[len(classes):]:
    ax.axis("off")
fig.supxlabel(r"openTECR $\Delta_rG'^\circ$, standardized (kJ mol$^{-1}$)", fontsize=18)
fig.supylabel(r"MetaG $\Delta_rG'^\circ$ (kJ mol$^{-1}$)", fontsize=18)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "scatter_by_class_grid.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote", os.path.join(HERE, "scatter_by_family.png"), "and scatter_by_class_grid.png")
