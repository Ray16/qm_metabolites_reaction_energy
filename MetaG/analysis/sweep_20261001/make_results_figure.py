"""Build the manuscript results overview from frozen analysis artifacts."""
import glob
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.environ.get(
    "METAG_RESULTS", os.path.join(HERE, "final_20261002_opentecr_calibrated")
)
OUT = os.environ.get(
    "RESULTS_FIGURE", os.path.join(os.path.dirname(ROOT), "MetaG_new", "manuscript", "figures", "results_overview.png")
)
ANALYSIS_OUT = os.environ.get(
    "RESULTS_ANALYSIS_FIGURE",
    os.path.join(HERE, "figures", "results_overview.png"),
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def load_records():
    return [
        json.load(open(path))
        for path in sorted(glob.glob(os.path.join(RESULTS, "*.json")))
    ]


records = load_records()
comparison = json.load(open(os.path.join(HERE, "common_reference_comparison.json")))
generality = json.load(open(os.path.join(HERE, "generality_report.json")))

fig, axes = plt.subplots(2, 2, figsize=(21, 17))
ax = axes[0, 0]
exp = np.array([r["exp"] for r in records])
pred = np.array([r["dG"] for r in records])
lo = min(exp.min(), pred.min()) - 5
hi = max(exp.max(), pred.max()) + 5
ax.plot([lo, hi], [lo, hi], color="#333333", linestyle="--", linewidth=1)
ax.scatter(exp, pred, s=40, color="#16856B", alpha=0.68, linewidth=0)
ax.set(xlim=(lo, hi), ylim=(lo, hi),
       xlabel=r"openTECR $\Delta_rG'^\circ$, standardized (kJ mol$^{-1}$)",
       ylabel=r"MetaG $\Delta_rG'^\circ$ (kJ mol$^{-1}$)")
ax.text(0.04, 0.96,
        f"MAE {np.mean(np.abs(pred-exp)):.2f}\n"
        f"median {np.median(np.abs(pred-exp)):.2f}\nn={len(records)}",
        transform=ax.transAxes, va="top")
ax.set_title("a", loc="left", fontweight="bold")

ax = axes[0, 1]
methods = comparison["methods"]
method_rows = [
    ("eQuilibrator", "in-sample", methods[
        "eQuilibrator CC (in-sample, I=0, pMg 14)"
    ]["MAE_vs_openTECR"], "#D55E00"),
    ("Group contribution", "in-sample", methods[
        "Group contribution (in-sample, I=0, pMg 14)"
    ]["MAE_vs_openTECR"], "#0072B2"),
    ("dGPredictor", "held-out", methods[
        "dGPredictor retrained (held-out, family-grouped CV)"
    ]["MAE_vs_openTECR"], "#8C6BB1"),
    ("MetaG", "", methods[
        "MetaG (never fit; policy selected on TECRDB -> development)"
    ]["MAE_vs_openTECR"], "#16856B"),
]
names = [name if not regime else f"{name}\n({regime})"
         for name, regime, _, _ in method_rows]
values = [value for _, _, value, _ in method_rows]
colors = [color for _, _, _, color in method_rows]
bars = ax.barh(range(len(values)), values, color=colors, height=0.62)
ax.set_yticks(range(len(values)), names)
ax.invert_yaxis()
ax.set_xlabel(f"MAE vs openTECR, common set n={comparison['n_common']}" + r" (kJ mol$^{-1}$)")
ax.set_xlim(0, max(values) * 1.22)
for bar, value in zip(bars, values):
    ax.text(value + 0.18, bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}", va="center")
ax.set_title("b", loc="left", fontweight="bold")

ax = axes[1, 0]
by_class = {}
for record in records:
    by_class.setdefault(record["class"], []).append(abs(record["err"]))
class_rows = sorted(
    ((name, statistics.mean(errors), len(errors))
     for name, errors in by_class.items()),
    key=lambda row: row[1],
)
labels = [name.replace("NAD(P)-redox(other)", "NAD(P)-redox")
          .replace("glycosyl/PRT/nucleoside", "glycosyl/PRT")
          .replace("amide/amidine-hydrolysis", "amide hydrolysis")
          for name, _, _ in class_rows]
maes = [value for _, value, _ in class_rows]
sizes = [n for _, _, n in class_rows]
y = np.arange(len(class_rows))
ax.barh(y, maes, color=["#B23A48" if value > 20 else "#4C78A8" for value in maes])
ax.set_yticks(y, labels)
ax.set_xlabel(r"class MAE (kJ mol$^{-1}$)")
ax.set_title("c", loc="left", fontweight="bold")
for yi, value, n in zip(y, maes, sizes):
    ax.text(value + 0.35, yi, f"{value:.1f} (n={n})", va="center")
ax.set_xlim(0, max(maes) * 1.55)

ax = axes[1, 1]
counts = [
    generality["n_estimates"],
    generality["n_failures"],
    generality["n_ood"],
]
labels = ["estimates", "failures", "OOD flags"]
colors = ["#16856B", "#B23A48", "#E69F00"]
bars = ax.bar(range(3), counts, color=colors, width=0.62)
ax.set_xticks(range(3), labels, rotation=12, ha="right")
ax.set_ylabel(f"ModelSEED reactions (of {generality['n_input']})")
ax.set_ylim(0, generality["n_input"] * 1.12)
for bar, value in zip(bars, counts):
    y = value - 22 if value > 50 else value + 5
    color = "white" if value > 50 else "black"
    ax.text(bar.get_x() + bar.get_width() / 2, y, str(value),
            ha="center", fontweight="bold", color=color)
ax.set_title("d", loc="left", fontweight="bold")

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y" if ax is axes[1, 1] else "x", color="#DDDDDD", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

fig.tight_layout(h_pad=3.0, w_pad=3.0)
os.makedirs(os.path.dirname(ANALYSIS_OUT), exist_ok=True)
fig.savefig(ANALYSIS_OUT, bbox_inches="tight")
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {ANALYSIS_OUT}")
print(f"wrote {OUT}")
