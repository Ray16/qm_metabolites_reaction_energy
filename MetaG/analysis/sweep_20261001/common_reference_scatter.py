"""Parity plots for all methods on the frozen 319-reaction common set.

Rows use the openTECR standardized reference (the benchmark reference) and the native-condition TECRDB values. Columns
separate evaluation regimes so in-sample, held-out and development results are
visually comparable without being presented as equivalent validation estimates.
"""
import glob
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TC = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RESULTS = os.environ.get(
    "METAG_RESULTS", os.path.join(HERE, "final_20261002_opentecr_calibrated")
)
OUT = os.environ.get(
    "COMMON_SCATTER_OUT",
    os.path.join(TC, "MetaG_new", "manuscript", "figures", "common_reference_scatter.png"),
)
ANALYSIS_OUT = os.environ.get(
    "COMMON_SCATTER_ANALYSIS_OUT",
    os.path.join(HERE, "figures", "common_reference_scatter.png"),
)

std = {
    rid: statistics.median(reaction["exp"])
    for rid, reaction in json.load(open(os.path.join(
        TC, "experiments", "qm_mlip_solvation", "scripts",
        "reactions_opentecr_std.json"
    ))).items()
}
native = json.load(open(os.path.join(
    TC, "results", "benchmark", "tecrdb_full_scored.json"
)))["experiment_kJ"]
metag = {
    record["reaction"]: record["dG"]
    for record in (
        json.load(open(path))
        for path in glob.glob(os.path.join(RESULTS, "*.json"))
    )
    if record.get("dG") is not None
}


def predictions(path):
    return {
        rid: value["dG_kJ"]
        for rid, value in json.load(open(path))["predictions"].items()
        if value.get("dG_kJ") is not None and abs(value["dG_kJ"]) < 1e6
    }


equilibrator = predictions(os.path.join(HERE, "eq_real_tecrdb_std.json"))
group_contribution = predictions(os.path.join(HERE, "gc_real_tecrdb_std.json"))
dgp = {
    rid: value["dgp_retrained_heldout"]
    for rid, value in json.load(open(os.path.join(
        TC, "results", "eq", "dgp_retrained_heldout.json"
    ))).items()
}

methods = [
    ("MetaG", "", metag, "#16856B"),
    ("dGPredictor", "held-out, family-grouped CV", dgp, "#8C6BB1"),
    ("eQuilibrator CC", "in-sample", equilibrator, "#D55E00"),
    ("Group contribution", "in-sample", group_contribution, "#0072B2"),
]
common = sorted(
    set(std) & set(native) & set.intersection(*(set(values) for _, _, values, _ in methods))
)
assert len(common) == 319, f"expected frozen common set of 319, found {len(common)}"

all_values = (
    [std[rid] for rid in common]
    + [native[rid] for rid in common]
    + [values[rid] for _, _, values, _ in methods for rid in common]
)
low, high = np.percentile(all_values, [0.5, 99.5])
pad = 12
limits = (low - pad, high + pad)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})
fig, axes = plt.subplots(2, 4, figsize=(26, 13.5), sharex=True, sharey=True)

for row, (reference_name, reference) in enumerate([
    ("openTECR standardized:\npH 7, I=0, no Mg", std),
    ("Native TECRDB conditions", native),
]):
    expected = np.array([reference[rid] for rid in common])
    for col, (method, regime, values, color) in enumerate(methods):
        ax = axes[row, col]
        predicted = np.array([values[rid] for rid in common])
        error = predicted - expected
        ax.plot(limits, limits, "--", color="#333333", linewidth=0.9)
        ax.scatter(
            expected, predicted, s=30, color=color, alpha=0.62,
            linewidth=0, rasterized=True
        )
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.text(
            0.04, 0.96,
            f"MAE {np.mean(np.abs(error)):.2f}\n"
            f"median {np.median(np.abs(error)):.2f}",
            transform=ax.transAxes, va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
        )
        if row == 0:
            title = method if not regime else f"{method}\n({regime})"
            ax.set_title(title, fontweight="bold")
        if col == 0:
            ax.set_ylabel(
                reference_name + "\n" + r"prediction (kJ mol$^{-1}$)"
            )
        if row == 1:
            ax.set_xlabel(r"reference $\Delta_rG'^\circ$ (kJ mol$^{-1}$)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

fig.tight_layout(w_pad=1.5, h_pad=2.0)
os.makedirs(os.path.dirname(ANALYSIS_OUT), exist_ok=True)
fig.savefig(ANALYSIS_OUT, bbox_inches="tight")
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {ANALYSIS_OUT}")
print(f"wrote {OUT}")
for method, regime, values, _ in methods:
    print(method, f"({regime})" if regime else "")
    for name, reference in (("openTECR standardized", std), ("native", native)):
        error = np.array([values[rid] - reference[rid] for rid in common])
        print(
            f"  {name}: MAE={np.mean(np.abs(error)):.2f}, "
            f"median={np.median(np.abs(error)):.2f}"
        )
