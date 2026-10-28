"""Render a readable slide figure from the preserved numerical audit."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    evidence = root.parent / "analysis/ncs_numerical_validation/evidence_20261008_final"
    data = json.loads((evidence / "audit.json").read_text())
    with (evidence / "minima_components.tsv").open() as stream:
        keto = [row for row in csv.DictReader(stream, delimiter="\t") if row["state"] == "ip_keto_anion"]
    assert len(keto) == 5
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.7), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=.075, right=.985, top=.80, bottom=.22, wspace=.32)
    gray, teal = "#9aa5ab", "#087f8c"
    methods = ("UMA_ALPB", "DFT_ALPB", "UMA_SMD", "DFT_SMD")
    labels = ("UMA\nALPB", "DFT\nALPB", "UMA\nSMD", "DFT\nSMD")
    ax = axes[0]
    ax.axhline(0, color="#617780", linewidth=.8)
    for offset, key, color, label in ((-.18, "selected_kj", gray, "Selected two / state"),
                                      (.18, "full_kj", teal, "All five saved / state")):
        values = [data["gaps_Z_enol_minus_keto"][m][key] for m in methods]
        x = np.arange(4) + offset
        ax.bar(x, values, width=.34, color=color, label=label)
        for xx, value in zip(x, values):
            ax.text(xx, value + (0.6 if value >= 0 else -0.6), f"{value:+.1f}",
                    ha="center", va="bottom" if value >= 0 else "top", fontsize=12)
    ax.set_xticks(np.arange(4), labels, fontsize=14)
    ax.set(ylim=(-18, 17), ylabel="State gap (kJ/mol)")
    ax.set_title("Z-enol minus keto", loc="left", fontsize=17, pad=32)
    ax.legend(frameon=False, fontsize=11, ncol=2, loc="lower left", bbox_to_anchor=(-.02, 1.015))
    ax = axes[1]
    ax.axvspan(-.5, 1.5, color="#edf1f3", zorder=0)
    for offset, key, color, label in ((-.18, "UMA_ALPB_weight", gray, "UMA / ALPB"),
                                      (.18, "DFT_SMD_weight", teal, "DFT / SMD")):
        values = [100 * float(r[key]) for r in keto]
        x = np.arange(5) + offset
        ax.bar(x, values, width=.34, color=color, label=label)
        for xx, value in zip(x, values):
            if value > 25:
                ax.text(xx, value+2, f"{value:.1f}%", ha="center", fontsize=12)
    ax.set_xticks(np.arange(5), [f"{int(r['index']):02d}" for r in keto], fontsize=14)
    ax.set(ylim=(0, 107), ylabel="Keto population (%)", xlabel="Saved minimum")
    ax.set_title("Conformer redistribution", loc="left", fontsize=17, pad=32)
    ax.legend(frameon=False, fontsize=11, ncol=2, loc="lower left", bbox_to_anchor=(-.02, 1.015))
    path = root / "figures/ncs_state_solvation_slide.png"
    fig.savefig(path, dpi=250, facecolor="white")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
