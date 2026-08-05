#!/usr/bin/env python
"""Grouped bar chart (experiment / dGPredictor / QM) for the 10
dGPredictor-TECRDB disagreement reactions.

Run:  python plot_comparison.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
CSV = os.path.join(THERMO, "results", "benchmark", "perreaction_dG.csv")
RXN_CSV = os.path.join(HERE, "top10_reactions_stereo_significant.csv")
OUT = os.path.join(THERMO, "results", "benchmark", "qm_vs_dgpredictor_top10.png")

# The reported baseline is "pH7 fixed species".  The pH-midpoint column is a
# sensitivity diagnostic and must not be presented as the model -- an earlier
# version of this figure plotted it and captioned it as "QM", which overstated
# the error (38.3 rather than the baseline 31.7).  Series absent from the CSV
# are skipped, so this works whichever --pH-mode produced it.
SERIES = [("exp", "TECRDB (experiment)", "#4C4C4C"),
          ("dGP", "dGPredictor", "#D1495B"),
          ("pH7 fixed species", "QM composite (MACE-POLAR-1)", "#2E86AB"),
          ("QM + external references [param-free]",
           "QM + external references", "#00916E")]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(CSV) as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r["rank"]))

    x = np.arange(len(rows))
    present = [s for s in SERIES if s[0] in rows[0]]
    width = 0.8 / len(present)
    series = {k: [float(r[k]) for r in rows] for k, _, _ in present}

    # Experimental uncertainty. A blank/0.0 sd means "not reported" (these are
    # single-measurement entries), NOT a zero-uncertainty measurement -- so those
    # get no error bar and are flagged separately rather than drawn as exact.
    meta = {r["modelseed_rxn"]: r for r in csv.DictReader(open(RXN_CSV))}
    sd, n_meas = [], []
    for r in rows:
        m = meta[r["rxn"]]
        s = float(m["tecrdb_dG_sd_kJ"] or 0.0)
        n = int(m["n_measurements"])
        sd.append(s if (s > 0 and n > 1) else 0.0)
        n_meas.append(n)

    fig, ax = plt.subplots(figsize=(13, 6))
    offset = (len(present) - 1) / 2.0
    for i, (key, label, color) in enumerate(present):
        if key == "exp":
            leg = label
        else:
            mae = np.mean([abs(p - e) for p, e in zip(series[key], series["exp"])])
            signs = sum(p * e > 0 for p, e in zip(series[key], series["exp"]))
            leg = f"{label}  (MAE {mae:.1f}, signs {signs}/{len(rows)})"
        ax.bar(x + (i - offset) * width, series[key], width, label=leg, color=color,
               edgecolor="white", linewidth=0.5,
               yerr=sd if key == "exp" else None,
               error_kw=dict(ecolor="black", capsize=3, lw=1.2))

    # mark the single-measurement reactions, whose uncertainty is simply unknown.
    # Pinned to the bottom of the axes (blended transform) so it never collides
    # with a bar, however tall the bars get.
    for xi, n in enumerate(n_meas):
        if n <= 1:
            ax.text(xi, 0.015, "n=1", ha="center", fontsize=7.5, color="gray",
                    transform=ax.get_xaxis_transform())

    ax.axhline(0, color="black", lw=0.8)
    ax.margins(y=0.18)
    ax.set_ylabel(r"$\Delta_r G'^{\circ}$ (kJ/mol)")
    ax.set_title("dGPredictor vs QM composite on 10 TECRDB disagreement reactions",
                 fontsize=13, pad=12)
    # Second label line is the reaction class: the corrections are per class, so
    # reading the figure without it hides why neighbouring bars behave alike.
    classes = {}
    class_path = os.path.join(HERE, "reaction_classes.json")
    if os.path.exists(class_path):
        pretty = {"thiolate_redox": "redox", "glycosyl_transfer": "glycosyl",
                  "phosphate_transfer": "nucleotidyl", "other": "glyoxalase"}
        raw = json.load(open(class_path))
        classes = {k: pretty.get(v, v) for k, v in raw.items()}
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['rxn']}\n{classes.get(r['rxn'], '')}".rstrip()
                        for r in rows], fontsize=9)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=9)
    ax.text(0.995, -0.16, "error bars = TECRDB sd; \"n=1\" = single measurement, "
            "uncertainty not reported", transform=ax.transAxes, ha="right",
            fontsize=8, color="gray")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
