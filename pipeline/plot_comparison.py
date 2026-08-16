#!/usr/bin/env python
"""Grouped bar chart on the 10 dGPredictor-disagreement reactions:
experiment / retrained-dGPredictor / UMA+truncation pipeline.

PROVENANCE (read before trusting this figure):
- These reactions are a CHERRY-PICKED subset selected where the retrained
  dGPredictor disagrees most with experiment (bench226 significant=True). They
  are NOT representative. Unbiased full-367-reaction MAEs are in the caption.
- "dGPredictor" here is Freiburger's RETRAINED-on-ModelSEED model (reproduced,
  results/eq/dgpredictor_retrained_full.json). Its worst-case errors here (40-90)
  are NOT its average (full-set 5.7); the STANDARD dGPredictor gets these to 0-10.
- "UMA+truncation pipeline" = current QM pipeline (systematic truncation + convergent
  sampling + solvation), pipeline/current_pipeline_top10.json. The old MACE-POLAR
  "QC composite" series was removed (superseded by this pipeline).
"""
from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
OUT = os.path.join(THERMO, "results", "benchmark", "qm_vs_dgpredictor_top10.png")

# The original 10 reactions, in order. rxn32133/rxn34788 are reaction-reversal
# duplicates of rxn00086/rxn00070 (kept to match the original figure); their
# values are exact negations, since every method here is antisymmetric.
RXNS = [("rxn00086", "redox", None), ("rxn32133", "redox", "rxn00086"),
        ("rxn00070", "redox", None), ("rxn34788", "redox", "rxn00070"),
        ("rxn00605", "glycosyl", None), ("rxn01713", "glycosyl", None),
        ("rxn01834", "glyoxalase", None), ("rxn00579", "glycosyl", None),
        ("rxn01675", "nucleotidyl", None), ("rxn01005", "nucleotidyl", None)]


def main():
    exp = {r: v["dG_kJ"] for r, v in json.load(open(f"{HERE}/tecrdb_full_experiment.json")).items()}
    meta = {r: v for r, v in json.load(open(f"{HERE}/tecrdb_full_experiment.json")).items()}
    rtr = json.load(open(f"{THERMO}/results/eq/dgpredictor_retrained_full.json"))
    cur = {k: v for k, v in json.load(open(f"{HERE}/current_pipeline_top10.json")).items()
           if not k.startswith("_")}                     # UMA+truncation pipeline
    ids = [r for r, _, _ in RXNS]
    # source value for each reaction; reverse-duplicates negate their parent
    def val(d, r, parent, key=None):
        src = parent if parent else r
        x = d[src][key] if key else d[src]
        return -x if parent else x
    E = np.array([val(exp, r, par) for r, _, par in RXNS])
    series = [
        ("TECRDB (experiment)", E, "#4C4C4C"),
        ("dGPredictor (retrained-ModelSEED)", np.array([val(rtr, r, par, "dG_kJ") for r, _, par in RXNS]), "#D1495B"),
        ("UMA + truncation pipeline", np.array([val(cur, r, par) for r, _, par in RXNS]), "#2A9D8F"),
    ]
    sd = [float(meta[par or r]["sd_kJ"] or 0.0) if (meta[par or r]["sd_kJ"] and meta[par or r]["n"] > 1) else 0.0
          for r, _, par in RXNS]

    x = np.arange(len(ids)); width = 0.8 / len(series); off = (len(series) - 1) / 2.0
    fig, ax = plt.subplots(figsize=(13, 6.2))
    for i, (label, vals, color) in enumerate(series):
        if label.startswith("TECRDB"):
            leg = label
        else:
            leg = f"{label}   (subset MAE {np.mean(np.abs(vals - E)):.0f})"
        ax.bar(x + (i - off) * width, vals, width, label=leg, color=color,
               edgecolor="white", linewidth=0.5,
               yerr=sd if label.startswith("TECRDB") else None,
               error_kw=dict(ecolor="black", capsize=3, lw=1.2))
    for xi, (r, _, par) in enumerate(RXNS):
        if meta[par or r]["n"] <= 1:
            ax.text(xi, 0.015, "n=1", ha="center", fontsize=7.5, color="gray",
                    transform=ax.get_xaxis_transform())
    ax.axhline(0, color="black", lw=0.8); ax.margins(y=0.18)
    ax.set_ylabel(r"$\Delta_r G'^{\circ}$ (kJ/mol)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n{c}" for r, c, _ in RXNS], fontsize=9)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8.5)
    cap = ("Cherry-picked disagreement subset (NOT representative). Unbiased full-367 MAE: "
           "retrained-dGP 5.7, standard-dGP 3.0, eQuilibrator 3.0 (UMA+truncation full-367 pending).  "
           "error bars = TECRDB sd; n=1 = single measurement.")
    ax.text(0.0, -0.20, cap, transform=ax.transAxes, ha="left", fontsize=7.6, color="gray")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    # also refresh the copies under figures/ (both the repo and the top-level dir)
    for d in (os.path.join(THERMO, "figures"),
              "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/figures"):
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "qm_vs_dgpredictor_top10.png"), dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
