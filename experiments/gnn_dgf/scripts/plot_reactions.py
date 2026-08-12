#!/usr/bin/env python
"""The 10-reaction disagreement figure + this work's GNN (held-out OOF).

Shows GNN-Δ tracking experiment on the cherry-picked hard subset where the
retrained-dGP (overfitting) and absolute QC (anion solvation) blow up.
Writes figures/qm_vs_dgpredictor_gnn.png.
"""
import _bootstrap  # noqa: F401
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnn import paths

RXNS = [("rxn00086", "redox", None), ("rxn32133", "redox", "rxn00086"),
        ("rxn00070", "redox", None), ("rxn34788", "redox", "rxn00070"),
        ("rxn00605", "glycosyl", None), ("rxn01713", "glycosyl", None),
        ("rxn01834", "glyoxalase", None), ("rxn00579", "glycosyl", None),
        ("rxn01675", "nucleotidyl", None), ("rxn01005", "nucleotidyl", None)]

meta = json.load(open(f"{paths.PIPE}/tecrdb_full_experiment.json"))
exp = {r: v["dG_kJ"] for r, v in meta.items()}
qc = json.load(open(f"{paths.RESULTS}/benchmark/tecrdb_full_scored.json"))["scored_kJ"]
rtr = json.load(open(f"{paths.RESULTS}/eq/dgpredictor_retrained_full.json"))
pred = json.load(open(paths.artifact("predictions.json")))
gnn_full = np.mean([abs(v["gnn"] - v["exp"]) for v in pred.values()])


def val(d, r, parent, key=None):
    src = parent or r
    x = d[src][key] if key else d[src]
    return -x if parent else x


E = np.array([val(exp, r, p) for r, _, p in RXNS])
series = [("TECRDB (experiment)", E, "#4C4C4C"),
          ("dGPredictor (retrained-ModelSEED)", np.array([val(rtr, r, p, "dG_kJ") for r, _, p in RXNS]), "#D1495B"),
          ("QC composite (MACE-POLAR + xtb-ALPB)", np.array([val(qc, r, p) for r, _, p in RXNS]), "#2E86AB"),
          ("GNN [this work, held-out]", np.array([val(pred, r, p, "gnn") for r, _, p in RXNS]), "#4C9F70")]
sd = [float(meta[p or r]["sd_kJ"] or 0.0) if (meta[p or r]["sd_kJ"] and meta[p or r]["n"] > 1) else 0.0
      for r, _, p in RXNS]

x = np.arange(len(RXNS)); width = 0.8 / len(series); off = (len(series) - 1) / 2.0
fig, ax = plt.subplots(figsize=(13.5, 6.4))
for i, (label, vals, color) in enumerate(series):
    leg = label if label.startswith("TECRDB") else f"{label}   (subset MAE {np.mean(np.abs(vals - E)):.0f})"
    ax.bar(x + (i - off) * width, vals, width, label=leg, color=color, edgecolor="white",
           linewidth=0.5, yerr=sd if label.startswith("TECRDB") else None,
           error_kw=dict(ecolor="black", capsize=3, lw=1.2))
for xi, (r, _, p) in enumerate(RXNS):
    if meta[p or r]["n"] <= 1:
        ax.text(xi, 0.015, "n=1", ha="center", fontsize=7.5, color="gray",
                transform=ax.get_xaxis_transform())
ax.axhline(0, color="black", lw=0.8); ax.margins(y=0.18)
ax.set_ylabel(r"$\Delta_r G'^{\circ}$ (kJ/mol)")
ax.set_xticks(x); ax.set_xticklabels([f"{r}\n{c}" for r, c, _ in RXNS], fontsize=9)
ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8.5)
cap = ("Cherry-picked disagreement subset (NOT representative). Held-out full-367 MAE: "
       f"GNN {gnn_full:.1f} (this work), retrained-dGP 5.7 / standard-dGP 3.0 / eQuilibrator 3.0 "
       "(all IN-SAMPLE), raw QC 36.")
ax.text(0.0, -0.22, cap, transform=ax.transAxes, ha="left", fontsize=7.2, color="gray")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for d in (paths.FIGURES, "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/figures"):
    os.makedirs(d, exist_ok=True)
    fig.savefig(f"{d}/qm_vs_dgpredictor_gnn.png", dpi=200, bbox_inches="tight")
print(f"wrote qm_vs_dgpredictor_gnn.png  (GNN held-out full-367 MAE {gnn_full:.2f})")
