#!/usr/bin/env python
"""Summary comparison: GNN (this work) vs eQuilibrator vs dGPredictor on all 367.

Predicted-vs-experiment scatter, one panel per method, with MAE annotated and
the honest in-sample / held-out label. eQ and dGP are IN-SAMPLE (trained on
TECRDB); the GNN is held-out (out-of-fold). Writes figures/gnn_vs_incumbents.png.
"""
import _bootstrap  # noqa: F401
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnn import paths

pred = json.load(open(paths.artifact("predictions.json")))
rxn_ids = list(pred)
exp = np.array([pred[r]["exp"] for r in rxn_ids])
gnn = np.array([pred[r]["gnn"] for r in rxn_ids])
eqf = json.load(open(f"{paths.RESULTS}/eq/equilibrator_full.json"))
dgs = json.load(open(f"{paths.RESULTS}/eq/dgpredictor_full.json"))
dgr = json.load(open(f"{paths.RESULTS}/eq/dgpredictor_retrained_full.json"))


def series(d):
    vals = [d.get(r, {}).get("dG_kJ") for r in rxn_ids]
    return np.array([v if v is not None else np.nan for v in vals], dtype=float)


panels = [
    ("eQuilibrator", series(eqf), "in-sample", "#2E8B57"),
    ("dGPredictor (standard)", series(dgs), "in-sample", "#2A6F97"),
    ("dGPredictor (retrained)", series(dgr), "in-sample", "#C0392B"),
    ("GNN (this work)", gnn, "held-out (OOF)", "#7B4FA3"),
]
lim = (-60, 60)
fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), sharex=True, sharey=True)
for ax, (name, p, kind, c) in zip(axes, panels):
    m = np.isfinite(p) & np.isfinite(exp)
    mae = np.abs(p[m] - exp[m]).mean()
    ax.plot(lim, lim, color="gray", lw=0.8, ls="--", zorder=0)
    ax.scatter(exp[m], p[m], s=14, c=c, alpha=0.55, edgecolors="none")
    ax.set_title(f"{name}\nMAE {mae:.1f}  ({kind})", fontsize=11)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel(r"experiment  $\Delta_r G'^{\circ}$ (kJ/mol)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel(r"predicted  $\Delta_r G'^{\circ}$ (kJ/mol)", fontsize=9)
fig.suptitle("367 TECRDB reactions — incumbents are IN-SAMPLE (trained on this data); "
             "the GNN is HELD-OUT", fontsize=11, y=1.02)
fig.tight_layout()
for d in (paths.FIGURES, "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/figures"):
    os.makedirs(d, exist_ok=True)
    fig.savefig(f"{d}/gnn_vs_incumbents.png", dpi=200, bbox_inches="tight")
print("wrote figures/gnn_vs_incumbents.png")
