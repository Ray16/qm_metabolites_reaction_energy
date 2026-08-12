#!/usr/bin/env python
"""FAIR comparison: GNN vs dGPredictor, both evaluated HELD-OUT on identical folds.

Both are out-of-fold (5-fold, seed 0): the GNN from artifacts/predictions.json,
dGPredictor as its own model class (ridge on the group-difference features)
refit on each training fold. No in-sample numbers are shown -- eQuilibrator is
omitted because it cannot be refit held-out here (its 3.0 is in-sample and not a
comparable evaluation).

Writes figures/gnn_vs_dgpredictor_heldout.png.
"""
import _bootstrap  # noqa: F401
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gnn import paths
from gnn.training import ridge_fit, kfold, compound_disjoint

# --- data + GNN held-out predictions ---
pred = json.load(open(paths.artifact("predictions.json")))
rxn_ids = list(pred)
exp = np.array([pred[r]["exp"] for r in rxn_ids])
gnn = np.array([pred[r]["gnn"] for r in rxn_ids])

d = torch.load(paths.artifact("data.pt"))
Xg = d["Xgroup"].numpy(); y = d["y"].numpy()
rxn_comps = d["rxn_comps"]; n_comp = d["n_comp"]; N = len(rxn_ids)
LAM = 10.0   # near-optimal, fixed a priori (not tuned on the test)


def dgp_oof(folds):
    p = np.full(N, np.nan)
    for tr, te in folds:
        if len(te):
            c = ridge_fit(Xg, y, tr, LAM)
            p[te] = Xg[te] @ c
    return p


rnd = kfold(N, 5, 0)
dgp = dgp_oof(rnd)                       # random-CV OOF (one prediction/reaction)

# compound-disjoint MAE for the caption (unique-reaction OOF)
cpd = compound_disjoint(rxn_comps, n_comp, 5, 0)
dgp_cpd = dgp_oof(cpd)


def mae(p, m=None):
    m = np.isfinite(p) if m is None else m & np.isfinite(p)
    return np.abs(p[m] - y[m]).mean()


# GNN compound-disjoint number is the reported 8.59 (from run_cv); random from OOF
panels = [("dGPredictor (group-linear)", dgp, "#C0392B"),
          ("GNN (this work)", gnn, "#7B4FA3")]
lim = (-60, 60)
fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharex=True, sharey=True)
for ax, (name, p, c) in zip(axes, panels):
    ax.plot(lim, lim, color="gray", lw=0.8, ls="--", zorder=0)
    ax.scatter(exp, p, s=16, c=c, alpha=0.55, edgecolors="none")
    ax.set_title(f"{name}\nheld-out MAE {mae(p):.1f} (random CV)", fontsize=11)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel(r"experiment  $\Delta_r G'^{\circ}$ (kJ/mol)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel(r"predicted  $\Delta_r G'^{\circ}$ (kJ/mol)", fontsize=9)
fig.suptitle("Fair comparison — both HELD-OUT (5-fold out-of-fold), identical folds", fontsize=11)
cap = (f"367 TECRDB reactions. dGPredictor = ridge on group-difference features (lam={LAM:g}), "
       f"refit per training fold. Compound-disjoint held-out MAE: dGP {mae(dgp_cpd):.1f}, GNN 8.6.  "
       "eQuilibrator omitted: cannot be refit held-out here (its 3.0 is in-sample, not comparable).")
fig.text(0.5, -0.04, cap, ha="center", fontsize=7.6, color="gray", wrap=True)
fig.tight_layout()
for od in (paths.FIGURES, "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/figures"):
    os.makedirs(od, exist_ok=True)
    fig.savefig(f"{od}/gnn_vs_dgpredictor_heldout.png", dpi=200, bbox_inches="tight")
print(f"wrote gnn_vs_dgpredictor_heldout.png  "
      f"held-out random MAE: dGP {mae(dgp):.2f}  GNN {mae(gnn):.2f}")
