#!/usr/bin/env python
"""Reproduce the 10-reaction disagreement figure + add this work's GNN-delta
(held-out OOF) as a fourth series.

Honesty: GNN-delta numbers are held-out (out-of-fold); eQ/standard-dGP get this
subset to 0-10 too. The point of THIS figure is that a well-regularized learned
model dodges BOTH failure modes shown -- retrained-dGP's overfitting blowups and
QC's absolute anion-solvation blowups.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(os.path.dirname(HERE))
PIPE = f"{THERMO}/pipeline"

RXNS = [("rxn00086", "redox", None), ("rxn32133", "redox", "rxn00086"),
        ("rxn00070", "redox", None), ("rxn34788", "redox", "rxn00070"),
        ("rxn00605", "glycosyl", None), ("rxn01713", "glycosyl", None),
        ("rxn01834", "glyoxalase", None), ("rxn00579", "glycosyl", None),
        ("rxn01675", "nucleotidyl", None), ("rxn01005", "nucleotidyl", None)]

meta = json.load(open(f"{PIPE}/tecrdb_full_experiment.json"))
exp = {r: v["dG_kJ"] for r, v in meta.items()}
qc = json.load(open(f"{THERMO}/results/benchmark/tecrdb_full_scored.json"))["scored_kJ"]
rtr = json.load(open(f"{THERMO}/results/eq/dgpredictor_retrained_full.json"))
pred = json.load(open(f"{HERE}/predictions.json"))   # this work, held-out OOF

# full-367 held-out MAE for caption
gnn_full = np.mean([abs(v["gnn"] - v["exp"]) for v in pred.values()])


def val(d, r, parent, key=None):
    src = parent if parent else r
    x = d[src][key] if key else d[src]
    return -x if parent else x


E = np.array([val(exp, r, par) for r, _, par in RXNS])
series = [
    ("TECRDB (experiment)", E, "#4C4C4C"),
    ("dGPredictor (retrained-ModelSEED)", np.array([val(rtr, r, par, "dG_kJ") for r, _, par in RXNS]), "#D1495B"),
    ("QC composite (MACE-POLAR + xtb-ALPB)", np.array([val(qc, r, par) for r, _, par in RXNS]), "#2E86AB"),
    ("GNN-Δ [this work, held-out]", np.array([val(pred, r, par, "gnn") for r, _, par in RXNS]), "#4C9F70"),
]
sd = [float(meta[par or r]["sd_kJ"] or 0.0) if (meta[par or r]["sd_kJ"] and meta[par or r]["n"] > 1) else 0.0
      for r, _, par in RXNS]

x = np.arange(len(RXNS)); width = 0.8 / len(series); off = (len(series) - 1) / 2.0
fig, ax = plt.subplots(figsize=(13.5, 6.4))
for i, (label, vals, color) in enumerate(series):
    leg = label if label.startswith("TECRDB") else f"{label}   (subset MAE {np.mean(np.abs(vals - E)):.0f})"
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
ax.set_xticks(x); ax.set_xticklabels([f"{r}\n{c}" for r, c, _ in RXNS], fontsize=9)
ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8.5)
cap = ("Cherry-picked disagreement subset (NOT representative). Held-out full-367 MAE: "
       f"GNN-Δ {gnn_full:.1f} (this work), retrained-dGP 5.7 / standard-dGP 3.0 / eQuilibrator 3.0 "
       "(all IN-SAMPLE), raw QC 36. GNN-Δ avoids both the retrained-dGP overfitting blowups and "
       "the QC solvation blowups. error bars = TECRDB sd; n=1 = single measurement.")
ax.text(0.0, -0.22, cap, transform=ax.transAxes, ha="left", fontsize=7.2, color="gray")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for d in (f"{THERMO}/figures", f"{THERMO}/results/benchmark",
          "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/figures"):
    os.makedirs(d, exist_ok=True)
    fig.savefig(f"{d}/qm_vs_dgpredictor_gnn.png", dpi=200, bbox_inches="tight")
print(f"wrote qm_vs_dgpredictor_gnn.png  (GNN-delta held-out full-367 MAE {gnn_full:.2f})")
