#!/usr/bin/env python
"""Benchmark the GNN component-contribution model against eQuilibrator and
dGPredictor on the 367 TECRDB reactions.

CRITICAL FAIRNESS NOTE
----------------------
eQuilibrator and dGPredictor are TRAINED ON TECRDB (component contribution
fit to this same data). Their numbers below are therefore *in-sample*.
The GNN (and the linear group-CC surrogate) numbers are *held-out CV*
(out-of-fold predictions, no test reaction seen in training). This is not a
like-for-like comparison of generalization -- it is the honest operating
comparison: "the deployed incumbent tool vs my held-out model". Both framings
are printed. `dgp_retrained` is our own group-linear refit (also in-sample).

Ground truth: pipeline/tecrdb_full_experiment.json[rxn]['dG_kJ']
              (median -RT ln K' over measurements), with n and sd per reaction.

No scipy: Spearman + bootstrap done in numpy.
"""
import _bootstrap  # noqa: F401  (adds repo paths)
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gnn import paths

ROOT = paths.THERMO
EXP  = json.load(open(f"{ROOT}/pipeline/tecrdb_full_experiment.json"))
GNN  = json.load(open(paths.artifact("predictions.json")))
EQ   = json.load(open(f"{ROOT}/results/eq/equilibrator_full.json"))
DGP  = json.load(open(f"{ROOT}/results/eq/dgpredictor_full.json"))
DGPR = json.load(open(f"{ROOT}/results/eq/dgpredictor_retrained_full.json"))
QC   = json.load(open(f"{ROOT}/results/benchmark/tecrdb_full_scored.json"))["scored_kJ"]

# Common reaction set (all four already align, but intersect defensively).
rxns = sorted(set(GNN) & set(EQ) & set(DGP) & set(EXP))
y    = np.array([EXP[r]["dG_kJ"] for r in rxns])
nmeas= np.array([EXP[r].get("n", 1) for r in rxns], float)
w    = np.log1p(nmeas)                    # same weighting used in training

def col(d, r, k):
    v = d[r].get(k)
    return float(v) if v is not None else np.nan


def qcval(r):
    v = QC.get(r)
    return float(v) if isinstance(v, (int, float)) else np.nan


METHODS = {
    "eQuilibrator  [in-sample]": np.array([col(EQ, r, "dG_kJ")  for r in rxns]),
    "dGPredictor   [in-sample]": np.array([col(DGP, r, "dG_kJ") for r in rxns]),
    "dGP-retrained [in-sample]": np.array([col(DGPR, r, "dG_kJ") for r in rxns]),
    "linear grp-CC [HELD-OUT]" : np.array([col(GNN, r, "linear") for r in rxns]),
    "GNN-delta     [HELD-OUT]" : np.array([col(GNN, r, "gnn")    for r in rxns]),
    "QC first-princ [no-fit]"  : np.array([qcval(r) for r in rxns]),
}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra**2).sum() * (rb**2).sum()))


def metrics(pred):
    ok = np.isfinite(pred)
    p, yy, ww = pred[ok], y[ok], w[ok]
    e = p - yy
    ae = np.abs(e)
    return dict(
        n=int(ok.sum()),
        MAE=ae.mean(),
        wMAE=(ww * ae).sum() / ww.sum(),
        RMSE=np.sqrt((e**2).mean()),
        medAE=np.median(ae),
        maxAE=ae.max(),
        sign=np.mean((np.sign(p) == np.sign(yy)) | (np.abs(yy) < 1e-6)) * 100,
        r2=1 - (e**2).sum() / ((yy - yy.mean())**2).sum(),
        rho=spearman(p, yy),
    )


rng = np.random.default_rng(0)
BOOT = 5000

print(f"\nTECRDB benchmark  |  n = {len(rxns)} reactions  "
      f"(median measurements/rxn = {int(np.median(nmeas))})")
print(f"Ground truth spread: sd(y) = {y.std():.1f} kJ/mol, "
      f"predict-zero MAE = {np.abs(y).mean():.2f}\n")

hdr = f"{'method':<26} {'n':>4} {'MAE':>6} {'wMAE':>6} {'RMSE':>6} {'medAE':>6} {'maxAE':>7} {'sign%':>6} {'R2':>6} {'rho':>6}"
print(hdr); print("-" * len(hdr))
results = {}
gnn_ae = np.abs(METHODS["GNN-delta     [HELD-OUT]"] - y)
for name, pred in METHODS.items():
    m = metrics(pred)
    results[name] = m
    print(f"{name:<26} {m['n']:>4} {m['MAE']:6.2f} {m['wMAE']:6.2f} {m['RMSE']:6.2f} "
          f"{m['medAE']:6.2f} {m['maxAE']:7.1f} {m['sign']:6.1f} {m['r2']:6.2f} {m['rho']:6.2f}")

# Paired bootstrap: MAE(method) - MAE(GNN) with 95% CI (positive => GNN better).
print("\nPaired ΔMAE vs GNN-delta  (MAE_method - MAE_GNN; +ve => GNN closer to exp)")
print(f"{'method':<26} {'ΔMAE':>7}  {'95% CI':>16}")
print("-" * 52)
for name, pred in METHODS.items():
    if "GNN" in name:
        continue
    ok = np.isfinite(pred) & np.isfinite(gnn_ae)
    d = np.abs(pred[ok] - y[ok]) - gnn_ae[ok]
    point = d.mean()
    bidx = rng.integers(0, len(d), size=(BOOT, len(d)))
    bs = d[bidx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    verdict = "GNN better" if lo > 0 else ("incumbent better" if hi < 0 else "tie")
    print(f"{name:<26} {point:+7.2f}  [{lo:+5.2f}, {hi:+5.2f}]  {verdict}")

# ---- Parity figure ---------------------------------------------------------
order = ["eQuilibrator  [in-sample]", "dGPredictor   [in-sample]",
         "GNN-delta     [HELD-OUT]", "linear grp-CC [HELD-OUT]"]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
lim = 60
for ax, name in zip(axes, order):
    pred = METHODS[name]; m = results[name]
    ax.axline((0, 0), slope=1, color="0.6", lw=1, zorder=0)
    ax.axhline(0, color="0.85", lw=0.7, zorder=0); ax.axvline(0, color="0.85", lw=0.7, zorder=0)
    ax.scatter(y, pred, s=14, alpha=0.55,
               c="#c0392b" if "in-sample" in name else "#2471a3", edgecolors="none")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_title(name.split("[")[0].strip(), fontsize=11)
    ax.set_xlabel("experimental ΔG′ (kJ/mol)")
    tag = "in-sample" if "in-sample" in name else "held-out CV"
    ax.text(0.04, 0.96, f"MAE {m['MAE']:.2f}\nRMSE {m['RMSE']:.1f}\n$R^2$ {m['r2']:.2f}\n({tag})",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
axes[0].set_ylabel("predicted ΔG′ (kJ/mol)")
fig.suptitle("TECRDB (n=367): GNN component-contribution vs incumbents  "
             "— red = trained on TECRDB (in-sample), blue = held-out CV",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = f"{ROOT}/../figures/gnn_vs_incumbents_tecrdb.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nfigure -> {out}")

json.dump({n: {k: float(v) for k, v in m.items()} for n, m in results.items()},
          open(paths.artifact("benchmark_incumbents.json"), "w"), indent=2)
print(f"metrics -> {paths.artifact('benchmark_incumbents.json')}")
