"""Predicted-vs-experiment Δ_rG'° comparison across ALL methods, from
gnn_dgf/artifacts/per_rxn_benchmark.json. Every output is a SINGLE-panel PNG (repo convention: no
subplot grids), no title, font>=18, dpi300. Same axes on every panel so they can be laid side by side.

Outputs (figures/):
  pred_vs_exp_<method>.png   one scatter per method (uma_qm, gnn, dgp_retrain, linear_cc, eq),
                             y=x diagonal + fitted line, slope+MAE+coverage annotated, |dG|>30 shaded.
  pred_vs_exp_shrinkage.png  the 2-method headline overlay (UMA-QM vs dGPredictor).
  slope_by_method.png        fitted pred-vs-exp slope per method (1.0 = tracks truth; <1 = shrink-to-mean).

Slope<1 = the method under-predicts large |dG| (regression to the mean) -> its error explodes on
far-from-equilibrium reactions even when its average MAE looks good. UMA-QM computes the magnitude
(slope~0.8); the fitted statistical methods shrink (slope~0.4-0.5). eQ is in-sample (leaked).
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
_STYLE = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
                      "rxnfp_clustering", "scripts")
try:
    sys.path.insert(0, _STYLE)
    import _style  # noqa
except Exception:
    matplotlib.rcParams.update({"font.size": 18, "savefig.dpi": 300, "figure.dpi": 300})
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
BENCH = os.path.join(EXP, os.pardir, "gnn_dgf", "artifacts", "per_rxn_benchmark.json")
FIGDIR = os.path.join(EXP, "figures")
LIM = 120

# method key -> (display label, color)
METHODS = {
    "uma_qm":      ("UMA-QM  (physics, no fit)",        "#3b6fb0"),
    "gnn":         ("GNN  (held-out)",                  "#2a9d8f"),
    "dgp_retrain": ("dGPredictor  (held-out)",          "#e07b39"),
    "linear_cc":   ("linear group-contrib.  (held-out)", "#9b5de5"),
    "eq":          ("eQuilibrator  (in-sample*)",       "#8d99ae"),
}


def _load():
    d = json.load(open(BENCH))
    rows = [r for r in d.values() if isinstance(r, dict) and r.get("exp") is not None]
    exp = np.array([r["exp"] for r in rows])
    return rows, exp


def _series(rows, exp, key):
    m = np.array([r.get(key) is not None for r in rows])
    return exp[m], np.array([r[key] for r in rows if r.get(key) is not None], float)


def _panel(x, y, label, color, fname):
    slope, icpt = np.polyfit(x, y, 1)
    mae = float(np.abs(y - x).mean())
    fig, ax = plt.subplots(figsize=(7.8, 7.6))
    ax.axvspan(-LIM, -30, color="#f2c94c", alpha=0.12)
    ax.axvspan(30, LIM, color="#f2c94c", alpha=0.12)
    ax.plot([-LIM, LIM], [-LIM, LIM], "-", color="#444", lw=2)
    ax.scatter(x, y, s=40, alpha=0.55, color=color, edgecolor="none")
    xs = np.array([-LIM, LIM])
    ax.plot(xs, slope * xs + icpt, "--", color=color, lw=2.6)
    ax.set_xlabel("experimental Δ$_r$G'°  (kJ/mol)")
    ax.set_ylabel("predicted Δ$_r$G'°  (kJ/mol)")
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.text(0.04, 0.96, label, transform=ax.transAxes, va="top", ha="left", fontsize=17)
    ax.text(0.04, 0.88, f"slope {slope:.2f}   MAE {mae:.1f}   n={len(x)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=16, color=color)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight")
    plt.close(fig)
    return slope, mae, len(x)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    rows, exp = _load()
    summ = {}
    for key, (label, color) in METHODS.items():
        x, y = _series(rows, exp, key)
        if len(x) < 5:
            continue
        summ[key] = _panel(x, y, label, color, f"pred_vs_exp_{key}.png")

    # headline 2-method overlay (UMA-QM vs dGPredictor)
    xu, yu = _series(rows, exp, "uma_qm")
    xd, yd = _series(rows, exp, "dgp_retrain")
    su = np.polyfit(xu, yu, 1); sd = np.polyfit(xd, yd, 1)
    fig, ax = plt.subplots(figsize=(8.3, 8))
    ax.axvspan(-LIM, -30, color="#f2c94c", alpha=0.12)
    ax.axvspan(30, LIM, color="#f2c94c", alpha=0.12)
    ax.plot([-LIM, LIM], [-LIM, LIM], "-", color="#444", lw=2, label="perfect (y = x)")
    ax.scatter(xd, yd, s=40, alpha=0.5, color="#e07b39", edgecolor="none",
               label=f"dGPredictor (slope {sd[0]:.2f})")
    ax.scatter(xu, yu, s=40, alpha=0.5, color="#3b6fb0", edgecolor="none",
               label=f"UMA-QM (slope {su[0]:.2f})")
    xs = np.array([-LIM, LIM])
    ax.plot(xs, sd[0]*xs+sd[1], "--", color="#e07b39", lw=2.6)
    ax.plot(xs, su[0]*xs+su[1], "--", color="#3b6fb0", lw=2.6)
    ax.set_xlabel("experimental Δ$_r$G'°  (kJ/mol)")
    ax.set_ylabel("predicted Δ$_r$G'°  (kJ/mol)")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
    ax.legend(loc="upper left", frameon=False, fontsize=15)
    ax.text(0.985, 0.03, "shaded = |Δ$_r$G'°|>30 (far from eq.)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=14, color="#8a6d1a")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "pred_vs_exp_shrinkage.png"), bbox_inches="tight")
    plt.close(fig)

    # slope-by-method bar (compact summary of the whole argument)
    order = [k for k in METHODS if k in summ]
    slopes = [summ[k][0] for k in order]
    labels = [METHODS[k][0].split("  ")[0] for k in order]
    colors = [METHODS[k][1] for k in order]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    xp = np.arange(len(order))
    ax.bar(xp, slopes, color=colors, alpha=0.85)
    ax.axhline(1.0, color="#444", lw=2, ls="-")
    ax.text(len(order)-0.5, 1.02, "tracks truth (slope=1)", ha="right", va="bottom", fontsize=15, color="#444")
    ax.set_xticks(xp); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("pred-vs-exp slope")
    ax.set_ylim(0, 1.2)
    for x0, s in zip(xp, slopes):
        ax.text(x0, s + 0.02, f"{s:.2f}", ha="center", va="bottom", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "slope_by_method.png"), bbox_inches="tight")
    plt.close(fig)

    # MAE vs |exp dG| — the crossover: fitted methods look great near equilibrium (small |dG|) but
    # their error climbs as reactions get far from equilibrium (shrinkage); UMA-QM stays flat.
    bins = [(0, 5), (5, 15), (15, 30), (30, 200)]
    blabels = ["0-5", "5-15", "15-30", ">30"]
    fig, ax = plt.subplots(figsize=(9, 6.8))
    for key in ("uma_qm", "dgp_retrain", "gnn"):
        x, y = _series(rows, exp, key)
        ae = np.abs(y - x)
        ax.plot(range(len(bins)),
                [ae[(np.abs(x) >= lo) & (np.abs(x) < hi)].mean() for lo, hi in bins],
                "-o", lw=2.6, ms=9, color=METHODS[key][1], label=METHODS[key][0].split("  ")[0])
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(blabels)
    ax.set_xlabel("|experimental Δ$_r$G'°|  bin  (kJ/mol)")
    ax.set_ylabel("MAE in bin  (kJ/mol)")
    ax.legend(loc="upper left", frameon=False, fontsize=16)
    ax.text(0.985, 0.03, "far-from-eq = right side (directionality matters)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=14, color="#666")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "mae_vs_dG_crossover.png"), bbox_inches="tight")
    plt.close(fig)

    print("wrote per-method pred_vs_exp panels:", ", ".join(order))
    print("  + pred_vs_exp_shrinkage.png, slope_by_method.png, mae_vs_dG_crossover.png")
    for k in order:
        print(f"  {k:12s} slope={summ[k][0]:.2f}  MAE={summ[k][1]:.1f}  n={summ[k][2]}")


if __name__ == "__main__":
    main()
