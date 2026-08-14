#!/usr/bin/env python
"""Phase 1a -- calibrated uncertainty for the standalone GNN predictor.

The ablation established that the seed-ensemble spread RANKS error (Spearman
~0.28, monotone selective prediction). Ranking is not enough for a package: a
user needs a usable *interval*, i.e. a sigma with the right MAGNITUDE. This
script turns the raw ensemble spread into a calibrated predictive sigma and
proves the calibration on held-out data.

Model of the predictive variance (two physically-distinct parts):

    sigma_total^2 = s^2 * sigma_ens^2 + tau^2
                     \\_____________/   \\____/
                       epistemic          aleatoric floor
                    (ensemble spread,   (irreducible TECRDB
                     rescaled by s)      measurement noise ~6 kJ)

(s, tau) are fit by Gaussian negative-log-likelihood on HELD-OUT out-of-fold
errors -- the method's own experimental residuals, NOT any other predictor.
To avoid fitting-and-scoring on the same points, the OOF reactions are split in
half: (s, tau) fit on one half, coverage/reliability scored on the other, both
directions averaged.

Reported for BOTH cv schemes:
  RANDOM        -- in-domain interpolation
  CPD-DISJOINT  -- extrapolation to unseen compounds == the coverage/frontier
                   regime the whole package exists for; this is the sigma that
                   travels to the QM-audit frontier.

Outputs: artifacts/uq_calibration.json  (s, tau per scheme + metrics),
         figures/uq_calibration.png     (reliability + selective-prediction).
Run:  CUDA_VISIBLE_DEVICES=0 python scripts/calibrate_uq.py
Quick: ... scripts/calibrate_uq.py --smoke
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os

import numpy as np
import torch

from gnn import data, paths
from gnn.model import Graph, DEV
from gnn.training import gnn_predict, kfold, compound_disjoint

SQRT2PI = np.sqrt(2.0 * np.pi)


def oof_stack(g, S, y, w, yn, folds, hp, ens, epochs, seed):
    """Out-of-fold ensemble mean + std per reaction (each reaction predicted by
    the fold that holds it out). Returns (mean, std, tested_mask)."""
    N = len(yn)
    mean_p = np.full(N, np.nan)
    std_p = np.full(N, np.nan)
    for tr, te in folds:
        if len(te) == 0:
            continue
        stack = gnn_predict(g, S, y, w, tr, epochs, hp, seed=seed,
                            n_ens=ens, return_stack=True)          # (ens, N)
        te_t = torch.as_tensor(te, device=DEV)
        mean_p[te] = stack[:, te_t].mean(0).cpu().numpy()
        std_p[te] = stack[:, te_t].std(0).cpu().numpy()
    return mean_p, std_p, ~np.isnan(mean_p)


def nll(err, sig_ens, s, tau):
    """Mean Gaussian NLL of residual `err` under sigma^2 = s^2 sig_ens^2 + tau^2."""
    var = (s * sig_ens) ** 2 + tau ** 2
    var = np.maximum(var, 1e-6)
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + err ** 2 / (2 * var)))


def fit_calibration(err, sig_ens):
    """Grid-search (s, tau) minimizing Gaussian NLL; coarse then refine. No scipy
    dependency (torch/rdkit env is fragile). s in [0.2,10], tau in [0,15] kJ."""
    def search(s_lo, s_hi, t_lo, t_hi, k=40):
        ss = np.linspace(s_lo, s_hi, k)
        tt = np.linspace(t_lo, t_hi, k)
        best = (1e18, 1.0, 0.0)
        for s in ss:
            for t in tt:
                v = nll(err, sig_ens, s, t)
                if v < best[0]:
                    best = (v, s, t)
        return best
    _, s0, t0 = search(0.2, 10.0, 0.0, 15.0)
    ds, dt = 10.0 / 40, 15.0 / 40
    nll1, s1, t1 = search(max(0.05, s0 - ds), s0 + ds, max(0.0, t0 - dt), t0 + dt)
    return float(s1), float(t1), float(nll1)


def coverage(err, sig_total, z):
    """Empirical fraction of |err| within z*sigma (nominal 2-sided coverage)."""
    return float(np.mean(np.abs(err) <= z * sig_total))


def spearman(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    rx = np.argsort(np.argsort(x[m])); ry = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(rx, ry)[0, 1])


# nominal two-sided z for 50/68/90/95 %
Z = {50: 0.674, 68: 0.994, 90: 1.645, 95: 1.960}


def evaluate_scheme(name, g, S, y, w, yn, mkfolds, hp, ens, epochs, seeds):
    """Aggregate OOF mean/std over CV seeds, fit calibration on half / score on
    the other (both directions), and compute reliability + selective prediction."""
    N = len(yn)
    means, stds = [], []
    for s in range(seeds):
        m, sd, mask = oof_stack(g, S, y, w, yn, mkfolds(s), hp, ens, epochs, seed=s)
        means.append(m); stds.append(sd)
    mean_p = np.nanmean(np.stack(means), 0)
    sig_ens = np.nanmean(np.stack(stds), 0)                 # avg spread over seeds
    m = ~(np.isnan(mean_p) | np.isnan(sig_ens))
    err = (mean_p - yn)[m]
    sig_ens = sig_ens[m]
    aerr = np.abs(err)

    rho = spearman(sig_ens, aerr)

    # honest calibration: split OOF points in two, fit on one / score on other
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(err))
    half = len(err) // 2
    A, B = idx[:half], idx[half:]
    s_A, t_A, _ = fit_calibration(err[A], sig_ens[A])
    s_B, t_B, _ = fit_calibration(err[B], sig_ens[B])
    # scored coverage uses the calibration fit on the OTHER half
    sig_tot = np.empty_like(err)
    sig_tot[B] = np.sqrt((s_A * sig_ens[B]) ** 2 + t_A ** 2)
    sig_tot[A] = np.sqrt((s_B * sig_ens[A]) ** 2 + t_B ** 2)
    # a single reported (s, tau) fit on all points (for the shipped predictor)
    s_all, t_all, nll_all = fit_calibration(err, sig_ens)
    sig_raw = sig_ens.copy()

    cov_raw = {p: coverage(err, np.maximum(sig_raw, 1e-6), Z[p]) for p in Z}
    cov_cal = {p: coverage(err, sig_tot, Z[p]) for p in Z}

    # reliability: bin by predicted sigma_total, compare to empirical RMS error
    order = np.argsort(sig_tot)
    nb = 5
    rel = []
    for b in range(nb):
        sel = order[b * len(order) // nb:(b + 1) * len(order) // nb]
        rel.append(dict(pred_sigma=float(sig_tot[sel].mean()),
                        emp_rms=float(np.sqrt((err[sel] ** 2).mean())),
                        n=int(len(sel))))

    # selective prediction: MAE over most-confident fraction (by sigma_total)
    o = np.argsort(sig_tot)
    sel_mae = {int(f * 100): float(aerr[o[:max(1, int(f * len(aerr)))]].mean())
               for f in (0.5, 0.7, 0.9, 1.0)}

    return dict(
        n=int(len(err)), mae=float(aerr.mean()), rmse=float(np.sqrt((err ** 2).mean())),
        spearman_spread_abserr=rho,
        calib=dict(scale_s=s_all, tau_kJ=t_all, nll=nll_all,
                   split_s=[s_A, s_B], split_tau=[t_A, t_B]),
        coverage_raw=cov_raw, coverage_calibrated=cov_cal,
        reliability=rel, selective_mae=sel_mae,
        mean_sigma_raw=float(sig_raw.mean()), mean_sigma_cal=float(sig_tot.mean()),
        _err=err.tolist(), _sig_tot=sig_tot.tolist(), _sig_raw=sig_raw.tolist())


def plot(results, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skip figure: {e})")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    colors = {"RANDOM": "#2c7fb8", "CPD-DISJOINT": "#d95f0e"}
    # reliability
    ax = axes[0]
    lo = hi = 0
    for name, r in results.items():
        xs = [b["pred_sigma"] for b in r["reliability"]]
        ys = [b["emp_rms"] for b in r["reliability"]]
        ax.plot(xs, ys, "o-", color=colors[name], label=name)
        hi = max(hi, max(xs + ys))
    ax.plot([0, hi], [0, hi], "k--", lw=1, alpha=.6, label="ideal")
    ax.set_xlabel("predicted $\\sigma_{total}$ (kJ/mol)")
    ax.set_ylabel("actual RMS error (kJ/mol)")
    ax.set_title("Are the error bars honest?\ndots on the dashed line = model says ±X, real error ≈ X",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    # selective prediction
    ax = axes[1]
    for name, r in results.items():
        fr = [50, 70, 90, 100]
        ax.plot(fr, [r["selective_mae"][k] for k in fr], "o-", color=colors[name],
                label=f"{name} ($\\rho$={r['spearman_spread_abserr']:.2f})")
    ax.set_xlabel("most-confident fraction kept (%)")
    ax.set_ylabel("MAE over kept (kJ/mol)")
    ax.set_title("Does dropping the unsure ones help?\ncurve sloping down-left = confident predictions really are more accurate",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Prediction uncertainty (ensemble spread), calibrated on held-out TECRDB error",
                 fontsize=13, fontweight="bold")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ens", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--level", choices=["none", "solv", "full", "rich"], default="rich",
                    help="'none' = graph-only (matches the deployable xtb-free model)")
    ap.add_argument("--replot", action="store_true",
                    help="redraw the figure from a saved uq_calibration.json (no recompute)")
    a = ap.parse_args()
    if a.smoke:
        a.seeds, a.ens, a.epochs, a.folds = 1, 4, 100, 4
    sfx = "" if a.level == "rich" else f"_{a.level}"
    json_name = f"uq_calibration{sfx}.json"
    fig_name = f"uq_calibration{sfx}.png"

    if a.replot:
        saved = json.load(open(paths.artifact(json_name)))
        results = {}
        for name, r in saved.items():
            if name.startswith("_"):
                continue
            r["selective_mae"] = {int(k): v for k, v in r["selective_mae"].items()}
            results[name] = r
        plot(results, os.path.join(paths.FIGURES, fig_name))
        plot(results, os.path.join(paths.ROOT, "figures", fig_name))
        return

    d = torch.load(paths.artifact("data.pt"))
    rxn_ids = d["rxn_ids"]; N = len(rxn_ids)
    S = d["S"].to(DEV); y = d["y"].to(DEV); n = d["n"].to(DEV)
    yn = y.cpu().numpy()
    g = Graph(d["graphs"][a.level])
    hp = dict(hidden=96, layers=3, drop=0.1, lr=3e-3, wd=1e-4)
    w = torch.log1p(n); w = w / w.mean()

    schemes = {"RANDOM": lambda s: kfold(N, a.folds, s)}
    if not a.smoke:
        schemes["CPD-DISJOINT"] = lambda s: compound_disjoint(
            d["rxn_comps"], d["n_comp"], a.folds, s)

    print(f"device={DEV}  N={N}  ens={a.ens}  seeds={a.seeds}  epochs={a.epochs}\n")
    results = {}
    for name, mkfolds in schemes.items():
        print(f"================ {name} ================")
        r = evaluate_scheme(name, g, S, y, w, yn, mkfolds, hp, a.ens, a.epochs, a.seeds)
        results[name] = r
        c = r["calib"]
        print(f"  n={r['n']}  MAE={r['mae']:.2f}  RMSE={r['rmse']:.2f}")
        print(f"  Spearman(spread,|err|) = {r['spearman_spread_abserr']:+.3f}")
        print(f"  calibration:  sigma_total^2 = ({c['scale_s']:.2f}*sigma_ens)^2 "
              f"+ ({c['tau_kJ']:.2f})^2   [s split {c['split_s'][0]:.2f}/"
              f"{c['split_s'][1]:.2f}, tau split {c['split_tau'][0]:.1f}/{c['split_tau'][1]:.1f}]")
        print(f"  mean sigma: raw {r['mean_sigma_raw']:.2f} -> calibrated {r['mean_sigma_cal']:.2f} kJ")
        print("  interval coverage (nominal -> raw / calibrated):")
        for p in (50, 68, 90, 95):
            print(f"     {p:3d}%  ->  {r['coverage_raw'][p]*100:4.0f}% / "
                  f"{r['coverage_calibrated'][p]*100:4.0f}%")
        print("  selective-prediction MAE (top confident %):")
        for k in (50, 70, 90, 100):
            print(f"     top {k:3d}%   MAE {r['selective_mae'][k]:.2f}")
        print()

    plot(results, os.path.join(paths.FIGURES, fig_name))
    # local copy in the experiment figures dir too
    plot(results, os.path.join(paths.ROOT, "figures", fig_name))

    # strip bulky per-point arrays before saving the summary json
    save = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in results.items()}
    save["_meta"] = dict(ens=a.ens, folds=a.folds, seeds=a.seeds, epochs=a.epochs, level=a.level,
                         model="sigma_total^2 = s^2 sigma_ens^2 + tau^2 (Gaussian NLL, "
                               "fit on held-out OOF residuals; TECRDB experiment only)")
    json.dump(save, open(paths.artifact(json_name), "w"), indent=2)
    print(f"wrote artifacts/{json_name}")


if __name__ == "__main__":
    main()
