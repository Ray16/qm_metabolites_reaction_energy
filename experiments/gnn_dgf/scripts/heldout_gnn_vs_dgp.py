#!/usr/bin/env python
"""Apples-to-apples HELD-OUT comparison: GNN vs dGPredictor on IDENTICAL folds.

The headline number people quote for dGPredictor/eQuilibrator (MAE ~3.0) is
IN-SAMPLE: those models are fit on TECRDB and then scored on TECRDB.  That is not
a generalization result.  This script removes that advantage by running
dGPredictor's OWN model -- BayesianRidge(fit_intercept=False) on its own 1114-dim
r1/r2 group-difference features (results/eq/dgp_group_features.json) -- through
the SAME cross-validation folds as the GNN, so BOTH are held-out and directly
comparable.  For context we also print the in-sample dGP/eQ numbers, clearly
labelled, so the reader sees exactly how much of the incumbents' edge is leakage.

Schemes (identical folds for every method, several seeds):
  RANDOM        5-fold reaction CV (interpolation)
  CPD-DISJOINT  held-out COMPOUNDS (extrapolation; the coverage regime)

Methods:
  GNN (held-out)          graph-only ensemble, S@f, per-fold retrained
  dGPredictor (held-out)  its BayesianRidge on its group features, per-fold refit
  dGPredictor (in-sample) same model fit on ALL 367, scored on all -- the leaked
                          number, for reference only
  eQuilibrator (in-sample) the deployed tool's stored predictions (also leaked)

Outputs: artifacts/heldout_gnn_vs_dgp.json, figures/heldout_gnn_vs_dgp.png
Run: CUDA_VISIBLE_DEVICES=1 python scripts/heldout_gnn_vs_dgp.py
Quick: ... --smoke
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import BayesianRidge

from gnn import paths
from gnn.model import Graph, DEV
from gnn.training import gnn_predict, kfold, compound_disjoint


def dgp_heldout(X, y, folds, have):
    """dGPredictor's own model, per-fold refit. `have` = bool mask of reactions
    whose features exist (retrained decomposition misses a few compounds). A
    reaction with no features is never fit and gets NaN in the OOF vector."""
    N = len(y)
    oof = np.full(N, np.nan)
    for tr, te in folds:
        tr = np.asarray(tr)[have[np.asarray(tr)]]
        te = np.asarray(te)[have[np.asarray(te)]]
        if len(te) == 0 or len(tr) == 0:
            continue
        m = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def dgp_insample(X, y, have):
    oof = np.full(len(y), np.nan)
    m = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
    m.fit(X[have], y[have])
    oof[have] = m.predict(X[have])
    return oof


def load_dgp_features(path, rxn_ids, dim):
    """Return (X float array with 0-rows where missing, have-mask)."""
    d = json.load(open(path))["X"]
    have = np.array([d.get(r) is not None for r in rxn_ids])
    X = np.array([d[r] if d.get(r) is not None else [0.0] * dim for r in rxn_ids], dtype=float)
    return X, have


def gnn_heldout(g, S, yt, w, folds, ens, epochs, seed):
    N = yt.shape[0]
    oof = np.full(N, np.nan)
    for tr, te in folds:
        if len(te) == 0:
            continue
        p = gnn_predict(g, S, yt, w, tr, epochs, seed=seed, n_ens=ens)
        oof[te] = p[torch.as_tensor(te, device=DEV)].cpu().numpy()
    return oof


def metrics(pred, y):
    m = ~np.isnan(pred)
    p, t = pred[m], y[m]
    e = p - t
    ss = 1 - (e ** 2).sum() / ((t - t.mean()) ** 2).sum()
    sign = np.mean(np.sign(p) == np.sign(t)) * 100
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                medAE=float(np.median(np.abs(e))), R2=float(ss), sign_pct=float(sign),
                n=int(m.sum()))


def paired_ci(err_a, err_b, iters=2000, seed=0):
    """95% CI on MAE(a)-MAE(b), bootstrap over reactions (paired)."""
    m = ~(np.isnan(err_a) | np.isnan(err_b))
    a, b = err_a[m], err_b[m]
    rng = np.random.default_rng(seed)
    d = np.array([a[idx].mean() - b[idx].mean()
                  for idx in (rng.integers(0, len(a), len(a)) for _ in range(iters))])
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["none", "rich"], default="none",
                    help="'none' = graph-only GNN (deployable); 'rich' = with xtb QC")
    ap.add_argument("--ens", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.ens, a.epochs, a.seeds = 2, 100, 1

    d = torch.load(paths.artifact("data.pt"))
    rxn_ids = d["rxn_ids"]; N = len(rxn_ids)
    S = d["S"].to(DEV); y = d["y"].to(DEV); n = d["n"].to(DEV)
    yn = y.cpu().numpy()
    w = torch.log1p(n); w = w / w.mean()
    g = Graph(d["graphs"][a.level])

    # dGPredictor's OWN features, aligned to rxn_ids -- BOTH variants:
    #   original  : KEGG group basis (decompose_vector_ac), 1114 non-constant cols
    #   retrained : ModelSEED group basis (Freiburger), 1421 cols, misses 3 rxns
    Xorig, have_orig = load_dgp_features(
        os.path.join(paths.RESULTS, "eq", "dgp_group_features.json"), rxn_ids, 1114)
    Xretr, have_retr = load_dgp_features(
        os.path.join(paths.RESULTS, "eq", "dgp_retrained_group_features.json"), rxn_ids, 1421)
    print(f"dGP features: original {have_orig.sum()}/{N} rxns (dim {Xorig.shape[1]}), "
          f"retrained {have_retr.sum()}/{N} rxns (dim {Xretr.shape[1]})")

    # eQ in-sample stored predictions (context)
    eqf = json.load(open(os.path.join(paths.RESULTS, "eq", "equilibrator_full.json")))
    eq_pred = np.array([eqf[r]["dG_kJ"] if r in eqf and eqf[r].get("dG_kJ") is not None
                        else np.nan for r in rxn_ids])

    schemes = {"RANDOM": lambda s: kfold(N, a.folds, s)}
    if not a.smoke:
        schemes["CPD-DISJOINT"] = lambda s: compound_disjoint(
            d["rxn_comps"], d["n_comp"], a.folds, s)

    print(f"device={DEV}  N={N}  GNN level={a.level}  ens={a.ens}  seeds={a.seeds}\n")
    results = {}

    # in-sample references (fold-independent) -- the "leaked" numbers people quote
    ref = {
        "dGP-original (in-sample, leaked)": metrics(dgp_insample(Xorig, yn, have_orig), yn),
        "dGP-retrained (in-sample, leaked)": metrics(dgp_insample(Xretr, yn, have_retr), yn),
        "eQuilibrator (in-sample, leaked)": metrics(eq_pred, yn),
    }

    for scheme, mkfolds in schemes.items():
        print(f"================ {scheme} (held-out) ================")
        gnn_s, orig_s, retr_s = [], [], []
        for s in range(a.seeds):
            folds = mkfolds(s)
            gnn_s.append(gnn_heldout(g, S, y, w, folds, a.ens, a.epochs, seed=s))
            orig_s.append(dgp_heldout(Xorig, yn, folds, have_orig))
            retr_s.append(dgp_heldout(Xretr, yn, folds, have_retr))
        gnn_oof = np.nanmean(np.stack(gnn_s), 0)
        orig_oof = np.nanmean(np.stack(orig_s), 0)
        retr_oof = np.nanmean(np.stack(retr_s), 0)

        mg = metrics(gnn_oof, yn)
        mo = metrics(orig_oof, yn)
        mr = metrics(retr_oof, yn)
        # paired GNN vs each dGP variant (per-reaction |err|); pair only where both defined
        def paired(other_oof):
            dlt, lo, hi = paired_ci(np.abs(gnn_oof - yn), np.abs(other_oof - yn))
            v = ("GNN better" if hi < 0 else "dGP better" if lo > 0 else "tie (within noise)")
            return dlt, lo, hi, v
        do = paired(orig_oof); dr = paired(retr_oof)

        for nm, m in (("GNN (held-out)", mg), ("dGP-original (held-out)", mo),
                      ("dGP-retrained (held-out)", mr)):
            print(f"  {nm:<28s} MAE {m['MAE']:5.2f}  RMSE {m['RMSE']:5.2f}  medAE {m['medAE']:4.2f}"
                  f"  sign {m['sign_pct']:4.0f}%  R2 {m['R2']:+.2f}  n={m['n']}")
        print(f"  paired  GNN-vs-original  ΔMAE {do[0]:+.2f} [{do[1]:+.2f},{do[2]:+.2f}]  {do[3]}")
        print(f"  paired  GNN-vs-retrained ΔMAE {dr[0]:+.2f} [{dr[1]:+.2f},{dr[2]:+.2f}]  {dr[3]}")
        print(f"  (in-sample ref: dGP-orig {ref['dGP-original (in-sample, leaked)']['MAE']:.2f}, "
              f"dGP-retr {ref['dGP-retrained (in-sample, leaked)']['MAE']:.2f}, "
              f"eQ {ref['eQuilibrator (in-sample, leaked)']['MAE']:.2f})\n")

        results[scheme] = dict(
            gnn_heldout=mg, dgp_original_heldout=mo, dgp_retrained_heldout=mr,
            paired_gnn_vs_original=dict(dMAE=do[0], ci=[do[1], do[2]], verdict=do[3]),
            paired_gnn_vs_retrained=dict(dMAE=dr[0], ci=[dr[1], dr[2]], verdict=dr[3]),
            _gnn_oof=gnn_oof.tolist(), _orig_oof=orig_oof.tolist(), _retr_oof=retr_oof.tolist())

    results["in_sample_reference"] = ref
    results["_meta"] = dict(level=a.level, ens=a.ens, seeds=a.seeds, epochs=a.epochs,
                            folds=a.folds,
                            note="GNN + BOTH dGPredictor variants (original KEGG basis, "
                                 "retrained ModelSEED basis) run held-out on IDENTICAL CV "
                                 "folds; dGP model = BayesianRidge(fit_intercept=False) on its "
                                 "own group features, coefficients refit per fold. In-sample "
                                 "rows are leaked (train==test), reference only.")
    save = {k: (v if not isinstance(v, dict) else
                {kk: vv for kk, vv in v.items() if not kk.startswith("_")})
            for k, v in results.items() if k != "_meta"}
    save["_meta"] = {k: v for k, v in results["_meta"].items() if k not in ("y",)}
    json.dump(save, open(paths.artifact("heldout_gnn_vs_dgp.json"), "w"), indent=2)
    print("wrote artifacts/heldout_gnn_vs_dgp.json")

    plot(results, os.path.join(paths.ROOT, "figures", "heldout_gnn_vs_dgp.png"))
    plot(results, os.path.join(paths.FIGURES, "heldout_gnn_vs_dgp.png"))


def plot(results, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skip figure: {e})"); return
    schemes = [k for k in results if k in ("RANDOM", "CPD-DISJOINT")]
    ref = results["in_sample_reference"]
    fig, axes = plt.subplots(1, len(schemes), figsize=(6.2 * len(schemes), 5.2),
                             squeeze=False)
    for ax, sch in zip(axes[0], schemes):
        r = results[sch]
        names = ["GNN\n(held-out)", "dGP-orig\n(held-out)", "dGP-retr\n(held-out)",
                 "dGP-orig\n(in-sample*)", "eQ\n(in-sample*)"]
        vals = [r["gnn_heldout"]["MAE"], r["dgp_original_heldout"]["MAE"],
                r["dgp_retrained_heldout"]["MAE"],
                ref["dGP-original (in-sample, leaked)"]["MAE"],
                ref["eQuilibrator (in-sample, leaked)"]["MAE"]]
        # held-out = solid; in-sample = pale (leaked)
        cols = ["#2a7fb8", "#2ca25f", "#66c2a4", "#c6dbef", "#c7e9c0"]
        bars = ax.bar(range(5), vals, color=cols)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.axhline(2.0, color="#888", ls=":", lw=1)
        ax.text(4.4, 2.1, "exp. noise floor ~2 kJ", color="#888", fontsize=8,
                va="bottom", ha="right")
        ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=8.5)
        ax.set_ylabel("MAE vs experiment (kJ/mol)")
        ax.set_title(f"{sch} cross-validation", fontsize=12, fontweight="bold")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Held-out, GNN matches dGPredictor; the incumbents' famous ~3 kJ is in-sample (leaked)",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005, "* in-sample = trained on TECRDB then scored on TECRDB (train=test), NOT "
             "generalization. Solid bars = held-out (honest). dGP variants use their own group "
             "basis, coefficients refit per fold on identical folds.",
             ha="center", fontsize=8.5, color="#555")
    fig.subplots_adjust(bottom=0.17, top=0.88)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
