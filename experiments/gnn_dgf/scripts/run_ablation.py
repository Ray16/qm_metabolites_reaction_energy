#!/usr/bin/env python
"""Controlled ablation: does each proposed improvement ACTUALLY move held-out MAE?

Every variant is trained on IDENTICAL CV folds against the same baseline, over
several CV seeds, and compared with a paired bootstrap CI on the per-reaction
residuals -- so a sub-noise delta is reported as "within noise", not as a win
(the failure mode that sank earlier claims).

Baseline = the production config in run_cv.py: level 'rich', DEFAULT_HP
(hidden 96 / 3 layers), w = log1p(n) weighting, MSE.

Variants:
  #6  robust / uncertainty-weighted loss   inv-variance w=n/sd2, count w=n, Huber
  #1  measurement-condition head           dG = S@f + h(pH, I, T, pMg)
  #4  QM injected into message passing      (vs readout-only)
  #7  ensemble-variance uncertainty         calibration + selective prediction

Also prints a metals audit (#3): the test set is enzyme reactions, so few metals
-> that improvement belongs to the (unbuilt) coverage eval, not TECRDB CV.

Run:  CUDA_VISIBLE_DEVICES=0 python scripts/run_ablation.py
Quick:                          ... scripts/run_ablation.py --smoke
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np
import torch

from gnn import data, paths
from gnn.model import Graph, DEV
from gnn.training import gnn_predict, kfold, compound_disjoint


# ----------------------------------------------------------------------------- CV
def oof_errors(g, S, y, yn, folds, hp, w, ens, epochs, seed, **kw):
    """Out-of-fold |pred - y| per reaction (mean over folds a reaction is tested
    in; folds are identical across variants so pairing is valid)."""
    N = len(yn)
    acc = [[] for _ in range(N)]
    for tr, te in folds:
        if len(te) == 0:
            continue
        p = gnn_predict(g, S, y, w, tr, epochs, hp, seed=seed, n_ens=ens, **kw)
        pe = p[torch.as_tensor(te, device=DEV)].cpu().numpy()
        for j, i in enumerate(te):
            acc[i].append(abs(pe[j] - yn[i]))
    return np.array([np.mean(a) if a else np.nan for a in acc])


def paired_ci(err_v, err_b, iters=2000, seed=0):
    """95% CI on MAE(variant) - MAE(baseline), bootstrapping over reactions."""
    m = ~(np.isnan(err_v) | np.isnan(err_b))
    ev, eb = err_v[m], err_b[m]
    rng = np.random.default_rng(seed)
    d = np.array([ev[idx].mean() - eb[idx].mean()
                  for idx in (rng.integers(0, len(ev), len(ev)) for _ in range(iters))])
    return float(ev.mean() - eb.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def verdict(delta, lo, hi):
    if hi < 0:
        return "HELPS"
    if lo > 0:
        return "HURTS"
    return "within noise"


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ens", type=int, default=3)
    ap.add_argument("--uq-ens", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="1 seed, ens1, 100 ep, random only")
    a = ap.parse_args()
    if a.smoke:
        a.seeds, a.ens, a.epochs, a.uq_ens = 1, 1, 100, 4

    d = torch.load(paths.artifact("data.pt"))
    rxn_ids = d["rxn_ids"]
    N = len(rxn_ids)
    S = d["S"].to(DEV); y = d["y"].to(DEV); n = d["n"].to(DEV)
    yn = y.cpu().numpy()
    g = Graph(d["graphs"]["rich"])

    # sd + conditions, aligned to data.pt's rxn_ids
    _, _, ens_qm, tgt, _ = data.load_tecrdb()
    _, _, sd = data.targets(tgt, rxn_ids)
    sd = sd.to(DEV)
    cond = data.load_conditions(rxn_ids, paths.artifact("rxn_conditions.json")).to(DEV)

    hp = dict(hidden=96, layers=3, drop=0.1, lr=3e-3, wd=1e-4)          # production HP
    w_log = torch.log1p(n); w_log = w_log / w_log.mean()               # baseline weight
    w_ivar = n / sd.clamp(min=1.0) ** 2; w_ivar = w_ivar / w_ivar.mean()
    w_cnt = n / n.mean()
    w_unit = torch.ones(N, device=DEV)

    schemes = {"RANDOM": lambda s: kfold(N, a.folds, s)}
    if not a.smoke:
        schemes["CPD-DISJOINT"] = lambda s: compound_disjoint(d["rxn_comps"], d["n_comp"], a.folds, s)

    # variant registry: name -> kwargs to oof_errors (baseline first)
    variants = {
        "baseline (log-n · MSE)":      dict(hp=hp, w=w_log, loss="mse"),
        "#6 inv-variance weight":      dict(hp=hp, w=w_ivar, loss="mse"),
        "#6 count weight (n)":         dict(hp=hp, w=w_cnt, loss="mse"),
        "#6 uniform weight":           dict(hp=hp, w=w_unit, loss="mse"),
        "#6 Huber (δ=6)":              dict(hp=hp, w=w_log, loss="huber", huber_delta=6.0),
        "#1 condition head":           dict(hp=hp, w=w_log, loss="mse", cond=cond),
        "#4 QM-in-messages":           dict(hp={**hp, "qm_in_messages": True}, w=w_log, loss="mse"),
    }

    print(f"device={DEV}  N={N}  seeds={a.seeds}  ens={a.ens}  epochs={a.epochs}\n")
    results = {}
    for scheme, mkfolds in schemes.items():
        print(f"================ {scheme} ================")
        # per-variant: stack of OOF error vectors over seeds
        err = {name: [] for name in variants}
        for s in range(a.seeds):
            folds = mkfolds(s)
            for name, cfg in variants.items():
                err[name].append(oof_errors(g, S, y, yn, folds, ens=a.ens,
                                            epochs=a.epochs, seed=s, **cfg))
        base = np.nanmean(np.stack(err["baseline (log-n · MSE)"]), 0)
        results[scheme] = {}
        for name in variants:
            stk = np.stack(err[name])                      # (seeds, N)
            mae_per_seed = np.nanmean(stk, 1)
            mean_err = np.nanmean(stk, 0)                   # pooled over seeds
            if name == "baseline (log-n · MSE)":
                dlt, lo, hi, vd = 0.0, 0.0, 0.0, "—"
            else:
                dlt, lo, hi = paired_ci(mean_err, base)
                vd = verdict(dlt, lo, hi)
            results[scheme][name] = dict(mae=float(mae_per_seed.mean()),
                                         mae_sd=float(mae_per_seed.std()),
                                         delta=dlt, ci=[lo, hi], verdict=vd)
            print(f"  {name:<26s} MAE {mae_per_seed.mean():5.2f} ±{mae_per_seed.std():.2f}"
                  f"   Δ {dlt:+5.2f} [{lo:+.2f},{hi:+.2f}]  {vd}")
        print()

    # ---- #7 uncertainty calibration (RANDOM CV, larger ensemble) --------------
    print("================ #7 ensemble-variance UQ (RANDOM) ================")
    folds = kfold(N, a.folds, 0)
    mean_p = np.full(N, np.nan); std_p = np.full(N, np.nan)
    for tr, te in folds:
        stack = gnn_predict(g, S, y, w_log, tr, a.epochs, hp, seed=0,
                            n_ens=a.uq_ens, return_stack=True)
        te_t = torch.as_tensor(te, device=DEV)
        mean_p[te] = stack[:, te_t].mean(0).cpu().numpy()
        std_p[te] = stack[:, te_t].std(0).cpu().numpy()
    aerr = np.abs(mean_p - yn)
    order = np.argsort(std_p)                               # most-confident first
    rho = _spearman(std_p, aerr)
    sel = {int(f * 100): float(aerr[order[:max(1, int(f * N))]].mean())
           for f in (0.5, 0.7, 0.9, 1.0)}
    print(f"  Spearman(ensemble std, |error|) = {rho:+.3f}  (>0 => spread tracks error)")
    print("  selective-prediction MAE by most-confident fraction:")
    for k in (50, 70, 90, 100):
        print(f"     top {k:3d}%   MAE {sel[k]:.2f}")
    results["UQ"] = dict(spearman=rho, selective=sel,
                         uq_ens=a.uq_ens)

    # ---- #3 metals audit ------------------------------------------------------
    print("\n================ #3 metals audit ================")
    metals = _element_hist([m.get("formula", "") for m in json.load(
        open(f"{paths.PIPE}/tecrdb_full_metabolites.json"))])
    print("  element -> #compounds:", metals["nonorganic"])
    print(f"  {metals['n_metal_cpd']}/{metals['n_cpd']} compounds contain a metal; "
          "TECRDB CV cannot demonstrate a metal-coverage gain -> belongs to the "
          "coverage eval, not here.")
    results["metals_audit"] = metals

    json.dump(results, open(paths.artifact("ablation_results.json"), "w"), indent=2)
    print("\nwrote artifacts/ablation_results.json")


def _spearman(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    rx = np.argsort(np.argsort(x[m])); ry = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(rx, ry)[0, 1])


def _element_hist(formulas):
    """Count non-organic elements from molecular formulas (avoids importing rdkit
    after torch, which hits a libstdc++/GLIBCXX conflict in this env)."""
    import re
    organic = {"C", "N", "O", "P", "S", "H", "F", "Cl", "Br", "I", "R"}
    counts, n_metal = {}, 0
    for f in formulas:
        els = set(re.findall(r"[A-Z][a-z]?", f or ""))
        extra = els - organic
        if extra:
            n_metal += 1
        for e in extra:
            counts[e] = counts.get(e, 0) + 1
    return dict(nonorganic=dict(sorted(counts.items(), key=lambda kv: -kv[1])),
                n_metal_cpd=n_metal, n_cpd=len(formulas))


if __name__ == "__main__":
    main()
