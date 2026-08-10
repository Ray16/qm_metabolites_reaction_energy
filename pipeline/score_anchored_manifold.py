#!/usr/bin/env python
"""Experimentally-anchored per-species correction of the charged/phosphate manifold.

Idea under test (user's factorization):
  - Phosphate charge-state / cofactor free energies -> pinned to experimental
    reaction data, NOT trusted from QC continuum solvation.
  - QC keeps the covalent/organic chemistry it is good at.

Because scoring is linear -- drG'(r) = sum_c nu_rc * dfG'_c -- "pinning species c
to data" is exactly an additive correction delta_c to that species' transformed
formation energy:

  pred(r) = baseline_QC(r) + sum_{c in H} nu_rc * delta_c

H is the *anchored set*. We fit delta_H to experimental reaction dG and evaluate
STRICTLY OUT-OF-SAMPLE (K-fold + leave-one-out). Using the training reactions as
the experimental anchor pool is the deployment scenario (a reference DB of
measured reactions -> predict a novel one) and the fair analogue to how
eQuilibrator is fit to TECRDB. In-sample fit is reported only as the ceiling.

This directly adjudicates the EXPERIMENTS_LOG counter-result (cofactor
cancellation made ATP/ADP worse -> "error is in the substrate centres that
change"): we compare anchoring only the recurring cofactors vs. anchoring all
phosphate / all multiply-charged species, out-of-sample.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)

from qm_thermo import config                                    # noqa: E402
from qm_thermo.composite import extract_ensemble_energy         # noqa: E402
from qm_thermo.reactions import Reaction, SpeciesInfo, reaction_dG  # noqa: E402


def load():
    bd = json.load(open(os.path.join(THERMO, "mlip", "G_aq_tecrdb_full.json")))
    rx = json.load(open(os.path.join(HERE, "tecrdb_full_reactions.json")))
    exp = json.load(open(os.path.join(HERE, "tecrdb_full_experiment.json")))
    spec = json.load(open(os.path.join(HERE, "tecrdb_full_species.json")))
    mets = {m["id"]: m for m in json.load(open(os.path.join(HERE, "tecrdb_full_metabolites.json")))}
    C = config.DEFAULT_CONDITIONS
    G = {}
    for c, rec in bd.items():
        G[c] = (extract_ensemble_energy(rec, temperature_K=C.temperature_K).gibbs_kJ
                if "conformers" in rec else float(rec["G_aq_kJ"]))
    S = {c: SpeciesInfo(c, n_hydrogens=int(v["n_hydrogens"]), charge=int(v["charge"]))
         for c, v in spec.items()}
    rids, base, y = [], [], []
    for rid, st in rx.items():
        if any(c not in G for c in st):
            continue
        b = reaction_dG(Reaction(rid, st), G, S, conditions=C).dG_transformed_kJ
        rids.append(rid)
        base.append(b)
        y.append(exp[rid]["dG_kJ"])
    return rids, np.array(base), np.array(y), rx, spec, mets


def design_matrix(rids, rx, anchored):
    """X[r, j] = stoichiometric coeff of anchored species j in reaction r."""
    cols = sorted(anchored)
    idx = {c: j for j, c in enumerate(cols)}
    X = np.zeros((len(rids), len(cols)))
    for i, rid in enumerate(rids):
        for c, v in rx[rid].items():
            if c in idx:
                X[i, idx[c]] += v
    return X, cols


def ridge_fit(X, t, lam):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ t)


def kfold_oos(X, base, y, lam, k, seed=0xF00D):
    """Return out-of-sample corrected predictions via K-fold ridge."""
    n = len(y)
    t = y - base                      # target the QC error; Xdelta should match it
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, k)
    pred = np.full(n, np.nan)
    for f in folds:
        te = f
        tr = np.setdiff1d(order, te)
        d = ridge_fit(X[tr], t[tr], lam)
        pred[te] = base[te] + X[te] @ d
    return pred


def loo_oos(X, base, y, lam):
    n = len(y)
    t = y - base
    pred = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        d = ridge_fit(X[tr], t[tr], lam)
        pred[i] = base[i] + X[i] @ d
    return pred


def stats(pred, y):
    e = pred - y
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e * e)))


def main():
    rids, base, y, rx, spec, mets = load()
    n = len(rids)
    mae0, rmse0 = stats(base, y)
    print(f"baseline QC:  n={n}  MAE={mae0:.2f}  RMSE={rmse0:.2f}\n")

    # ---- anchored-set definitions -------------------------------------------
    cofactor14 = {"cpd00003", "cpd00004", "cpd00006", "cpd00005",      # NAD(H)/NADP(H)
                  "cpd00002", "cpd00008", "cpd00018",                  # ATP/ADP/AMP
                  "cpd00009", "cpd00012",                              # Pi / PPi
                  "cpd00010", "cpd00022",                              # CoA / acetyl-CoA
                  "cpd00042", "cpd00111",                              # GSH / GSSG
                  "cpd00102", "cpd00095", "cpd00103"}                  # GAP / DHAP / PRPP
    has_P = {c for c, m in mets.items() if "P" in (m.get("smiles") or "")}
    multi2 = {c for c, v in spec.items() if abs(int(v["charge"])) >= 2}
    charged = {c for c, v in spec.items() if int(v["charge"]) != 0}

    sets = {
        "cofactor14 (recurring cofactors only)": cofactor14,
        "phosphorus-bearing (all P species)": has_P,
        "multivalent |z|>=2": multi2,
        "phosphorus OR |z|>=2": has_P | multi2,
        "all charged z!=0": charged,
    }

    # frequency: how many reactions touch each anchored species -> OOS learnability
    print(f"{'anchored set':40s} {'|H|':>4s} {'seen>=3':>7s} "
          f"{'in-sample':>9s} {'10fold':>7s} {'LOO':>7s}  {'LOO RMSE':>8s}")
    print("-" * 92)
    results = {}
    for name, H in sets.items():
        Hs = {c for c in H if c in spec}          # only scorable species
        X, cols = design_matrix(rids, rx, Hs)
        seen = int(((X != 0).sum(axis=0) >= 3).sum())
        lam = 10.0                                 # ridge; scanned below for winner
        insample = base + X @ ridge_fit(X, y - base, lam)
        mae_in, _ = stats(insample, y)
        p10 = kfold_oos(X, base, y, lam, k=10)
        mae10, _ = stats(p10, y)
        ploo = loo_oos(X, base, y, lam)
        maeloo, rmseloo = stats(ploo, y)
        results[name] = dict(H=Hs, X=X, cols=cols)
        print(f"{name:40s} {len(cols):4d} {seen:7d} "
              f"{mae_in:9.2f} {mae10:7.2f} {maeloo:7.2f}  {rmseloo:8.2f}")

    # ---- lambda scan on the most complete set (phosphorus OR |z|>=2) --------
    print("\nridge lambda scan (LOO) on 'phosphorus OR |z|>=2':")
    H = (has_P | multi2) & set(spec)
    X, cols = design_matrix(rids, rx, H)
    for lam in (0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
        mae, rmse = stats(loo_oos(X, base, y, lam), y)
        print(f"  lam={lam:6.1f}   LOO MAE={mae:6.2f}   RMSE={rmse:6.2f}")

    # ---- residual structure after best anchoring (LOO) ----------------------
    lam = 10.0
    ploo = loo_oos(X, base, y, lam)
    e = ploo - y
    order = np.argsort(-np.abs(e))
    print(f"\ntop-12 remaining LOO errors after anchoring (phosphorus OR |z|>=2, lam={lam}):")
    nm = {c: v["name"] for c, v in spec.items()}
    for i in order[:12]:
        rid = rids[i]
        parts = " + ".join(f"{v:+g} {nm.get(c, c)}" for c, v in rx[rid].items())
        print(f"  {rid}  base_err={base[i]-y[i]:+7.1f}  anchored_err={e[i]:+7.1f}   {parts[:80]}")


if __name__ == "__main__":
    main()
