"""Fit the per-reaction feature uncertainty model (UQ_MODEL=features) -> metag/data/uq_feature_model.json.

Model: log(|err| + 1) ~ ridge(standardized features) gives a reaction scale s(x) = exp(prediction).
Intervals are split-conformal: half-width(level) = q_level * s(x), with q_level the finite-sample
quantile of |err| / s over OUT-OF-FOLD scores from near-duplicate-grouped CV (metag.tools.calibrate.
reaction_groups), so the reported coverage is not a Gaussian assumption. The artifact also records the
nested-CV evaluation (refit per outer fold, conformal q from inner folds) so the advertised coverage and
ranking skill are held-out numbers, not in-sample ones.

Calibration data are TECRDB residuals of the deployed estimator: like the class sigma it replaces, this is
in-distribution calibration, not external validation.

    python -m metag.tools.fit_uq SWEEP_DIR REACTIONS_JSON [--out PATH] [--repeats 10]
"""
import argparse
import glob
import json
import math
import os
from datetime import date

import numpy as np

from metag.uq_features import FEATURES, FEATURE_VERSION, reaction_features, transform

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ARTIFACT = os.path.join(_DATA, "uq_feature_model.json")
EPS = 1.0
ALPHAS = np.logspace(-1, 3, 25)
LEVELS = (0.68, 0.95)


def records_to_rows(sweep_dir, reactions):
    rows = {}
    for path in sorted(glob.glob(os.path.join(sweep_dir, "*.json"))):
        r = json.load(open(path))
        rid = r.get("reaction")
        if r.get("dG") is None or r.get("exp") is None or rid not in reactions:
            continue
        exp = r["exp"][0] if isinstance(r["exp"], list) else r["exp"]
        raw = reaction_features(reactions[rid]["species"], r.get("species_scored") or {},
                                r.get("routes"), r.get("stages"), r.get("U_samp"))
        rows[rid] = {"x": transform(raw), "err": float(r["dG"]) - float(exp), "config": r.get("config")}
    return rows


def _matrix(rows, rids, med=None):
    X = np.array([[np.nan if v is None else v for v in rows[r]["x"]] for r in rids], float)
    if med is None:
        med = np.nanmedian(X, axis=0)
    i = np.where(np.isnan(X))
    X[i] = np.take(med, i[1])
    return X, med


def _fit(rows, rids):
    from sklearn.linear_model import RidgeCV
    X, med = _matrix(rows, rids)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    y = np.log(np.abs([rows[r]["err"] for r in rids]) + EPS)
    m = RidgeCV(alphas=ALPHAS).fit((X - mu) / sd, y)
    return {"median": med, "mean": mu, "scale": sd, "coef": m.coef_, "intercept": float(m.intercept_),
            "alpha": float(m.alpha_), "lo": X.min(0), "hi": X.max(0)}


def _scale(p, rows, rids):
    X, _ = _matrix(rows, rids, p["median"])
    return np.exp(((X - p["mean"]) / p["scale"]) @ p["coef"] + p["intercept"])


def _folds(rids, groups, k, rng):
    g = sorted({groups[r] for r in rids})
    rng.shuffle(g)
    a = {gg: i % k for i, gg in enumerate(g)}
    return [[r for r in rids if a[groups[r]] == f] for f in range(k)]


def _q(scores, level):
    n = len(scores)
    return float(np.sort(scores)[min(n, math.ceil((n + 1) * level)) - 1])


def _oof_scores(rows, rids, groups, k, rng):
    s = []
    for f in _folds(rids, groups, k, rng):
        fs = set(f)
        p = _fit(rows, [r for r in rids if r not in fs])
        s += list(np.abs([rows[r]["err"] for r in f]) / _scale(p, rows, f))
    return np.array(s)


def nested_cv(rows, groups, repeats=10, k=5, seed=0):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    rids = sorted(rows)
    ae = np.abs([rows[r]["err"] for r in rids])
    res = {"coverage68": [], "coverage95": [], "mean_halfwidth95": [], "spearman": [], "auroc_err_gt20": []}
    for _ in range(repeats):
        h = {lv: {} for lv in LEVELS}
        sc = {}
        for test in _folds(rids, groups, k, rng):
            ts = set(test)
            train = [r for r in rids if r not in ts]
            oof = _oof_scores(rows, train, groups, k, rng)
            p = _fit(rows, train)
            for r, s in zip(test, _scale(p, rows, test)):
                sc[r] = s
                for lv in LEVELS:
                    h[lv][r] = s * _q(oof, lv)
        s = np.array([sc[r] for r in rids])
        h68 = np.array([h[0.68][r] for r in rids]); h95 = np.array([h[0.95][r] for r in rids])
        res["coverage68"].append(np.mean(ae <= h68)); res["coverage95"].append(np.mean(ae <= h95))
        res["mean_halfwidth95"].append(np.mean(h95)); res["spearman"].append(spearmanr(s, ae)[0])
        res["auroc_err_gt20"].append(roc_auc_score(ae > 20, s))
    return {k: round(float(np.mean(v)), 3) for k, v in res.items()} | {"repeats": repeats, "folds": k}


def build(sweep_dir, reactions_file, out=ARTIFACT, repeats=10):
    from metag.tools.calibrate import reaction_groups
    reactions = json.load(open(reactions_file))
    rows = records_to_rows(sweep_dir, reactions)
    configs = {json.dumps(r["config"], sort_keys=True) for r in rows.values()}
    if len(configs) != 1:
        raise SystemExit(f"sweep records carry {len(configs)} different configs; refusing to fit")
    groups = reaction_groups({r: reactions[r] for r in rows})
    rids = sorted(rows)
    p = _fit(rows, rids)
    oof = _oof_scores(rows, rids, groups, 5, np.random.default_rng(0))
    art = {
        "model": "log(|err|+1) ~ ridge(standardized features); scale = exp(pred); split-conformal width",
        "feature_version": FEATURE_VERSION, "features": FEATURES,
        "median": p["median"].tolist(), "mean": p["mean"].tolist(), "scale": p["scale"].tolist(),
        "coef": p["coef"].tolist(), "intercept": p["intercept"], "ridge_alpha": p["alpha"],
        "train_min": p["lo"].tolist(), "train_max": p["hi"].tolist(),
        "conformal_q": {str(lv): _q(oof, lv) for lv in LEVELS},
        "n_reactions": len(rids), "n_groups": len(set(groups.values())),
        "nested_cv": nested_cv(rows, groups, repeats=repeats),
        "config": json.loads(next(iter(configs))),
        "calibration_basis": "TECRDB internal nested grouped CV (not an external dataset)",
        "source": {"sweep_dir": os.path.abspath(sweep_dir), "reactions": os.path.abspath(reactions_file),
                   "built": date.today().isoformat()},
    }
    with open(out + ".tmp", "w") as fh:
        json.dump(art, fh, indent=1)
    os.replace(out + ".tmp", out)
    return art


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sweep_dir")
    ap.add_argument("reactions")
    ap.add_argument("--out", default=ARTIFACT)
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()
    art = build(a.sweep_dir, a.reactions, a.out, a.repeats)
    print(json.dumps({k: art[k] for k in ("n_reactions", "n_groups", "ridge_alpha", "conformal_q",
                                          "nested_cv")}, indent=2))


if __name__ == "__main__":
    main()
