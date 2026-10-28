"""Fit the per-reaction feature uncertainty model (UQ_MODEL=features) -> metag/data/uq_feature_model.json.

Model: log(|err| + 1) ~ ridge(standardized features) gives a reaction scale s(x) = exp(prediction).
Intervals use pooled out-of-fold calibration: half-width(level) = q_level * s(x), with q_level the finite-sample
quantile of |err| / s over OUT-OF-FOLD scores from near-duplicate-grouped CV (metag.tools.calibrate.
reaction_groups), so the reported coverage is not a Gaussian assumption. The artifact also records the
nested-CV evaluation (refit per outer fold, conformal q from inner folds) so the advertised coverage and
ranking skill are held-out numbers, not in-sample ones.

This is cross-validated residual calibration, not an independent split-conformal
calibration set. The standard split-conformal coverage guarantee does not apply;
coverage here is evaluated empirically by nested grouped cross-validation.

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

from metag.uq_features import FEATURES, FEATURE_VERSION, primitive_reaction_features, transform

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
        if r.get('suspect'):
            continue
        values = r['exp'] if isinstance(r['exp'], list) else [r['exp']]
        if not values:
            continue
        if len(values) != 1:
            raise ValueError('Calibration requires one explicitly aggregated reference per reaction; multiple measurements are not extra targets')
        exp = values[0]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in (r['dG'], exp)):
            raise ValueError('Calibration estimates and references must be finite numbers')
        if rid in rows:
            raise ValueError('Duplicate calibration output for reaction: ' + str(rid))
        raw, extent = primitive_reaction_features(reactions[rid]['species'], r.get('species_scored') or {},
                                                 r.get('routes'), r.get('stages'), r.get('U_samp'))
        rows[rid] = {'x': transform(raw), 'err': (float(r['dG']) - float(exp)) / extent,
                     'config': r.get('config'), 'input_extent': extent}
    return rows


def _matrix(rows, rids, med=None):
    if not rids:
        raise ValueError('Uncertainty fitting requires nonempty training rows')
    X = np.array([[np.nan if v is None else v for v in rows[r]["x"]] for r in rids], float)
    if X.shape != (len(rids), len(FEATURES)) or np.isinf(X).any():
        raise ValueError('Uncertainty features must have the declared shape and no infinite values')
    if med is None:
        med = np.array([np.median(column[~np.isnan(column)]) if (~np.isnan(column)).any() else 0.
                        for column in X.T])
    if np.shape(med) != (len(FEATURES),) or not np.isfinite(med).all():
        raise ValueError('Uncertainty imputation values must be finite and complete')
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
            "alpha": float(m.alpha_), "lo": X.min(0), "hi": X.max(0),
            "unobserved_features": [FEATURES[i] for i in range(len(FEATURES))
                                    if all(rows[r]['x'][i] is None or math.isnan(rows[r]['x'][i]) for r in rids)]}


def _scale(p, rows, rids):
    X, _ = _matrix(rows, rids, p["median"])
    return np.exp(((X - p["mean"]) / p["scale"]) @ p["coef"] + p["intercept"])


def _folds(rids, groups, k, rng):
    g = sorted({groups[r] for r in rids})
    if type(k) is not int or k < 2 or len(g) < k:
        raise ValueError('Grouped uncertainty CV requires at least k independent groups and k >= 2')
    rng.shuffle(g)
    a = {gg: i % k for i, gg in enumerate(g)}
    return [[r for r in rids if a[groups[r]] == f] for f in range(k)]


def _q(scores, level):
    values = np.asarray(scores)
    if (values.ndim != 1 or not len(values) or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError('Calibration scores must be a nonempty finite nonnegative vector')
    if isinstance(level, bool) or not isinstance(level, (int, float)) or not math.isfinite(level) or not 0 < level < 1:
        raise ValueError('Calibration level must be strictly between zero and one')
    rank = math.ceil((len(values) + 1) * level)
    if rank > len(values):
        raise ValueError('Insufficient calibration scores for a finite interval at this level; maximum-score clamping is not valid')
    return float(np.sort(values)[rank - 1])


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
    if type(repeats) is not int or repeats < 1:
        raise ValueError('Uncertainty assessment requires at least one repeat')
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
        res["mean_halfwidth95"].append(np.mean(h95))
        if np.ptp(s) > 0 and np.ptp(ae) > 0:
            res['spearman'].append(spearmanr(s, ae)[0])
        if np.unique(ae > 20).size == 2:
            res['auroc_err_gt20'].append(roc_auc_score(ae > 20, s))
    return {name: round(float(np.mean(values)), 3) if values else None for name, values in res.items()} | {"repeats": repeats, "folds": k}


def build(sweep_dir, reactions_file, out=ARTIFACT, repeats=10):
    from metag.tools.calibrate import reaction_groups
    reactions = json.load(open(reactions_file))
    rows = records_to_rows(sweep_dir, reactions)
    configs = {json.dumps(r["config"], sort_keys=True) for r in rows.values()}
    if len(configs) != 1:
        raise SystemExit(f"sweep records carry {len(configs)} different configs; refusing to fit")
    config = json.loads(next(iter(configs)))
    from metag.validation.calibration_artifacts import configuration_issues
    issues = configuration_issues(config)
    if issues:
        raise ValueError('; '.join(issues))
    groups = reaction_groups({r: reactions[r] for r in rows})
    rids = sorted(rows)
    p = _fit(rows, rids)
    oof = _oof_scores(rows, rids, groups, 5, np.random.default_rng(0))
    art = {
        "model": "log(|err|+1) ~ ridge(standardized features); scale = exp(pred); pooled OOF calibrated width",
        "feature_version": FEATURE_VERSION, "features": FEATURES,
        "median": p["median"].tolist(), "mean": p["mean"].tolist(), "scale": p["scale"].tolist(),
        "coef": p["coef"].tolist(), "intercept": p["intercept"], "ridge_alpha": p["alpha"],
        "unobserved_features": p['unobserved_features'],
        "missingness_patterns": sorted({sum(1 << i for i, value in enumerate(rows[r]['x']) if value is None)
                                        for r in rids}),
        "train_min": p["lo"].tolist(), "train_max": p["hi"].tolist(),
        "conformal_q": {str(lv): _q(oof, lv) for lv in LEVELS},
        "n_reactions": len(rids), "n_groups": len(set(groups.values())),
        "nested_cv": nested_cv(rows, groups, repeats=repeats),
        "config": json.loads(next(iter(configs))),
        "calibration_basis": "TECRDB internal nested grouped CV (not an external dataset)",
        "quantile_rule": "ceil((n+1)*level), without maximum-score clamping",
        "formal_coverage_guarantee": False,
        "external_validation_available": False,
        "source": {"sweep_dir": os.path.abspath(sweep_dir), "reactions": os.path.abspath(reactions_file),
                   "built": date.today().isoformat()},
    }
    with open(out + ".tmp", "w") as fh:
        json.dump(art, fh, indent=1, allow_nan=False)
    os.replace(out + ".tmp", out)
    return art


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sweep_dir")
    ap.add_argument("reactions")
    ap.add_argument("--out", default=ARTIFACT)
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args(argv)
    art = build(a.sweep_dir, a.reactions, a.out, a.repeats)
    print(json.dumps({k: art[k] for k in ("n_reactions", "n_groups", "ridge_alpha", "conformal_q",
                                          "nested_cv")}, indent=2))


if __name__ == "__main__":
    main()
