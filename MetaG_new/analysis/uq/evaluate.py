"""Compare per-reaction uncertainty models for MetaG under nested, near-duplicate-grouped CV.

Each model gives a scale s(x) > 0. Intervals are split-conformal on the TRAINING fold: the 95% (68%)
half-width is q * s(x), with q the finite-sample-corrected quantile of |err|/s over out-of-fold training
scores (inner grouped CV), so coverage does not rely on a Gaussian assumption. Everything that touches
the target -- class sigmas, regression weights, conformal q -- is refit inside each outer training fold.

Models
  class_frozen   current deployed scheme: per-class RMS (class = structural family first, then enzyme-
                 name/EC keywords), shrunk toward the global RMS; refit per fold.
  constant       one global scale (pure conformal; the no-information reference).
  feat_linear    log|err| ~ ridge on structural/routing/solvation features (no annotations).
  feat_gbm       same features, shallow gradient boosting (regularized).
  feat+class     feat_linear plus the class-frozen log-scale as one extra feature.

    python evaluate.py [--repeats 20]   -> evaluation.json (+ printed table)
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

FEATURES = [
    "solv_spread", "solv_d_cpcmx", "solv_d_cosmo", "n_species", "max_rotb", "sum_heavy", "max_heavy",
    "gross_anionO", "gross_charge", "net_abs_charge_change", "max_abs_charge", "abs_pka_transform",
    "d_PN", "d_POP", "n_multiP_species", "max_P_per_species", "d_hemiacetal", "d_cationN", "d_aromN",
    "d_amine", "d_amide", "d_guan", "d_carbonyl", "d_ketoacid", "d_aldehyde", "U_samp", "truncated",
    "ntp_core", "cofactor_ring", "prefer_full", "ph0", "scored_max_heavy",
]
SKEWED = {"solv_spread", "solv_d_cpcmx", "solv_d_cosmo", "sum_heavy", "max_heavy", "max_rotb",
          "gross_anionO", "gross_charge", "abs_pka_transform", "scored_max_heavy", "U_samp"}
EPS = 1.0          # kJ/mol floor inside log|err| (experimental noise scale)


def design(rows, rids, med=None):
    X = []
    for rid in rids:
        r = rows[rid]
        v = []
        for k in FEATURES:
            x = r.get(k)
            x = np.nan if x is None else float(x)
            v.append(math.log1p(x) if (k in SKEWED and not np.isnan(x)) else x)
        X.append(v)
    X = np.array(X)
    if med is None:
        med = np.nanmedian(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(med, idx[1])
    return X, med


def conformal_q(scores, level):
    n = len(scores)
    k = min(n, math.ceil((n + 1) * level))
    return float(np.sort(scores)[k - 1])


class ClassScale:
    """Per-class RMS shrunk to the global RMS with pseudo-count 4 (the deployed recipe)."""
    def fit(self, rows, rids):
        e = np.array([rows[r]["err"] for r in rids])
        g = math.sqrt(np.mean(e ** 2))
        by = {}
        for r in rids:
            by.setdefault(rows[r]["class_frozen"], []).append(rows[r]["err"])
        self.s = {c: math.sqrt((sum(x * x for x in v) + 4 * g * g) / (len(v) + 4)) for c, v in by.items()}
        self.g = g
        return self

    def scale(self, rows, rids):
        return np.array([self.s.get(rows[r]["class_frozen"], self.g) for r in rids])


class ConstScale:
    def fit(self, rows, rids):
        self.g = math.sqrt(np.mean([rows[r]["err"] ** 2 for r in rids])); return self

    def scale(self, rows, rids):
        return np.full(len(rids), self.g)


class FeatScale:
    def __init__(self, kind="linear", with_class=False):
        self.kind, self.with_class = kind, with_class

    def _X(self, rows, rids, fit=False):
        X, med = design(rows, rids, None if fit else self.med)
        if fit:
            self.med = med
        if self.with_class:
            X = np.column_stack([X, np.log(self.cls.scale(rows, rids))])
        return X

    def fit(self, rows, rids):
        if self.with_class:
            self.cls = ClassScale().fit(rows, rids)
        X = self._X(rows, rids, fit=True)
        y = np.log(np.abs([rows[r]["err"] for r in rids]) + EPS)
        self.sc = StandardScaler().fit(X)
        Xs = self.sc.transform(X)
        if self.kind == "linear":
            self.m = RidgeCV(alphas=np.logspace(-1, 3, 25)).fit(Xs, y)
        else:
            self.m = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.03,
                                               subsample=0.7, min_samples_leaf=15, random_state=0).fit(Xs, y)
        # log|err| regression predicts E[log|e|]; the scale is exp(pred) up to a constant that the
        # conformal quantile absorbs, so no bias correction is needed.
        return self

    def scale(self, rows, rids):
        return np.exp(self.m.predict(self.sc.transform(self._X(rows, rids))))


MODELS = {
    "class_frozen": lambda: ClassScale(),
    "constant": lambda: ConstScale(),
    "feat_linear": lambda: FeatScale("linear"),
    "feat_gbm": lambda: FeatScale("gbm"),
    "feat+class": lambda: FeatScale("linear", with_class=True),
}


def group_folds(rids, groups, k, rng):
    g = sorted({groups[r] for r in rids})
    rng.shuffle(g)
    assign = {gg: i % k for i, gg in enumerate(g)}
    return [[r for r in rids if assign[groups[r]] == f] for f in range(k)]


def fit_with_conformal(make, rows, train, groups, rng, levels=(0.68, 0.95), inner_k=5):
    """Fit on train; conformal q from out-of-fold training scores (inner grouped CV)."""
    scores = []
    for f in group_folds(train, groups, inner_k, rng):
        tr = [r for r in train if r not in set(f)]
        m = make().fit(rows, tr)
        s = m.scale(rows, f)
        scores += list(np.abs([rows[r]["err"] for r in f]) / s)
    model = make().fit(rows, train)
    return model, {lv: conformal_q(np.array(scores), lv) for lv in levels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    rows = json.load(open(os.path.join(HERE, "features_tecrdb.json")))
    from metag.tools.calibrate import reaction_groups
    inputs = json.load(open(os.path.join(HERE, "..", "..", "..", "experiments", "qm_mlip_solvation",
                                         "scripts", "reactions_opentecr_std.json")))
    groups = reaction_groups({r: inputs[r] for r in rows})
    rids = sorted(rows)
    ae = np.abs([rows[r]["err"] for r in rids])
    out = {}
    for name, make in MODELS.items():
        rng = np.random.default_rng(0)
        cov68, cov95, hw95, rho, auc, nll, top_cov, by_class = [], [], [], [], [], [], [], {}
        for rep in range(a.repeats):
            pred_s, q68, q95 = {}, {}, {}
            for test in group_folds(rids, groups, a.k, rng):
                train = [r for r in rids if r not in set(test)]
                m, q = fit_with_conformal(make, rows, train, groups, rng)
                for r, s in zip(test, m.scale(rows, test)):
                    pred_s[r], q68[r], q95[r] = s, q[0.68], q[0.95]
            s = np.array([pred_s[r] for r in rids])
            h68 = s * np.array([q68[r] for r in rids]); h95 = s * np.array([q95[r] for r in rids])
            cov68.append(np.mean(ae <= h68)); cov95.append(np.mean(ae <= h95)); hw95.append(np.mean(h95))
            rho.append(spearmanr(s, ae)[0]); auc.append(roc_auc_score(ae > 20, s))
            sig = h68                                  # 68% conformal half-width as a sigma
            nll.append(np.mean(np.log(sig) + 0.5 * (ae / sig) ** 2 + 0.5 * math.log(2 * math.pi)))
            top = s >= np.quantile(s, 0.75)
            top_cov.append((np.mean(ae[top] <= h95[top]), np.mean(ae[~top] <= h95[~top])))
            for c in sorted({rows[r]["class_frozen"] for r in rids}):
                sel = np.array([rows[r]["class_frozen"] == c for r in rids])
                by_class.setdefault(c, []).append(np.mean(ae[sel] <= h95[sel]))
        out[name] = {
            "coverage68": float(np.mean(cov68)), "coverage95": float(np.mean(cov95)),
            "mean_halfwidth95": float(np.mean(hw95)), "spearman_s_abs_err": float(np.mean(rho)),
            "auroc_err_gt20": float(np.mean(auc)), "gaussian_nll": float(np.mean(nll)),
            "coverage95_top_quartile_s": float(np.mean([t[0] for t in top_cov])),
            "coverage95_rest": float(np.mean([t[1] for t in top_cov])),
            "worst_class_coverage95": min((float(np.mean(v)), c) for c, v in by_class.items()),
            "n": len(rids), "groups": len(set(groups.values())), "repeats": a.repeats, "folds": a.k,
        }
    json.dump(out, open(os.path.join(HERE, "evaluation.json"), "w"), indent=2)
    print(f"{'model':14s} {'cov68':>6s} {'cov95':>6s} {'hw95':>6s} {'rho':>5s} {'AUROC>20':>8s} {'NLL':>5s} "
          f"{'cov95 topQ/rest':>16s}  worst-class cov95")
    for k, v in out.items():
        print(f"{k:14s} {v['coverage68']:6.3f} {v['coverage95']:6.3f} {v['mean_halfwidth95']:6.1f} "
              f"{v['spearman_s_abs_err']:5.2f} {v['auroc_err_gt20']:8.3f} {v['gaussian_nll']:5.2f} "
              f"{v['coverage95_top_quartile_s']:7.3f}/{v['coverage95_rest']:.3f}  "
              f"{v['worst_class_coverage95'][0]:.2f} ({v['worst_class_coverage95'][1]})")


if __name__ == "__main__":
    main()
