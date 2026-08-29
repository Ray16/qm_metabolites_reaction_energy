"""Regenerate the uncertainty calibration (metag/data/sigma_class_calibrated.json) from a set of
per-reaction residuals. Decoupled from any benchmark: give it rows of (mechanism_class, residual_kJ)
where residual = predicted - experiment for the DEPLOYED pipeline, and it produces the calibrated
per-class sigma, the heavy-tail floor q95abs, and the nested-CV symmetric interval multiplier.

This is the exact logic that produced the shipped artifact (see the module history); the only thing that
changes on recalibration (new sweep, or a ModelSEED validation set) is the input rows.

    from metag.tools.calibrate import calibrate_classes, write_artifact
    rows = [(rid, mech_class(note, smiles), pred - exp), ...]
    calib = calibrate_classes(rows)
    write_artifact(calib)          # -> metag/data/sigma_class_calibrated.json
"""
import os
import json
import hashlib
import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ARTIFACT = os.path.join(_DATA, "sigma_class_calibrated.json")
KQ = 8.0            # quantile pooling strength (small n -> global shape)


def _fit_mult(pairs, target=0.965):
    """Smallest symmetric multiplier m with coverage>=target of half-width max(m*sigma, q95abs)."""
    for m in np.arange(1.8, 3.01, 0.05):
        if np.mean([ae <= max(m * s, q) for ae, s, q in pairs]) >= target:
            return round(float(m), 2)
    return 3.0


def calibrate_classes(rows):
    """rows = iterable of (rid, class, residual). Returns the calibration dict:
    {overall, classes:{cls:{n,sigma,rms,mae,bias,medAE,center,q95abs}}, interval_sigma_mult,
     cv_coverage_interval95, cv_coverage_1sigma, cv_coverage_2sigma, default_sigma}."""
    rows = [(rid, c, float(e)) for rid, c, e in rows if abs(float(e)) <= 200]
    allerr = np.array([e for *_, e in rows])
    orms = float(np.sqrt((allerr ** 2).mean()))
    omae = float(np.abs(allerr).mean())
    by = {}
    for _, c, e in rows:
        by.setdefault(c, []).append(e)

    within = []
    for _c, v in by.items():
        v = np.array(v); within.extend(list(v - np.median(v)))
    g95abs = float(np.quantile(np.abs(allerr), 0.95))

    classes = {}
    for c, v in by.items():
        v = np.array(v); n = len(v)
        rms = float(np.sqrt((v ** 2).mean()))
        shrunk = float(np.sqrt((n * rms ** 2 + 4 * orms ** 2) / (n + 4)))
        sigma = round(max(shrunk, rms, 6.0), 1)
        cq95 = float(np.quantile(np.abs(v), 0.95))
        classes[c] = {"n": n, "sigma": sigma, "rms": round(rms, 1),
                      "mae": round(float(np.abs(v).mean()), 1), "bias": round(float(v.mean()), 1),
                      "medAE": round(float(np.median(np.abs(v))), 1),
                      "center": round(float(np.median(v)), 1),
                      "q95abs": round((n * cq95 + KQ * g95abs) / (n + KQ), 1)}

    # nested CV: sigma, q95abs, and the multiplier all fit on train folds, scored held-out
    fold = {rid: int(hashlib.md5((rid or str(i)).encode()).hexdigest(), 16) % 5
            for i, (rid, _, _) in enumerate(rows)}

    def _sig(c, tr, trr):
        v = tr.get(c, []); n = len(v); cr = trr.get(c, orms)
        return max(np.sqrt((n * cr ** 2 + 4 * orms ** 2) / (n + 4)) if n else orms, cr, 6.0)

    def _q(c, tr, gq):
        v = np.abs(tr.get(c, [])); n = len(v)
        return ((n * float(np.quantile(v, 0.95)) + KQ * gq) / (n + KQ)) if n else gq

    got1 = got2 = 0; ho = []
    for f in range(5):
        tr = {}
        for rid, c, e in rows:
            if fold[rid] != f:
                tr.setdefault(c, []).append(e)
        trr = {c: float(np.sqrt(np.mean(np.square(v)))) for c, v in tr.items()}
        tabs = np.abs([e for v in tr.values() for e in v])
        gq = float(np.quantile(tabs, 0.95)) if len(tabs) else 2 * orms
        trp, hop = [], []
        for rid, c, e in rows:
            s = _sig(c, tr, trr); q = _q(c, tr, gq)
            if fold[rid] == f:
                hop.append((abs(e), s, q)); got1 += abs(e) <= s; got2 += abs(e) <= 2 * s
            else:
                trp.append((abs(e), s, q))
        m_tr = _fit_mult(trp)
        ho.append(np.mean([ae <= max(m_tr * s, q) for ae, s, q in hop]))
    m95 = _fit_mult([(abs(e), classes[c]["sigma"], classes[c]["q95abs"]) for _, c, e in rows])

    return {"overall": {"MAE": round(omae, 2), "RMS": round(orms, 2), "bias": round(float(allerr.mean()), 2)},
            "n_reactions": len(rows), "default_sigma": round(orms, 1), "classes": classes,
            "interval_sigma_mult": m95, "cv_coverage_interval95": round(float(np.mean(ho)), 3),
            "cv_coverage_1sigma": round(got1 / len(rows), 3), "cv_coverage_2sigma": round(got2 / len(rows), 3)}


def write_artifact(calib, path=ARTIFACT):
    json.dump(calib, open(path, "w"), indent=2)
    return path
