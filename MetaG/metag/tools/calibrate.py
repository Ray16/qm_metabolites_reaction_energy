"""Regenerate the uncertainty calibration (metag/data/sigma_class_calibrated.json) from per-reaction
results of the DEPLOYED pipeline, with a FULLY NESTED cross-validation of the complete estimator.

What is calibrated on the benchmark (TECRDB) -- and therefore what the nested CV must refit per fold:
  1. the TECRDB-referenced ANCHOR OFFSETS (metag.routing.anchor.ANCHORS entries with ref == "tecrdb":
     offset = mean(dG_raw - exp) over the anchor pool). Externally-referenced anchors (ref == "external",
     the adenylylate +25 kJ indirect-cycle target) use no TECRDB experiment and stay fixed;
  2. per-class sigma (shrunk toward the overall RMS, which is itself a training-fold quantity);
  3. the per-class heavy-tail floor q95abs;
  4. the symmetric interval multiplier m.
Routing choices (truncation / pH-0 / cofactor gates) and the hydro-lyase water constant are structural
rules, not fitted to experiment, so they are fixed across folds.

In each fold, the held-out reaction's anchored dG is rebuilt with the TRAINING-fold offset
(dG_fold = dG_deployed + offset_deployed - offset_fold; no correction if no pool member is in training),
all widths are fit on training residuals, and coverage/MAE are scored on the held-out fold only.

The interval half-width is max(m * sqrt(sigma_class^2 + U_samp^2), q95abs) -- the same form
metag.uncertainty.prediction_interval deploys, so the CV scores what ships.

    from metag.tools.calibrate import calibrate, write_artifact
    records = [{"rid":..., "class":..., "dG":..., "dG_raw":..., "exp":..., "U_samp":...,
                "anchor": {"subclass":..., "offset":...} or None}, ...]
    write_artifact(calibrate(records))
"""
import os
import json
import hashlib
import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ARTIFACT = os.path.join(_DATA, "sigma_class_calibrated.json")
KQ = 8.0            # quantile pooling strength (small n -> global shape)
K_SIG = 4.0         # sigma shrinkage pseudo-count toward the overall RMS
SIGMA_MIN = 6.0     # sigma floor (kJ): experimental noise + method floor
MAX_ABS_ERR = 200.0 # rows beyond this are pipeline failures, not residuals (reported, excluded)
N_FOLDS = 5


def _fit_mult(pairs, target=0.965):
    """Smallest symmetric multiplier m with coverage>=target of half-width max(m*s_eff, q95abs)."""
    for m in np.arange(1.8, 3.01, 0.05):
        if np.mean([ae <= max(m * s, q) for ae, s, q in pairs]) >= target:
            return round(float(m), 2)
    return 3.0


def _anchor_refs():
    """{subclass: (ref, anchor_rids)} from the live anchor table (ref defaults to 'tecrdb')."""
    try:
        from metag.routing.anchor import ANCHORS
    except Exception:
        return {}
    return {k: (v.get("ref", "tecrdb"), set(v.get("anchor_rids", []))) for k, v in ANCHORS.items()}


def _normalize(records):
    """Accept dict records, or legacy (rid, class, residual) tuples (no anchors, U_samp=0)."""
    out = []
    for r in records:
        if isinstance(r, dict):
            exp = r.get("exp")
            if isinstance(exp, (list, tuple)):
                exp = exp[0] if exp else None
            if exp is None or r.get("dG") is None:
                continue
            a = r.get("anchor") or {}
            out.append({"rid": r.get("rid", r.get("reaction")), "class": r["class"], "dG": float(r["dG"]),
                        "dG_raw": float(r.get("dG_raw", r["dG"])), "exp": float(exp),
                        "U": float(r.get("U_samp") or 0.0),
                        # "offset" = the SIGNED correction applied (dG_raw - dG); "direction" = +1 if the
                        # reaction is written in its class's canonical direction, -1 if reversed
                        "sc": a.get("subclass"), "off": float(a.get("offset") or 0.0),
                        "dir": int(a.get("direction", 1))})
        else:
            rid, c, e = r
            out.append({"rid": rid, "class": c, "dG": float(e), "dG_raw": float(e), "exp": 0.0,
                        "U": 0.0, "sc": None, "off": 0.0, "dir": 1})
    return out


def _fold_offsets(rows, train_idx, refs):
    """TECRDB-referenced anchor offsets refit on the training rows only: {subclass: offset or None}."""
    off = {}
    for sc, (ref, pool) in refs.items():
        if ref != "tecrdb":
            continue
        v = [rows[i]["dir"] * (rows[i]["dG_raw"] - rows[i]["exp"]) for i in train_idx   # canonical direction
             if rows[i]["sc"] == sc and rows[i]["rid"] in pool]
        off[sc] = float(np.mean(v)) if v else None
    return off


def _residual(r, fold_off, refs):
    """Residual of row r under a given set of (fold) anchor offsets."""
    sc = r["sc"]
    if sc is None or sc not in refs or refs[sc][0] != "tecrdb":
        return r["dG"] - r["exp"]                          # un-anchored or externally referenced: fixed
    o = fold_off.get(sc)
    dg = r["dG"] + r["off"] - r["dir"] * (o if o is not None else 0.0)   # no pool member in training -> dG_raw
    return dg - r["exp"]


def _class_fit(res_by_class, orms, gq):
    """Per-class sigma (shrunk) and q95abs (pooled toward the global quantile gq)."""
    sig, q95 = {}, {}
    for c, v in res_by_class.items():
        v = np.asarray(v); n = len(v)
        rms = float(np.sqrt((v ** 2).mean()))
        sig[c] = max(float(np.sqrt((n * rms ** 2 + K_SIG * orms ** 2) / (n + K_SIG))), rms, SIGMA_MIN)
        q95[c] = (n * float(np.quantile(np.abs(v), 0.95)) + KQ * gq) / (n + KQ)
    return sig, q95


def calibrate(records, n_folds=N_FOLDS):
    """Full calibration dict (artifact format) + fully nested CV of the complete deployed estimator."""
    refs = _anchor_refs()
    rows_all = _normalize(records)
    rows = [r for r in rows_all if abs(r["dG"] - r["exp"]) <= MAX_ABS_ERR]
    excluded = sorted(r["rid"] for r in rows_all if abs(r["dG"] - r["exp"]) > MAX_ABS_ERR)
    n = len(rows)

    # ---- deployed (full-data) calibration: residuals of the shipped estimator
    err = np.array([r["dG"] - r["exp"] for r in rows])
    orms = float(np.sqrt((err ** 2).mean()))
    gq = float(np.quantile(np.abs(err), 0.95))
    by = {}
    for r, e in zip(rows, err):
        by.setdefault(r["class"], []).append(e)
    sig, q95 = _class_fit(by, orms, gq)
    classes = {}
    for c, v in by.items():
        v = np.asarray(v)
        classes[c] = {"n": len(v), "sigma": round(sig[c], 1), "rms": round(float(np.sqrt((v ** 2).mean())), 1),
                      "mae": round(float(np.abs(v).mean()), 1), "bias": round(float(v.mean()), 1),
                      "medAE": round(float(np.median(np.abs(v))), 1),
                      "center": round(float(np.median(v)), 1), "q95abs": round(q95[c], 1)}
    m95 = _fit_mult([(abs(e), np.hypot(sig[r["class"]], r["U"]), q95[r["class"]]) for r, e in zip(rows, err)])

    # ---- fully nested CV: anchors + sigma + q95 + m all refit on the training folds
    fold = [int(hashlib.md5((r["rid"] or str(i)).encode()).hexdigest(), 16) % n_folds
            for i, r in enumerate(rows)]
    ho_cover, ho_c1, ho_c2, ho_abs = [], 0, 0, []
    for f in range(n_folds):
        tr = [i for i in range(n) if fold[i] != f]
        te = [i for i in range(n) if fold[i] == f]
        if not tr or not te:
            continue
        foff = _fold_offsets(rows, tr, refs)
        res = {i: _residual(rows[i], foff, refs) for i in range(n)}
        etr = np.array([res[i] for i in tr])
        orms_f = float(np.sqrt((etr ** 2).mean()))
        gq_f = float(np.quantile(np.abs(etr), 0.95))
        byf = {}
        for i in tr:
            byf.setdefault(rows[i]["class"], []).append(res[i])
        sig_f, q95_f = _class_fit(byf, orms_f, gq_f)

        def width(i):
            c = rows[i]["class"]
            return np.hypot(sig_f.get(c, orms_f), rows[i]["U"]), q95_f.get(c, gq_f)

        m_f = _fit_mult([(abs(res[i]), *width(i)) for i in tr])
        for i in te:
            s, q = width(i); ae = abs(res[i])
            ho_cover.append(ae <= max(m_f * s, q))
            ho_c1 += ae <= s; ho_c2 += ae <= 2 * s
            ho_abs.append(ae)

    return {"overall": {"MAE": round(float(np.abs(err).mean()), 2), "RMS": round(orms, 2),
                        "bias": round(float(err.mean()), 2), "medAE": round(float(np.median(np.abs(err))), 2)},
            "n_reactions": n, "excluded_abs_err_gt_200": excluded,
            "default_sigma": round(orms, 1), "global_q95abs": round(gq, 1), "classes": classes,
            "interval_sigma_mult": m95,
            "interval_form": "half_width = max(m * sqrt(sigma_class^2 + U_samp^2), q95abs)",
            "cv": "fully nested: TECRDB-referenced anchor offsets, sigma, q95abs, m refit per training fold",
            "cv_heldout_MAE": round(float(np.mean(ho_abs)), 2) if ho_abs else None,
            "cv_coverage_interval95": round(float(np.mean(ho_cover)), 3) if ho_cover else None,
            "cv_coverage_1sigma": round(ho_c1 / max(len(ho_abs), 1), 3),
            "cv_coverage_2sigma": round(ho_c2 / max(len(ho_abs), 1), 3)}


def calibrate_classes(rows):
    """Legacy API: rows = iterable of (rid, class, residual) -> calibrate() with no anchors, U_samp=0."""
    return calibrate(list(rows))


def refit_anchor_offsets(records):
    """Full-data TECRDB-referenced anchor offsets from a sweep: {subclass: {"offset", "n", "std"}}.
    Use to refresh metag.routing.anchor.ANCHORS after a pipeline change (the deployed offsets)."""
    refs = _anchor_refs()
    rows = _normalize(records)
    out = {}
    for sc, (ref, pool) in refs.items():
        if ref != "tecrdb":
            continue
        v = [r["dir"] * (r["dG_raw"] - r["exp"]) for r in rows if r["sc"] == sc and r["rid"] in pool]
        if v:
            out[sc] = {"offset": round(float(np.mean(v)), 1), "n": len(v),
                       "std": round(float(np.std(v, ddof=1)), 1) if len(v) > 1 else None}
    return out


def write_artifact(calib, path=ARTIFACT):
    with open(path, "w") as fh:
        json.dump(calib, fh, indent=2)
    return path


def load_sweep(sweep_dir):
    """Per-reaction records written by analysis/tecrdb_rescore.py. Returns (usable, excluded) where
    excluded = {rid: reason} for errored / suspect (no-estimate) / experiment-less records."""
    import glob
    usable, excluded = [], {}
    for f in sorted(glob.glob(os.path.join(sweep_dir, "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        rid = r.get("reaction") or os.path.basename(f)[:-5]
        if "error" in r:
            excluded[rid] = f"error: {r['error'][:120]}"
        elif r.get("dG") is None:
            excluded[rid] = f"no estimate (suspect: {r.get('suspect')})"
        elif r.get("exp") is None:
            excluded[rid] = "no experiment"
        else:
            usable.append(dict(r, rid=rid))
    return usable, excluded


def build_artifact(sweep_dir, path=ARTIFACT):
    """Regenerate the shipped calibration from ONE sweep of the deployed pipeline, with provenance.
    Refuses a sweep whose records disagree on (or lack) the pipeline configuration fingerprint -- the
    artifact must describe exactly one estimator, and metag.uncertainty checks it at run time."""
    import datetime
    import subprocess
    recs, excluded = load_sweep(sweep_dir)
    cfgs = {json.dumps(r.get("config"), sort_keys=True) for r in recs}
    if len(cfgs) != 1 or "null" in cfgs:
        raise ValueError(f"sweep {sweep_dir} has {len(cfgs)} distinct configuration fingerprints "
                         f"(or records without one): re-run the sweep with one configuration")
    calib = calibrate(recs)
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                             capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        sha = None
    from metag.routing.anchor import ANCHORS
    calib.update({
        "config": json.loads(cfgs.pop()),
        "source": {"sweep_dir": os.path.abspath(sweep_dir), "n_records": len(recs),
                   "excluded": excluded, "git_sha": sha,
                   "built": datetime.datetime.now().isoformat(timespec="seconds")},
        "anchors": {k: {"offset": v["offset"], "sigma": v["sigma"], "ref": v.get("ref", "tecrdb")}
                    for k, v in ANCHORS.items()},
        "per_reaction": [{k: r.get(k) for k in ("rid", "class", "dG", "dG_raw", "exp", "U_samp", "anchor")}
                         for r in recs],
    })
    return write_artifact(calib, path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Rebuild metag/data/sigma_class_calibrated.json from a sweep, "
                                             "or print refreshed anchor offsets (--refit-anchors).")
    ap.add_argument("sweep_dir")
    ap.add_argument("--out", default=ARTIFACT)
    ap.add_argument("--refit-anchors", action="store_true")
    a = ap.parse_args()
    if a.refit_anchors:
        print(json.dumps(refit_anchor_offsets(load_sweep(a.sweep_dir)[0]), indent=2))
    else:
        print("wrote", build_artifact(a.sweep_dir, a.out))
