"""Regenerate the uncertainty calibration (metag/data/sigma_class_calibrated.json) from per-reaction
results of a specified pipeline, with nested cross-validation of fitted offsets and uncertainty.
This does not cross-validate benchmark-guided development of the physical estimator.
Anchor refitting is retained for historical reproduction; it is not a physical
point-estimator improvement and must not be used to tune the current workflow.

What is calibrated on the benchmark (TECRDB) -- and therefore what the nested CV must refit per fold:
  1. the TECRDB-referenced ANCHOR OFFSETS (metag.routing.anchor.ANCHORS entries with ref == "tecrdb":
     offset = mean(dG_raw - exp) over the anchor pool). Externally-referenced anchors (ref == "external",
     the adenylylate +25 kJ indirect-cycle target) use no TECRDB experiment and stay fixed;
  2. per-class sigma (shrunk toward the overall RMS, which is itself a training-fold quantity);
  3. the per-class heavy-tail floor q95abs;
  4. the symmetric interval multiplier m.
Routing choices (truncation / pH-0 / cofactor gates) and the hydro-lyase water constant are structural
rules held fixed across folds. Their benchmark-guided selection is development exposure,
not independent validation of the complete estimator.

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
import math
import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ARTIFACT = os.path.join(_DATA, "sigma_class_calibrated.json")
KQ = 8.0            # quantile pooling strength (small n -> global shape)
K_SIG = 4.0         # sigma shrinkage pseudo-count toward the overall RMS
SIGMA_MIN = 6.0     # sigma floor (kJ): experimental noise + method floor
N_FOLDS = 5
GROUP_SIM = 0.7     # reactions with difference-fingerprint Tanimoto >= this share a CV group (near-twins)
GROUP_VERSION = "2026-10-06-direction-extent"


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
                if len(exp) > 1:
                    raise ValueError('Calibration requires one explicitly aggregated reference per reaction')
                exp = exp[0] if exp else None
            if exp is None or r.get("dG") is None:
                continue
            a = r.get("anchor")
            if a is None:
                a = {}
            if not isinstance(a, dict):
                raise ValueError('Calibration anchor must be an object or null')
            from metag.reactions import stoichiometric_extent
            species = r.get('original_species') or r.get('species_scored')
            extent = stoichiometric_extent(species) if species else 1.
            spread = r.get('U_samp')
            spread = 0. if spread is None else spread
            raw = r.get('dG_raw', r['dG'])
            offset = a.get('offset', 0.)
            anchor_extent = a.get('extent', 1.)
            direction = a.get('direction', 1)
            for name, value in (('dG', r['dG']), ('raw dG', raw), ('reference', exp),
                                ('sampling spread', spread), ('anchor offset', offset),
                                ('anchor extent', anchor_extent)):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError('Calibration ' + name + ' must be finite')
            if spread < 0:
                raise ValueError('Calibration sampling spread must be nonnegative')
            if anchor_extent <= 0 or isinstance(direction, bool) or direction not in (-1, 1):
                raise ValueError('Calibration anchor extent must be positive and direction must be +/-1')
            out.append({"rid": r.get("rid", r.get("reaction")), "class": r["class"], "dG": float(r["dG"]),
                        "dG_raw": float(raw), "exp": float(exp),
                        "U": float(spread),
                        # "offset" = the SIGNED correction applied (dG_raw - dG); "direction" = +1 if the
                        # reaction is written in its class's canonical direction, -1 if reversed
                        "sc": a.get("subclass"), "off": float(offset),
                        "dir": int(direction), "ext": float(anchor_extent)})
            if extent != 1.:
                for name in ('dG', 'dG_raw', 'exp', 'U', 'off', 'ext'):
                    out[-1][name] /= extent
        else:
            rid, c, e = r
            if isinstance(e, bool) or not isinstance(e, (int, float)) or not math.isfinite(e):
                raise ValueError('Calibration residual must be finite')
            out.append({"rid": rid, "class": c, "dG": float(e), "dG_raw": float(e), "exp": 0.0,
                        "U": 0.0, "sc": None, "off": 0.0, "dir": 1, "ext": 1.0})
    ids = [row['rid'] for row in out]
    if any(not isinstance(rid, str) or not rid for rid in ids) or len(set(ids)) != len(ids):
        raise ValueError('Calibration requires one unique nonempty reaction identity per target')
    if any(not isinstance(row['class'], str) or not row['class'] for row in out):
        raise ValueError('Calibration requires a nonempty class identity')
    return out


def _fold_offsets(rows, train_idx, refs):
    """TECRDB-referenced anchor offsets refit on the training rows only: {subclass: offset or None}."""
    off = {}
    for sc, (ref, pool) in refs.items():
        if ref != "tecrdb":
            continue
        v = [rows[i]["dir"] * (rows[i]["dG_raw"] - rows[i]["exp"]) / rows[i]["ext"]   # per canonical
             for i in train_idx if rows[i]["sc"] == sc and rows[i]["rid"] in pool]     # transformation
        off[sc] = float(np.mean(v)) if v else None
    return off


def _residual(r, fold_off, refs):
    """Residual of row r under a given set of (fold) anchor offsets."""
    sc = r["sc"]
    if sc is None or sc not in refs or refs[sc][0] != "tecrdb":
        return r["dG"] - r["exp"]                          # un-anchored or externally referenced: fixed
    o = fold_off.get(sc)
    dg = r["dG"] + r["off"] - r["dir"] * r["ext"] * (o if o is not None else 0.0)   # no pool member -> dG_raw
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


def reaction_groups(reactions, threshold=GROUP_SIM):
    """{rid: group id} clustering near-duplicate reactions (single linkage on difference-fingerprint
    Tanimoto >= threshold in either direction), so CV folds never split near-twins between training and test -- random folds
    over TECRDB leak same-family reactions and overstate held-out coverage.

    `reactions` = {rid: {"species": {name: [coeff, q, smi]}}}. The reaction fingerprint is the signed
    count-Morgan difference Σν·fp (water excluded); similarity is Tanimoto on its positive and negative parts."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from metag.chem import is_water
    from metag.reactions import unit_stoichiometry
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rids = sorted(reactions)
    fps = []
    identities = []
    for rid in rids:
        v = np.zeros(2048)
        normalized, _ = unit_stoichiometry(reactions[rid]["species"])
        equation = {}
        for c, q, smi in normalized.values():
            m = Chem.MolFromSmiles(smi)
            if m is None or is_water(smi, q):
                continue
            cs = Chem.MolToSmiles(m)
            equation[cs] = equation.get(cs, 0) + c
            v += c * gen.GetCountFingerprintAsNumPy(m).astype(float)
        forward = tuple(sorted((s, c) for s, c in equation.items() if c))
        reverse = tuple((s, -c) for s, c in forward)
        identities.append(min(forward, reverse))
        fps.append(np.concatenate([np.clip(v, 0, None), np.clip(-v, 0, None)]))
    F = np.array(fps)
    parent = list(range(len(rids)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    identical = {}
    for i, identity in enumerate(identities):
        if identity in identical:
            parent[find(i)] = find(identical[identity])
        identical[identity] = i
    for i in range(len(rids)):
        mn = np.minimum(F[i], F[i + 1:]).sum(1); mx = np.maximum(F[i], F[i + 1:]).sum(1)
        reverse = np.concatenate([F[i, 2048:], F[i, :2048]])
        mn_rev = np.minimum(reverse, F[i + 1:]).sum(1)
        mx_rev = np.maximum(reverse, F[i + 1:]).sum(1)
        similar = ((mn >= threshold * np.maximum(mx, 1e-12))
                   | (mn_rev >= threshold * np.maximum(mx_rev, 1e-12)))
        for j in np.nonzero(similar)[0] + i + 1:
            parent[find(i)] = find(j)
    return {rid: rids[find(i)] for i, rid in enumerate(rids)}


def calibrate(records, n_folds=N_FOLDS, groups=None):
    """`groups` = {rid: group} (see reaction_groups): folds are assigned per group, so near-duplicate
    reactions are always held out together. Without it folds are per reaction (leaky; reported as such)."""
    """Full calibration dict (artifact format) + fully nested CV of the complete deployed estimator."""
    refs = _anchor_refs()
    # NO outcome-based exclusion: every record with an estimate is a residual, however large (a large
    # chemically valid failure belongs in the tail the interval must cover). Records are excluded upstream
    # only by prediction-time validity (suspect -> no estimate; see load_sweep).
    rows = _normalize(records)
    n = len(rows)
    if not n:
        raise ValueError('Calibration requires at least one usable target')
    if type(n_folds) is not int or n_folds < 2:
        raise ValueError('Calibration requires at least two held-out folds')
    if groups is not None and any(r['rid'] not in groups for r in rows):
        raise ValueError('Every calibration reaction must have a declared group')
    gkey = [groups[r['rid']] if groups is not None else r['rid'] for r in rows]
    if any(not isinstance(g, str) or not g for g in gkey):
        raise ValueError('Calibration groups must be nonempty identities')
    unique_groups = sorted(set(gkey), key=lambda g: (hashlib.sha256(g.encode()).hexdigest(), g))
    if len(unique_groups) < n_folds:
        raise ValueError('Calibration requires at least n_folds independent groups')
    group_fold = {group: i % n_folds for i, group in enumerate(unique_groups)}
    fold = [group_fold[group] for group in gkey]

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
    ho_cover, ho_c1, ho_c2, ho_abs = [], 0, 0, []
    for f in range(n_folds):
        tr = [i for i in range(n) if fold[i] != f]
        te = [i for i in range(n) if fold[i] == f]
        assert tr and te
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

    assert len(ho_abs) == n
    return {"overall": {"MAE": round(float(np.abs(err).mean()), 2), "RMS": round(orms, 2),
                        "bias": round(float(err.mean()), 2), "medAE": round(float(np.median(np.abs(err))), 2)},
            "n_reactions": n, "n_abs_err_gt_200": int(np.sum(np.abs(err) > 200)),   # kept, not excluded
            "cv_grouping_version": GROUP_VERSION if groups else None,
            "cv_fold_policy": "2026-10-21-whole-groups-sorted-by-sha256-round-robin",
            "n_cv_folds": n_folds, "n_cv_heldout": len(ho_abs), "n_cv_groups": len(unique_groups),
            "cv_group_assignments": group_fold,
            "formal_coverage_guarantee": False, "external_validation_available": False,
            "cv_grouping": (f"near-duplicate clusters (direction/extent invariant difference-fingerprint Tanimoto >= {GROUP_SIM}), "
                            f"{len(set(gkey))} groups" if groups else "per reaction (NOT grouped: leaky)"),
            "default_sigma": round(orms, 1), "global_q95abs": round(gq, 1), "classes": classes,
            "interval_sigma_mult": m95,
            "interval_form": "half_width = max(m * sqrt(sigma_class^2 + U_samp^2), q95abs)",
            "cv": "fully nested: TECRDB-referenced anchor offsets, sigma, q95abs, m refit per training fold",
            "cv_coverage_note": "unconditional over every record with an estimate; no outcome-based exclusion",
            "cv_heldout_MAE": round(float(np.mean(ho_abs)), 2) if ho_abs else None,
            "cv_heldout_MAE_note": ("held-out only with respect to the parameters refit per fold (TECRDB-referenced "
                                    "anchors, sigma). With no anchors this equals the development MAE: it is NOT a "
                                    "generalization estimate; routing/policy choices were made on the same TECRDB."),
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
        v = [r["dir"] * (r["dG_raw"] - r["exp"]) / r["ext"] for r in rows if r["sc"] == sc and r["rid"] in pool]
        if v:
            out[sc] = {"offset": round(float(np.mean(v)), 1), "n": len(v),
                       "std": round(float(np.std(v, ddof=1)), 1) if len(v) > 1 else None}
    return out


def write_artifact(calib, path=ARTIFACT):
    """Publish a complete finite JSON snapshot with an atomic replacement."""
    from pathlib import Path
    import tempfile
    destination = Path(path)
    content = json.dumps(calib, indent=2, allow_nan=False) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=destination.parent,
                                         prefix=destination.name + '.', suffix='.tmp', delete=False) as fh:
            temporary = Path(fh.name)
            fh.write(content)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
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
        elif r.get("dG") is None or r.get('suspect'):
            excluded[rid] = f"no estimate (suspect: {r.get('suspect')})"
        elif r.get("exp") is None:
            excluded[rid] = "no experiment"
        else:
            usable.append(dict(r, rid=rid))
    return usable, excluded


def build_artifact(sweep_dir, path=ARTIFACT, reactions_file=None):
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
    from metag.validation.calibration_artifacts import configuration_issues
    issues = configuration_issues(json.loads(next(iter(cfgs))))
    if issues:
        raise ValueError('; '.join(issues))
    groups = None
    if reactions_file:
        with open(reactions_file) as fh:
            groups = reaction_groups(json.load(fh))
    calib = calibrate(recs, groups=groups)
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


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Rebuild metag/data/sigma_class_calibrated.json from a sweep, "
                                             "or print refreshed anchor offsets (--refit-anchors).")
    ap.add_argument("sweep_dir")
    ap.add_argument("--out", default=ARTIFACT)
    ap.add_argument("--reactions", help="input reactions JSON of the sweep: enables near-duplicate-grouped CV "
                                        "folds (strongly recommended)")
    ap.add_argument("--refit-anchors", action="store_true")
    a = ap.parse_args(argv)
    if a.refit_anchors:
        print(json.dumps(refit_anchor_offsets(load_sweep(a.sweep_dir)[0]), indent=2))
    else:
        print("wrote", build_artifact(a.sweep_dir, a.out, reactions_file=a.reactions))


if __name__ == '__main__':
    main()
