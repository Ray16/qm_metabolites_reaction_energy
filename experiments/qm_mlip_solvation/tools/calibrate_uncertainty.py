"""Calibrate the per-mechanism-class prediction uncertainty (sigma_class) from the clean
full-367 `logs/production` sweep, and write it to `artifacts/sigma_class_calibrated.json`,
which `scripts/uncertainty.py` loads at import.

sigma_class = RMS of the RAW residual (uma - exp) per class. RAW (not bias-subtracted) because the
pipeline does NOT apply a per-class bias correction (that would be fitting TECRDB) — so the honest
1-sigma the prediction carries is sqrt(std^2 + bias^2) = RMS of the residual. This is exactly the
number a TFA solver needs to treat a hard class (phosphagen/CoA/glycosyl) as UNCONSTRAINED rather
than confidently wrong.

Prints a calibration table + a coverage check (what fraction of |err| falls within 1 and 2 sigma_class;
well-calibrated ~= 68% / 95%). ZERO GPU.

Usage: python tools/calibrate_uncertainty.py [LOGDIR=logs/production]
"""
import os
import re
import sys
import json
import importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
LOGDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(EXP, "logs", "production")
RXN_FILE = os.path.join(EXP, "scripts", "reactions_tecrdb_all.json")
OUT = os.path.join(EXP, "artifacts", "sigma_class_calibrated.json")

# import the deployment classifier from scripts/uncertainty.py (clean import, no side effects)
_s = importlib.util.spec_from_file_location("uncertainty",
                                            os.path.join(EXP, "scripts", "uncertainty.py"))
unc = importlib.util.module_from_spec(_s)
_s.loader.exec_module(unc)

_DG = re.compile(r"ΔG = ([+-]?\d+\.\d+)")

# The deployed pipeline applies aldehyde-hydration + anchor corrections on top of the raw logged ΔG, so
# σ_class must be calibrated on the DEPLOYED residual, not the raw one (else the anchored classes --
# phosphagen/phosphatase/thioester -- get a stale, too-large σ). We apply the same two corrections here.
sys.path.insert(0, os.path.join(EXP, "tools"))
sys.path.insert(0, os.path.join(EXP, "scripts"))
import where_lacking as _wl          # ald_delta(rid) reads logs/ah367_on
from route_anchor import anchor_correct as _anchor


def read_dG(rid, species=None):
    p = os.path.join(LOGDIR, f"{rid}.log")
    if not os.path.exists(p):
        return None
    m = _DG.search(open(p, errors="ignore").read())
    if not m:
        return None
    dG = float(m.group(1)) + _wl.ald_delta(rid)          # deployed: + aldehyde hydration
    if species is not None:
        ac = _anchor(dG, species)                        # deployed: + anchor correction
        if ac is not None:
            dG = ac[0]
    return dG


def main():
    d = json.load(open(RXN_FILE))
    rows = []          # (rid, cls, err)
    missing = 0
    for rid, rec in d.items():
        u = read_dG(rid, rec["species"])
        if u is None:
            missing += 1
            continue
        exp = rec["exp"][0]
        err = u - exp
        if abs(err) > 200:                        # QM garbage / loader failure -> not a class stat
            continue
        smis = [s[2] for s in rec["species"].values()]
        cls = unc.mech_class(rec["note"], smis)
        rows.append((rid, cls, err))

    allerr = np.array([e for *_, e in rows])
    N = len(allerr)
    overall_rms = float(np.sqrt((allerr ** 2).mean()))
    overall_mae = float(np.abs(allerr).mean())

    by = {}
    for _, cls, e in rows:
        by.setdefault(cls, []).append(e)

    classes = {}
    for cls, es in by.items():
        es = np.array(es)
        rms = float(np.sqrt((es ** 2).mean()))
        mae = float(np.abs(es).mean())
        bias = float(es.mean())
        med = float(np.median(np.abs(es)))
        # sigma = residual RMS, made SAFE for small n. Shrinkage toward the overall RMS
        # (sigma = sqrt((n*rms^2 + k*overall_rms^2)/(n+k)), k=4 pseudo-counts) protects a tiny-n class
        # whose RMS is spuriously SMALL from being overconfident. But it must NEVER pull a legitimately
        # LARGE class down (e.g. phosphagen RMS 60 -> would wrongly become 44 and under-cover its real
        # +60 error), so we take the MAX of the shrunk value and the raw RMS -> one-sided (inflate-only)
        # shrinkage. Floored at 6 kJ so a near-exact class isn't reported as certain.
        n = len(es)
        k = 4.0
        shrunk = float(np.sqrt((n * rms ** 2 + k * overall_rms ** 2) / (n + k)))
        sigma = round(max(shrunk, rms, 6.0), 1)
        # NON-PARAMETRIC cross-check: empirical quantiles of |residual|. If the Gaussian sigma is right,
        # q68 ~= sigma and q95 ~= 2*sigma. (Noisy for small n; reported as a diagnostic, not the bar.)
        aes = np.abs(es)
        q68 = float(np.quantile(aes, 0.68))
        q95 = float(np.quantile(aes, 0.95))
        classes[cls] = {"n": n, "sigma": sigma, "rms": round(rms, 1), "mae": round(mae, 1),
                        "bias": round(bias, 1), "medAE": round(med, 1),
                        "sigma_raw_rms": round(rms, 1),
                        "emp_q68": round(q68, 1), "emp_q95": round(q95, 1)}

    default_sigma = round(overall_rms, 1)

    # ---- HONEST calibration: k-fold CV coverage. sigma_class fit on TRAIN folds only, coverage
    # measured on held-out folds -> not circular. Deterministic fold assignment (hash of rid), no RNG.
    def _cv_coverage(n_folds=5):
        import hashlib
        fold = {rid: int(hashlib.md5(rid.encode()).hexdigest(), 16) % n_folds
                for rid, _, _ in rows}
        got1 = got2 = 0
        for f in range(n_folds):
            tr = {}
            for rid, cls, e in rows:
                if fold[rid] != f:
                    tr.setdefault(cls, []).append(e)
            tr_rms = {c: float(np.sqrt(np.mean(np.square(v)))) for c, v in tr.items()}
            for rid, cls, e in rows:
                if fold[rid] != f:
                    continue
                v = tr.get(cls, [])
                n = len(v)
                crms = tr_rms.get(cls, overall_rms)
                shrunk = np.sqrt((n * crms ** 2 + 4 * overall_rms ** 2) / (n + 4)) if n else overall_rms
                s = max(shrunk, crms, 6.0)   # one-sided (inflate-only) shrinkage, matches deploy rule
                got1 += abs(e) <= s
                got2 += abs(e) <= 2 * s
        return got1 / len(rows), got2 / len(rows)

    cv1, cv2 = _cv_coverage()

    out = {
        "source": os.path.relpath(LOGDIR, EXP),
        "n_reactions": N,
        "n_missing_log": missing,
        "overall": {"MAE": round(overall_mae, 2), "RMS": round(overall_rms, 2),
                    "bias": round(float(allerr.mean()), 2),
                    "medAE": round(float(np.median(np.abs(allerr))), 2)},
        "cv_coverage_1sigma": round(cv1, 3),
        "cv_coverage_2sigma": round(cv2, 3),
        "default_sigma": default_sigma,
        "classes": classes,
        "per_reaction": [{"rid": rid, "class": cls, "err": round(float(e), 2),
                          "sigma": classes[cls]["sigma"]} for rid, cls, e in rows],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)

    # ---- report ----
    print(f"calibrated from {out['source']}  N={N}  ({missing} logs missing/unreadable)")
    print(f"OVERALL  MAE={overall_mae:.1f}  RMS={overall_rms:.1f}  "
          f"bias={allerr.mean():+.1f}  medAE={np.median(np.abs(allerr)):.1f}")
    print(f"\n{'class':26s} {'n':>3s} {'MAE':>6s} {'bias':>7s} {'RMS':>6s} {'sigma':>6s}")
    for cls, v in sorted(classes.items(), key=lambda kv: -kv[1]["sigma"]):
        print(f"{cls:26s} {v['n']:3d} {v['mae']:6.1f} {v['bias']:+7.1f} {v['rms']:6.1f} {v['sigma']:6.1f}")
    print(f"{'(default / unknown class)':26s} {'':3s} {'':6s} {'':7s} {'':6s} {default_sigma:6.1f}")

    # coverage: fraction of reactions whose |err| <= m * sigma_class  (well-calibrated: ~0.68 / 0.95)
    print("\nCALIBRATION COVERAGE (|err| within m*sigma_class):")
    sig = np.array([classes[cls]["sigma"] for _, cls, _ in rows])
    for m in (1.0, 2.0):
        cov = float((np.abs(allerr) <= m * sig).mean())
        print(f"  in-sample within {m:.0f} sigma: {cov*100:5.1f}%   (ideal {'68' if m==1 else '95'}%)")
    print(f"  5-fold CV  within 1 sigma: {cv1*100:5.1f}%   (ideal 68%)   <- the honest number")
    print(f"  5-fold CV  within 2 sigma: {cv2*100:5.1f}%   (ideal 95%)")

    # does a continuous structural feature carry residual-magnitude signal BEYOND the class label?
    # (if strong, a within-class refinement term would help; if weak, class-only sigma is justified.)
    print("\nstructural signal in |err| BEYOND class (Spearman of |within-class-resid| vs feature):")
    d2 = json.load(open(RXN_FILE))
    feats = {"|dG_pred|": [], "max|charge|": [], "n_heavy": [], "|resid|": []}
    for rid, cls, e in rows:
        rec = d2[rid]
        u = read_dG(rid)
        cmae = classes[cls]["mae"]
        feats["|resid|"].append(abs(abs(e) - cmae))          # deviation from class-typical error
        feats["|dG_pred|"].append(abs(u))
        feats["max|charge|"].append(max(abs(s[1]) for s in rec["species"].values()))
        feats["n_heavy"].append(sum(len(s[2]) for s in rec["species"].values()))
    try:
        from scipy.stats import spearmanr
        for f in ("|dG_pred|", "max|charge|", "n_heavy"):
            rho, p = spearmanr(feats[f], feats["|resid|"])
            print(f"  {f:12s} rho={rho:+.2f}  p={p:.2f}")
    except Exception as ex:
        print(f"  (scipy unavailable: {ex})")
    print(f"\nwrote {os.path.relpath(OUT, EXP)}")


if __name__ == "__main__":
    main()
