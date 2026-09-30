"""Calibrated per-reaction uncertainty for a predicted reaction ΔrG'° — for downstream
thermodynamic flux / TFA, where the ΔG CONFIDENCE INTERVAL (not the point estimate) decides
feasibility.

WHY this is needed: the pipeline's `U_samp` is ONLY the conformer-sampling spread (~1-3 kJ). The
TRUE prediction error is the residual vs experiment (class-dependent, ~7-60 kJ), dominated by
SYSTEMATIC method error that U_samp cannot see: charged-solute + neutral-conformer aqueous
solvation (the phosphate/CoA wall), the glycosyl/thioester electronic-reference ceiling, and
experimental scatter. Reporting ΔG ± U_samp alone would make a TFA solver ~10x overconfident on
exactly the hard classes (phosphagen, CoA, glycosyl) — flagging them "confidently wrong" instead of
"unconstrained". This module returns a CALIBRATED σ that reflects the real expected
|predicted − experiment| for the reaction's MECHANISM class.

DESIGN
- The class is assigned STRUCTURALLY when the reaction matches an anchor sub-class (SMARTS, via
  metag.routing.anchor), otherwise from the reaction NOTE (enzyme name / EC keywords) + species SMILES.
  No reaction-id lookup. Note-based classes are only as good as the note: a cryptic/absent note falls to
  "other/clean".
- σ_class is the shrunk residual RMS per class of the DEPLOYED pipeline on TECRDB, written by
  `metag/tools/calibrate.py` to `metag/data/sigma_class_calibrated.json` (fully nested CV: anchor
  offsets, σ, q95 and the multiplier refit per fold). Recalibrate after every pipeline change.
- σ_total = sqrt(U_samp^2 + σ_class^2)   [independent sources in quadrature]; the 95% interval uses
  the same σ_total: half-width = max(m·σ_total, q95abs).

SCOPE (read before using intervals for TFA/MDF): the calibration is EMPIRICAL and IN-DISTRIBUTION —
it measures the error on TECRDB-like chemistry. There is no computed (ensemble / cycle-consistency)
uncertainty yet, so for reactions carrying OOD flags or in a class absent from the calibration set, the
interval is reported with `externally_calibrated = False`: a nominal width, not a validated coverage.

COST: FREE per prediction — σ_class is a lookup, the class is a keyword+SMARTS match, U_samp is
already computed by the pipeline. No extra QM.
"""
import os
import re
import json
import math
from statistics import NormalDist

_HERE = os.path.dirname(os.path.abspath(__file__))
_CALIB_PATH = os.path.join(_HERE, "data", "sigma_class_calibrated.json")

# Fallback σ (kJ/mol) if the calibrated artifact is missing. These are the 2026-08-21 per-class
# residual RMS from the six-method benchmark (pre-CoA-truncation) — conservative placeholders only;
# tools/calibrate_uncertainty.py OVERWRITES them from the current sweep.
_FALLBACK_SIGMA = {
    "phosphagen(P-N/Mg)":      60.0,
    "CoA-thioester":           27.0,
    "glycosyl/PRT/nucleoside": 25.0,
    "hydratase":               19.0,
    "phosphatase":             17.0,
    "aldolase":                19.0,
    "amide/amidine-hydrolysis":16.0,
    "reductive-amination-DH":  20.0,
    "carbamoyltransfer":       12.0,
    "ammonia-lyase":           17.0,
    "flavin/FAD-redox":        14.0,
    "kinase/phosphotransfer":  18.0,
    "NAD(P)-redox(other)":     16.0,
    "transaminase":            11.0,
    "isomerase/mutase":        12.0,
    "other/clean":             17.0,
}
DEFAULT_SIGMA = 30.0    # class not in the table -> conservative

_COA_SMI_TAG = "SCCNC(=O)CCNC(=O)".lower()   # pantetheine arm — CoA fingerprint in SMILES
_COA_WORD = re.compile(r"(?<![a-z])coa(?![a-z])")    # "acetyl-CoA", "CoA ligase"; not "glucoamylase"


def _load_sigma():
    """Load the calibrated per-class σ; fall back to the baked-in placeholders."""
    try:
        with open(_CALIB_PATH) as fh:
            calib = json.load(fh)
        # artifact format: {class: {"sigma": float, "n": int, "rms": float, ...}, ...}
        out = {k: float(v["sigma"]) for k, v in calib.get("classes", {}).items()}
        if out:
            return out, calib.get("default_sigma", DEFAULT_SIGMA)
    except (OSError, ValueError, KeyError):
        pass
    return dict(_FALLBACK_SIGMA), DEFAULT_SIGMA


SIGMA_CLASS, _DEFAULT_SIGMA = _load_sigma()


# Structural anchor sub-classes (SMARTS-detected in metag.routing.anchor) -> the calibrated σ-class name.
# When a reaction STRUCTURALLY matches an anchor class, the class MUST come from structure, not the note:
# the note mis-labels (adenylyltransferase -> "kinase") or is cryptic ("ENTF-RXN.c" -> "other/clean"),
# which gave the SAME chemistry two different σ. This keeps the σ-class coherent with the anchor and
# note-independent (works on the poorly-annotated GC-silent ModelSEED target).
# 2026-08-30: anchor.subclass() now returns finer names (thioester_ppi/_pi, adenylylate_aliphatic/
# _aminoacid) after the mechanism-mixing/staleness audit -- both map to the SAME coarse σ-class bucket
# here, since the uncertainty artifact (sigma_class_calibrated.json) is calibrated at the coarse
# taxonomy level, not per anchor sub-class. phosphatase_monoester_cationic is DELIBERATELY left out: it
# now falls through to the note-based classifier below (still lands on "phosphatase" for any reaction
# whose note mentions it), which is an approximation -- its point estimate is dG_raw (uncorrected), not
# the same distribution the "phosphatase" σ was calibrated against. Needs a real recalibration sweep to
# get this right; flagged, not silently trusted.
_ANCHOR_TO_CLASS = {
    "phosphagen":            "phosphagen(P-N/Mg)",
    "thioester_ppi":         "CoA-thioester",
    "thioester_pi":          "CoA-thioester",
    "phosphatase_monoester": "phosphatase",
    # adenylylate offsets are referenced to an EXTERNAL indirect cycle and no member is in the TECRDB
    # calibration set, so this bucket is never calibrated -> class_calibrated=False and the wide
    # uncalibrated width (>= overall σ, >= the other/clean class). It used to share a bucket with carboxyP,
    # whose two TECRDB rows then made "adenylylate" look calibrated on data containing no adenylylate.
    "adenylylate_aliphatic": "adenylylate(external-ref)",
    "adenylylate_aminoacid": "adenylylate(external-ref)",
    "carboxyP":              "carboxyP",
    "amide_hydrolysis":      "amide/amidine-hydrolysis",
}
_DECARBOXYLASE = re.compile(r"(?<!de)carboxylase")   # "carboxylase" but not "decarboxylase"


def mech_class(note, species_smiles, species=None):
    """Assign the mechanism/uncertainty class. STRUCTURAL FIRST: if `species` (the full
    {name:[coeff,charge,smi]} dict) is given and structurally matches an anchor sub-class, that wins
    (note-independent, coherent with the anchor). Otherwise fall back to the NOTE (enzyme name / EC) +
    SMILES keyword taxonomy. Deployment-safe (no reaction-id). `species_smiles` = iterable of SMILES.
    """
    if species:
        from metag.routing.anchor import subclass as _anchor_subclass
        sc = _anchor_subclass(species)
        if sc in _ANCHOR_TO_CLASS:
            return _ANCHOR_TO_CLASS[sc]
    n = (note or "").lower()
    smis = species_smiles if isinstance(species_smiles, str) else " ".join(species_smiles)
    has_coa = bool(_COA_WORD.search(n)) or _COA_SMI_TAG in smis.lower()   # whole word: not "glucoamylase"

    if ("phosphoribosyl" in n or "nucleosidase" in n
            or ("phosphorylase" in n and any(k in n for k in ("uridine", "purine", "nucleoside")))):
        return "glycosyl/PRT/nucleoside"
    if has_coa or any(k in n for k in [
            "-coa", "coa ", "coa)", "acetyltransferase", "citrate (si)-synthase", "citrate(pro",
            "succinate-coa", "acetate-coa", "carnitine o-", "formate c-acetyl", "propanoyl-coa",
            "malyl-coa", "oxoacid coa", "c-acetyltransferase", "phosphate acetyltransferase",
            "serine acetyltransferase", "choline o-acetyl"]):
        return "CoA-thioester"
    if any(k in n for k in ["creatine kinase", "arginine kinase", "taurocyamine", "lombricine",
                            "phosphagen"]):
        return "phosphagen(P-N/Mg)"
    if ("ammonia-lyase" in n or "aspartase" in n or "tryptophanase" in n
            or ("lyase" in n and ("argininosucc" in n or "arginosucc" in n)) or "adenylosuccinate lyase" in n
            or ("cyclase" in n and "glutamate" in n)):
        return "ammonia-lyase"
    if (("dehydrogenase" in n and any(k in n for k in [
            "glutamate", "alanine deh", "leucine", "octopine", "alanopine", "saccharopine",
            "diaminopentanoate", "diaminohexanoat"]))
            or ("reductase" in n and any(k in n for k in ["piperidine", "pyrroline"]))):
        return "reductive-amination-DH"
    if "carbamoyltransfer" in n or "carbamoyl-transfer" in n:
        return "carbamoyltransfer"
    if any(k in n for k in ["amidohydrolase", "amidase", "aminoacylase", "urease", "deaminase",
                            "pantothenase", "allantoicase", "asparaginase", "glutaminase"]):
        return "amide/amidine-hydrolysis"
    if any(k in n for k in ["flavin", "dihydroorotate dehydrogenase", "methylenetetrahydrofolat",
                            "dihydrolipoamide"]):
        return "flavin/FAD-redox"
    if ("hydratase" in n or "dehydratase" in n or "hydro-lyase" in n
            or any(k in n for k in ["fumarase", "fumarate hydratase", "enolase", "aconitase"])):
        return "hydratase"
    if "aldolase" in n or "aldol" in n:
        return "aldolase"
    if "transaminase" in n or "aminotransferase" in n or "transamin" in n:
        return "transaminase"
    if "phosphatase" in n:
        return "phosphatase"
    if any(k in n for k in ["isomerase", "epimerase", "mutase", "cycloisomerase", "racemase"]):
        return "isomerase/mutase"
    if (any(k in n for k in ["kinase", "dikinase", "adenylyltransferase", "pyrophospho", "diphospho"])
            or _DECARBOXYLASE.search(n)):          # ATP-dependent carboxylases; decarboxylases are NOT kinases
        return "kinase/phosphotransfer"
    if "dehydrogenase" in n or "oxidase" in n or "reductase" in n:
        return "NAD(P)-redox(other)"
    return "other/clean"


def _load_class_stats():
    """Full per-class stats (bias_shrunk + signed residual quantiles) for the de-biased asymmetric
    prediction interval. Empty if the calibrated artifact is missing (then prediction_interval falls
    back to the symmetric sigma)."""
    try:
        with open(_CALIB_PATH) as fh:
            return json.load(fh).get("classes", {})
    except (OSError, ValueError):
        return {}


CLASS_STATS = _load_class_stats()


def _load_global_q95():
    try:
        with open(_CALIB_PATH) as fh:
            c = json.load(fh)
        if "global_q95abs" in c:
            return float(c["global_q95abs"])
        return max((v.get("q95abs", 0.0) for v in c.get("classes", {}).values()), default=0.0)
    except (OSError, ValueError):
        return 0.0


_GLOBAL_Q95 = _load_global_q95()


def _load_calib_config():
    try:
        with open(_CALIB_PATH) as fh:
            return json.load(fh).get("config")
    except (OSError, ValueError):
        return None


CALIB_CONFIG = _load_calib_config()


def calibration_mismatch(config):
    """List of reasons the runtime pipeline configuration differs from the one the artifact was calibrated
    on (empty = same estimator). `config` = metag.pipeline.effective_config(). No config supplied, or an
    artifact without a fingerprint, counts as a mismatch: coverage is only validated for the exact
    estimator that was calibrated, so an unverifiable configuration must not be reported as calibrated."""
    if config is None:
        return ["runtime configuration not supplied (pass config=metag.pipeline.effective_config())"]
    if not CALIB_CONFIG:
        return ["calibration artifact records no configuration fingerprint (re-run metag.tools.calibrate)"]
    keys = sorted(set(config) | set(CALIB_CONFIG))
    return [f"{k}: runtime {config.get(k)!r} != calibrated {CALIB_CONFIG.get(k)!r}"
            for k in keys if config.get(k) != CALIB_CONFIG.get(k)]


def _class_width(cls):
    """(sigma_class, q95abs, calibrated). A class absent from the calibration set (e.g. a new anchor
    bucket before the next sweep) gets at least the overall σ and the global heavy-tail floor -- never a
    narrower interval than the calibrated classes (previously adenylylate got σ=11, q95=0: the tightest
    interval of all, on its least-validated class)."""
    st = CLASS_STATS.get(cls)
    if st is not None:
        return float(st.get("sigma", SIGMA_CLASS.get(cls, _DEFAULT_SIGMA))), float(st.get("q95abs", 0.0)), True
    # never narrower than the global statistics OR the "other/clean" fallback class (where un-annotated
    # novel reactions land anyway), whichever is wider
    oc = CLASS_STATS.get("other/clean", {})
    s = max(float(SIGMA_CLASS.get(cls, _DEFAULT_SIGMA)), float(_DEFAULT_SIGMA), float(oc.get("sigma", 0.0)))
    return s, max(_GLOBAL_Q95, float(oc.get("q95abs", 0.0))), False

# sigma multiplier for the (symmetric) 95% interval, calibrated so held-out coverage >= 95% (~2.1).
_INTERVAL_MULT = 2.1
try:
    with open(_CALIB_PATH) as _fh:
        _INTERVAL_MULT = float(json.load(_fh).get("interval_sigma_mult", 2.1))
except (OSError, ValueError):
    pass


def _ood(species):
    if not species:
        return None
    from metag.routing.applicability import ood_assessment
    return ood_assessment(species)


def _scope(calibrated, ood_info, mismatch=()):
    """Is the interval externally calibrated (validated coverage) or nominal?"""
    flags = (ood_info or {}).get("flags") or []
    ext = bool(calibrated and not flags and not mismatch)
    if ext:
        note = "in-distribution: nested-CV coverage on TECRDB applies"
    elif mismatch:
        note = ("interval not externally calibrated: runtime configuration differs from the calibrated one ("
                + "; ".join(mismatch) + ")")
    elif not calibrated:
        note = "interval not externally calibrated: class absent from calibration set (nominal width)"
    else:
        note = "interval not externally calibrated: OOD features " + "; ".join(flags)
    return ext, note


def prediction_interval(note, species_smiles, dG, level=95, species=None, U_samp=0.0, config=None):
    """Symmetric prediction interval for the TRUE ΔrG'° given the pipeline's dG. Returns
    (lo, hi, center, breakdown); center == dG (no de-biasing).

    half-width = max(m · sqrt(σ_class² + U_samp²), q95abs): the SAME σ_total as reaction_sigma(), so
    ci95 and sigma_pred are consistent (a floppy species with large U_samp widens both). m is the
    nested-CV multiplier (held-out coverage >= 95%), q95abs a per-class heavy-tail floor. Only level=95
    is CV-calibrated; other levels scale m and q95abs by the Gaussian z-ratio and are reported with
    level_calibrated=False. An asymmetric de-biased interval was tried and CV-rejected (OOD-fragile).
    `species` (full {name:[coeff,charge,smi]} dict) enables structural class detection; recommended.
    `config` = metag.pipeline.effective_config(); externally_calibrated requires it to match the artifact.
    """
    cls = mech_class(note, species_smiles, species)
    st = CLASS_STATS.get(cls, {})
    s_cls, q95abs, calibrated = _class_width(cls)
    ood_info = _ood(species)
    s = math.hypot(s_cls, float(U_samp or 0.0))
    if ood_info:                                             # flag-only today (sigma_floor = 0)
        s = max(s, ood_info.get("sigma_floor", 0.0))
    if level == 95:
        m, q = _INTERVAL_MULT, q95abs
    else:
        zr = NormalDist().inv_cdf(0.5 + level / 200.0) / NormalDist().inv_cdf(0.975)
        m, q = _INTERVAL_MULT * zr, q95abs * zr
    hw = max(m * s, q)
    mismatch = calibration_mismatch(config)
    ext, scope = _scope(calibrated, ood_info, mismatch)
    return round(dG - hw, 1), round(dG + hw, 1), round(dG, 1), {"class": cls, "level": level,
            "level_calibrated": level == 95,
            "sigma": round(s, 1), "sigma_class": s_cls, "U_samp": float(U_samp or 0.0),
            "sigma_mult": round(m, 3), "half_width": round(hw, 1), "q95abs": round(q, 1),
            "class_calibrated": calibrated, "externally_calibrated": ext, "calibration_scope": scope,
            "config_mismatch": mismatch,
            "ood": bool(ood_info and ood_info["ood"]),
            "ood_reasons": (ood_info["reasons"] if ood_info else []),
            "ood_flags": (ood_info["flags"] if ood_info else []),
            "point_bias": st.get("bias"), "point_bias_note": "mean residual of the DEPLOYED (post-anchor) "
            "estimator on the calibration set, in-distribution only; the interval already covers it -- "
            "do NOT also widen"}


def reaction_sigma(note, species_smiles, U_samp=0.0, species=None, config=None):
    """Return (sigma_total_kJ, breakdown_dict). Independent error sources added in quadrature:
      sigma_total = sqrt(U_samp^2 + sigma_class^2)
    `species_smiles` = final (possibly truncated/neutralised) species SMILES. `U_samp` = the
    pipeline's conformer-sampling spread for this reaction (kJ/mol; 0 if unknown).
    `species` (full {name:[coeff,charge,smi]} dict) enables structural class detection; recommended.
    """
    cls = mech_class(note, species_smiles, species)
    s_class, _q, calibrated = _class_width(cls)
    terms = {"U_samp": float(U_samp), "sigma_class": float(s_class)}
    sigma = math.sqrt(sum(v * v for v in terms.values()))
    br = {"class": cls, **terms}
    # OOD gate: if the reaction is structurally unlike the calibration set, FLOOR sigma (never narrow).
    if species:
        from metag.routing.applicability import ood_assessment
        oa = ood_assessment(species)
        br["ood"] = oa["ood"]; br["ood_reasons"] = oa["reasons"]
        if oa["flags"]:
            br["ood_flags"] = oa["flags"]               # informational only -- do NOT affect sigma
        if oa["sigma_floor"] > sigma:
            br["sigma_floored_from"] = round(sigma, 1)
            sigma = oa["sigma_floor"]
    else:
        oa = None
    br["class_calibrated"] = calibrated
    br["externally_calibrated"], br["calibration_scope"] = _scope(calibrated, oa, calibration_mismatch(config))
    return round(sigma, 1), br


if __name__ == "__main__":
    print(f"loaded {len(SIGMA_CLASS)} calibrated classes from "
          f"{'artifact' if os.path.exists(_CALIB_PATH) else 'FALLBACK placeholders'}")
    for note, smis, u in [
        ("creatine kinase | EC=2.7.3.2", ["O=P([O-])([O-])O"], 1.8),
        ("acetyl-CoA C-acetyltransferase | EC=2.3.1.9", ["CC(=O)SCCNC(=O)CCNC(=O)"], 2.5),
        ("triose-phosphate isomerase | EC=5.3.1.1", ["CC(=O)C(=O)[O-]"], 1.0),
        ("some novel hydratase | EC=4.2.1.-", ["OCC(O)C"], 2.0),
    ]:
        print(f"  {note[:40]:40s} -> {reaction_sigma(note, smis, u)}")
