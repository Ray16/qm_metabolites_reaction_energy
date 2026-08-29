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
- The class is assigned from the reaction NOTE (enzyme name / EC) + species SMILES — no reaction-id
  lookup — so it works on novel ModelSEED reactions, not just TECRDB. Taxonomy = the mechanism
  classes used in the six-method benchmark (fix_map.mech), which track the physics failure modes.
- σ_class is the residual RMS per class, CALIBRATED ONCE from the clean full-367 `logs/production`
  sweep by `tools/calibrate_uncertainty.py`, which writes `artifacts/sigma_class_calibrated.json`.
  Recalibration = re-run that script after a new sweep; NO edit to this file is needed.
- σ_total = sqrt(U_samp^2 + σ_class^2)   [independent sources in quadrature]

COST: FREE per prediction — σ_class is a lookup, the class is a keyword+SMARTS match, U_samp is
already computed by the pipeline. No extra QM.
"""
import os
import json
import math

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
# adenylyl-transfer is a NEW structural class not yet in the calibration artifact; use the anchor's
# LOO+reference-inflated residual (metag.routing.anchor ANCHORS["adenylylate"]["sigma"] = 11.0) until a
# full sweep recalibrates it. setdefault so a future artifact value wins.
SIGMA_CLASS.setdefault("adenylylate", 11.0)


# Structural anchor sub-classes (SMARTS-detected in metag.routing.anchor) -> the calibrated σ-class name.
# When a reaction STRUCTURALLY matches an anchor class, the class MUST come from structure, not the note:
# the note mis-labels (adenylyltransferase -> "kinase") or is cryptic ("ENTF-RXN.c" -> "other/clean"),
# which gave the SAME chemistry two different σ. This keeps the σ-class coherent with the anchor and
# note-independent (works on the poorly-annotated GC-silent ModelSEED target).
_ANCHOR_TO_CLASS = {
    "phosphagen":            "phosphagen(P-N/Mg)",
    "thioester":             "CoA-thioester",
    "phosphatase_monoester": "phosphatase",
    "adenylylate":           "adenylylate",
}


def mech_class(note, species_smiles, species=None):
    """Assign the mechanism/uncertainty class. STRUCTURAL FIRST: if `species` (the full
    {name:[coeff,charge,smi]} dict) is given and structurally matches an anchor sub-class, that wins
    (note-independent, coherent with the anchor). Otherwise fall back to the NOTE (enzyme name / EC) +
    SMILES keyword taxonomy. Deployment-safe (no reaction-id). `species_smiles` = iterable of SMILES.
    """
    if species:
        try:
            from metag.routing.anchor import subclass as _anchor_subclass
            sc = _anchor_subclass(species)
            if sc in _ANCHOR_TO_CLASS:
                return _ANCHOR_TO_CLASS[sc]
        except Exception:
            pass                                            # structural detection is best-effort; fall back
    n = (note or "").lower()
    smis = species_smiles if isinstance(species_smiles, str) else " ".join(species_smiles)
    has_coa = "coa" in n or _COA_SMI_TAG in smis.lower()

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
            or ("lyase" in n and "arginosucc" in n) or "adenylosuccinate lyase" in n
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
                            "pantothenase", "allantoicase"]):
        return "amide/amidine-hydrolysis"
    if any(k in n for k in ["flavin", "dihydroorotate dehydrogenase", "methylenetetrahydrofolat",
                            "dihydrolipoamide"]):
        return "flavin/FAD-redox"
    if "hydratase" in n or "dehydratase" in n or "hydro-lyase" in n:
        return "hydratase"
    if "aldolase" in n or "aldol" in n:
        return "aldolase"
    if "transaminase" in n or "aminotransferase" in n or "transamin" in n:
        return "transaminase"
    if "phosphatase" in n:
        return "phosphatase"
    if any(k in n for k in ["isomerase", "epimerase", "mutase", "cycloisomerase", "racemase"]):
        return "isomerase/mutase"
    if any(k in n for k in ["kinase", "dikinase", "adenylyltransferase", "pyrophospho", "diphospho",
                            "carboxylase"]):
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

# sigma multiplier for the (symmetric) 95% interval, calibrated so held-out coverage >= 95% (~2.1).
_INTERVAL_MULT = 2.1
try:
    with open(_CALIB_PATH) as _fh:
        _INTERVAL_MULT = float(json.load(_fh).get("interval_sigma_mult", 2.1))
except (OSError, ValueError):
    pass


def prediction_interval(note, species_smiles, dG, level=95, species=None):
    """DE-BIASED, ASYMMETRIC, tail-aware interval for the TRUE ΔrG'° given the pipeline's dG.

    The class residuals (pred - exp) are generally biased (hydratase pred ~ +16 high; glycosyl ~ -14 low),
    so a symmetric dG ± sigma is MIS-CENTERED -- for TFA that is a directional feasibility error. Instead we
    invert the empirical SIGNED residual distribution: exp lies in [dG - q_hi, dG - q_lo] with the empirical
    quantiles, which centers on dG - median_residual (de-biased) and is asymmetric (captures the heavier
    tail). Returns (lo, hi, center, breakdown). Falls back to the symmetric class sigma if uncalibrated.
    `species` (full {name:[coeff,charge,smi]} dict) enables structural class detection; recommended.
    """
    cls = mech_class(note, species_smiles, species)
    st = CLASS_STATS.get(cls, {})
    s = st.get("sigma", SIGMA_CLASS.get(cls, _DEFAULT_SIGMA))
    q95abs = st.get("q95abs", 0.0)
    ood_info = None
    if species:                                              # OOD gate floors the interval sigma too
        try:
            from metag.routing.ood import ood_assessment
            ood_info = ood_assessment(species)
            s = max(s, ood_info["sigma_floor"])
        except Exception:
            ood_info = None
    # SYMMETRIC, NESTED-CV-validated interval centred on the (physics+anchor) prediction. Half-width =
    # max(m*sigma, q95abs): m is the multiplier (nested-CV to >=95% held-out), q95abs is a per-class
    # heavy-tail floor for classes (e.g. reductive-amination-DH) that k*sigma under-covers. An asymmetric
    # de-biased interval was tried and CV-REJECTED (92.4% < symmetric 95%; a first-moment fit, OOD-fragile).
    m = _INTERVAL_MULT if level == 95 else _INTERVAL_MULT / 2.0
    hw = max(m * s, q95abs) if level == 95 else m * s
    # class_bias is POINT-ESTIMATE metadata, NOT a coverage correction: the symmetric interval already
    # covers biased classes (wide sigma). It says the POINT is off (hydratase ~ +18 high) so a consumer who
    # wants a sharper estimate MAY recenter dG - class_bias; do not also widen for it (double-applying).
    return round(dG - hw, 1), round(dG + hw, 1), round(dG, 1), {"class": cls, "level": level,
            "sigma": s, "sigma_mult": m, "half_width": round(hw, 1),
            "ood": bool(ood_info and ood_info["ood"]),
            "ood_reasons": (ood_info["reasons"] if ood_info else []),
            "ood_flags": (ood_info["flags"] if ood_info else []),
            "point_bias": st.get("bias"), "point_bias_note": "point-estimate metadata (in-distribution); "
            "interval already covers -- optional recenter dG-point_bias, do NOT also widen"}


def reaction_sigma(note, species_smiles, U_samp=0.0, species=None):
    """Return (sigma_total_kJ, breakdown_dict). Independent error sources added in quadrature:
      sigma_total = sqrt(U_samp^2 + sigma_class^2)
    `species_smiles` = final (possibly truncated/neutralised) species SMILES. `U_samp` = the
    pipeline's conformer-sampling spread for this reaction (kJ/mol; 0 if unknown).
    `species` (full {name:[coeff,charge,smi]} dict) enables structural class detection; recommended.
    """
    cls = mech_class(note, species_smiles, species)
    s_class = SIGMA_CLASS.get(cls, _DEFAULT_SIGMA)
    terms = {"U_samp": float(U_samp), "sigma_class": float(s_class)}
    sigma = math.sqrt(sum(v * v for v in terms.values()))
    br = {"class": cls, **terms}
    # OOD gate: if the reaction is structurally unlike the calibration set, FLOOR sigma (never narrow).
    if species:
        try:
            from metag.routing.ood import ood_assessment
            oa = ood_assessment(species)
            br["ood"] = oa["ood"]; br["ood_reasons"] = oa["reasons"]
            if oa["flags"]:
                br["ood_flags"] = oa["flags"]           # informational only -- do NOT affect sigma
            if oa["sigma_floor"] > sigma:
                br["sigma_floored_from"] = round(sigma, 1)
                sigma = oa["sigma_floor"]
        except Exception:
            pass
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
