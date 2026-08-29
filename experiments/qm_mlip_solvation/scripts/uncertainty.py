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
_CALIB_PATH = os.path.join(_HERE, os.pardir, "artifacts", "sigma_class_calibrated.json")

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


def mech_class(note, species_smiles):
    """Assign the mechanism/uncertainty class from the reaction note (enzyme name / EC) plus the
    species SMILES. Deployment-safe (no reaction-id). Mirrors tools/fix_map.mech ordering —
    most-specific class first. `species_smiles` = iterable of SMILES strings (or one joined string).
    """
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


def prediction_interval(note, species_smiles, dG, level=95):
    """DE-BIASED, ASYMMETRIC, tail-aware interval for the TRUE ΔrG'° given the pipeline's dG.

    The class residuals (pred - exp) are generally biased (hydratase pred ~ +16 high; glycosyl ~ -14 low),
    so a symmetric dG ± sigma is MIS-CENTERED -- for TFA that is a directional feasibility error. Instead we
    invert the empirical SIGNED residual distribution: exp lies in [dG - q_hi, dG - q_lo] with the empirical
    quantiles, which centers on dG - median_residual (de-biased) and is asymmetric (captures the heavier
    tail). Returns (lo, hi, center, breakdown). Falls back to the symmetric class sigma if uncalibrated.
    """
    cls = mech_class(note, species_smiles)
    st = CLASS_STATS.get(cls)
    if st and "resid_q025" in st:
        qlo, qhi = ("resid_q025", "resid_q975") if level == 95 else ("resid_q16", "resid_q84")
        lo = dG - st[qhi]                       # exp = dG - residual; high residual -> low exp bound
        hi = dG - st[qlo]
        center = dG - st.get("resid_q50", 0.0)  # de-biased point estimate (median residual removed)
        return round(lo, 1), round(hi, 1), round(center, 1), {"class": cls, "level": level,
                "resid_q_lo": st[qlo], "resid_q_hi": st[qhi], "resid_med": st.get("resid_q50")}
    # fallback: symmetric sigma
    s = SIGMA_CLASS.get(cls, _DEFAULT_SIGMA)
    k = 2.0 if level == 95 else 1.0
    return round(dG - k * s, 1), round(dG + k * s, 1), round(dG, 1), {"class": cls, "level": level,
            "sigma": s, "note": "symmetric fallback (uncalibrated)"}


def reaction_sigma(note, species_smiles, U_samp=0.0):
    """Return (sigma_total_kJ, breakdown_dict). Independent error sources added in quadrature:
      sigma_total = sqrt(U_samp^2 + sigma_class^2)
    `species_smiles` = final (possibly truncated/neutralised) species SMILES. `U_samp` = the
    pipeline's conformer-sampling spread for this reaction (kJ/mol; 0 if unknown).
    """
    cls = mech_class(note, species_smiles)
    s_class = SIGMA_CLASS.get(cls, _DEFAULT_SIGMA)
    terms = {"U_samp": float(U_samp), "sigma_class": float(s_class)}
    sigma = math.sqrt(sum(v * v for v in terms.values()))
    return round(sigma, 1), {"class": cls, **terms}


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
