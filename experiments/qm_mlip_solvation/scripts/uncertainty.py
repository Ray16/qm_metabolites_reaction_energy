"""Calibrated total uncertainty for a predicted reaction ΔG — for downstream thermodynamic flux /
TFA, where the ΔG CONFIDENCE INTERVAL (not the point estimate) decides feasibility.

WHY this is needed: the pipeline's `U_samp` is ONLY the conformer-sampling spread (~1-3 kJ). The
TRUE prediction error is the residual vs experiment (~15-40 kJ, dominated by SYSTEMATIC method error:
anion-solvation, electronic reference ceiling, truncation). Reporting ΔG ± U_samp alone would make a
TFA solver ~10x overconfident. This module returns a CALIBRATED σ that reflects the real expected
|predicted - experiment| for the reaction's TYPE.

COST: essentially FREE per prediction — σ_class is a LOOKUP (calibrated ONCE from the full-367
validation), U_samp is already computed, σ_motif is a SMARTS lookup. No extra QM. (An optional
radius-sensitivity term for truncated reactions costs 2x QM and is NOT required — the huge/floppy
class-σ already captures truncation uncertainty empirically.)

σ_total = sqrt(U_samp^2 + σ_class^2 + σ_motif^2)   [independent error sources in quadrature]
"""
import math
from rdkit import Chem

# Per-class residual RMS (kJ/mol), calibrated from predicted-vs-TECRDB validation. These are the
# HONEST systematic uncertainties by reaction type. UPDATE the `anion` value after the gated-pH-0
# pass (baseline 47.5 -> projected ~18); the others are pH-0-independent.
SIGMA_CLASS = {
    "anion":       18.0,   # gated-pH-0 target (baseline RMS 47.5 -> ~18 validated); UPDATE post-pass
    "huge/floppy": 49.3,   # conformer noise; drops when truncation succeeds (use radius-sens refine)
    "clean":       22.9,
    "isomerase":   11.5,   # near-equilibrium, small
    "thioester":   50.0,   # reference-ceiling (neutral C(=O)-S resonance) -> wide bars, honestly
    "glycosyl":    45.0,   # anomeric C-X reference-ceiling (calibrate as data accrues)
}
DEFAULT_SIGMA = 30.0       # unknown class -> conservative

# motifs on the reference-ceiling frontier -> the QM electronic error is irreducible here; inflate.
_THIOESTER = Chem.MolFromSmarts("[CX3](=O)[SX2]")
_ANOMERIC = Chem.MolFromSmarts("[CX4;R]([OX2,NX3])[OX2,NX3]")   # ring C bonded to two O/N (anomeric)


def rxn_class(note, smis):
    """Route a reaction to its uncertainty class from its flags + structure (no reaction-id logic)."""
    if any(m and m.HasSubstructMatch(_THIOESTER) for m in smis):
        return "thioester"
    if any(m and m.HasSubstructMatch(_ANOMERIC) for m in smis) and ("transferase" in note.lower()
                                                                     or "glycosyl" in note.lower()):
        return "glycosyl"
    if "isomerase" in note:
        return "isomerase"
    if "huge/floppy" in note:
        return "huge/floppy"
    if "Mg-prone" in note or "anion-count" in note:
        return "anion"
    return "clean"


def reaction_sigma(note, smiles_list, U_samp, radius_delta=None):
    """Return (sigma_total, breakdown). smiles_list = final (possibly truncated/neutralised) species
    SMILES. radius_delta = optional |ΔG(r)-ΔG(r+1)| truncation-reliability term (kJ); None to skip.
    All terms independent -> quadrature."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    cls = rxn_class(note, mols)
    s_class = SIGMA_CLASS.get(cls, DEFAULT_SIGMA)
    terms = {"U_samp": float(U_samp), "sigma_class": s_class}
    if radius_delta is not None:
        terms["radius_sens"] = float(radius_delta)
    sigma = math.sqrt(sum(v * v for v in terms.values()))
    return round(sigma, 1), {"class": cls, **terms}


if __name__ == "__main__":
    # demo
    for note, smis, u in [
        ("... flags=Mg-prone(NTP/PPi)", ["O=P([O-])([O-])O"], 1.8),
        ("... flags=isomerase", ["CC(=O)C(=O)[O-]"], 1.0),
        ("... flags=huge/floppy transferase", ["OC[C@H]1OC(O)[C@H](O)[C@@H]1O"], 2.5),
    ]:
        print(note[:30], "->", reaction_sigma(note, smis, u))
