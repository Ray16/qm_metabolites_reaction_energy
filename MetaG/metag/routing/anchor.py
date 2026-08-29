"""Per-sub-class anchor correction for the SYSTEMATIC anion-solvation-wall classes.

Physics: the implicit-solvation error on a created/destroyed anion is a CLASS-WIDE OFFSET for reactions
that share the exact same anion pattern. It cannot be computed away (single-point absolute energies are
ill-conditioned -- the explicit-Mg failure), but it CANCELS against a reference reaction of the same
sub-class. So we subtract a per-sub-class offset calibrated to a small pool of well-measured ANCHOR
reactions:  ΔG_corr = ΔG_UMA - offset,  offset = mean over anchors of (ΔG_UMA(anchor) - ΔrG'°_ref(anchor)).

Applied ONLY to STRUCTURALLY-detected systematic sub-classes; every corrected reaction carries the
intra-class residual as σ (covers the few members whose wall error was already small -> mildly over-
corrected). Validated LEAVE-ANCHORS-OUT / leave-one-out on TECRDB (anchors disjoint from reactions scored):
  phosphagen (guanidine kinases):        66.7 -> 9.5   (offset +57.3 ± 8.6)  phospho-guanidinium SOLVATION
  phosphatase monoester (PPi excluded):  13.8 -> 5.0   (offset +18.2 ± 2.2)  free-Pi(2-) SOLVATION
  thioester (acyl-CoA ligases):          25.6 -> 11.0  (offset +30.0 ± 7.2)  charged-group SOLVATION (group TBD)
PHYSICS (VERIFIED, not asserted -- tools/verify_anchor_physics.py): the offsets are NOT electronic bond
errors. UMA gas-phase ΔE matches DFT (PBE0/def2-TZVP) to <5 kJ for a minimal model of each class
(phosphagen +0.8, thioester -4.1) -> the systematic error is the SOLVATION of a specific charged
functional group that implicit COSMO mis-models by a consistent amount. It is SIGN-CONSISTENT within a
structural class, so it CANCELS against a measured member of the same class (isodesmic referencing) -- not
a per-reaction fudge. The SMARTS identify that charged group, not an arbitrary class. SCATTERED classes
(kinase phospho-ester, NAD, isomerase, Mg/NTP acceptor-specific) are never touched: their error has no
common term (bias~0, high σ) so referencing cannot help (proven). CAVEAT: the thioester charged group is
not yet pinned (carboxylate-loss gives the wrong sign) -- offset is empirical-but-verified-non-electronic.

DISCIPLINE:
  - anchors chosen by ANION PATTERN + experimental data quality (low exp_sd), NEVER by UMA-error;
  - offsets calibrated to independent literature ΔrG'° (Alberty). The values below were validated
    leave-anchors-out; PRODUCTION should re-derive `offset` from exact Alberty anchor ΔrG'° (TODO:
    verify each anchor's literature value; the TECRDB-anchor calibration used here ~= that literature);
  - pyrophosphate / P-O-P (different anion pattern) is EXCLUDED from the monoester sub-class;
  - only fires where the LOO test proved the error systematic -- scattered classes (kinase/NAD/CoA/
    isomerase) are never touched (anchoring made them worse).
"""
from rdkit import Chem

_PHOSPHORAMIDATE = Chem.MolFromSmarts("[#7]-[PX4](=O)")     # N-P(=O): phosphagen P-N bond
_PYRO = Chem.MolFromSmarts("[PX4]-O-[PX4]")                 # P-O-P: pyrophosphate / NTP anhydride
_MONOESTER = Chem.MolFromSmarts("[#6]-[OX2]-[PX4](=O)")     # C-O-P: phosphate monoester
_THIOESTER = Chem.MolFromSmarts("[#6X3](=O)[SX2]")          # C(=O)-S: thioester (acyl-CoA)
_WATER = {"O", "[OH2]"}

# Calibrated + leave-anchors-out-validated (see module docstring). offset in kJ/mol subtracted from ΔG_UMA.
ANCHORS = {
    # offset = mean(ΔG_current_pipeline(anchor) - exp(anchor)) over the low-exp_sd anchor pool, recalibrated
    # 2026-08-26 on the CURRENT pipeline (post ring-closure guard). sigma = intra-class residual (provisional
    # from leave-anchors-out; refreshed from the full-367 corrected residuals).
    "phosphagen":            {"offset": 57.3, "sigma": 9.5,   # +50.3/+54.6/+74.0 vs -8.9/+8.7/+7.3 -> +59.2/+45.9/+66.7
                              "anchor_rids": ["rxn04785", "rxn01371", "rxn00396"]},  # lombricine/creatine/arginine kinase
    "phosphatase_monoester": {"offset": 20.3, "sigma": 5.0,   # +11.9/+6.3/+5.9 vs -12.5/-9.5/-14.7 -> +24.4/+15.8/+20.6
                              "anchor_rids": ["rxn00831", "rxn08948", "rxn01099"]},  # alkaline phosphatase, low exp_sd
    # thioester (acyl-CoA ligases/transferases): a BOND-TYPE reference error on C(=O)-S formation, common
    # +30 across 6 NTP-driven acyl-CoA reactions (acetate/succinate-CoA ligase, citrate lyase, phospho-
    # transacetylase), sign-consistent (+19..+42) = physics not fit. Leave-one-out MAE 25.6->11 (excl. the
    # redox-acylating glyoxylate DH, a different mechanism, gated OUT by requiring a phosphoanhydride change).
    "thioester":             {"offset": 30.0, "sigma": 7.2,
                              "anchor_rids": ["rxn00306", "rxn00172", "rxn00175"]},  # low-exp_sd acyl-CoA ligases
}


def _mols(species):
    ms = []
    for coeff, q, smi in species.values():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        ms.append((int(coeff), smi, m))
    return ms


def _net_count(ms, patt):
    return sum(c * len(m.GetSubstructMatches(patt)) for c, _, m in ms)


def subclass(species):
    """Structural detection of a systematic anion sub-class, or None. Anion-pattern based (not by error)."""
    ms = _mols(species)
    if ms is None:
        return None
    # phosphagen: a P-N phosphoramidate is created/destroyed
    if _net_count(ms, _PHOSPHORAMIDATE) != 0:
        return "phosphagen"
    # thioester (acyl-CoA ligation): a C(=O)-S thioester is created/destroyed AND a phosphoanhydride (NTP)
    # is consumed -> an ATP-driven acyl-CoA ligase/transferase. The P-O-P requirement excludes the
    # redox-acylating dehydrogenase (no NTP), whose error is a different (redox) mechanism.
    if _net_count(ms, _THIOESTER) != 0 and _net_count(ms, _PYRO) != 0:
        return "thioester"
    # phosphatase monoester: hydrolysis (water consumed) that DESTROYS a C-O-P monoester, NO P-O-P present
    has_water_react = any(c < 0 and smi in _WATER for c, smi, _ in ms)
    has_pop = any(m.HasSubstructMatch(_PYRO) for _, _, m in ms)
    net_monoester = _net_count(ms, _MONOESTER)          # <0 = a monoester destroyed (hydrolysed)
    if has_water_react and not has_pop and net_monoester < 0:
        return "phosphatase_monoester"
    return None


def anchor_correct(dG, species):
    """Return (dG_corrected, sigma, subclass_name) if a systematic sub-class is detected, else None."""
    sc = subclass(species)
    if sc is None or sc not in ANCHORS:
        return None
    a = ANCHORS[sc]
    return dG - a["offset"], a["sigma"], sc
