"""Per-sub-class anchor correction for the SYSTEMATIC anion-solvation-wall classes.

Physics: the implicit-solvation error on a created/destroyed anion is a CLASS-WIDE OFFSET for reactions
that share the exact same anion pattern. It cannot be computed away (single-point absolute energies are
ill-conditioned -- the explicit-Mg failure), but it CANCELS against a reference reaction of the same
sub-class. So we subtract a per-sub-class offset calibrated to a small pool of well-measured ANCHOR
reactions:  ΔG_corr = ΔG_UMA - offset,  offset = mean over anchors of (ΔG_UMA(anchor) - ΔrG'°_ref(anchor)).

Applied ONLY to STRUCTURALLY-detected systematic sub-classes; every corrected reaction carries the
intra-class residual as σ. 2026-08-30 AUDIT (each anchor scored against: n too thin? indirect reference?
stale offset? mechanism-homogeneous anchor pool?) found and fixed three concrete defects -- not just
"more anchors would be nice" but specific, verified bugs:
  1. phosphatase_monoester offset (+20.3, from 3 anchors) was STALE: the 9 clean TECRDB members of this
     class (all 11 minus 2 outliers, see #2) have raw mean +15.7 -- the 3-anchor number over-corrected by
     ~4.6 kJ on average. Recalibrated to +15.7 (n=9, uses data already in TECRDB, no new anchors needed).
  2. phosphatase_monoester silently mixed two populations: 9 plain sugar-/nucleotide-phosphate hydrolyses
     (raw mean +15.7, σ 2.4, tight) vs 2 cases with a permanent/protonated CATION adjacent to the
     hydrolysed phosphate (phosphocholine, phosphoserine; raw +4.7/-0.7 -- their RAW pipeline output is
     already near-correct, and the blanket offset was making them WORSE, -15.6/-21.0). Now gated OUT
     structurally (a cation in the same species as the monoester) and reported as dG_raw, uncorrected.
  3. thioester's 3 original anchors mixed two mechanisms 2:1 (not a sample-size problem, a composition
     problem): rxn00175 releases free PPi (acyl-adenylate intermediate, same mechanism as adenylylate)
     while rxn00172/rxn00306 release ADP/GDP+Pi directly (no adenylylate intermediate). Split by
     `_produces_ppi` (already used for the adenylylate gate) using all 6 TECRDB members of this class:
     thioester_ppi offset +42.8 (n=2, σ 3.2), thioester_pi offset +28.1 (n=4, σ 4.2). LOO MAE 7.9->5.3
     weighted vs the old single-offset scheme.
  4. adenylylate's 5-point calibration also split cleanly (not previously acted on): aliphatic acids
     (acetate/propanoate, raw 41.5/39.7) vs alpha-amino acids (seryl/Tyr/Ile, raw 49.8/48.1/53.8) --
     the alpha-ammonium adjacent to the forming mixed anhydride shifts the offset by ~10 kJ, same
     "adjacent fixed charge" theme as #2. Split: adenylylate_aliphatic +15.6 (n=2), adenylylate_aminoacid
     +25.6 (n=3). LOO MAE (intra-class only) 6.0->2.7 weighted. The ±8-10 kJ reference-uncertainty
     inflation (indirect 6-parent-ligase cycle, no direct measured Keq for either sub-class) still
     applies to BOTH -- carried in sigma (search for a direct reference came up empty: aminoacyl-adenylates
     are too reactive to have a published solution-phase Keq; the literature only reports enzyme-bound
     Kd / ATP-PPi exchange kinetics, a different quantity). NOT reducible by more of the same kind of anchor.
Prior (pre-audit) validation, still the basis for what's kept: LEAVE-ANCHORS-OUT / leave-one-out on
TECRDB (anchors disjoint from reactions scored):
  phosphagen (guanidine kinases):        66.7 -> 9.5   (offset +57.3 ± 8.6)  phospho-guanidinium SOLVATION
PHYSICS (VERIFIED, not asserted -- tools/verify_anchor_physics.py, re-run live 2026-08-30, same numbers
reproduced): UMA gas-phase ΔE matches DFT (PBE0/def2-TZVP) to <5 kJ for phosphagen (+0.8) and thioester
(-4.1) -> solvation, not an electronic bond error. phosphatase_monoester's gap is +7.9 kJ (39% of its
20.3 kJ offset) -- passes the <10 kJ "electronic FINE" bar but is meaningfully weaker than the other
three; flagged, not treated as equally solid. adenylylate: +1.1 kJ (analysis/verify_adenylylate_physics.py).
It is SIGN-CONSISTENT within a structural class, so it CANCELS against a measured member of the same
class (isodesmic referencing) -- not a per-reaction fudge. SCATTERED classes (kinase phospho-ester, NAD,
isomerase, Mg/NTP acceptor-specific) are never touched: their error has no common term (bias~0, high σ)
so referencing cannot help (proven).

DISCIPLINE:
  - anchors chosen by ANION PATTERN + experimental data quality (low exp_sd), NEVER by UMA-error;
  - offsets calibrated to independent literature ΔrG'° (Alberty) or, for phosphagen/phosphatase/thioester,
    to TECRDB anchors validated leave-anchors-out;
  - pyrophosphate / P-O-P (different anion pattern) is EXCLUDED from the monoester sub-class;
  - a species carrying an adjacent fixed/permanent charge (cation next to a monoester phosphate; alpha-
    ammonium next to a forming mixed anhydride) gets its OWN sub-class or is excluded, never averaged
    into the majority offset -- this was the actual bug behind items 2-4 above, not insufficient data;
  - only fires where the LOO test proved the error systematic -- scattered classes (kinase/NAD/CoA/
    isomerase) are never touched (anchoring made them worse).
"""
from metag.routing.reaction_families import (  # noqa: F401  (re-exported: legacy public API)
    subclass, subclass_dir, subclass_extent,
)

# Calibrated + leave-anchors-out-validated (see module docstring). offset in kJ/mol subtracted from ΔG_UMA.
# 2026-09-26 REFRESH: TECRDB-referenced offsets/sigmas re-derived (metag.tools.calibrate.refit_anchor_offsets)
# on the conformer-dedup pipeline (Boltzmann over unique minima; TECRDB sweep analysis/tecrdb_sweep_v2). The
# old sample-count degeneracy was larger for big floppy species and had been absorbed by the offsets
# (phosphatase +15.7 -> +23.3, amide -10.9 -> -6.6). The adenylylate offsets use the external indirect-cycle
# reference with non-TECRDB pseudo-reactions and were NOT re-derived (needs QM on adyl_* members) -- flagged.
# "ref": "tecrdb" (default) = offset is mean(dG_raw - TECRDB exp) over anchor_rids -> the nested CV in
# metag.tools.calibrate REFITS it per training fold; "external" = reference is not TECRDB exp -> fixed.
ANCHORS = {
    # offset = mean(ΔG_current_pipeline(anchor) - exp(anchor)) over the low-exp_sd anchor pool, recalibrated
    # 2026-08-26 on the CURRENT pipeline (post ring-closure guard). sigma = intra-class residual (provisional
    # from leave-anchors-out; refreshed from the full-367 corrected residuals).
    "phosphagen":            {"offset": 56.1, "sigma": 11.1,   # +50.3/+54.6/+74.0 vs -8.9/+8.7/+7.3 -> +59.2/+45.9/+66.7
                              "anchor_rids": ["rxn04785", "rxn01371", "rxn00396"]},  # lombricine/creatine/arginine kinase
    # RECALIBRATED 2026-08-30 (item 1+2 above): n widened 3->9 (all clean TECRDB members, not just the
    # original 3 anchors), the 2 cationic-adjacent outliers (phosphocholine, phosphoserine) excluded --
    # see phosphatase_monoester_cationic below. Old offset +20.3/σ5.0 (n=3) was stale; raw mean over the
    # 9 clean members is +15.7±2.4.
    "phosphatase_monoester": {"offset": 23.3, "sigma": 2.9,
                              "anchor_rids": ["rxn00132", "rxn00549", "rxn08430", "rxn01491", "rxn00831",
                                              "rxn00220", "rxn08948", "rxn01099", "rxn12188"]},
    # thioester SPLIT 2026-08-30 (item 3 above): the original 3-anchor pool mixed two mechanisms 2:1.
    # thioester_ppi: PPi-releasing (acyl-adenylate intermediate, same mechanism family as adenylylate) --
    # acetate-CoA ligase AMP-forming. All 2 TECRDB members used (n too small to hold any out).
    "thioester_ppi":         {"offset": 44.6, "sigma": 4.6,
                              "anchor_rids": ["rxn00175", "rxn00674"]},
    # thioester_pi: ADP/GDP+Pi-releasing (direct acyl-phosphate, no adenylate intermediate) -- succinate/
    # malate-CoA ligase, ATP citrate lyase, phosphate acetyltransferase. All 4 TECRDB members used.
    "thioester_pi":          {"offset": 30.9, "sigma": 5.9,
                              "anchor_rids": ["rxn00172", "rxn00306", "rxn00285", "rxn00257"]},
    # adenylyl-transfer (acyl/aminoacyl-adenylate synthetases): ATP + carboxylate -> acyl-AMP + PPi.
    # A bond-type reference error on the acyl-adenylate MIXED ANHYDRIDE (C(=O)-O-P), NOT solvation
    # (pH-0 already neutralises the phosphates) nor sampling (U_samp ~2). PHYSICS VERIFIED
    # (analysis/verify_adenylylate_physics.py): UMA≈DFT (PBE0/def2-TZVP) to +1.1 kJ on the neutral
    # mixed-anhydride bond-swap -> the ~+20 error is SOLVATION, same KIND as the other verified anchors.
    # SPLIT 2026-08-30 (item 4 above): the 5-point calibration separates cleanly by whether the activated
    # acid is a plain aliphatic carboxylate or carries its own alpha-ammonium (amino acid). Reference for
    # BOTH sub-classes is the SAME external +25 kJ target (six measured parent ligases, indirect cycle --
    # no direct adenylylation Keq exists in the literature; confirmed by search 2026-08-30, not just
    # unsearched). sigma = sqrt(intra-class std^2 + 8.5^2) to carry that shared reference uncertainty.
    "adenylylate_aliphatic": {"offset": 15.6, "sigma": 8.6,   # raw 41.5/39.7 vs ref 25 -> intra std 1.3
                              "ref": "external",       # indirect +25 kJ cycle, NOT TECRDB exp -> fixed in CV
                              "anchor_rids": ["rxn00226", "adyl_propanoate"]},  # acetate, propanoate
    "adenylylate_aminoacid": {"offset": 25.6, "sigma": 9.0,   # raw 49.8/48.1/53.8 vs ref 25 -> intra std 2.9
                              "ref": "external",
                              "anchor_rids": ["rxn00418", "adyl_Tyr", "adyl_Ile"]},  # seryl, Tyr, Ile
    # carboxy-phosphate (biotin/ATP-dependent carboxylases) ADDED 2026-08-30: CO2/HCO3- + ATP + acceptor
    # -> carboxylated acceptor + ADP + Pi. The carboxyphosphate is an INTERMEDIATE (not in the net eqn);
    # the systematic error is the same MIXED-ANHYDRIDE SOLVATION family as adenylylate -- PHYSICS VERIFIED
    # (analysis/verify_class_physics.py --class mg_carboxylase: UMA-DFT -1.2 kJ, electronic FINE -> the
    # +25 is solvation, not an MLIP bond mis-rank). Both TECRDB members used (n=2, NO leave-one-out
    # possible -- treat like thioester_ppi). 100% sign-consistent (raw +27.8/+22.5). offset = mean raw err;
    # sigma = sqrt(intra_std^2 + mean_exp_sd^2) = 7.8 -- DIRECT TECRDB references (not the adenylylate
    # indirect cycle), but n=2 + one member's large exp_sd (11.9) dominate the uncertainty. offset
    # recalibrated on the CURRENT (post-hydratase-H+-fix, MAE 10.9) pipeline dG_raw: rxn00250 +28.9,
    # rxn51768 +23.5 -> mean 26.2 (the stale 11.60-baseline gave +25.1; ~1 kJ conformer/era drift). NOTE:
    # the sibling acyl-phosphate class (GAPDH/kinase C(=O)-O-P) was DELIBERATELY NOT anchored -- its n=8
    # pool is scatter (bias -0.5, std 18.2, 75% sign: redox-DH/kinase/carbamoyltransfer lumped by one
    # SMARTS), so it is flagged, not corrected (candidate_anchor_scan.py).
    "carboxyP":              {"offset": 25.2, "sigma": 6.8,
                             "anchor_rids": ["rxn00250", "rxn51768"]},  # pyruvate / propanoyl-CoA carboxylase
    # amide hydrolysis (ACYCLIC acyl/carbamoyl amide + H2O -> carboxylic acid + amine) ADDED 2026-09-20.
    # A SOLVATION offset on the CREATED carboxylic-acid + amine groups -- PHYSICS VERIFIED
    # (analysis/verify_class_physics.py --class amide: UMA-DFT +2.9 kJ, electronic FINE -> the ~-11 error is
    # SOLVATION, same KIND as the phosphatase/thioester anchors, NOT an MLIP bond mis-rank). 100%
    # sign-consistent (all 7 TECRDB acyclic-amide members negative). offset = mean dG_raw err over the 5
    # MECHANISM-HOMOGENEOUS non-huge members (LOO residual MAE 3.6, tight like phosphatase). NEGATIVE offset:
    # the pipeline is too NEGATIVE (amide hydrolysis scored too favorable) -> dG_corr = dG - (-10.9) = dG+10.9.
    # DISCIPLINE (docstring items 2-4): the class is SPLIT to stay homogeneous --
    #   * cyclic-amide hydrolysis (hydantoinase/dihydropyrimidinase, carbonyl IN a ring) is a SEPARATE
    #     near-zero-error population (raw -0.1/-2.9); a [CX3;!R] gate excludes it so it is NOT over-corrected;
    #   * aromatic-ring amidine DEAMINASES (cytidine/adenosine, raw -3.2/+1.8) don't match [CX3;!R] at all;
    #   * the 2 huge/floppy anandamide-amidohydrolase members (raw -22/-30) DO get the offset (improves them
    #     to -11/-19, never over-corrected) but carry their large conformer residual via U_samp.
    "amide_hydrolysis":      {"offset": -6.6, "sigma": 4.3,
                             "anchor_rids": ["rxn01792", "rxn39443", "rxn45677", "rxn00189", "rxn36656"]},
}


def anchor_correct(dG, species):
    """Return (dG_corrected, sigma, subclass_name, direction) if a systematic sub-class is detected AND has
    a validated offset, else None (caller reports dG_raw). The offset is defined for the canonical
    direction; a reversed reaction gets the opposite-signed correction, so the anchored ΔG stays
    antisymmetric: anchor(-ΔG, reversed) == -anchor(ΔG, forward). A subclass name detected but absent from
    ANCHORS -- e.g. phosphatase_monoester_cationic -- deliberately falls through to None: it is a
    recognized, structurally distinct case whose correction is not yet trustworthy enough to apply."""
    sc, direction, extent = subclass_extent(species)
    if sc is None or sc not in ANCHORS:
        return None
    a = ANCHORS[sc]
    return dG - direction * extent * a["offset"], a["sigma"], sc, direction
