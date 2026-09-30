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
from rdkit import Chem

from metag.chem import is_water

_PHOSPHORAMIDATE = Chem.MolFromSmarts("[#7]-[PX4](=O)")     # N-P(=O): phosphagen P-N bond
_PYRO = Chem.MolFromSmarts("[PX4]-O-[PX4]")                 # P-O-P: pyrophosphate / NTP anhydride
# C-O-P phosphate MONOESTER: the P carries exactly one ester O plus two terminal OH/O-. The looser
# "[#6]-[OX2]-[PX4](=O)" also matched both C-O-P links of a DIESTER (cAMP, glycerophosphoinositol), so
# phosphodiesterases (diester -> monoester, net monoester +1 per link) were mis-detected as phosphatases.
_MONOESTER = Chem.MolFromSmarts("[#6]-[OX2]-[PX4](=[OX1])(-[OX1-,OX2H1])-[OX1-,OX2H1]")
_THIOESTER = Chem.MolFromSmarts("[#6X3](=O)[SX2]")          # C(=O)-S: thioester (acyl-CoA)
_MIXEDANHYDRIDE = Chem.MolFromSmarts("[#6X3](=O)[OX2][PX4]")  # C(=O)-O-P: acyl/aminoacyl-adenylate anhydride
_ACYCLIC_AMIDE = Chem.MolFromSmarts("[CX3;!R](=[OX1])[NX3]")  # acyl/carbamoyl amide, carbonyl NOT in a ring
_CATION = Chem.MolFromSmarts("[N+]")                         # permanent/protonated cationic nitrogen
_ALPHA_AMINO_ACID = Chem.MolFromSmarts("[NX3,NX4+][CX4][CX3](=O)[OX1-,OX2H1]")  # alpha-amino acid backbone
_CARBOXYL = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")     # carboxylic acid / carboxylate


def _is_ppi(m):
    """True if the molecule is pyrophosphate (any protonation state): all heavy atoms P or O, exactly 2 P.
    Distinguishes free PPi (adenylyl-transfer product) from a nucleotide's internal P-O-P (ATP/ADP)."""
    zs = [a.GetAtomicNum() for a in m.GetAtoms()]
    return zs.count(15) == 2 and all(z in (8, 15) for z in zs)

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


def _mols(species):
    ms = []
    for coeff, q, smi in species.values():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        ms.append((float(coeff), smi, m))            # fractional coefficients (ModelSEED 0.5 O2) kept
    return ms


def _net_count(ms, patt):
    return sum(c * len(m.GetSubstructMatches(patt)) for c, _, m in ms)


def _produces_ppi(ms):
    """Net free pyrophosphate produced (coeff>0) minus consumed -- pins adenylyl-transfer direction."""
    return sum(c for c, _, m in ms if _is_ppi(m))


def _is_co2_like(m):
    """CO2 / carbonic acid / bicarbonate cosubstrate: <=4 heavy atoms, exactly 1 C, only C/O, >=2 O.
    Matches O=C=O, O=C(O)O, O=C([O-])O regardless of protonation."""
    zs = [a.GetAtomicNum() for a in m.GetAtoms()]
    return m.GetNumHeavyAtoms() <= 4 and zs.count(6) == 1 and zs.count(8) >= 2 and all(z in (6, 8) for z in zs)


def _net_skeleton_carboxyl(ms):
    """Net carboxyl groups created on CARBON-SKELETON molecules (>4 heavy atoms). Excludes the CO2/
    bicarbonate cosubstrate, which itself matches the carboxyl SMARTS -- counting it would cancel the
    carboxyl created on the acceptor to a spurious net 0 (the bug that silently zeroed a naive gate)."""
    return sum(c * len(m.GetSubstructMatches(_CARBOXYL)) for c, _, m in ms if m.GetNumHeavyAtoms() > 4)


def _detect(ms):
    """Sub-class of a reaction written in the class's CANONICAL direction, or None.

    Every class is defined in ONE direction -- the direction all its TECRDB anchor members are written
    in (P-N created, thioester created, P-O-P consumed, monoester / amide hydrolysed, PPi released) -- so
    the offset has one sign. The reverse reaction is handled by subclass_dir(), which flips the sign.
    (Detecting on "net change != 0" fired in BOTH directions with the same-signed offset: reversed
    creatine kinase got -56 kJ instead of +56, a 112 kJ error, and the one-directional classes left
    reversed reactions uncorrected, so ΔG(reverse) != -ΔG(forward).)"""
    ppi = _produces_ppi(ms)
    # adenylyl-transfer (acyl/aminoacyl-adenylate synthetases): free PPi is PRODUCED and a mixed
    # anhydride C(=O)-O-P is FORMED (ATP + carboxylate -> acyl-AMP + PPi). CHECKED BEFORE phosphagen:
    # an N-adenylate's P-N would otherwise be mis-caught by the phosphagen SMARTS and given the
    # creatine-kinase offset (wrong). SPLIT by whether the activated acid carries its own alpha-amino
    # group (amino acid) vs a plain aliphatic/aromatic carboxylate -- see ANCHORS docstring, item 4.
    if ppi > 0 and _net_count(ms, _MIXEDANHYDRIDE) > 0:
        is_aminoacid = any(c < 0 and m.HasSubstructMatch(_ALPHA_AMINO_ACID) for c, _, m in ms)
        return "adenylylate_aminoacid" if is_aminoacid else "adenylylate_aliphatic"
    # phosphagen: a P-N phosphoramidate is CREATED (ATP + guanidine -> ADP + phosphagen) -- but NOT an
    # adenylyl-transfer (which produces PPi, not ADP). The PPi guard stops N-adenylates being
    # mis-corrected as creatine kinase.
    if _net_count(ms, _PHOSPHORAMIDATE) > 0 and ppi <= 0:
        return "phosphagen"
    # thioester (acyl-CoA ligation): a C(=O)-S thioester is CREATED AND a phosphoanhydride (NTP) is
    # consumed -> an ATP-driven acyl-CoA ligase. The P-O-P requirement excludes the redox-acylating
    # dehydrogenase (no NTP), whose error is a different (redox) mechanism. SPLIT by whether free PPi is
    # released (acyl-adenylate intermediate) or ADP/GDP+Pi (direct acyl-phosphate) -- see ANCHORS
    # docstring, item 3.
    if _net_count(ms, _THIOESTER) > 0 and _net_count(ms, _PYRO) < 0:
        return "thioester_ppi" if ppi > 0 else "thioester_pi"
    # carboxy-phosphate (biotin/ATP carboxylase): CO2/bicarbonate consumed + ATP P-O-P consumed + a
    # carboxyl CREATED on a carbon skeleton. The mixed anhydride is an intermediate (not in the net eqn)
    # so gate on the NET transformation. Carboxyls counted only on >4-heavy mols so the consumed CO2/
    # bicarbonate cosubstrate (which itself matches the carboxyl SMARTS) does NOT cancel the created
    # carboxyl to a spurious net 0. Mutually exclusive with the classes above: adenylylate releases PPi
    # (ppi>0) not Pi; thioester needs a net thioester change (0 for propanoyl-CoA carboxylase, thioester
    # on both sides); phosphagen needs a P-N. Verified to hit EXACTLY {rxn00250, rxn51768} in TECRDB-367.
    if (any(c < 0 and _is_co2_like(m) for c, _, m in ms) and _net_count(ms, _PYRO) < 0
            and _net_skeleton_carboxyl(ms) > 0):
        return "carboxyP"
    # amide hydrolysis (acyclic acyl/carbamoyl amide + H2O -> carboxylic acid + amine): a SOLVATION offset
    # on the created COOH + amine (PHYSICS VERIFIED UMA-DFT +2.9). Only ACYCLIC amides ([CX3;!R]): cyclic
    # hydantoinases are a separate near-zero population and aromatic-ring deaminases don't match. Checked
    # AFTER the phosphate/thioester/carboxyP classes (all require a phosphate this class lacks) so those win
    # any overlap; the _PYRO guard is belt-and-suspenders. Water-consuming + net acyclic amide destroyed.
    # NOTE: the pool includes carbamate hydrolysis (rxn45677, carbamate -> bicarbonate + NH4+), i.e.
    # "carbamoyl amide" is in scope by design, not by accident.
    water_consumed = any(c < 0 and is_water(smi) for c, smi, _ in ms)
    has_pop = any(m.HasSubstructMatch(_PYRO) for _, _, m in ms)
    if water_consumed and _net_count(ms, _ACYCLIC_AMIDE) < 0 and not has_pop:
        return "amide_hydrolysis"
    # phosphatase monoester: hydrolysis (water consumed) that DESTROYS a C-O-P monoester, NO P-O-P present.
    # A monoester bearing its own adjacent permanent/protonated cation (phosphocholine, phosphoserine)
    # is a DIFFERENT, unresolved sub-case -- detected but excluded from ANCHORS (see docstring item 2):
    # its raw pipeline output is already near-correct, and the majority offset makes it worse.
    if water_consumed and not has_pop and _net_count(ms, _MONOESTER) < 0:
        cationic = any(c < 0 and m.HasSubstructMatch(_MONOESTER) and m.HasSubstructMatch(_CATION)
                       for c, _, m in ms)
        return "phosphatase_monoester_cationic" if cationic else "phosphatase_monoester"
    return None


def subclass_dir(species):
    """(sub-class, direction) for a reaction, or (None, 0). direction = +1 if the reaction is written in
    the class's canonical direction (see _detect), -1 if it is the reverse. The forward reading wins if
    both readings match (never observed on TECRDB)."""
    ms = _mols(species)
    if ms is None:
        return None, 0
    sc = _detect(ms)
    if sc is not None:
        return sc, +1
    sc = _detect([(-c, smi, m) for c, smi, m in ms])
    if sc is not None:
        return sc, -1
    return None, 0


def subclass(species):
    """Structural detection of a systematic anion sub-class (either direction), or None. Anion-pattern
    based (not by error). Returns a name that may or may not be a key in ANCHORS -- a detected-but-excluded
    case (e.g. a cationic-adjacent phosphatase monoester) returns its own name so callers/diagnostics can
    see it was recognized, but anchor_correct() will not apply a correction unless the name is in ANCHORS."""
    return subclass_dir(species)[0]


def anchor_correct(dG, species):
    """Return (dG_corrected, sigma, subclass_name, direction) if a systematic sub-class is detected AND has
    a validated offset, else None (caller reports dG_raw). The offset is defined for the canonical
    direction; a reversed reaction gets the opposite-signed correction, so the anchored ΔG stays
    antisymmetric: anchor(-ΔG, reversed) == -anchor(ΔG, forward). A subclass name detected but absent from
    ANCHORS -- e.g. phosphatase_monoester_cationic -- deliberately falls through to None: it is a
    recognized, structurally distinct case whose correction is not yet trustworthy enough to apply."""
    sc, direction = subclass_dir(species)
    if sc is None or sc not in ANCHORS:
        return None
    a = ANCHORS[sc]
    return dG - direction * a["offset"], a["sigma"], sc, direction
