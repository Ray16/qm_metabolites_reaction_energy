"""Auto-generate the pH-0 / pKa-transform version of a reaction.

WHY (physics, not fitting): the Mg/NTP/PPi/phosphoryl class is UMA's softest regime --
its thermochemistry is dominated by charge delocalisation on a FORMAL ANIONIC charge, which
hits three UMA weaknesses at once: (1) locality can't represent non-local anionic charge
delocalisation, (2) charge is a global embedding not physics, (3) def2-TZVPD is only lightly
diffuse so even the wB97M-V reference has an anion ceiling. Implicit continuum then mis-solvates
each polyphosphate charge state by +-20-50 kJ, and because phosphoryl transfer CHANGES the
charge concentration the errors do NOT cancel (rxn00695 -96, rxn10427 +61 -- opposite signs).

THE FIX (Jinich/Alberty pH-0 route): protonate every anionic site to its NEUTRAL microspecies
BEFORE the QM step (UMA's comfortable regime -- no formal charge, no diffuse-anion basis need,
continuum-solvation valid), then bridge to pH 7 ANALYTICALLY with the site's EXPERIMENTAL pKa:

    dG'(pH7) = dG_QM(all-neutral) + SUM_sites  sign * RT ln10 * (pH - pKa)      (sign: +react, -prod)

Nothing is fitted to the thermodynamic database: pKa's are textbook functional-group values.

NO ATOM-MAPPER NEEDED: we neutralise EVERY anionic site and emit a per-side pKa term for each.
A spectator anion (charge/site-matched partner on the opposite side) emits ['react',pKa] and
['prod',pKa] of the SAME class -> the two contributions cancel exactly. Only the NET charge-state
change (the created/destroyed anion) survives -- reproducing the hand annotations automatically.

SCOPE: only the CHARGED (ionisable-anion) subclass. Thioester (neutral C(=O)-S resonance) and
glycosyl anomeric (neutral stereoelectronic) have no ionisable proton -> untouched (returns the
species unchanged for those; those stay on the DFT-electronics frontier).
"""
import os
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as _rdMD


def is_isomerization(species):
    """GATE for pH-0: True if the reaction is an ISOMERIZATION (every reactant molecular-formula
    has a matching product formula -> a rearrangement with conserved anionic groups). pH-0 must be
    SKIPPED for these -- there's no anion-solvation change to fix, so neutralising the (spectator)
    anions only injects neutral-vs-anion sampling noise (validated: pH-0 hurts isomerases). General
    structural rule (no flags / no reaction-id) -> works on ModelSEED too. Wire into the pipeline as:
        if os.environ.get("PH0_AUTO") and not is_isomerization(rx["species"]) and not rx.get("pka_sites"):
    """
    def formula(s):
        m = Chem.MolFromSmiles(s)
        return _rdMD.CalcMolFormula(m) if m else None
    R = [formula(s) for c, q, s in species.values() if c < 0 for _ in range(abs(int(c)))]
    P = [formula(s) for c, q, s in species.values() if c > 0 for _ in range(abs(int(c)))]
    if None in R or None in P:
        return False
    return sorted(R) == sorted(P)

# ---- textbook functional-group pKa's (experimental; NOT fitted to any dG) ---------------------
# Each entry: the pKa of REMOVING one proton from the neutral acid at that site.
# For a phosphate P centre we distinguish the near-neutral TERMINAL deprotonation (~6.5, the one
# that actually straddles pH 7 and drives pH-dependence) from the strongly-acidic earlier ones
# (~1.8, essentially always ionised at pH 7). Carboxylate ~4.75. Sulfonate/sulfate ~ -1 (strong).
# Full experimental pKa LADDERS per group type (NOT fitted to any dG). After max-anion
# canonicalisation each site is deprotonated, so we assign the group's COMPLETE ladder and use
# the EXACT Alberty form RT*ln(1+10^(pH-pKa)) per proton -- correct near AND above pH~pKa (the
# linear (pH-pKa) form wrongly makes a high-pKa site like Pi's 12.35 contribute -5 kJ instead of ~0).
# Phosphate ladders keyed by #bridging-O on the P (ester/anhydride links to C or another P):
#   0 bridge = free phosphate (Pi):            H3PO4  pKa 2.15, 7.20, 12.35
#   1 bridge = terminal monoester/anhydride:   ROPO3H2 pKa ~1.5, ~6.5   (the 6.5 straddles pH7)
#   2 bridge = internal diester/anhydride:     one acidic proton ~1.5
P_LADDER = {0: [2.15, 7.20, 12.35], 1: [1.50, 6.50], 2: [1.50], 3: [1.50]}
# FREE pyrophosphate H4P2O7 (all heavy atoms P/O, 2 P): its own macroscopic ladder (I->0), NOT two
# terminal-phosphate ladders ({1.5,1.5,6.5,6.5} over-counts the transform by ~4.4 kJ per free PPi).
PPI_LADDER = [0.91, 2.10, 6.70, 9.32]
CARBOXYL_PKA = 4.75
# Carboxyl pKa by alpha-environment (textbook macroscopic values, I->0; NOT fitted to any dG). An
# electron-withdrawing alpha substituent acidifies the carboxyl by 1-2.5 units, which is up to ~14 kJ
# per created/destroyed site in the Alberty transform -- the flat 4.75 was systematically wrong for
# alpha-keto (pyruvate 2.49, 2-oxoglutarate 2.47, oxaloacetate 2.22) and alpha-amino acids.
#   alpha-ammonium (N stays PROTONATED in the QM microspecies): glycine/alanine pKa1 2.34 -> 2.3
#   alpha-amine NEUTRALIZED too (base path): the MICROSCOPIC pKa of COOH next to a neutral NH2,
#       pk(COOH|NH2) = pKa2 - log K_taut = 9.78 - 5.35 = 4.4 (glycine microconstants); with the amine's
#       9.6 base term the independent-site product then reproduces the zwitterion-dominated binding
#       polynomial (the coupled microstate sum) to < 0.1 pKa unit.
#   alpha-oxo (C=O on the alpha C: alpha-keto acids / glyoxylate): 2.5
#   alpha-oxygen (hydroxy / ester / phosphate on an sp3 alpha C: lactate 3.86, glycolate 3.83,
#       glycerate 3.52, malate pKa1 3.40): 3.8
#   formate (carboxyl C bears no carbon): 3.75;  aromatic alpha C (benzoate 4.20): 4.2
CARBOXYL_PKA_ALPHA = {"ammonium": 2.3, "amine_neutralized": 4.4, "oxo": 2.5, "oxygen": 3.8,
                      "formate": 3.75, "aromatic": 4.2}
_BASIC_N = Chem.MolFromSmarts("[NX4+;H1,H2,H3,H0;!$(N~[#6]=[#7,#8])]")   # ammonium (not amidinium)
_AMINE_N = Chem.MolFromSmarts("[NX3;H1,H2;!$(N[#6]=[#7,#8,#16]);!$(N-a)]")  # basic sp3 amine (not amide/aniline)


def _carboxyl_env(mol, o_idx):
    """Classify the alpha environment of the carboxylate whose O- is o_idx (see CARBOXYL_PKA_ALPHA)."""
    o = mol.GetAtomWithIdx(o_idx)
    c = next((n for n in o.GetNeighbors() if n.GetSymbol() == "C"), None)
    if c is None:
        return None
    alpha = [n for n in c.GetNeighbors() if n.GetSymbol() == "C"]
    if not alpha:
        return "formate"
    a = alpha[0]
    if a.GetIsAromatic():
        return "aromatic"
    for nb in a.GetNeighbors():
        if nb.GetIdx() == c.GetIdx():
            continue
        bond = mol.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx())
        if nb.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 2:
            return "oxo"                                   # alpha-keto acid
    if a.GetHybridization() == Chem.HybridizationType.SP3:
        for nb in a.GetNeighbors():
            if nb.GetSymbol() == "N" and nb.GetFormalCharge() == 1 and nb.GetTotalNumHs() >= 1:
                return "ammonium"
        for nb in a.GetNeighbors():
            if nb.GetSymbol() == "N" and nb.GetFormalCharge() == 0 and nb.GetIdx() in _amine_ns(mol):
                return "amine_neutral"                     # neutral NH2 in the QM species: microscopic pKa
        if any(nb.GetSymbol() == "O" and nb.GetIdx() != c.GetIdx() for nb in a.GetNeighbors()):
            return "oxygen"
    return None


def _amine_ns(mol):
    return {m[0] for m in mol.GetSubstructMatches(_AMINE_N)}


def _pka_env_enabled():
    """PKA_ENV (default OFF). The environment-specific carboxyl pKa's and the free-PPi ladder are each
    textbook-correct, but applied as a PARTIAL table they break the error cancellation between reaction
    partners: e.g. fumarase -- malate's alpha-hydroxy carboxyl is corrected (4.75 -> 3.8) while fumarate's
    alpha,beta-unsaturated carboxyls (exp 3.03/4.44) have no rule and stay at 4.75, so every hydratase
    moved +5.4 kJ (TECRDB A/B 2026-09-26: MAE 11.25 -> 11.91, 47 worse / 16 better). pKa corrections must
    be applied as a COMPLETE, uniformly-validated set (per-compound macroscopic pKa's validated against an
    independent pKa reference), not piecemeal. Kept for that work; not deployed."""
    v = os.environ.get("PKA_ENV")
    return v is not None and v.strip().lower() not in ("", "0", "off", "false", "no")


def carboxyl_pka(mol, o_idx, amine_neutralized=False):
    """Environment-specific carboxyl pKa for the carboxylate O- at o_idx (textbook; see table).
    Returns the flat CARBOXYL_PKA unless PKA_ENV is enabled (see _pka_env_enabled)."""
    if not _pka_env_enabled():
        return CARBOXYL_PKA
    env = _carboxyl_env(mol, o_idx)
    # the relevant constant depends on the alpha-N's state IN THE QM MICROSPECIES: still protonated ->
    # macroscopic pKa1 (2.3); neutral (drawn neutral, or neutralized by the base path) -> microscopic 4.4
    if env == "ammonium":
        return CARBOXYL_PKA_ALPHA["amine_neutralized" if amine_neutralized else "ammonium"]
    if env == "amine_neutral":
        return CARBOXYL_PKA_ALPHA["amine_neutralized"]
    return CARBOXYL_PKA_ALPHA.get(env, CARBOXYL_PKA)
SULFONATE_PKA = -1.5
# Sulfate proton ladders (experimental strong-acid pKa's; NOT fitted to any dG). Like phosphate we
# assign the k most-acidic entries where k = #deprotonated O on that S. FREE sulfate (H2SO4, both O
# terminal) gets the full 2-proton ladder; a sulfate MONOESTER (R-O-SO3(-), one bridging O) has one
# ionisable proton. Both pKa's sit far below pH 7, so the exact-Alberty form recovers the SO4(2-)
# free energy at pH 7 exactly from the neutral H2(SO4) reference used for the QM step.
SULFATE_LADDER = [-3.0, 1.99]        # H2SO4 pKa1, pKa2

# SMARTS for a deprotonated (anionic) oxygen of each class, matched on the [O-] atom (first atom).
_ANION_SMARTS = [
    ("carboxyl",  "[$([OX1-][CX3]=O)]"),                 # carboxylate O-
    ("sulfonate", "[$([OX1-][SX4](=O)(=O)[#6])]"),        # R-SO3-  (S bears a carbon)
    ("sulfate",   "[$([OX1-][SX4](=O)(=O)[#8])]"),        # sulfate ester R-O-SO3- AND free SO4(2-):
    #                                                       S bears a 4th OXYGEN (ester OX2 or terminal
    #                                                       O-). Free sulfate O=S(=O)([O-])[O-] was
    #                                                       previously UNMATCHED (needed a bridging
    #                                                       OX2) -> never neutralised -> H-imbalance ->
    #                                                       pH-0 REFUSED -> charged-COSMO anion
    #                                                       catastrophe (rxn00379 sulfate
    #                                                       adenylyltransferase -70).
    ("phosphate", "[$([OX1-][P])]"),                      # any P-O-  (sub-classified below)
]


def _anion_sites(mol):
    """[(o_idx, group)] for every anionic O, group in {carboxyl, sulfonate, sulfate, phosphate}."""
    claimed, sites = set(), []
    for cls, sm in _ANION_SMARTS:
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts(sm)):
            o = match[0]
            if o not in claimed:
                claimed.add(o); sites.append((o, cls))
    return sites


def _is_free_ppi(mol):
    zs = [a.GetAtomicNum() for a in mol.GetAtoms()]
    return zs.count(15) == 2 and all(z in (8, 15) for z in zs)


_SITE_MODEL = None


def _site_model():
    """Fitted generic site-pKa model (metag/tools/fit_site_pka.py -> data/site_pka_model.json), or None."""
    global _SITE_MODEL
    if _SITE_MODEL is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                            "site_pka_model.json")
        try:
            import json as _json
            _SITE_MODEL = _json.load(open(path))["params"]
        except Exception:
            _SITE_MODEL = {}
    return _SITE_MODEL or None


def _pka_model_site():
    v = os.environ.get("PKA_MODEL", "table").strip().lower()
    return v == "site"


def _classify_species_site(mol, amine_neutralized):
    """PKA_MODEL=site: per-site microscopic pKa's from the generic inductive + electrostatic model."""
    from metag.tools.fit_site_pka import site_features, effective_pkas
    params = dict(_site_model())
    if amine_neutralized:
        params["carb_ammonium"] = params.get("carb_amine", 4.4)
    feats = site_features(mol, _anion_sites(mol))
    eff = effective_pkas(mol, feats, params)
    return [(f["o"], round(float(p), 3)) for f, p in zip(feats, eff)]


def _classify_species(smi, amine_neutralized=False):
    """Return (mol, list_of_(atom_idx, pKa_value)) for every anionic O in the (max-anion) molecule.
    Each phosphate P gets its group's FULL pKa ladder chosen by #bridging-O (free/terminal/internal);
    carboxyl/sulfonate/sulfate get their single pKa. Assumes the SMILES is already max-anion so the
    ladder length matches the deprotonated-O count on each P (source-protonation independent)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, []
    if _pka_model_site() and _site_model():
        return mol, _classify_species_site(mol, amine_neutralized)
    claimed = set()
    sites = []
    for cls, sm in _ANION_SMARTS:
        patt = Chem.MolFromSmarts(sm)
        for match in mol.GetSubstructMatches(patt):
            o = match[0]
            if o in claimed:
                continue
            claimed.add(o)
            sites.append((o, cls))
    resolved = []
    p_groups = {}                                   # P atom idx -> list of its anionic O idx
    s_groups = {}                                   # S atom idx -> list of its anionic O idx (sulfate)
    for o, cls in sites:
        if cls == "carboxyl":
            resolved.append((o, carboxyl_pka(mol, o, amine_neutralized))); continue
        if cls == "sulfonate":
            resolved.append((o, SULFONATE_PKA)); continue
        if cls == "sulfate":                         # group per S, ladder assigned below
            oa = mol.GetAtomWithIdx(o)
            s = next((n.GetIdx() for n in oa.GetNeighbors() if n.GetSymbol() == "S"), None)
            s_groups.setdefault(s, []).append(o); continue
        oa = mol.GetAtomWithIdx(o)                   # phosphate: group per P
        p = next((n.GetIdx() for n in oa.GetNeighbors() if n.GetSymbol() == "P"), None)
        p_groups.setdefault(p, []).append(o)
    if _pka_env_enabled() and _is_free_ppi(mol) and p_groups:   # free PPi ladder (PKA_ENV; see above)
        all_o = sorted(o for os in p_groups.values() for o in os)
        for o, pka in zip(all_o, PPI_LADDER + [PPI_LADDER[-1]] * max(0, len(all_o) - 4)):
            resolved.append((o, pka))
        p_groups = {}
    for p, os in p_groups.items():
        pa = mol.GetAtomWithIdx(p)
        n_bridge = sum(1 for n in pa.GetNeighbors()
                       if n.GetSymbol() == "O" and n.GetDegree() >= 2)   # ester/anhydride links
        ladder = list(P_LADDER.get(min(n_bridge, 3), [1.50]))
        k = len(os)
        # assign the k most-acidic entries of the ladder to the k deprotonated O on this P
        pkas = sorted(ladder)[:k] if k <= len(ladder) else sorted(ladder) + [1.50] * (k - len(ladder))
        for o, pka in zip(sorted(os), pkas):
            resolved.append((o, pka))
    for s, os in s_groups.items():                  # sulfate: k most-acidic ladder entries (k = #O-)
        k = len(os)
        pkas = SULFATE_LADDER[:k] if k <= len(SULFATE_LADDER) \
            else SULFATE_LADDER + [SULFATE_LADDER[-1]] * (k - len(SULFATE_LADDER))
        for o, pka in zip(sorted(os), pkas):
            resolved.append((o, pka))
    return mol, resolved


# protonated acid groups that the SOURCE may draw ionised or not (inconsistently across
# ATP/ADP/AMP). Canonicalising to the FULLY-DEPROTONATED max-anion first makes the pKa ladder
# source-independent so matched phosphates cancel exactly (fixes the FRAGILE charge-state class).
_ACID_OH_SMARTS = [
    "[OX2H][PX4]",                 # phosphate/anhydride P-OH
    "[OX2H][CX3]=[OX1]",           # carboxyl C(=O)OH
    "[OX2H][SX4](=O)(=O)",         # sulfonic/sulfate S-OH
]


def _canonicalize_maxanion(smi):
    """Deprotonate every remaining acidic O-H (phosphate/carboxyl/sulfonyl) to reach the
    FULLY-DEPROTONATED max-anion form, so downstream pKa bookkeeping does not depend on the
    source's (inconsistent) drawn protonation. Returns canonical SMILES (or the input on failure)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    rw = Chem.RWMol(mol)
    changed = False
    claimed = set()
    for sm in _ACID_OH_SMARTS:
        patt = Chem.MolFromSmarts(sm)
        for match in mol.GetSubstructMatches(patt):
            o = match[0]
            if o in claimed:
                continue
            a = rw.GetAtomWithIdx(o)
            nH = a.GetTotalNumHs()
            if nH >= 1:
                claimed.add(o)
                a.SetFormalCharge(-1)
                a.SetNoImplicit(True)
                a.SetNumExplicitHs(nH - 1)
                changed = True
    if not changed:
        return smi
    m2 = rw.GetMol()
    try:
        Chem.SanitizeMol(m2)
    except Exception:
        return smi
    return Chem.MolToSmiles(m2)


def _hcount(smi):
    """Total hydrogen count (explicit + implicit) of a SMILES, or None on parse failure.
    Used by build_ph0_reaction's mass-balance guard."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    m = Chem.AddHs(m)
    return sum(1 for a in m.GetAtoms() if a.GetSymbol() == "H")


def _neutralize(smi):
    """Canonicalise to max-anion, then protonate every anionic O to its neutral acid; return
    (neutral_smiles, list_of_pKa, net_charge_of_neutralised_form). Cationic centres (e.g.
    [N+]) are left untouched (they carry their own conjugate-acid pKa handling / stay charged)."""
    smi = _canonicalize_maxanion(smi)
    mol, sites = _classify_species(smi)
    if mol is None:
        return None, [], None
    pkas = [pka for _, pka in sites]
    rw = Chem.RWMol(mol)
    for o, _ in sites:
        a = rw.GetAtomWithIdx(o)
        a.SetFormalCharge(0)
        a.SetNumExplicitHs(a.GetNumExplicitHs() + 1)
    m2 = rw.GetMol()
    try:
        Chem.SanitizeMol(m2)
    except Exception:
        return None, [], None
    return Chem.MolToSmiles(m2), pkas, Chem.GetFormalCharge(m2)


# ---- BASE (cation) pKa's: conjugate-acid pKa of protonated N centres (textbook, NOT fitted) ----
# The deamination/transaminase/lyase classes create/destroy CATIONS (NH4+, alpha-amino -NH3+,
# guanidinium). The current _neutralize only protonates ANIONS -> leaves these charged -> the
# mass-balance guard refuses (net-proton reaction). v2 also DEPROTONATES the bases to neutral and
# emits a base pKa term using the EXACT Alberty form for a base: -RT ln(1+10^(pKa-pH)) (protonated
# form favoured below pKa), the mirror of the acid form -RT ln(1+10^(pH-pKa)).
AMMONIA_PKA = 9.25          # NH4+/NH3
AAA_AMINE_PKA = 9.60        # alpha-amino-acid -NH3+ (electron-withdrawing COOH lowers it, ~9.1-9.9)
PRIMARY_AMINE_PKA = 10.6    # plain primary aliphatic amine
IMIDAZOLE_PKA = 6.5         # histidine imidazolium (straddles pH 7)
GUANIDINIUM_PKA = 12.5      # arginine/creatine guanidinium
_ALPHA_AMINO_ACID = Chem.MolFromSmarts("[NX3,NX4+;H1,H2,H3][CX4][CX3](=O)[OX1,OX2]")  # N-C-COOH
_GUAN_C = Chem.MolFromSmarts("[#7][CX3](=[#7,#7+])")                                  # amidinium/guanidinium C

def _amine_pka(mol, n_idx):
    """Per-environment base pKa for a protonated amine N. alpha-amino-acid amine (N on a C bearing a
    carboxyl) ~9.6; plain primary amine ~10.6."""
    aa_ns = {m[0] for m in mol.GetSubstructMatches(_ALPHA_AMINO_ACID)}
    return AAA_AMINE_PKA if n_idx in aa_ns else PRIMARY_AMINE_PKA


def _classify_cations(mol):
    """Every deprotonatable protonated-N cation: (atom_idx, pKa, kind). Quaternary/aromatic N with
    no H (e.g. NAD+ N-ribosyl pyridinium) is NOT deprotonatable -> skipped (stays charged; the redox
    couple is validated as-is)."""
    sites = []
    # carbons of a guanidinium/amidinium (C bonded to >=2 N, one of them double-bonded). An N is a
    # guanidinium N only if IT is bonded to such a carbon (per atom -- the old whole-molecule test gave
    # arginine's alpha-ammonium the guanidinium pKa 12.5 instead of 9.6).
    guan_c = {m[1] for m in mol.GetSubstructMatches(_GUAN_C)}
    for a in mol.GetAtoms():
        if a.GetSymbol() != "N" or a.GetFormalCharge() != 1 or a.GetTotalNumHs() < 1:
            continue
        in_guan = any(nb.GetIdx() in guan_c for nb in a.GetNeighbors())
        heavy = [nb for nb in a.GetNeighbors() if nb.GetSymbol() != "H"]
        if any(nb.GetSymbol() == "C" and nb.GetIsAromatic() and nb.GetDegree() >= 3 for nb in heavy) \
           and a.GetTotalNumHs() == 0:
            continue
        if a.GetIsAromatic():                                 # imidazolium / aromatic N-H (his)
            sites.append((a.GetIdx(), IMIDAZOLE_PKA, "base"))
        elif in_guan:
            sites.append((a.GetIdx(), GUANIDINIUM_PKA, "base"))
        elif a.GetTotalNumHs() == 4 or (len(heavy) == 0):     # NH4+
            sites.append((a.GetIdx(), AMMONIA_PKA, "base"))
        else:
            sites.append((a.GetIdx(), _amine_pka(mol, a.GetIdx()), "base"))
    return sites


def _neutralize_v2(smi):
    """Neutralize BOTH anions (protonate O-) AND cations (deprotonate protonated N) to the fully
    neutral microspecies; return (neutral_smiles, acid_pkas, base_pkas, net_charge)."""
    smi = _canonicalize_maxanion(smi)
    mol, anion_sites = _classify_species(smi, amine_neutralized=True)
    if mol is None:
        return None, [], [], None
    cation_sites = _classify_cations(mol)
    acid_pkas = [pka for _, pka in anion_sites]
    base_pkas = [pka for _, pka, _ in cation_sites]
    rw = Chem.RWMol(mol)
    for o, _ in anion_sites:                                   # protonate anion O-
        a = rw.GetAtomWithIdx(o); a.SetFormalCharge(0); a.SetNumExplicitHs(a.GetNumExplicitHs() + 1)
    for n, _, _ in cation_sites:                               # deprotonate cation N+
        a = rw.GetAtomWithIdx(n); a.SetFormalCharge(0)
        if a.GetNumExplicitHs() > 0:
            a.SetNumExplicitHs(a.GetNumExplicitHs() - 1)
        else:
            a.SetNoImplicit(True); a.SetNumExplicitHs(max(0, a.GetTotalNumHs() - 1))
    m2 = rw.GetMol()
    try:
        Chem.SanitizeMol(m2)
    except Exception:
        return None, [], [], None
    return Chem.MolToSmiles(m2), acid_pkas, base_pkas, Chem.GetFormalCharge(m2)


# BASIC aliphatic amine/ammonium on an sp3 carbon ONLY: this is what protonates to a real cation at
# physiological pH and whose creation/destruction leaves an unmatched charged-species solvation error.
# Excludes (correctly) amides (!$(NC=O)), the NAD(P)H dihydropyridine ring N (an enamine on sp2 C -> not
# [CX4], not basic), aromatic/pyridinium N, and imines (N=*).
_AMINE_ON_C = Chem.MolFromSmarts("[CX4]-[NX3;H1,H2;!$(NC=O);!$(N=*)]")
_AMMONIUM_ON_C = Chem.MolFromSmarts("[CX4]-[NX4+;H1,H2,H3]")
def _amine_cn_change(species):
    """Net change in the count of chargeable amine-on-carbon C-N bonds across the reaction
    (Sum coeff*count). NON-zero = a C-N amine is CREATED/DESTROYED (deamination/amination/lyase) --
    exactly where neutralizing the cation removes an UNMATCHED charged-species solvation error. ZERO
    = the cations (if any) are spectators (e.g. malate-DH carboxylates, or a transaminase that just
    moves the amine) -> v2 would only inject neutral-vs-ion sampling noise -> don't fire. Generic,
    no atom-mapper (mirrors the anion pH-0's created/destroyed logic)."""
    net = 0
    for coeff, q, smi in species.values():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return 0
        cnt = len(m.GetSubstructMatches(_AMINE_ON_C)) + len(m.GetSubstructMatches(_AMMONIUM_ON_C))
        net += int(coeff) * cnt
    return net


# --- physics exclusions for the base fix (NOT residual-fitting) --------------------------------
# The pH-0 BASE correction is justified ONLY when a nitrogen that is genuinely CATIONIC at pH 7
# (basic amine, pKa ~9-11) is created/destroyed and its cation solvation is unmatched. Two amine
# C-N changes look like that to _amine_cn_change but physically are NOT, and firing there injects a
# spurious base-pKa term (validated as over-firing on the 41-rxn sweep):
_AMIDE = Chem.MolFromSmarts("[NX3][CX3]=[OX1]")      # amide / carbamoyl / urea N: pKa~0, NEUTRAL at pH7
_IMINE = Chem.MolFromSmarts("[#6]=[NX2,NX3+]")        # C=N imine/iminium (double bond, not aromatic ring N)
_NICOTINAMIDE = Chem.MolFromSmarts("[n,N]1cccc(c1)C(=O)[NX3]")   # NAD(P) nicotinamide -> a redox reaction
# formamidinium bridge [N+]=CH-N (H on the bridging C): the CATIONIC methenyl-THF bridge (charge +1 at
# pH7) that hydrolyses to a neutral formyl amide (methenyl-THF cyclohydrolase, rxn01211). Distinct from
# guanidinium (bridging C has no H -> not matched). When it is created/destroyed the anion-only pH-0 path
# FALSELY REFUSES (retained bridge cation breaks H-balance -> baseline anion -> implicit COSMO over-
# solvates the folate dicarboxylate by ~120 kJ; rxn01211 -128 vs exp -1.5). The BASE path handles it
# (protonates the carboxylates, carries the residual as n_H+) -> validated -7. Narrow trigger below.
_FORMAMIDINIUM = Chem.MolFromSmarts("[NX3+]=[CX3H1][NX3]")

def _has_nicotinamide(species):
    for coeff, q, smi in species.values():
        m = Chem.MolFromSmiles(smi)
        if m is not None and m.HasSubstructMatch(_NICOTINAMIDE):
            return True
    return False

def _count_change(species, pat):
    net = 0
    for coeff, q, smi in species.values():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        net += int(coeff) * len(m.GetSubstructMatches(pat))
    return net

def _has_free_ammonia(species):
    """A free NH3/NH4+ species (single heavy N) is present -> a genuine deamination/lyase released
    ammonia to solvent (the base fix's home ground); distinguishes it from in-place imine reduction."""
    for coeff, q, smi in species.values():
        m = Chem.MolFromSmiles(smi)
        if m is not None and m.GetNumHeavyAtoms() == 1 and m.GetAtomWithIdx(0).GetSymbol() == "N":
            return True
    return False


def _base_gate(species):
    """Should the pH-0 build ALSO neutralize basic cations / take the base path (the old v2/PH0_BASES
    path), or stay anion-only (the old v1 path)? PHYSICS: fire when a NITROGEN that is genuinely CATIONIC
    at pH 7 is CREATED/DESTROYED, so its unmatched cation solvation (and the proton bookkeeping) needs the
    base path. Two families of such cation, both textbook, nothing fitted to the DB:
      (i) basic AMINE / ammonium on sp3 C (pKa 9-11) -> _amine_cn_change; guarded by the amide + NAD-imine
          exclusions below (an amine that becomes a neutral amide, or an NAD-driven in-place imine, must
          NOT emit a spurious base-pKa).
      (ii) NON-DEPROTONATABLE heterocyclic AMIDINIUM/FORMAMIDINIUM (e.g. the methenyl-THF bridge
          [N+]=CH-N, +1 at pH7) that is created/destroyed. GENERAL, not rxn01211-specific: any
          formamidinium/amidinium hydrolysis in folate one-carbon chemistry and beyond. It is physically
          distinct from an amine: it has NO removable N-H, so _neutralize_v2 emits NO base-pKa for it -->
          it is EXEMPT from the amide exclusion (which only guards against a spurious amine base-pKa), and
          the base path's only effect is to neutralise the accompanying ANIONS (that the anion-only path
          FALSELY refuses on, collapsing to the over-solvated multi-anion) and carry the residual as n_H+.
          Gated off nicotinamide (NAD-coupled formamidinium formation is a redox that cofactor-ring
          references). Validated: methenyl-THF cyclohydrolase rxn01211 -128 -> ~-7 (exp -1.5)."""
    if _count_change(species, _FORMAMIDINIUM) and not _has_nicotinamide(species):
        return True                                           # (ii) formamidinium/amidinium hydrolysis
    if _amine_cn_change(species) == 0:                        # (i) GATE: only when a C-N amine is
        return False                                          # created/destroyed (else spectator noise)
    # PHYSICS exclusion 1: an amide/carbamoyl/urea N is created/destroyed. That N is neutral at pH 7
    # (pKa~0), NOT a basic cation -> the base premise fails. Covers carbamoyltransfer, amide
    # hydrolysis / amidohydrolase, aminoacylase, arginosuccinate synthase.
    # NOTE (2026-09-19): an attempt to fire the base path on amide HYDROLYSIS (amide_ch<0), on the
    # theory that a liberated free amine's pH-7 protonation is an unmatched contribution, was TESTED
    # and REVERTED -- it regressed EVERY amide-hydrolysis reaction (penicillin amidase -11->-29,
    # anandamide -30->-40, aminoacylase -7->-16, pantothenase -17->-23; 0 improvements). Root cause:
    # the anion-only path already scores the PROTONATED amine ([NH3+]) directly in QM (the pH-7 form),
    # so the base transform double-shifts ADD a spurious ~-14 kJ; and these floppy fatty-acid amides
    # already carry a large same-signed error (exp is POSITIVE, QM strongly negative). The amide
    # exclusion is correct as a blanket rule.
    if _count_change(species, _AMIDE):                        # None or nonzero -> don't fire
        return False
    # PHYSICS exclusion 2: an NAD(P)-driven in-place imine reduction (C=N -> C-N on the same skeleton,
    # no free NH3 released, e.g. cyclic imino-acid reductases). The amine forms by hydride transfer;
    # the nicotinamide cofactor core already references that redox -> a base-pKa term double-counts.
    # Gated on nicotinamide present so a PLP transaminase whose product merely cyclizes to an imine
    # (e.g. ornithine-oxo-acid transaminase -> P5C) still fires.
    imine_ch = _count_change(species, _IMINE)
    if imine_ch and not _has_free_ammonia(species) and _has_nicotinamide(species):
        return False
    return True


_ZW_CATION = Chem.MolFromSmarts("[N+;!H0]")     # protonated (deprotonatable) N
_ZW_ANION = Chem.MolFromSmarts("[O-]")


def is_zwitterion(smi):
    """True if the species carries BOTH a protonated N and an anionic O. Such a species is NOT a minimum
    in the gas phase: UMA relaxation transfers the N-H proton to the O- in every conformer (verified for
    Gly/Ala/Ser/Glu/Asp/phosphoserine/ethanolamine-P), so the scored species silently becomes the
    neutral tautomer (~30 kJ above the aqueous zwitterion for glycine)."""
    m = Chem.MolFromSmiles(smi)
    return m is not None and m.HasSubstructMatch(_ZW_CATION) and m.HasSubstructMatch(_ZW_ANION)


def has_zwitterion(species):
    return any(is_zwitterion(s) for _, _, s in species.values())


def is_ionized(smi):
    """True if the species carries an ANIONIC O (includes zwitterions). Scope is EVIDENCE-BASED: on TECRDB
    (2026-09-26), reactions whose QM species include an anion have MAE 15.1 (n=48) / 14.3 with a cation too
    (n=29), while cation-only reactions have MAE 8.2 (n=35) -- better than neutral-only (10.6). Protonated
    cations are therefore left charged (continuum solvation handles them), consistent with the earlier
    decomposition (anion error ~31 kJ vs cation ~4)."""
    m = Chem.MolFromSmiles(smi)
    return m is not None and m.HasSubstructMatch(_ZW_ANION)


def has_ionized(species):
    return any(is_ionized(s) for _, _, s in species.values())


def _redox_proton_enabled():
    v = os.environ.get("PH0_REDOX_PROTON")
    return v is not None and v.strip().lower() not in ("", "0", "off", "false", "no")


def build_ph0_reaction(species, n_Hplus=0, base=True, force_base=False):
    """species: {name: [coeff, q, smi]}, n_Hplus of the CHARGED reaction  ->
    (new_species, pka_sites, n_Hplus_neutral) or None.

    UNIFIED pH-0 build (the former v1 anion-only and v2 anion+base paths merged; the DISPATCH is the
    only thing that changed). new_species protonates every ionisable site to its NEUTRAL microspecies;
    pka_sites = [side, pKa, kind] with kind in {'acid','base'} telling the pipeline which exact-Alberty
    form to use (the consumer treats a bare 2-tuple as 'acid', so this is numerically identical to the
    old v1 output). In the Alberty transformed framework G(H+) is ZERO, so proton exchange with the
    pH7 bath is carried by the pKa-transform terms.

      * ANIONS are ALWAYS neutralized (the phosphoryl/carboxyl-transfer class). For this anion-only
        case a MASS-BALANCE GUARD forces n_H+=0 and REFUSES (-> None -> baseline) when the neutralised
        reaction isn't H-balanced -- else re-protonating leaks a spurious +G_HPLUS ~= +1170 kJ per
        unbalanced proton (validated: adenylate kinase +1173; rxn00184 +1112; rxn00070/86 +/-1150).
      * BASIC CATIONS are additionally neutralized and base-pKa terms emitted ONLY when base=True AND
        _base_gate passes (a real amine acid/base change: deamination/transaminase/lyase). There the
        net (redox) proton is carried by n_H+ = -h_residual instead of the mass-balance refusal.

    base=False reproduces the pure anion-only path (ablation). Returns None if no ionisable site is
    present, on any parse failure, or on the anion mass-balance refusal (caller keeps the charged path)."""
    # force_base: the reaction contains a ZWITTERION, which must not enter gas-phase QM (see
    # is_zwitterion). The full neutral-microspecies (base) path is the Alberty-consistent treatment: both
    # the acid and the amine are neutralized and the zwitterion-dominated aqueous macro-state is
    # reconstructed analytically by the acid + base pKa terms (independent-site product with the
    # microscopic carboxyl pKa reproduces the tautomer constant; see CARBOXYL_PKA_ALPHA notes).
    use_base = (base and _base_gate(species)) or force_base
    new_species = {}
    pka_sites = []
    any_ionizable = False
    h_residual = 0                                    # Sum coeff*H over NEUTRAL microspecies
    q_residual = 0                                    # Sum coeff*charge over NEUTRAL microspecies
    for name, (coeff, q, smi) in species.items():
        if use_base:
            neutral, acids, bases, netq = _neutralize_v2(smi)   # anions AND cations
        else:
            neutral, acids, netq = _neutralize(smi)             # anions only
            bases = []
        if neutral is None:
            return None                              # bail safely on any parse failure
        nH = _hcount(neutral)
        if nH is None:
            return None
        h_residual += int(coeff) * nH
        q_residual += int(coeff) * int(netq)
        if acids or bases:
            any_ionizable = True
        side = "react" if coeff < 0 else "prod"
        for pka in acids:
            for _ in range(abs(int(coeff))):
                pka_sites.append([side, pka, "acid"])
        for pka in bases:
            for _ in range(abs(int(coeff))):
                pka_sites.append([side, pka, "base"])
        new_species[name] = [coeff, netq, neutral]
    if not any_ionizable:
        return None
    if use_base:
        return new_species, pka_sites, -h_residual   # n_H+ carries the net (redox) proton
    # anion-only mass-balance guard (see docstring): n_H+=0 only valid when H-balanced, else refuse.
    # NOTE (2026-08-20): a charge-closure relaxation (fire with n_H+=-Σcoeff·q_neutral when charge- and
    # H-derived n_H+ agree) was TESTED to reach rxn01211 (folate cyclohydrolase, false-refused -> -144).
    # VALIDATED NO-GO: it fixes rxn01211 (-144->-7) but REINTRODUCES the ~+1170 kJ leak on cofactor
    # reactions (glutathione reductase +2351, sulfate adenylyltransferase +100, nicotinate PRT -59) via
    # the composition with COFACTOR_RING's own proton bookkeeping, and mildly regresses redox DHs /
    # carbamoyltransferase. The charge/H consistency check does NOT separate the leaks (they pass it).
    # The conservative refusal stays. rxn01211's correct value (-7, base path) is a documented known
    # outlier; a narrow, separately-validated iminium-aware base gate is the future fix, not this.
    if h_residual != 0:
        # REDOX PROTON (PH0_REDOX_PROTON): a net-proton reaction (e.g. NAD(P)+ + 2e- + H+ -> NAD(P)H with the
        # substrate's anions neutralized) is H-imbalanced by the proton exchanged with the pH-7 bath. Alberty-
        # consistent bookkeeping carries it explicitly as n_H+ = -h_residual (G_HPLUS includes the pH term),
        # exactly as the base path does -- PROVIDED it also closes the charge (q_residual == h_residual), so the
        # count is a real proton and not a mis-neutralized species. Otherwise refuse (the +/-1170 kJ leak guard).
        if _redox_proton_enabled() and q_residual == h_residual:
            return new_species, pka_sites, -h_residual
        return None
    return new_species, pka_sites, 0                 # n_H+=0: transforms carry all proton exchange


if __name__ == "__main__":
    import json, sys
    # self-test: reproduce the hand pH-0 annotations + preview the phosphate failures
    tests = {
        "rxn01713 (ester/AcOH, hand: react pKa4.4)": {
            "AcOH": [-1, -1, "CC(=O)[O-]"],
            "Pglc": [-1, -1, "O=P([O-])(O)O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"],
            "ester": [1, 0, "CC(=O)O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"],
            "Pi": [1, -1, "O=P([O-])(O)O"]},
        "ach (hand: prod pKa4.76)": {
            "ach": [-1, 1, "CC(=O)OCC[N+](C)(C)C"], "H2O": [-1, 0, "O"],
            "AcOH": [1, 0, "CC(=O)O"], "choline": [1, 1, "OCC[N+](C)(C)C"]},
    }
    for label, sp in tests.items():
        out = build_ph0_reaction(sp)
        print("===", label, "===")
        if out is None:
            print("  (no anion / bail)"); continue
        ns, pk, nh = out
        for nm, (c, q, s) in ns.items():
            print(f"   {nm:10s} coeff={c:+d} q={q:+d}  {s}")
        print("   pka_sites:", pk, "| n_Hplus_neutral:", nh)
        # pk entries are [side, pKa, kind]; acid term uses (pH-pKa), base term uses (pKa-pH)
        def _term(site):
            sd, p = site[0], site[1]
            kind = site[2] if len(site) > 2 else "acid"
            expo = (7.0 - p) if kind == "acid" else (p - 7.0)
            return (1 if sd == "react" else -1) * expo
        net = sum(_term(s) for s in pk)
        print(f"   net (pH-pKa) sum (x5.71 = kJ): {net:.2f}  -> {net*5.71:+.1f} kJ/mol\n")
