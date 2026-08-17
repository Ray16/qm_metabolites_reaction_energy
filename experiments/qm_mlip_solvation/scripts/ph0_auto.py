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
CARBOXYL_PKA = 4.75
SULFONATE_PKA = -1.5
SULFATE_PKA = -3.0

# SMARTS for a deprotonated (anionic) oxygen of each class, matched on the [O-] atom (first atom).
_ANION_SMARTS = [
    ("carboxyl",  "[$([OX1-][CX3]=O)]"),                 # carboxylate O-
    ("sulfonate", "[$([OX1-][SX4](=O)(=O)[#6])]"),        # R-SO3-
    ("sulfate",   "[$([OX1-][SX4](=O)(=O)[OX2])]"),       # R-O-SO3-
    ("phosphate", "[$([OX1-][P])]"),                      # any P-O-  (sub-classified below)
]


def _classify_species(smi):
    """Return (mol, list_of_(atom_idx, pKa_value)) for every anionic O in the (max-anion) molecule.
    Each phosphate P gets its group's FULL pKa ladder chosen by #bridging-O (free/terminal/internal);
    carboxyl/sulfonate/sulfate get their single pKa. Assumes the SMILES is already max-anion so the
    ladder length matches the deprotonated-O count on each P (source-protonation independent)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, []
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
    for o, cls in sites:
        if cls == "carboxyl":
            resolved.append((o, CARBOXYL_PKA)); continue
        if cls == "sulfonate":
            resolved.append((o, SULFONATE_PKA)); continue
        if cls == "sulfate":
            resolved.append((o, SULFATE_PKA)); continue
        oa = mol.GetAtomWithIdx(o)                   # phosphate: group per P
        p = next((n.GetIdx() for n in oa.GetNeighbors() if n.GetSymbol() == "P"), None)
        p_groups.setdefault(p, []).append(o)
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


def build_ph0_reaction(species, n_Hplus=0):
    """species: {name: [coeff, q, smi]}, n_Hplus of the CHARGED reaction  ->
    (new_species, pka_sites, n_Hplus_neutral) or None.

    new_species has every ionisable site protonated to its NEUTRAL microspecies; pka_sites lists
    ['react'|'prod', pKa] (one per neutralised proton, per |coeff|). In the Alberty transformed
    framework the transformed formation energy of H+ is ZERO, so ALL proton exchange with the pH7
    bath is carried by the pKa-transform terms and the explicit n_H+*G_HPLUS term MUST be dropped:
    n_Hplus_neutral = 0. (Keeping the CHARGED reaction's n_H+ would double-count -- it added a
    spurious +G_HPLUS ~= +1170 kJ on e.g. adenylate kinase, giving ΔG +1173 instead of ~0.)
    This matches the validated hand pH-0 reactions (ach_ph0, rxn01713_ph0), all of which set n_H+=0.
    Returns None if no anionic site is present (nothing to do -> caller keeps the original path)."""
    new_species = {}
    pka_sites = []
    any_anion = False
    for name, (coeff, q, smi) in species.items():
        neutral, pkas, netq = _neutralize(smi)
        if neutral is None:
            return None                              # bail safely on any parse failure
        if pkas:
            any_anion = True
        side = "react" if coeff < 0 else "prod"
        for pka in pkas:
            for _ in range(abs(int(coeff))):
                pka_sites.append([side, pka])
        new_species[name] = [coeff, netq, neutral]
    if not any_anion:
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
        net = sum((1 if sd == "react" else -1) * (7.0 - p) for sd, p in pk)
        print(f"   net (pH-pKa) sum (x5.71 = kJ): {net:.2f}  -> {net*5.71:+.1f} kJ/mol\n")
