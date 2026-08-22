"""Canonical cofactor cores -- a table-driven, extensible generalization of the ring-cofactor fix.

WHY a curated table (not pure MCS localization): the general MCS localizer (localize.py) works on
the substrate tail, but FAILS on the ubiquitous cofactors -- NAD/NADP/FAD/CoA are large and
SYMMETRIC (two riboses + adenine), so RDKit's atom-map picks the wrong symmetric copy and the
reactive core can't be isolated (measured: it keeps the full cofactor). Those same cofactors are
the top ~10 compounds in ModelSEED (~10,600 of 86,775 species-instances), so a curated canonical
core per cofactor is (a) reliable where MCS is not and (b) essentially FREE at scale (compute the
small core once, cache, reuse). Adding a cofactor = one table row, no new code.

Principle (per couple): the large scaffold (ADP-ribose-phosphate tail, glutathione peptide) is
IDENTICAL on the oxidised and reduced forms, so it cancels in ΔG. Replace each form with its small
redox-active core -- an isodesmic substitution -- removing the floppy-tail conformer noise that does
NOT cancel numerically in full-molecule QM. Experiment-free (textbook cores). Gated to a genuine
ox/red COUPLE (both forms present) so it never mis-fires on biosynthesis/salvage. Multiple couples
compose: glutathione reductase (NAD + GSH) gets BOTH substitutions automatically.

Validated: NAD dehydrogenases 44.6->10.4 kJ; glutathione reductase (NAD+GSH) +34.7->+19.2 (err +7).
"""
from rdkit import Chem


def _m(sm):
    return Chem.MolFromSmarts(sm)


# ---- couple registry -------------------------------------------------------------------------
# Each entry: detect the OXIDISED and REDUCED form of a cofactor by substructure; give the small
# canonical core (SMILES, formal charge) for each. Substitute only when a reaction contains BOTH.
COUPLES = [
    dict(
        name="nicotinamide",                                   # NAD+/NADP+  <->  NADH/NADPH
        ox_pat=_m("[n+]1cccc(c1)C(=O)[NX3]"),                  # N-substituted pyridinium carboxamide
        red_pat=_m("[NX3][CX3](=O)C1=CN([#6])C=CC1"),          # N-substituted 1,4-dihydronicotinamide
        ox_core=("NC(=O)c1ccc[n+](C)c1", 1),                   # 1-methylnicotinamide cation
        red_core=("NC(=O)C1=CN(C)C=CC1", 0),                   # 1-methyl-1,4-dihydronicotinamide
    ),
    dict(
        name="cysteine-thiol",                                 # 2 GSH  <->  GSSG  (thiol/disulfide)
        # oxidised = cystine disulfide on a cysteinyl (S-S-CH2-CH(N)-C=O); reduced = free thiol/thiolate
        ox_pat=_m("[#16X2]-[#16X2]-[CH2]-[CH]([#7])-[#6]=O"),
        red_pat=_m("[#16X2H1,#16X1H0-]-[CH2]-[CH]([#7])-[#6]=O"),
        ox_core=("CC(=O)NC(CSSCC(NC(C)=O)C(=O)NC)C(=O)NC", 0), # Ac-Cys-NHMe disulfide dimer
        red_core=("CC(=O)NC(CS)C(=O)NC", 0),                   # Ac-Cys-NHMe thiol
    ),
    # TODO (unvalidated stubs): flavin (FAD/FADH2 -> lumiflavin), lipoate.
]

# ---- CoA thioester carrier (NOT a redox couple) --------------------------------------------------
# CoA differs from NAD/GSH: the ADP-3'-phosphate-pantetheine tail is the spectator, but the ACYL
# group on the thioester VARIES per reaction and IS the chemistry -- so a fixed core string can't be
# used. Instead truncate the invariant tail to N-ACETYLCYSTEAMINE (the textbook experimental CoA
# surrogate): keep the acyl-thioester (or free thiol) + cysteamine, cap the first amide with methyl.
# Isodesmic: the -S-CH2CH2-NH-C(=O)-CH3 cap is IDENTICAL on every CoA-derived species (acyl-CoA and
# free CoA-SH alike) so the tail cancels in ΔG; only the reactive acyl-thioester vs free-thiol
# difference survives. Double win: also drops the tail's -4 phosphate charge (which was itself
# driving UMA's charged-solute solvation error), and removes the ~50-atom floppy tail whose
# independent conformer sampling on each side leaves a large NON-cancelling residual in full-molecule
# QM (e.g. methylmalonyl-CoA epimerase: exp ΔG≈0, full-molecule UMA +54 -- a pure sampling artifact).
# Anchor: S-CH2-CH2-NH-C(=O)-C(tail); break carbonyl->tail bond, keep the sulfur fragment, cap the
# carbonyl carbon with a methyl. Applied to EVERY CoA species, but ONLY when a CoA tail is present on
# BOTH sides (mass-balanced -> the cap cancels); otherwise left untouched.
_COA_ANCHOR = _m("[#16X2,#16X1;!$([#16]~[#16])]-[CH2]-[CH2]-[NX3]-[CX3](=[OX1])-[#6]")


def _coa_to_nac(smi):
    """Truncate a CoA-tail molecule to its N-acetylcysteamine analogue. Returns (smiles, charge) or
    None if it carries no CoA tail (caller should keep the original)."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    matches = m.GetSubstructMatches(_COA_ANCHOR)
    if not matches:
        return None
    s_idx, carbonyl_c, tail_c = matches[0][0], matches[0][4], matches[0][6]
    rw = Chem.RWMol(m)
    rw.RemoveBond(carbonyl_c, tail_c)
    me = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(carbonyl_c, me, Chem.BondType.SINGLE)
    keep = next(f for f in Chem.GetMolFrags(rw.GetMol(), asMols=False, sanitizeFrags=False)
                if s_idx in f)
    em = Chem.RWMol(rw.GetMol())
    for a in sorted([i for i in range(em.GetNumAtoms()) if i not in keep], reverse=True):
        em.RemoveAtom(a)
    out = em.GetMol()
    Chem.SanitizeMol(out)
    return Chem.MolToSmiles(out), Chem.GetFormalCharge(out)


def _formula_counter(smi):
    """Element -> atom count (incl. explicit H) for a SMILES, as a Counter."""
    from collections import Counter
    m = Chem.MolFromSmiles(smi)
    c = Counter()
    if m is None:
        return c
    for a in Chem.AddHs(m).GetAtoms():
        c[a.GetSymbol()] += 1
    return c


def _coa_truncate(species):
    """Replace every CoA-derived species by its N-acetylcysteamine analogue, but only when a CoA tail
    appears on BOTH sides AND the removed tail is IDENTICAL on both sides (so it cancels in ΔG).
    Returns a name->(smiles,charge) replacement dict (empty if it does not fire)."""
    from collections import Counter
    nac = {n: _coa_to_nac(s) for n, (c, q, s) in species.items()}
    coa = [n for n, r in nac.items() if r is not None]
    react = any(species[n][0] < 0 for n in coa)
    prod = any(species[n][0] > 0 for n in coa)
    if not (react and prod):
        return {}
    # MASS-BALANCE GATE: the truncated-away tail (original − NAC) must CANCEL across the reaction.
    # If a reaction MODIFIES the tail (dephospho-CoA kinase adds a 3'-phosphate; CoA-biosynthesis),
    # the removed tails differ between reactant and product and do NOT cancel -> truncation destroys
    # the reaction center -> ~+1.49M-kJ leak. Require Σ coeff·(removed formula) == 0, else refuse.
    net = Counter()
    for n in coa:
        c, q, s = species[n]
        removed = _formula_counter(s)
        removed.subtract(_formula_counter(nac[n][0]))
        for el, cnt in removed.items():
            net[el] += c * cnt
    if any(v != 0 for v in net.values()):
        return {}       # tail is modified by the reaction -> not isodesmic -> keep full molecules
    return {n: nac[n] for n in coa}


def _has(mol, pat):
    return pat is not None and mol is not None and mol.HasSubstructMatch(pat)


def cofactor_ring(species):
    """species: {name:(coeff,charge,SMILES)} -> new dict with every cofactor whose ox/red COUPLE is
    present replaced by its canonical core (charge set from the table). Returns the ORIGINAL dict
    (identity) unchanged if no couple fires. Composes across couples (double-redox handled)."""
    mols = {n: Chem.MolFromSmiles(s) for n, (c, q, s) in species.items()}
    repl = {}                                                  # name -> (core_smiles, core_charge)
    # The cysteine-thiol couple denoises the glutathione-REDUCTASE cofactor and is only valid alongside
    # a nicotinamide redox: its symmetric-cystine core is ISODESMIC for GSSG (glutathione homodimer)
    # but MANGLES a MIXED disulfide (e.g. CoA-S-S-glutathione in the standalone transhydrogenase
    # rxn00824 -> cored to plain cystine, dropping the CoA half -> ~+7.7M kJ garbage). Gate it on
    # nicotinamide presence so standalone thiol-disulfide exchanges fall back to full molecules.
    _nic = COUPLES[0]
    has_nicotinamide = any(_has(mol, _nic["ox_pat"]) or _has(mol, _nic["red_pat"])
                           for mol in mols.values())
    for cp in COUPLES:
        if cp["name"] == "cysteine-thiol" and not has_nicotinamide:
            continue                                           # accessory couple needs a nicotinamide redox
        ox = [n for n, mol in mols.items() if _has(mol, cp["ox_pat"])]
        red = [n for n, mol in mols.items()
               if _has(mol, cp["red_pat"]) and not _has(mol, cp["ox_pat"])]
        if not (ox and red):                                   # need a genuine couple -> skip
            continue
        for n in ox:
            repl[n] = cp["ox_core"]
        for n in red:
            repl[n] = cp["red_core"]
    # CoA thioester carrier truncation (independent of the redox couples; touches disjoint species so
    # it composes -- e.g. acetaldehyde dehydrogenase (acylating) gets BOTH nicotinamide and CoA cores).
    for n, core in _coa_truncate(species).items():
        repl[n] = core
    if not repl:
        return species
    new = {}
    for n, (c, q, s) in species.items():
        if n in repl:
            core, cq = repl[n]
            new[n] = (c, cq, core)
        else:
            new[n] = (c, q, s)
    return new


if __name__ == "__main__":
    NAD = "NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1"
    NADH = "NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1"
    GSH = "[NH3+][C@@H](CCC(=O)N[C@@H](C[S-])C(=O)NCC(=O)[O-])C(=O)[O-]"
    GSSG = "[NH3+][C@@H](CCC(=O)N[C@@H](CSSC[C@H](NC(=O)CC[C@H]([NH3+])C(=O)[O-])C(=O)N)C(=O)N)C(=O)[O-]"
    NICOTINAMIDE = "NC(=O)c1cccnc1"; FAD = "Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C)c2cc1C"
    # simple NAD dehydrogenase -> only nicotinamide fires
    r1 = cofactor_ring({"NAD": (-1, -1, NAD), "NADH": (1, -2, NADH), "S": (-1, 0, "CCO")})
    assert r1["NAD"][2] == "NC(=O)c1ccc[n+](C)c1" and r1["NADH"][2] == "NC(=O)C1=CN(C)C=CC1"
    assert r1["S"][2] == "CCO"
    # glutathione reductase -> BOTH couples fire
    r2 = cofactor_ring({"NADP": (-1, -1, NAD), "GSH": (-2, -2, GSH),
                        "NADPH": (1, -2, NADH), "GSSG": (1, -2, GSSG)})
    assert r2["NADP"][2] == "NC(=O)c1ccc[n+](C)c1", "NADP ring"
    assert r2["GSH"][2] == "CC(=O)NC(CS)C(=O)NC", "GSH thiol core"
    assert "SS" in r2["GSSG"][2], "GSSG disulfide core"
    # no reduced partner (biosynthesis) or non-cofactor -> unchanged
    assert cofactor_ring({"NAD": (-1, -1, NAD), "x": (1, 0, "CCO")}) == {"NAD": (-1, -1, NAD), "x": (1, 0, "CCO")}
    assert not _has(Chem.MolFromSmiles(NICOTINAMIDE), COUPLES[0]["ox_pat"]), "free nicotinamide skip"
    assert not _has(Chem.MolFromSmiles(FAD), COUPLES[0]["ox_pat"]), "FAD not nicotinamide"
    # --- CoA thioester carrier truncation ---
    ACCOA = "CC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)([O-])[O-]"
    COASH = "SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)([O-])[O-]"
    assert _coa_to_nac(ACCOA) == ("CC(=O)NCCSC(C)=O", 0), "acetyl-CoA -> S-acetyl-NAC"
    assert _coa_to_nac(COASH) == ("CC(=O)NCCS", 0), "free CoA -> NAC thiol"
    assert _coa_to_nac("CCO") is None, "non-CoA untouched"
    # acyltransferase: acetyl-CoA (reactant) <-> CoA-SH (product) -> BOTH truncated, tail cancels
    r3 = cofactor_ring({"AcCoA": (-1, -4, ACCOA), "CoA": (1, -4, COASH), "S": (-1, 0, "CC(=O)O")})
    assert r3["AcCoA"] == (-1, 0, "CC(=O)NCCSC(C)=O") and r3["CoA"] == (1, 0, "CC(=O)NCCS"), "CoA acyl-transfer"
    # CoA on ONE side only -> not mass-balanced -> left untouched (no false truncation)
    r4 = cofactor_ring({"AcCoA": (-1, -4, ACCOA), "x": (1, 0, "CCO")})
    assert r4["AcCoA"] == (-1, -4, ACCOA), "one-sided CoA untouched"
    # composes with nicotinamide: acetaldehyde dehydrogenase (acylating) gets BOTH cores
    r5 = cofactor_ring({"NAD": (-1, -1, NAD), "CoA": (-1, -4, COASH),
                        "NADH": (1, -2, NADH), "AcCoA": (1, -4, ACCOA)})
    assert r5["NAD"][2] == "NC(=O)c1ccc[n+](C)c1" and r5["CoA"][2] == "CC(=O)NCCS", "NAD+CoA compose"
    assert r5["AcCoA"][2] == "CC(=O)NCCSC(C)=O", "acetyl-CoA core alongside NAD"
    print("canonical-cores self-test PASSED (nicotinamide + cysteine-thiol + CoA-NAC; all compose)")
