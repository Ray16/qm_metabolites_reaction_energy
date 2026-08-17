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
    # TODO (unvalidated stubs): flavin (FAD/FADH2 -> lumiflavin), CoA thioester core, lipoate.
]


def _has(mol, pat):
    return pat is not None and mol is not None and mol.HasSubstructMatch(pat)


def cofactor_ring(species):
    """species: {name:(coeff,charge,SMILES)} -> new dict with every cofactor whose ox/red COUPLE is
    present replaced by its canonical core (charge set from the table). Returns the ORIGINAL dict
    (identity) unchanged if no couple fires. Composes across couples (double-redox handled)."""
    mols = {n: Chem.MolFromSmiles(s) for n, (c, q, s) in species.items()}
    repl = {}                                                  # name -> (core_smiles, core_charge)
    for cp in COUPLES:
        ox = [n for n, mol in mols.items() if _has(mol, cp["ox_pat"])]
        red = [n for n, mol in mols.items()
               if _has(mol, cp["red_pat"]) and not _has(mol, cp["ox_pat"])]
        if not (ox and red):                                   # need a genuine couple -> skip
            continue
        for n in ox:
            repl[n] = cp["ox_core"]
        for n in red:
            repl[n] = cp["red_core"]
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
    print("canonical-cores self-test PASSED (nicotinamide + cysteine-thiol; double-redox composes)")
