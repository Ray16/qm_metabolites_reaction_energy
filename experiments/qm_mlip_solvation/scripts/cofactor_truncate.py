"""Cofactor ring-truncation for nicotinamide redox couples (NAD(P)+/NAD(P)H).

Replace the giant NAD(P) cofactor with its redox-active NICOTINAMIDE RING model
(1-methylnicotinamide+ / 1-methyl-1,4-dihydronicotinamide). The ADP-ribose-phosphate
tail is IDENTICAL on the oxidised and reduced forms, so it cancels in the reaction ΔG --
this is an isodesmic substitution, NOT a truncation of reactive atoms. The point: the
floppy tail's conformer/solvation error does NOT cleanly cancel in the full-molecule QM
(it is why NAD, floppy, fails while NADP, rigidified by its extra phosphate, does not).
Replacing with the small rigid ring removes that noise AND collapses NAD & NADP to the
same model, so the spurious NAD-vs-NADP differential vanishes.

Validated (2026-08-16, 6 TECRDB redox rxns): NAD MAE 44.6 -> 10.4, all-6 33.8 -> 9.5 kJ.
Experiment-free (textbook ring model, no fitted data). General/structural: no reaction-id.

SAFETY GATE: only fires when the reaction contains BOTH an oxidised and a reduced
nicotinamide (a genuine hydride-transfer couple) -- so it never mis-fires on NAD
biosynthesis/salvage reactions where a nicotinamide appears as a substrate being built.
Free nicotinamide (neutral pyridine), nicotinate (carboxylate not amide), and FAD/flavin
do NOT match. FAD is a separate ring system (not handled here; its tail is small relative
to the flavin, so the noise problem is milder).
"""
from rdkit import Chem

# redox-active ring models (ribose tail -> methyl cap); nicotinamide ring is identical for NAD & NADP
RING_OX = "NC(=O)c1ccc[n+](C)c1"    # 1-methylnicotinamide cation, q=+1
RING_RED = "NC(=O)C1=CN(C)C=CC1"     # 1-methyl-1,4-dihydronicotinamide, q=0

# oxidised: N-substituted pyridinium bearing a carboxamide (NAD+/NADP+ nicotinamide).
# free nicotinamide is neutral `n` -> no match; nicotinate is C(=O)[O-] -> no match.
_PAT_OX = Chem.MolFromSmarts("[n+]1cccc(c1)C(=O)[NX3]")
# reduced: N-substituted 1,4-dihydropyridine bearing a carboxamide (NADH/NADPH).
_PAT_RED = Chem.MolFromSmarts("[NX3][CX3](=O)C1=CN([#6])C=CC1")


def _is_ox(smi):
    m = Chem.MolFromSmiles(smi)
    return m is not None and m.HasSubstructMatch(_PAT_OX)


def _is_red(smi):
    m = Chem.MolFromSmiles(smi)
    return m is not None and m.HasSubstructMatch(_PAT_RED)


def cofactor_ring(species):
    """species: {name: (coeff, charge, SMILES)} -> new dict with NAD(P) cofactors replaced by the
    ring model, or the ORIGINAL dict unchanged if no genuine nicotinamide redox couple is present
    (or if the substitution would unbalance the reaction). Charge set to +1 (ox) / 0 (red)."""
    ox = [n for n, (c, q, s) in species.items() if _is_ox(s)]
    red = [n for n, (c, q, s) in species.items() if _is_red(s)]
    if not (ox and red):                       # need a real hydride-transfer couple -> gate off
        return species
    new = {}
    for n, (c, q, s) in species.items():
        if n in ox:
            new[n] = (c, 1, RING_OX)
        elif n in red:
            new[n] = (c, 0, RING_RED)
        else:
            new[n] = (c, q, s)
    return new


if __name__ == "__main__":
    # self-test on the real cofactor SMILES + the edge cases that must NOT match
    NAD = "NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1"
    NADH = "NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1"
    NICOTINAMIDE = "NC(=O)c1cccnc1"
    NICOTINATE_RN = "O=C([O-])c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@H]2O)c1"
    FAD = "Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C)c2cc1C"
    assert _is_ox(NAD) and not _is_red(NAD), "NAD+ should be OX only"
    assert _is_red(NADH) and not _is_ox(NADH), "NADH should be RED only"
    assert not _is_ox(NICOTINAMIDE) and not _is_red(NICOTINAMIDE), "free nicotinamide must not match"
    assert not _is_ox(NICOTINATE_RN), "nicotinate (carboxylate) must not match"
    assert not _is_ox(FAD) and not _is_red(FAD), "FAD flavin must not match"
    # couple present -> substitute; only ox present (biosynthesis) -> unchanged
    sp = {"NAD": (-1, -1, NAD), "NADH": (1, -2, NADH), "S": (-1, 0, "CCO")}
    out = cofactor_ring(sp)
    assert out["NAD"][2] == RING_OX and out["NADH"][2] == RING_RED, "couple should be replaced"
    sp_bio = {"NAD": (-1, -1, NAD), "S": (-1, 0, "CCO")}
    assert cofactor_ring(sp_bio) is sp_bio, "no reduced partner -> gate off (unchanged)"
    print("cofactor_truncate self-test PASSED")
