"""Chemical-microstate enumeration for the sugar-forms / tautomer study (analysis only; no src/ edits).

Two families of states that do NOT share a molecular graph with the input SMILES, so conformer sampling
can never reach them (see metag.energetics.microstates):

1. Sugar ring-chain states (mutarotation): a free reducing centre (hemiacetal/hemiketal C bearing an OH and a
   ring O, or an open aldehyde/ketone) equilibrates in water among the open carbonyl and every 5-/6-membered
   hemiacetal ring that a free hydroxyl can close, each as two anomers. Glycosides / glycosyl phosphates
   (anomeric O substituted) cannot mutarotate and get no alternative states.

2. Heteroatom prototropic tautomers: H moves only between N/O/S (lactam/lactim, amino/imino, imidazole N-H).
   Keto-enol (C-H <-> O-H) is excluded by requiring every carbon's H count to be unchanged; all defined
   stereocentres must be preserved.

Every function returns the INPUT state description too, so the caller always retains it.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

_HEMI = Chem.MolFromSmarts("[CX4;R:1]([OX2;R:2])[OX2H1:3]")       # ring hemiacetal / hemiketal centre
_CARBONYL = Chem.MolFromSmarts("[CX3:1](=[OX1:2])([#6,#1])")        # aldehyde or ketone (filtered below)


def _canon(m):
    return Chem.MolToSmiles(m)


def _is_carboxyl_like(atom):
    """A carbonyl C also bonded to an O/N single-bonded heteroatom (acid, ester, amide) is not a sugar carbonyl."""
    for nb in atom.GetNeighbors():
        b = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
        if nb.GetSymbol() in ("O", "N", "S") and b.GetBondType() == Chem.BondType.SINGLE:
            return True
    return False


def open_chain(smi):
    """Open every free ring hemiacetal/hemiketal of `smi` (one at a time; we only treat mono-reducing
    sugars). Returns (open_smiles, opened) -- opened=False if `smi` has no free reducing ring centre."""
    m = Chem.MolFromSmiles(smi)
    hits = m.GetSubstructMatches(_HEMI)
    hits = [h for h in hits if m.GetAtomWithIdx(h[1]).IsInRingSize(5) or m.GetAtomWithIdx(h[1]).IsInRingSize(6)]
    if not hits:
        return smi, False
    if len({h[0] for h in hits}) > 1:
        raise ValueError(f"{smi}: more than one free reducing centre (not handled)")
    c, o_ring, o_h = hits[0]
    rw = Chem.RWMol(m)
    rw.RemoveBond(c, o_ring)
    rw.GetBondBetweenAtoms(c, o_h).SetBondType(Chem.BondType.DOUBLE)
    a_c = rw.GetAtomWithIdx(c); a_c.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    a_oh = rw.GetAtomWithIdx(o_h); a_oh.SetNumExplicitHs(0); a_oh.SetNoImplicit(True)
    a_or = rw.GetAtomWithIdx(o_ring); a_or.SetNoImplicit(False); a_or.SetNumExplicitHs(0)
    for a in rw.GetAtoms():
        a.SetIsAromatic(a.GetIsAromatic())
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return _canon(out), True


def _ring_closures(open_smi):
    """All mono-cyclic 5/6-membered hemiacetal closures of the open-chain carbonyl, each anomer explicit.
    Returns list of (smiles, ring_size, anomeric_atom_idx_in_open, ring_O_idx_in_open)."""
    m = Chem.MolFromSmiles(open_smi)
    dm = Chem.GetDistanceMatrix(m)
    out, done = [], set()
    for c, o_c, _ in m.GetSubstructMatches(_CARBONYL):
        if c in done:
            continue
        done.add(c)
        ca = m.GetAtomWithIdx(c)
        if _is_carboxyl_like(ca) or ca.GetIsAromatic() or ca.IsInRing():
            continue
        for o in m.GetAtoms():
            if o.GetSymbol() != "O" or o.GetTotalNumHs() != 1 or o.GetDegree() != 1:
                continue
            nb = o.GetNeighbors()[0]
            if nb.GetSymbol() != "C" or nb.GetHybridization() != Chem.HybridizationType.SP3:
                continue
            d = int(dm[c, o.GetIdx()])
            if d not in (4, 5):                        # 5-ring (furanose) / 6-ring (pyranose)
                continue
            path = Chem.GetShortestPath(m, c, o.GetIdx())
            if not all(m.GetAtomWithIdx(i).GetSymbol() == "C" and
                       m.GetAtomWithIdx(i).GetHybridization() == Chem.HybridizationType.SP3
                       for i in path[1:-1]):
                continue
            for tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
                rw = Chem.RWMol(m)
                rw.GetBondBetweenAtoms(c, o_c).SetBondType(Chem.BondType.SINGLE)
                rw.AddBond(c, o.GetIdx(), Chem.BondType.SINGLE)
                ao = rw.GetAtomWithIdx(o.GetIdx()); ao.SetNoImplicit(True); ao.SetNumExplicitHs(0)
                aoc = rw.GetAtomWithIdx(o_c); aoc.SetNoImplicit(False)
                rw.GetAtomWithIdx(c).SetChiralTag(tag)
                mm = rw.GetMol()
                Chem.SanitizeMol(mm)
                Chem.AssignStereochemistry(mm, cleanIt=True, force=True)
                out.append((_canon(mm), d + 1, c, o.GetIdx()))
    return out


def anomer_cip(ring_smi):
    """CIP label of the anomeric centre. For D-glucose, D-fructose and D-ribose (pyranose AND furanose)
    beta == R (checked against PubChem beta-D-glucopyranose/fructopyranose/fructofuranose/ribofuranose)."""
    from rdkit.Chem import rdCIPLabeler
    m = Chem.MolFromSmiles(ring_smi); rdCIPLabeler.AssignCIPLabels(m)
    c = m.GetSubstructMatches(_HEMI)[0][0]
    return m.GetAtomWithIdx(c).GetProp("_CIPCode")


def _anomer_label_geometric_unused(ring_smi):
    """alpha/beta for a D/L sugar ring by the Fischer rule (alpha: anomeric exocyclic O on the same Fischer side
    as the O of the configurational reference atom). Implemented geometrically: in a ring, two substituents
    on the same Fischer side of ADJACENT-numbered ring carbons are cis. We return 'cis'/'trans' of the anomeric
    OH relative to the nearest ring-carbon OH (C2 for aldoses, C3 for 2-ketoses); the caller maps cis/trans
    to alpha/beta per sugar (glucose/ribose: alpha=1,2-cis; fructose: alpha=2,3-trans)."""
    m = Chem.AddHs(Chem.MolFromSmiles(ring_smi))
    hit = m.GetSubstructMatches(_HEMI)
    c, o_ring, o_h = hit[0]
    ring = [r for r in m.GetRingInfo().AtomRings() if c in r and o_ring in r][0]
    nbr_c = [n.GetIdx() for n in m.GetAtomWithIdx(c).GetNeighbors()
             if n.GetIdx() in ring and n.GetSymbol() == "C"][0]
    oh2 = [n.GetIdx() for n in m.GetAtomWithIdx(nbr_c).GetNeighbors()
           if n.GetSymbol() == "O" and n.GetIdx() not in ring]
    if not oh2:
        return None
    AllChem.EmbedMolecule(m, randomSeed=7)
    X = m.GetConformer().GetPositions()
    P = X[list(ring)]; cen = P.mean(0)
    n = np.linalg.svd(P - cen)[2][-1]
    s1 = np.sign(np.dot(X[o_h] - X[c], n)); s2 = np.sign(np.dot(X[oh2[0]] - X[nbr_c], n))
    return "cis" if s1 == s2 else "trans"


def sugar_states(smi):
    """Ring-chain states of a species. Returns dict:
       input: canonical input; input_kind: 'ring_unspecified'|'ring_defined'|'open'|'none';
       alternatives: list of {smiles, kind, ring_size} NOT already represented by the input.
    For an input ring with an UNSPECIFIED anomeric centre, the production ensemble already samples both
    anomers of that ring size (ETKDG assigns unspecified centres randomly; verified in the anomer audit),
    so the alternatives are the other ring size(s) and the open chain."""
    inp = _canon(Chem.MolFromSmiles(smi))
    open_smi, opened = open_chain(smi)
    if opened:
        m = Chem.MolFromSmiles(smi)
        c = m.GetSubstructMatches(_HEMI)[0][0]
        spec = m.GetAtomWithIdx(c).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        ring_size = 5 if m.GetAtomWithIdx(c).IsInRingSize(5) else 6
        kind = "ring_defined" if spec else "ring_unspecified"
    else:
        kind = "open"
    closures = _ring_closures(open_smi)
    if not closures:
        return {"input": inp, "input_kind": "none" if not opened else kind, "alternatives": []}
    alts = []
    if opened:
        alts.append({"smiles": open_smi, "kind": "open", "ring_size": None})
    for s, rs, _, _ in closures:
        if s == inp:
            continue                                       # the input itself (defined anomer)
        if opened and kind == "ring_unspecified" and rs == ring_size:
            continue                                       # already in the input's sampled ensemble
        alts.append({"smiles": s, "kind": f"ring{rs}", "ring_size": rs})
    # de-duplicate (two OH can close the same ring only by symmetry)
    seen, uniq = set(), []
    for a in alts:
        if a["smiles"] not in seen:
            seen.add(a["smiles"]); uniq.append(a)
    return {"input": inp, "input_kind": kind, "alternatives": uniq}


def all_sugar_forms(smi):
    """Every defined ring-chain state (open + all ring anomers) -- for the population validation."""
    open_smi, _ = open_chain(smi)
    forms = [{"smiles": open_smi, "kind": "open", "ring_size": None}]
    for s, rs, _, _ in _ring_closures(open_smi):
        forms.append({"smiles": s, "kind": f"ring{rs}", "ring_size": rs})
    return forms


def _carbon_h(m):
    return tuple(a.GetTotalNumHs() for a in m.GetAtoms() if a.GetSymbol() == "C")


def heteroatom_tautomers(smi, cap=None):
    """Input + heteroatom-only prototropic tautomers (same charge, carbon H counts unchanged, defined
    stereocentres preserved). Ranked by RDKit tautomer score; input first."""
    m = Chem.MolFromSmiles(smi)
    inp = _canon(m)
    p = rdMolStandardize.CleanupParameters()
    p.tautomerRemoveSp3Stereo = False
    p.tautomerRemoveBondStereo = False
    p.maxTautomers = 2000
    te = rdMolStandardize.TautomerEnumerator(p)
    q0 = Chem.GetFormalCharge(m)
    ch0 = Chem.MolToSmiles(m)  # noqa
    ref_h = None
    # compare atom-mapped carbon H counts via canonical ranking-independent approach: use the enumerator's
    # products, which keep atom order of the input
    ref_h = _carbon_h(m)
    ref_st = sorted(Chem.FindMolChiralCenters(m, includeUnassigned=False, useLegacyImplementation=False))
    res = te.Enumerate(m)
    cands = []
    for t in res:
        if Chem.GetFormalCharge(t) != q0 or _carbon_h(t) != ref_h:
            continue
        st = sorted(Chem.FindMolChiralCenters(t, includeUnassigned=False, useLegacyImplementation=False))
        if st != ref_st:
            continue
        cands.append((-te.ScoreTautomer(t), _canon(t)))
    cands.sort()
    out = [inp] + [s for _, s in cands if s != inp]
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq if cap is None else uniq[:cap]


if __name__ == "__main__":
    tests = {
        "glucose_open": "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO",
        "fructose_open": "OCC(=O)[C@@H](O)[C@H](O)[C@H](O)CO",
        "ribose_furanose": "OC[C@H]1OC(O)[C@H](O)[C@@H]1O",
        "FBP": "O=P(O)(O)OC[C@H]1OC(O)(COP(=O)(O)O)[C@@H](O)[C@@H]1O",
        "S7P": "O=C(CO)[C@@H](O)[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)O",
        "KDG": "O=C(O)C(=O)C[C@H](O)[C@H](O)CO",
        "R1P": "O=P(O)(O)O[C@H]1O[C@H](CO)[C@@H](O)[C@H]1O",
        "GAP": "O=C[C@H](O)COP(=O)(O)O",
    }
    for k, s in tests.items():
        st = sugar_states(s)
        print(k, st["input_kind"], [(a["kind"], a["smiles"]) for a in st["alternatives"]])
    for s in ["Nc1ncnc2[nH]cnc12", "Nc1nc2[nH]cnc2c(=O)[nH]1", "NC(=O)c1cccnc1"]:
        print(s, heteroatom_tautomers(s))
