"""Global-map reaction truncation — extends truncate.py to the 20% of TECRDB reactions the
1:1-MCS pairing refuses outright (BEFORE trying): multi-coefficient (e.g. 2 dADP) and
unequal-side (A -> B + C atom splits like NTP -> NDP + PPi, or folate cyclohydrolase).

Idea: don't pair species 1:1. EXPAND multi-coeff to unit species, then COMBINE all reactants
into one super-molecule and all products into another, and take ONE global MCS. That conserved
substructure is the reaction's spectator scaffold across the WHOLE reaction (splits and merges
included). Reaction center = atoms outside the global conserved map or with changed bonding;
truncate each ORIGINAL species to its own atoms within `radius` of that center, cap severed bonds.
Reuses truncate.py's grow / capping / balance / n_H+ guards unchanged.

This is the MCS-flavoured stand-in for a full atom-mapper (RXNMapper): cheaper, deterministic,
no ML dep. Falls back to None (caller uses full molecules) whenever the global map is too small,
the truncation is unbalanced, or it changes the net proton count -- same safety contract as v1.
"""
from __future__ import annotations
from collections import Counter
from rdkit import Chem
from rdkit.Chem import rdFMCS
import truncate as T


def _expand_multicoeff(species_dict):
    """[(name, side(+1/-1), q, smi)] with every |coeff|>1 species duplicated to unit copies."""
    out = []
    for name, (c, q, smi) in species_dict.items():
        side = 1 if c > 0 else -1
        for k in range(abs(int(c))):
            out.append((f"{name}#{k}" if abs(c) > 1 else name, side, q, smi))
    return out


def _combine(smis):
    """Combined mol of a dot-joined SMILES + list of (species_index, local_atom_idx) per global atom.
    RDKit preserves atom order across '.' so offsets map global->(species,local)."""
    mol = Chem.MolFromSmiles(".".join(smis))
    if mol is None:
        return None, None
    offsets, base = [], 0
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None, None
        for local in range(m.GetNumAtoms()):
            offsets.append((i, local))
        base += m.GetNumAtoms()
    if len(offsets) != mol.GetNumAtoms():
        return None, None
    return mol, offsets


def _global_map(superR, superP, timeout=20):
    res = rdFMCS.FindMCS([superR, superP], timeout=timeout,
                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareOrder,
                         ringMatchesRingOnly=True, completeRingsOnly=True)
    if res.numAtoms == 0:
        return {}, 0
    patt = Chem.MolFromSmarts(res.smartsString)
    ma = superR.GetSubstructMatch(patt)
    mb = superP.GetSubstructMatch(patt)
    return dict(zip(ma, mb)), res.numAtoms


def build_truncated_reaction_v2(species_dict, radius=2, min_conserved_frac=0.15):
    """Same contract as truncate.build_truncated_reaction: -> (new_species, n_Hplus) or None.
    Handles multi-coeff + unequal-sides via a single global MCS over combined molecules."""
    units = _expand_multicoeff(species_dict)
    R = [(n, q, s) for (n, side, q, s) in units if side < 0]
    P = [(n, q, s) for (n, side, q, s) in units if side > 0]
    if not R or not P:
        return None
    full_nH = T._full_nHplus(species_dict)
    if full_nH is None:
        return None
    superR, offR = _combine([s for _, _, s in R])
    superP, offP = _combine([s for _, _, s in P])
    if superR is None or superP is None:
        return None
    amap, nmcs = _global_map(superR, superP)
    if nmcs < min_conserved_frac * max(superR.GetNumAtoms(), superP.GetNumAtoms()):
        return None                                       # too little conserved -> not truncatable
    inv = {v: k for k, v in amap.items()}
    cR = T.reaction_center(superR, amap, superP)
    cP = T.reaction_center(superP, inv, superR)
    keepR = T.grow(superR, cR, radius)
    keepP = T.grow(superP, cP, radius) | {amap[i] for i in keepR if i in amap}
    keepR = keepR | {inv[j] for j in keepP if j in inv}

    def per_species(smis, offs, keep_global):
        """Split a global keep-set back to each original species; truncate+cap each.
        Returns (caps, removed_fragment_multiset) or (None, None)."""
        by_sp = {}
        for g, (si, local) in enumerate(offs):
            if g in keep_global:
                by_sp.setdefault(si, set()).add(local)
        caps = []
        removed = Counter()
        for si, smi in enumerate(smis):
            keep_local = by_sp.get(si, set())
            if not keep_local:
                return None, None                         # a whole species dropped -> ill-posed here
            cap, rem = T.truncate_species(smi, keep_local)
            cm = Chem.MolFromSmiles(cap)
            if cm is None:
                return None, None
            caps.append(cap)
            for fr in rem:                                # canonicalise each dropped piece
                for piece in fr.split("."):
                    pm = Chem.MolFromSmiles(piece)
                    removed[Chem.MolToSmiles(pm) if pm else piece] += 1
        return caps, removed

    r_caps, rem_r = per_species([s for _, _, s in R], offR, keepR)
    p_caps, rem_p = per_species([s for _, _, s in P], offP, keepP)
    if r_caps is None or p_caps is None:
        return None
    # CONSISTENCY GUARD: the removed spectator must be the SAME multiset of fragments on both sides,
    # else it does not cancel in ΔG (balance + n_H+ can pass yet the cut be asymmetric -> garbage:
    # rxn00065 barely shrank but flipped +25 -> -27 with a non-cancelling cut). Reject -> fall back.
    if rem_r != rem_p:
        return None

    new = {}
    Hr = Hp = qr = qp = 0
    def q_h(smi):
        m = Chem.MolFromSmiles(smi)
        return (Chem.GetFormalCharge(m), sum(a.GetTotalNumHs() for a in m.GetAtoms())) if m else (None, None)
    for (n, _, _), cap in zip(R, r_caps):
        q, h = q_h(cap)
        if q is None: return None
        new[n + "_t"] = [-1, int(q), cap]; Hr += h; qr += q
    for (n, _, _), cap in zip(P, p_caps):
        q, h = q_h(cap)
        if q is None: return None
        new[n + "_t"] = [1, int(q), cap]; Hp += h; qp += q
    nHplus_H = Hr - Hp
    if nHplus_H != qr - qp:                                # not proton-consistent
        return None
    if nHplus_H != full_nH:                                # GUARD: truncation changed net H+ -> bad cut
        return None
    # NOTE: balance + n_H+ + removed-fragment-consistency are all NECESSARY but not SUFFICIENT --
    # rxn00065 passes them yet flips +25 -> -27 (a cut that touches the reactive context). The
    # RIGOROUS, reaction-agnostic validity test is RADIUS-SENSITIVITY: a true spectator removal
    # leaves ΔG invariant to the cut radius, so the caller should score at radius R and R+1 and
    # trust the truncation only when |ΔΔG| < tol (truncate.py guard C). Implemented in the pipeline
    # via TRUNC_VALIDATE, not as a tuned structural threshold here.
    return new, int(nHplus_H)
