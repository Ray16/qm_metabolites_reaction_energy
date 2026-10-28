"""Systematic reaction-level spectator truncation.

Given a reaction (reactant SMILES + product SMILES), build the SMALLEST QM model
that preserves the reactive center on both sides, replacing the conserved spectator
moiety (nucleotide tail, peptide backbone, ...) with a small methyl cap.

Why this is well-posed *at the reaction level*: we have BOTH sides in hand, so the
spectator is exactly the sub-structure that is atom-mapped and bonding-unchanged
across the reaction. Cutting it at the same point on both sides makes its energy AND
its conformer noise cancel in ΔG (validated by hand: redox 49->7 kJ noise; nucleotidyl
solved). This module automates the hand construction and adds machine guards.

Algorithm (deterministic):
  1. pair reactants<->products by maximum common substructure (greedy, MCS atom count)
  2. per pair, get the ATOM MAP from the shared MCS
  3. reaction center = {unmapped atoms} U {mapped atoms whose bonding changed}
  4. keep = reaction center grown by `radius` bonds, then RING-CLOSED (no partial rings);
     FragmentOnBonds at the boundary; cap the severed valence with methyl
Guards (all automated):
  A. balance      : truncated rxn is atom + charge + H balanced          (hard reject)
  B. consistency  : each removed fragment is identical on both sides       (else no cancel)
  C. sensitivity  : caller compares ΔG(radius) vs ΔG(radius+1) < tol       (hook: emit both)
  D. rigidity     : rotatable bonds in each reacting core                  (report, not gate)
  E. ring-closure : cuts fall ONLY on acyclic bonds (`_ring_close`)        (a priori structural)
                    -- a severed ring bond opens the ring + changes hybridization (cyclic lactone
                    -> acyclic methyl ester), a stably-wrong cut B/C cannot catch. Fixes the
                    aldose-dehydrogenase +33 kJ artifact (D-Glucose_t COC(O)C(C)O -> ring intact).

Cap-length (methyl at radius R vs the extra bond at R+1) IS the Me/Et sensitivity knob.
"""
from __future__ import annotations
from collections import Counter
from functools import lru_cache
from rdkit import Chem
from rdkit.Chem import rdFMCS, Descriptors
from rdkit.Chem import rdMolDescriptors as rdMD


# ---------------------------------------------------------------- MCS atom map
def _mcs(a, b, timeout=30):
    return rdFMCS.FindMCS(
        [a, b], ringMatchesRingOnly=True, completeRingsOnly=True, timeout=timeout,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder)


def mcs_atom_map(a, b):
    """Return (a_idx -> b_idx) correspondence for the largest common substructure."""
    res = _mcs(a, b)
    if res.numAtoms == 0:
        return {}, 0
    patt = Chem.MolFromSmarts(res.smartsString)
    ma = a.GetSubstructMatch(patt)
    mb = b.GetSubstructMatch(patt)
    return dict(zip(ma, mb)), res.numAtoms


# ------------------------------------------------------------- pairing species
@lru_cache(maxsize=4096)
def _pair_maps(reactants, products):
    """Cached greedy pairing as immutable ``(reactant index, product index, map)`` records."""
    R = [Chem.MolFromSmiles(s) for s in reactants]
    P = [Chem.MolFromSmiles(s) for s in products]
    scores = []
    for i, r in enumerate(R):
        for j, p in enumerate(P):
            amap, n = mcs_atom_map(r, p)
            scores.append((n, i, j, tuple(amap.items())))
    scores.sort(key=lambda row: row[:3], reverse=True)
    used_r, used_p, pairs = set(), set(), []
    for _, i, j, amap in scores:
        if i in used_r or j in used_p:
            continue
        used_r.add(i); used_p.add(j)
        pairs.append((i, j, amap))
    return tuple(pairs)


def pair_by_mcs(reactants, products):
    """Greedy max-MCS bipartite pairing. Returns list of (SMILES, mols, atom map).

    MCS is the dominant CPU cost in routing.  Cache immutable atom maps by the
    input SMILES and never recompute the winning maps after candidate scoring.
    """
    reactants, products = tuple(reactants), tuple(products)
    R = [Chem.MolFromSmiles(s) for s in reactants]
    P = [Chem.MolFromSmiles(s) for s in products]
    return [(reactants[i], products[j], R[i], P[j], dict(amap))
            for i, j, amap in _pair_maps(reactants, products)]


# ---------------------------------------------------------- reaction center
def reaction_center(a, amap, b):
    """Atoms in `a` at the reaction center: unmapped, or mapped-but-bonding-changed."""
    center = set()
    rev = amap  # a_idx -> b_idx
    for at in a.GetAtoms():
        i = at.GetIdx()
        if i not in rev:                       # unmapped -> reacting group
            center.add(i); continue
        # mapped: compare neighbor identity across the map
        a_nb_mapped = {rev[n.GetIdx()] for n in at.GetNeighbors() if n.GetIdx() in rev}
        has_unmapped_nb = any(n.GetIdx() not in rev for n in at.GetNeighbors())
        b_at = b.GetAtomWithIdx(rev[i])
        b_nb = {n.GetIdx() for n in b_at.GetNeighbors()}
        # bond broken/formed to a conserved atom, or a neighbor left the map
        if a_nb_mapped != (b_nb & set(rev.values())) or has_unmapped_nb:
            center.add(i)
    return center


def has_anomeric_ring_reaction_center(species_dict):
    """Return True when a changed bond is attached to an anomeric sugar carbon.

    A two-bond shell around an N/O-glycosidic reaction center can end inside the
    ribose substituents.  In particular, it can remove a conserved 5'-phosphate
    even though that charged group remains conformationally and electrostatically
    coupled to the reacting sugar.  Such a core is not locally converged.

    This is deliberately a topology test, not a reaction-name or database-ID
    rule: an anomeric center is a changed ring carbon adjacent to a ring oxygen.
    """
    items = list(species_dict.values())
    if any(abs(c) != 1 for c, _, _ in items):
        return False
    # Cheap rejection before the MCS work: require a ring carbon next to a ring
    # oxygen and an exocyclic N/O substituent, i.e. a possible hemiacetal/acetal
    # center.  Most metabolic reactions have no such atom.
    possible = False
    for _, _, smiles in items:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        ring_atoms = {i for ring in mol.GetRingInfo().AtomRings() for i in ring}
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() != "C":
                continue
            neighbors = list(atom.GetNeighbors())
            if (any(nb.GetSymbol() == "O" and nb.GetIdx() in ring_atoms for nb in neighbors)
                    and any(nb.GetSymbol() in ("N", "O") and nb.GetIdx() not in ring_atoms
                            for nb in neighbors)):
                possible = True
                break
        if possible:
            break
    if not possible:
        return False

    reactants = [s for c, _, s in items if c < 0]
    products = [s for c, _, s in items if c > 0]
    if len(reactants) != len(products):
        return False

    for _, _, reactant, product, amap in pair_by_mcs(reactants, products):
        for mol, atom_map, other in (
                (reactant, amap, product),
                (product, {v: k for k, v in amap.items()}, reactant)):
            ring_atoms = {i for ring in mol.GetRingInfo().AtomRings() for i in ring}
            for idx in reaction_center(mol, atom_map, other):
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() != "C" or idx not in ring_atoms:
                    continue
                if any(nb.GetSymbol() == "O" and nb.GetIdx() in ring_atoms
                       for nb in atom.GetNeighbors()):
                    return True
    return False


def truncation_radius(species_dict, default=2):
    """Choose the smallest chemically adequate reactive-core radius."""
    return max(int(default), 3) if has_anomeric_ring_reaction_center(species_dict) else int(default)


def grow(a, seed, radius, within=None):
    """BFS grow `seed` by `radius` bonds. If `within` given, stay inside that atom set."""
    keep = set(seed)
    frontier = set(seed)
    for _ in range(radius):
        nxt = set()
        for i in frontier:
            for nb in a.GetAtomWithIdx(i).GetNeighbors():
                k = nb.GetIdx()
                if within is not None and k not in within:
                    continue
                if k not in keep:
                    keep.add(k); nxt.add(k)
        frontier = nxt
    return keep


def _cut_close(mol, keep):
    """Expand `keep` until every bond crossing the cut is a single C-C bond whose KEPT carbon is sp3, so the
    H cap only turns R-CH2- into R-CH3 and never changes a functional group. Cutting other bonds rewrote the
    chemistry being scored: a C-O cut made a phosphoester a free H3PO4 (AMP -> H3PO4 in PPDK, adding a free-Pi
    species) or stripped a sugar's OH groups (F6P -> bare tetrahydrofuran); a C-C cut on a carbonyl or aryl
    carbon would turn a ketone into an aldehyde. Fixpoint, like _ring_close."""
    keep = set(keep)
    changed = True
    while changed:
        changed = False
        for i in list(keep):
            a = mol.GetAtomWithIdx(i)
            for nb in a.GetNeighbors():
                j = nb.GetIdx()
                if j in keep:
                    continue
                b = mol.GetBondBetweenAtoms(i, j)
                ok = (a.GetSymbol() == "C" and nb.GetSymbol() == "C"
                      and b.GetBondType() == Chem.BondType.SINGLE and not b.GetIsAromatic()
                      and a.GetHybridization() == Chem.HybridizationType.SP3)
                if not ok:
                    keep.add(j); changed = True
    return keep


def _close(mol, keep):
    """Ring closure and functional-group-preserving cut closure, to a joint fixpoint."""
    while True:
        new = _ring_close(mol, _cut_close(mol, keep))
        if new == keep:
            return keep
        keep = new


def _ring_close(mol, keep):
    """Expand `keep` so NO ring is partially included: if any ring shares an atom with keep,
    add the whole ring (iterating to a fixpoint to absorb fused systems). This guarantees the
    truncation cut falls only on ACYCLIC bonds. A severed ring bond cannot be faithfully
    methyl-capped -- it opens the ring and changes topology + hybridization (e.g. a pyranose
    hemiacetal -> an acyclic methyl-hemiacetal, and a cyclic lactone -> an acyclic methyl ester),
    which does NOT cancel in ΔG (the aldose-dehydrogenase +33 kJ artifact). Only rings NEAR the
    reaction center are touched (keep is radius-bounded), so distant spectator rings are untouched."""
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    keep = set(keep)
    changed = True
    while changed:
        changed = False
        for rs in rings:
            if (rs & keep) and not (rs <= keep):
                keep |= rs
                changed = True
    return keep


# ------------------------------------------------------------------- capping
def _removed_frags(smiles, keep):
    """SMILES of the pieces that get CUT AWAY (everything not in keep), for the
    cap-consistency guard. Independent of core capping."""
    m = Chem.MolFromSmiles(smiles)
    drop = [i for i in range(m.GetNumAtoms()) if i not in keep]
    if not drop:
        return set()
    # emit each dropped connected component as its own H-capped SMILES
    return {Chem.MolFragmentToSmiles(m, atomsToUse=drop)}


def truncate_species(smiles, keep):
    """Keep atom-set `keep`; H-fill the severed valences (cutting a C-C bond thus
    yields a methyl cap automatically). Returns (capped_smiles, removed_frag_set)."""
    m = Chem.MolFromSmiles(smiles)
    keep = {i for i in keep if 0 <= i < m.GetNumAtoms()}
    if not keep:                                      # empty reaction centre on this species:
        return None, set()                            # not truncatable (was a ValueError crash)
    capped = Chem.MolFragmentToSmiles(m, atomsToUse=sorted(keep))
    # round-trip to canonicalise + validate: a core that does not re-parse (e.g. a cut aromatic ring that
    # cannot be kekulized) is not a valid molecule -> not truncatable (was a crash downstream)
    cm = Chem.MolFromSmiles(capped)
    if cm is None:
        return None, set()
    return Chem.MolToSmiles(cm), _removed_frags(smiles, keep)


# --------------------------------------------------------------------- guards
def formula_charge(smiles):
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    f = Counter(a.GetSymbol() for a in m.GetAtoms())
    q = Chem.GetFormalCharge(m)
    return f, q


def check_balance(react_smis, prod_smis):
    """Atom + charge balance across a (already truncated) reaction. H included."""
    lf, lq = Counter(), 0
    for s in react_smis:
        f, q = formula_charge(s); lf += f; lq += q
    rf, rq = Counter(), 0
    for s in prod_smis:
        f, q = formula_charge(s); rf += f; rq += q
    atom_ok = lf == rf
    dH = rf.get("H", 0) - lf.get("H", 0)
    return dict(atom_balanced=atom_ok, charge_balanced=(lq == rq),
                dH=dH, dq=rq - lq, left=dict(lf), right=dict(rf))


def rotatable_in_core(smiles):
    return rdMD.CalcNumRotatableBonds(Chem.MolFromSmiles(smiles))


# ---------------------------------------------- pipeline preprocessing hook
def _full_nHplus(species_dict):
    """Net H+ released by the FULL reaction (from microspecies H/charge balance)."""
    Hr = Hp = qr = qp = 0
    for _, (c, q, s) in species_dict.items():
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        h = sum(a.GetTotalNumHs() for a in m.GetAtoms())
        if c < 0: Hr += -c * h; qr += -c * q
        else:     Hp += c * h;  qp += c * q
    return (Hr - Hp) if (Hr - Hp) == (qr - qp) else None


_THIOESTER_SM = Chem.MolFromSmarts("[CX3](=O)[SX2]")   # C(=O)-S thioester (acyl-CoA reaction centre)


def _count_thioesters(species):
    """Total thioester (C(=O)-S) bonds in a species dict, weighted by |coeff|."""
    n = 0
    for _, (c, q, s) in species.items():
        m = Chem.MolFromSmiles(s)
        if m is not None:
            n += abs(c) * len(m.GetSubstructMatches(_THIOESTER_SM))
    return n


def _truncation_degenerate(new):
    """True if the truncated cores form the SAME canonical multiset on both sides -> the reaction
    cancels to ΔG≡0 and the measured chemistry has been truncated away. This happens on acyl-transfer
    reactions (e.g. 3-oxoacid CoA-transferase rxn00290): an acyl-CoA is truncated down to its free acid,
    which is exactly the spectator acid on the OTHER side, so succinyl-CoA+acetoacetate -> succinate+
    acetoacetyl-CoA collapses to succinate+acetoacetate -> acetoacetate+succinate = 0. Reject such a cut
    so the caller falls back to full molecules (noisier, but not a spurious 0)."""
    def canon(smi):
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m else smi
    r = Counter(canon(s) for _, (c, q, s) in new.items() if c < 0)
    p = Counter(canon(s) for _, (c, q, s) in new.items() if c > 0)
    return r == p


def _truncation_invalid(species_full, new):
    """Reject a truncation that (a) collapses both sides to the same cores (ΔG≡0), or (b) DROPS a
    thioester bond -- the C(=O)-S is the reaction centre of every acyl-CoA reaction, so cutting it
    (succinyl-CoA -> succinate, or the v2 mangle succinyl-CoA -> butyrate / acetoacetyl-CoA -> methyl
    ketone) destroys exactly the chemistry being scored. Truncation may trim the CoA TAIL but must
    keep the thioester; if the thioester count drops, the cut hit the reaction centre -> reject."""
    if _truncation_degenerate(new):
        return True
    if _count_thioesters(new) < _count_thioesters(species_full):
        return True
    return False


def build_truncated_reaction(species_dict, radius=2, fg_cuts=False):
    """Convert a pipeline species dict {name:[coeff,charge,SMILES]} into its TRUNCATED
    reactive-core form for scoring. General preprocessing heuristic (no per-reaction tuning):
    removes the conserved spectator backbone so catastrophic cancellation + its conformer
    noise drop. Returns (new_species_dict, n_Hplus) or None if not cleanly truncatable (caller
    falls back to full molecules). Handles unit-coefficient reactions; multi-coeff -> None.

    QUALITY GUARD: truncation only removes CONSERVED spectators, so it must NOT change the
    reaction's net proton count. If truncated n_H+ != full n_H+, the cut was asymmetric / the
    reaction center was mis-detected (demonstrated: rxn00545/00216 invented n_H+=2 -> garbage
    -80 kJ). Such truncations are REJECTED -> fall back to full molecules."""
    items = list(species_dict.items())
    if any(abs(c) != 1 for _, (c, q, s) in items):
        return None                                   # multi-coeff: not handled -> fallback
    full_nH = _full_nHplus(species_dict)
    R = [(n, s) for n, (c, q, s) in items if c < 0]
    P = [(n, s) for n, (c, q, s) in items if c > 0]
    if len(R) != len(P):                              # unequal sides -> pairing ill-posed
        return None
    res = truncate_reaction([s for _, s in R], [s for _, s in P], radius=radius, fg_cuts=fg_cuts)
    if res.get("invalid"):
        return None
    # GUARDS A+B (computed by truncate_reaction, previously never enforced): the removed spectator must be
    # the SAME fragment multiset on both sides (else it does not cancel -- e.g. fragments differing in
    # stereochemistry), and the cores must balance in heavy atoms. H and charge are closed by n_H+, which
    # is checked below (check_balance's own atom/charge flags ignore the proton, so they are not used).
    if not res["consistent"]:
        return None
    heavy = lambda side: {k: v for k, v in res["balance"][side].items() if k != "H"}
    if heavy("left") != heavy("right"):
        return None
    caps = res["species"]
    if (sum(d["side"] == "reactant" for d in caps) != len(R)
            or sum(d["side"] == "product" for d in caps) != len(P)):
        return None
    def chg(smi):
        m = Chem.MolFromSmiles(smi); return Chem.GetFormalCharge(m) if m else None
    def nH(smi):
        m = Chem.MolFromSmiles(smi); return sum(a.GetTotalNumHs() for a in m.GetAtoms()) if m else None
    # Match each cap back to its ORIGINAL species by orig-SMILES, NOT by position: pair_by_mcs
    # reorders species relative to the input dict, so a positional zip(R, r_caps) mislabels the
    # cores (the ΔG sum is unaffected -- coeff+SMILES travel together -- but the logs were wrong,
    # which sent an earlier debug chasing a phantom cache collision). Pop matches to handle
    # duplicate SMILES on a side.
    r_pool, p_pool = list(R), list(P)
    new = {}
    Hr = Hp = qr = qp = 0
    for capd in caps:
        side, orig, cap = capd["side"], capd["orig"], capd["capped"]
        pool = r_pool if side == "reactant" else p_pool
        idx = next((k for k, (nm, smi) in enumerate(pool) if smi == orig), None)
        if idx is None:                                # cap's original not found -> ill-posed
            return None
        name, _ = pool.pop(idx)
        q = chg(cap); h = nH(cap)
        if q is None or h is None: return None
        if side == "reactant":
            new[name + "_t"] = [-1, int(q), cap]; Hr += h; qr += q
        else:
            new[name + "_t"] = [1, int(q), cap]; Hp += h; qp += q
    if r_pool or p_pool:                               # every original must be matched exactly once
        return None
    nHplus_H = Hr - Hp
    if nHplus_H != qr - qp:                            # truncated rxn not proton-consistent
        return None
    if full_nH is None or nHplus_H != full_nH:         # GUARD: truncation changed net H+ -> bad cut
        return None
    if _truncation_invalid(species_dict, new):         # GUARD: collapsed sides or dropped a thioester
        return None
    return new, int(nHplus_H)


# ------------------------------------------------------------ top-level driver
def truncate_reaction(reactants, products, radius=2, cap="C", fg_cuts=False):
    """Truncate every species; return per-species caps + guard report."""
    pairs = pair_by_mcs(reactants, products)
    out = {"radius": radius, "species": [], "removed": []}
    for r_smi, p_smi, R, P, amap in pairs:
        inv = {v: k for k, v in amap.items()}
        c_r = reaction_center(R, amap, P)
        c_p = reaction_center(P, inv, R)
        # keep on each side = reaction center grown by `radius`, then RING-CLOSED so no cut
        # falls on a ring bond (a methyl cap can't represent a severed ring -> topology change).
        close = _close if fg_cuts else _ring_close         # fg_cuts: functional-group-preserving cuts (A/B)
        keep_r = close(R, grow(R, c_r, radius))
        keep_p = close(P, grow(P, c_p, radius))
        # MIRROR the cut through the atom map so the removed spectator is IDENTICAL on both sides
        # (this is what makes it cancel in ΔG), then re-close, iterating to a fixpoint so the kept
        # core is both map-symmetric AND ring-complete on both sides.
        for _ in range(8):
            new_r = close(R, keep_r | {inv[j] for j in keep_p if j in inv})
            new_p = close(P, keep_p | {amap[i] for i in new_r if i in amap})
            if new_r == keep_r and new_p == keep_p:
                break
            keep_r, keep_p = new_r, new_p
        cr, rem_r = truncate_species(r_smi, keep_r)
        cp, rem_p = truncate_species(p_smi, keep_p)
        if cr is None or cp is None:                  # "not applicable": caller keeps full molecules
            out["invalid"] = f"no valid core for pair {r_smi} / {p_smi}"
            return out
        out["species"].append(dict(side="reactant", orig=r_smi, capped=cr,
                                    n_center=len(c_r), rot_core=rotatable_in_core(cr)))
        out["species"].append(dict(side="product", orig=p_smi, capped=cp,
                                   n_center=len(c_p), rot_core=rotatable_in_core(cp)))
        out["removed"].append(dict(reactant=r_smi, product=p_smi,
                                   removed_from_reactant=sorted(rem_r),
                                   removed_from_product=sorted(rem_p),
                                   consistent=(rem_r == rem_p)))
    caps_r = [s["capped"] for s in out["species"] if s["side"] == "reactant"]
    caps_p = [s["capped"] for s in out["species"] if s["side"] == "product"]
    out["balance"] = check_balance(caps_r, caps_p)
    # GLOBAL cap-consistency: spectators cancel across the WHOLE reaction, not per pair
    # (one reactant's atoms may split across several products). Compare the multiset of
    # removed fragments on each side after canonicalising each dropped piece to atoms.
    def norm(fr):
        c = Counter()
        for s in fr:
            for piece in s.split("."):
                m = Chem.MolFromSmiles(piece)
                if m is None:  # dropped fragment may be a bare radical; canon best-effort
                    c[piece] += 1
                else:
                    c[Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(m)))] += 1
        return c
    rem_r = Counter()
    rem_p = Counter()
    for d in out["removed"]:
        rem_r += norm(d["removed_from_reactant"])
        rem_p += norm(d["removed_from_product"])
    out["consistent"] = (rem_r == rem_p)
    out["removed_reactant_global"] = dict(rem_r)
    out["removed_product_global"] = dict(rem_p)
    return out


TESTS = {
    # rigid center, truncation KNOWN to work by hand (nucleotidyl / uridylyl transfer)
    "nucleotidyl_2.7.7.9": dict(  # UTP + Glc-1-P -> UDP-Glc + PPi
        reactants=[
            "O=c1ccn([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]2O)c(=O)[nH]1",
            "OC[C@H]1O[C@H](OP(=O)([O-])[O-])[C@H](O)[C@@H](O)[C@@H]1O"],
        products=[
            "OC[C@H]1O[C@@H](OP(=O)([O-])OP(=O)([O-])OC[C@H]2O[C@H](n3ccc(=O)[nH]c3=O)[C@@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O",
            "[O-]P(=O)([O-])OP(=O)([O-])[O-]"]),
    # rigid center (redox); full NAD+/NADH would be huge -> expect uridine-like tail cut
    "redox_MNA": dict(  # 1-methylnicotinamide+ + H- -> 1,4-dihydro (proxy w/ Me already)
        reactants=["C[n+]1cccc(C(N)=O)c1"],
        products=["O=C(N)C1=CN(C)C=CC1"]),
    # floppy sugar-sugar center: truncation EXPECTED to underperform (stress case)
    "glycosyl": dict(  # UDP-Glc + Fructose -> UDP + Sucrose
        reactants=[
            "OC[C@H]1O[C@@H](OP(=O)([O-])OP(=O)([O-])OC[C@H]2O[C@H](n3ccc(=O)[nH]c3=O)[C@@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O",
            "OC[C@H]1OC(O)(CO)[C@@H](O)[C@@H]1O"],
        products=[
            "OC[C@H]1O[C@H](n2ccc(=O)[nH]c2=O)[C@@H](O)[C@H]1OP(=O)([O-])OP(=O)([O-])O",
            "OC[C@H]1O[C@@H](O[C@]2(CO)O[C@H](CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"]),
}

if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else None
    radii = [int(x) for x in sys.argv[2:]] or [2, 3]
    for name, rx in TESTS.items():
        if which and which != name:
            continue
        print(f"\n########## {name} ##########")
        for R in radii:
            res = truncate_reaction(rx["reactants"], rx["products"], radius=R)
            print(f"  --- radius {R} ---")
            for s in res["species"]:
                print(f"    {s['side']:8s} rot={s['rot_core']} c={s['n_center']:2d}  {s['capped']}")
            b = res["balance"]
            print(f"    balance atom={b['atom_balanced']} charge={b['charge_balanced']} "
                  f"dH={b['dH']} dq={b['dq']}   cap-consistent={res['consistent']}")
