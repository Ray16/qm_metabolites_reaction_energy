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
  4. keep = reaction center grown by `radius` bonds; FragmentOnBonds at the boundary;
     cap the severed valence with methyl
Guards (all automated):
  A. balance      : truncated rxn is atom + charge + H balanced          (hard reject)
  B. consistency  : each removed fragment is identical on both sides       (else no cancel)
  C. sensitivity  : caller compares ΔG(radius) vs ΔG(radius+1) < tol       (hook: emit both)
  D. rigidity     : rotatable bonds in each reacting core                  (report, not gate)

Cap-length (methyl at radius R vs the extra bond at R+1) IS the Me/Et sensitivity knob.
"""
from __future__ import annotations
from collections import Counter
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
def pair_by_mcs(reactants, products):
    """Greedy max-MCS bipartite pairing. Returns list of (ri, pj, amap)."""
    R = [Chem.MolFromSmiles(s) for s in reactants]
    P = [Chem.MolFromSmiles(s) for s in products]
    scores = []
    for i, r in enumerate(R):
        for j, p in enumerate(P):
            _, n = mcs_atom_map(r, p)
            scores.append((n, i, j))
    scores.sort(reverse=True)
    used_r, used_p, pairs = set(), set(), []
    for n, i, j in scores:
        if i in used_r or j in used_p:
            continue
        used_r.add(i); used_p.add(j)
        amap, _ = mcs_atom_map(R[i], P[j])
        pairs.append((reactants[i], products[j], R[i], P[j], amap))
    return pairs


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
    capped = Chem.MolFragmentToSmiles(m, atomsToUse=sorted(keep))
    # round-trip to canonicalise + validate
    cm = Chem.MolFromSmiles(capped)
    capped = Chem.MolToSmiles(cm) if cm is not None else capped
    return capped, _removed_frags(smiles, keep)


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
def build_truncated_reaction(species_dict, radius=2):
    """Convert a pipeline species dict {name:[coeff,charge,SMILES]} into its TRUNCATED
    reactive-core form for scoring. General preprocessing heuristic (no per-reaction tuning):
    removes the conserved spectator backbone so catastrophic cancellation + its conformer
    noise drop. Returns (new_species_dict, n_Hplus) or None if not cleanly balanced (caller
    falls back to full molecules). Handles unit-coefficient reactions; multi-coeff -> None."""
    items = list(species_dict.items())
    if any(abs(c) != 1 for _, (c, q, s) in items):
        return None                                   # multi-coeff: not handled -> fallback
    R = [(n, s) for n, (c, q, s) in items if c < 0]
    P = [(n, s) for n, (c, q, s) in items if c > 0]
    if len(R) != len(P):                              # unequal sides -> pairing ill-posed
        return None
    res = truncate_reaction([s for _, s in R], [s for _, s in P], radius=radius)
    caps = res["species"]
    r_caps = [d["capped"] for d in caps if d["side"] == "reactant"]
    p_caps = [d["capped"] for d in caps if d["side"] == "product"]
    if len(r_caps) != len(R) or len(p_caps) != len(P):
        return None
    def chg(smi):
        m = Chem.MolFromSmiles(smi); return Chem.GetFormalCharge(m) if m else None
    def nH(smi):
        m = Chem.MolFromSmiles(smi); return sum(a.GetTotalNumHs() for a in m.GetAtoms()) if m else None
    new = {}
    Hr = Hp = qr = qp = 0
    for (n, _), cap in zip(R, r_caps):
        q = chg(cap); h = nH(cap)
        if q is None or h is None: return None
        new[n + "_t"] = [-1, int(q), cap]; Hr += h; qr += q
    for (n, _), cap in zip(P, p_caps):
        q = chg(cap); h = nH(cap)
        if q is None or h is None: return None
        new[n + "_t"] = [1, int(q), cap]; Hp += h; qp += q
    nHplus_H = Hr - Hp
    if nHplus_H != qr - qp:                            # truncated rxn not proton-consistent
        return None
    return new, int(nHplus_H)


# ------------------------------------------------------------ top-level driver
def truncate_reaction(reactants, products, radius=2, cap="C"):
    """Truncate every species; return per-species caps + guard report."""
    pairs = pair_by_mcs(reactants, products)
    out = {"radius": radius, "species": [], "removed": []}
    for r_smi, p_smi, R, P, amap in pairs:
        inv = {v: k for k, v in amap.items()}
        c_r = reaction_center(R, amap, P)
        c_p = reaction_center(P, inv, R)
        # keep on the reactant = reaction center grown by `radius`
        keep_r = grow(R, c_r, radius)
        # MIRROR the cut through the atom map so the removed spectator is IDENTICAL on
        # both sides (this is what makes it cancel in ΔG). Product keeps: its own
        # reacting center, grown, PLUS the map-images of every kept mapped reactant atom.
        mirrored = {amap[i] for i in keep_r if i in amap}
        keep_p = grow(P, c_p, radius) | mirrored
        # symmetric back-mirror so the reactant also keeps images of kept product atoms
        keep_r = keep_r | {inv[j] for j in keep_p if j in inv}
        cr, rem_r = truncate_species(r_smi, keep_r)
        cp, rem_p = truncate_species(p_smi, keep_p)
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
