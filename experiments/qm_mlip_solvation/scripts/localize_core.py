"""GENERAL reactive-core localization by conserved-substructure capping (no per-species detectors).

The one rule behind spectator-truncation, the ring-cofactor fix, and thiol-truncation:
  find the CONSERVED substructure of a transformation (the MCS between a reactant and its paired
  product, matched with EXACT bond orders so a redox change breaks the match), keep the REACTIVE
  part (atoms not in the conserved MCS) plus a `radius`-bond shell, and replace the conserved distal
  scaffold with a uniform methyl cap. Because the same conserved scaffold is capped identically on
  both sides, it cancels in ΔG -- an isodesmic localization, experiment-free.

Reproduces (target, no cofactor names):
  NAD(P)+/NAD(P)H  -> the nicotinamide ring model   (the ring differs in bond order -> not in MCS ->
                      reactive; the ADP-ribose-phosphate tail is the conserved MCS -> capped)
  GSH/GSSG         -> the capped cysteine            (S-H vs S-S differs -> reactive; peptide tail
                      is conserved -> capped)
  phosphoryl transfer, etc. -> the reactive core     (backbone conserved -> capped)

This is a PROTOTYPE localizer meant to test whether ONE general rule subsumes the hand-coded fixes.
"""
from rdkit import Chem
from rdkit.Chem import rdFMCS


def _grow(mol, seed, radius):
    keep = set(seed)
    frontier = set(seed)
    for _ in range(radius):
        nxt = set()
        for i in frontier:
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                if nb.GetIdx() not in keep:
                    keep.add(nb.GetIdx()); nxt.add(nb.GetIdx())
        frontier = nxt
    return keep


def _cap_fragment(smi, keep):
    """Keep `keep` atoms; every bond crossing the boundary is cut and the kept side capped with a
    methyl carbon. Returns capped-fragment SMILES, or None on failure."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    keep = {i for i in keep if 0 <= i < m.GetNumAtoms()}
    if not keep:
        return None
    rw = Chem.RWMol(m)
    # find boundary bonds (one atom kept, one dropped)
    caps = []
    for b in m.GetBonds():
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (a in keep) != (c in keep):
            caps.append(a if a in keep else c)
    # drop non-kept atoms (highest index first so indices stay valid)
    drop = sorted((i for i in range(m.GetNumAtoms()) if i not in keep), reverse=True)
    for i in drop:
        rw.RemoveAtom(i)
    # remap kept-atom old->new indices
    old2new, n = {}, 0
    for i in range(m.GetNumAtoms()):
        if i in keep:
            old2new[i] = n; n += 1
    # add a methyl cap on each boundary atom
    for old in caps:
        cidx = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(old2new[old], cidx, Chem.BondType.SINGLE)
    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def localize_pair(r_smi, p_smi, radius=2, mcs_timeout=20):
    """Localize one reactant->product transformation. Returns (r_core_smi, p_core_smi) or None."""
    R, P = Chem.MolFromSmiles(r_smi), Chem.MolFromSmiles(p_smi)
    if R is None or P is None:
        return None
    res = rdFMCS.FindMCS([R, P], bondCompare=rdFMCS.BondCompare.CompareOrderExact,
                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                         matchValences=False, ringMatchesRingOnly=True,
                         completeRingsOnly=True, timeout=mcs_timeout)
    if res.canceled or res.numAtoms == 0:
        return None
    q = Chem.MolFromSmarts(res.smartsString)
    rm, pm = R.GetSubstructMatch(q), P.GetSubstructMatch(q)
    if not rm or not pm:
        return None
    r_react = set(range(R.GetNumAtoms())) - set(rm)      # atoms whose bonding changed
    p_react = set(range(P.GetNumAtoms())) - set(pm)
    if not r_react and not p_react:
        return (r_smi, p_smi)                             # nothing changed (spectator) -> unchanged
    r_core = _cap_fragment(r_smi, _grow(R, r_react, radius))
    p_core = _cap_fragment(p_smi, _grow(P, p_react, radius))
    if r_core is None or p_core is None:
        return None
    return (r_core, p_core)


if __name__ == "__main__":
    # test: does the general rule reproduce the nicotinamide ring on NAD (no cofactor name)?
    NAD = "NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1"
    NADH = "NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1"
    out = localize_pair(NAD, NADH, radius=2)
    print("NAD+  core:", out[0] if out else None)
    print("NADH  core:", out[1] if out else None)
    print("(target ~ N-capped nicotinamide ring, tail replaced by methyl)")
