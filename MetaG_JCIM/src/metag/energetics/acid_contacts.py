"""Structural helpers for the optional inter-acid hydrogen-bond diagnostic."""
import numpy as np


def acid_groups(template):
    """{atom idx: group id} for the acidic O-H hydrogens' oxygens and all oxygens of every acid group
    (each P atom and each carboxyl C is its own group) of an H-explicit RDKit molecule."""
    group_of_o, donors = {}, []
    # a P-O-P chain (pyro/triphosphate) is ONE acid group: the P-OH...O=P contacts between neighbouring
    # phosphoryls of a chain are intrinsic to it; only contacts between DISTINCT acid units are excluded.
    chain = {}
    for a in template.GetAtoms():
        if a.GetSymbol() == "P" and a.GetIdx() not in chain:
            stack, comp = [a.GetIdx()], []
            while stack:
                x = stack.pop()
                if x in chain:
                    continue
                chain[x] = a.GetIdx(); comp.append(x)
                for o in template.GetAtomWithIdx(x).GetNeighbors():
                    for y in o.GetNeighbors():
                        if y.GetSymbol() == "P" and y.GetIdx() not in chain:
                            stack.append(y.GetIdx())
    for a in template.GetAtoms():
        if a.GetSymbol() == "P" or (a.GetSymbol() == "C" and is_carboxyl(a)):
            gid = chain.get(a.GetIdx(), a.GetIdx())
            for o in a.GetNeighbors():
                if o.GetSymbol() != "O":
                    continue
                if a.GetSymbol() == "P" and any(n.GetSymbol() == "P" and n.GetIdx() != a.GetIdx()
                                                 for n in o.GetNeighbors()):
                    continue                                 # bridging P-O-P oxygen: shared, skip
                group_of_o[o.GetIdx()] = gid
                for h in o.GetNeighbors():
                    if h.GetSymbol() == "H":
                        donors.append((h.GetIdx(), gid))
    return group_of_o, donors


def is_carboxyl(c):
    """Carboxylic acid carbon: one C=O and one O-H oxygen (not a hemiacetal/ester carbon)."""
    os_ = [n for n in c.GetNeighbors() if n.GetSymbol() == "O"]
    if len(os_) != 2:
        return False
    mol = c.GetOwningMol()
    dbl = [o for o in os_ if mol.GetBondBetweenAtoms(c.GetIdx(), o.GetIdx()).GetBondTypeAsDouble() == 2]
    oh = [o for o in os_ if any(h.GetSymbol() == "H" for h in o.GetNeighbors())]
    return len(dbl) == 1 and len(oh) == 1


def has_interacid_hbond(atoms, groups, cutoff):
    group_of_o, donors = groups
    pos = atoms.get_positions()
    for h, g in donors:
        for o, go in group_of_o.items():
            if go != g and np.linalg.norm(pos[h] - pos[o]) < cutoff:
                return True
    return False


