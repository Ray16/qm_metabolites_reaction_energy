"""Shared species/reaction helpers: canonical SMILES, water detection, element + charge balance.

One definition of each, used by the pipeline, the routing gates and the anchors, so that the same
molecule written differently (water as "O", "[OH2]" or "[H]O[H]") is always treated the same, and every
routing step can be checked against the one invariant a rewritten reaction must keep: element and charge
balance with the proton count carried in n_H+.

Coefficients may be fractional (ModelSEED writes e.g. 0.5 O2); nothing here truncates them to int.
"""
from collections import Counter
from functools import lru_cache

from rdkit import Chem

_TOL = 1e-9


@lru_cache(maxsize=None)
def canonical(smi):
    """Canonical SMILES, or None if RDKit cannot parse it."""
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def is_water(smi, q=0):
    """True for a neutral water molecule however it is written ("O", "[OH2]", "[H]O[H]", ...)."""
    return q == 0 and canonical(smi) == "O"


@lru_cache(maxsize=None)
def formula_charge(smi):
    """(element -> count incl. H, formal charge) for a SMILES, or (None, None) if unparseable."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    c = Counter()
    for a in m.GetAtoms():
        c[a.GetSymbol()] += 1
        c["H"] += a.GetTotalNumHs()
    return dict(c), Chem.GetFormalCharge(m)


def reaction_residual(species, n_hplus=0):
    """Element and charge residual of Σ coeff·species + n_H+·H+ (coeff > 0 = product).

    Returns ({element: residual}, charge_residual), both empty/zero for a balanced reaction, or
    (None, None) if any SMILES does not parse. The charge used is the SMILES formal charge, so a species
    whose declared charge disagrees with its SMILES shows up as a "declared_charge_mismatch" entry."""
    net, q_net = Counter(), 0.0
    for coeff, q, smi in species.values():
        f, fq = formula_charge(smi)
        if f is None:
            return None, None
        for el, k in f.items():
            net[el] += coeff * k
        q_net += coeff * fq
        if fq != q:                                   # declared charge must match the structure
            net["declared_charge_mismatch"] += 1
    net["H"] += n_hplus
    q_net += n_hplus
    return {el: v for el, v in net.items() if abs(v) > _TOL}, q_net


def is_balanced(species, n_hplus=0):
    res, q = reaction_residual(species, n_hplus)
    return res is not None and not res and abs(q) <= _TOL


def fmt_num(x):
    """Signed compact number for logs: integral values without decimals (works for fractional coeffs)."""
    return f"{x:+g}"
