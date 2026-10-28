"""Shared chemical-domain boundaries for pKa providers (no inference/training imports)."""

class UnsupportedSiteError(ValueError):
    """The selected model has no parameterization for a chemical environment."""


def validate_phosphoryl_domain(mol, pa):
    """Reject phosphonate/phosphite/etc. rather than borrowing phosphate pKas.

    Both providers currently describe tetra-coordinate P(V) with one terminal
    P=O and O substituents, or one N substituent (table provider only).
    This is a model-domain check, not a chemical-validity judgment.
    """
    neighbors = list(pa.GetNeighbors())
    oxo = [n for n in neighbors if n.GetSymbol() == "O" and n.GetDegree() == 1
           and mol.GetBondBetweenAtoms(pa.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2]
    if (len(neighbors) != 4 or len(oxo) != 1 or pa.GetFormalCharge() != 0
            or any(n.GetSymbol() not in {"O", "N"} for n in neighbors)
            or sum(n.GetSymbol() == "N" for n in neighbors) > 1
            or any(mol.GetBondBetweenAtoms(pa.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() != 1
                   for n in neighbors if n not in oxo)):
        raise UnsupportedSiteError(
            f"unsupported phosphoryl environment at atom {pa.GetIdx()}: "
            "no pKa parameters for this P coordination/substitution")

