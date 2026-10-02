"""Independent checks for the pKa rule table."""

from rdkit import Chem


_BASIC = Chem.MolFromSmarts(
    "[NX3,NX4+;!$(N-C=[O,N,S]);!$(N-a);!$(N=*);!$(N#*);!$(N-[S,P]=O)]"
)
_AROMATIC_N = Chem.MolFromSmarts("[n;H0;X2]")


def is_acid_only(molecule):
    """Whether a molecule has an unambiguous acid-only binding polynomial."""
    if any(atom.GetFormalCharge() > 0 for atom in molecule.GetAtoms()):
        return False
    return (
        not molecule.HasSubstructMatch(_BASIC)
        and not molecule.HasSubstructMatch(_AROMATIC_N)
    )

