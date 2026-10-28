"""CPU inspection of declared stereochemical identity, without assigning isomers.

RDKit FindPotentialStereo recognizes tetrahedral atoms and double bonds. This
inspection is not proof that a molecule has no atropisomers or other unmodeled
states. Relative/racemic enhanced stereo groups need explicit pool semantics.
Atom indices refer to the parsed input SMILES, not canonical atom order.
"""
from rdkit import Chem, rdBase

POLICY_VERSION = '2026-10-12-declared-stereo-v2'


def _equivalent_phosphate_acid_oxygens(atom):
    """Find labile, same-isotope terminal oxygens at an ordinary phosphate.

    A localized P=O/P-OH drawing of a phosphate acid does not define persistent
    phosphorus chirality: proton transfer interchanges its nonbridging oxygens.
    This exception affects only stereo admission, not the owned QM microstate.
    Phosphorothioates, triesters and isotope-labelled distinguishable ligands
    are not covered. Explicit isotope-labelled hydrogens are not interchangeable.
    """
    if (atom.GetAtomicNum() != 15 or atom.GetDegree() != 4
            or any(n.GetAtomicNum() != 8 for n in atom.GetNeighbors())):
        return []
    terminal = []
    for oxygen in atom.GetNeighbors():
        if any(n.GetAtomicNum() == 1 for n in oxygen.GetNeighbors()):
            continue
        if oxygen.GetDegree() != 1:
            continue
        bond = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), oxygen.GetIdx())
        state = (bond.GetBondType(), oxygen.GetFormalCharge(), oxygen.GetTotalNumHs())
        if state in ((Chem.BondType.DOUBLE, 0, 0),
                     (Chem.BondType.SINGLE, -1, 0),
                     (Chem.BondType.SINGLE, 0, 1)):
            terminal.append(oxygen)
    # Equivalent isotopes are required; never erase isotope-induced chirality.
    for i, left in enumerate(terminal):
        for right in terminal[i + 1:]:
            if left.GetIsotope() == right.GetIsotope():
                return [left.GetIdx(), right.GetIdx()]
    return []


def inspect_stereochemistry(smiles):
    if not isinstance(smiles, str) or not smiles:
        raise ValueError('SMILES must be a nonempty string')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid SMILES for stereochemistry inspection')
    # Maps are identifiers, not physical substituent differences.
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    canonical = Chem.MolToSmiles(mol)
    resonance_normalized = []
    # A localized P=O / P-O(-) drawing does not make phosphate P chiral.
    # Use the charge-separated resonance form ONLY for stereo perception;
    # never alter the chemical input used for QM, protonation, or caching.
    # Preserve isotopes, substituents and total charge (including genuine
    # isotope-induced P chirality and chiral phosphorothioate diesters).
    for phosphorus in mol.GetAtoms():
        if phosphorus.GetAtomicNum() != 15 or phosphorus.GetDegree() != 4:
            continue
        oxygens = [o for o in phosphorus.GetNeighbors() if o.GetAtomicNum() == 8
                   and o.GetDegree() == 1 and o.GetTotalNumHs() == 0]
        if not any(o.GetFormalCharge() == -1 for o in oxygens):
            continue
        for oxygen in oxygens:
            bond = mol.GetBondBetweenAtoms(phosphorus.GetIdx(), oxygen.GetIdx())
            if bond.GetBondType() == Chem.BondType.DOUBLE and oxygen.GetFormalCharge() == 0:
                bond.SetBondType(Chem.BondType.SINGLE)
                phosphorus.SetFormalCharge(phosphorus.GetFormalCharge() + 1)
                oxygen.SetFormalCharge(-1)
                resonance_normalized.append(phosphorus.GetIdx())
    if resonance_normalized:
        Chem.SanitizeMol(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeAtomMaps=False))
    unresolved, specified, unsupported, prototropy_equivalent = [], [], [], []
    for stereo in Chem.FindPotentialStereo(mol, cleanIt=False, flagPossible=True):
        kind, index = str(stereo.type), int(stereo.centeredOn)
        record = {'kind': kind, 'index': index, 'specified': str(stereo.specified)}
        if kind == 'Atom_Tetrahedral':
            equivalent = _equivalent_phosphate_acid_oxygens(mol.GetAtomWithIdx(index))
            if equivalent:
                prototropy_equivalent.append({'phosphorus_index': index,
                                              'terminal_oxygen_indices': equivalent})
                continue
        if kind == 'Bond_Double':
            bond = mol.GetBondWithIdx(index)
            record['atom_indices'] = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            # Some RDKit versions report terminal C=N with two equivalent real
            # substituents at carbon. Such a bond has no two distinct E/Z forms.
            symmetric = False
            for atom, other in ((bond.GetBeginAtom(), bond.GetEndAtomIdx()),
                                (bond.GetEndAtom(), bond.GetBeginAtomIdx())):
                neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != other]
                if len(neighbors) == 2 and ranks[neighbors[0]] == ranks[neighbors[1]]:
                    symmetric = True
            if symmetric:
                continue
        if kind not in ('Atom_Tetrahedral', 'Bond_Double'):
            unsupported.append(record)
        elif stereo.specified != Chem.StereoSpecified.Specified:
            unresolved.append(record)
        else:
            specified.append(record)
    groups = [{'type': str(g.GetGroupType()), 'atom_indices': [a.GetIdx() for a in g.GetAtoms()]}
              for g in mol.GetStereoGroups() if g.GetGroupType() != Chem.StereoGroupType.STEREO_ABSOLUTE]
    issues = []
    if unresolved: issues.append('unspecified_stereochemistry')
    if unsupported: issues.append('unsupported_stereochemistry')
    if groups: issues.append('stereoisomer_pool_not_defined')
    return {'policy': POLICY_VERSION, 'rdkit_version': rdBase.rdkitVersion,
            'status': 'unresolved' if issues else 'no_unresolved_detected',
            'canonical_isomeric_smiles': canonical,
            'phosphate_resonance_normalized_for_perception': resonance_normalized,
            'phosphate_prototropy_equivalent_for_perception': prototropy_equivalent,
            'unresolved': unresolved, 'specified': specified, 'unsupported': unsupported,
            'relative_stereo_groups': groups, 'issues': issues,
            'scope': 'Declared tetrahedral and double-bond stereochemistry only; no isomer assignment, mixture inference or completeness claim.'}


def unresolved_species(species):
    """Return named state-definition problems for a validated species mapping."""
    records = {name: inspect_stereochemistry(entry[2]) for name, entry in species.items()}
    return {name: row for name, row in records.items() if row['issues']}
