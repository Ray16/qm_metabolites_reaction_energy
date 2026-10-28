"""Enumerate a narrow fixed-charge ring-N proton equilibrium from structure.

The supported move is [nH]-c(NH2)-n <-> n-c(NH2)-[nH] in an aromatic
six-membered pyrimidinone ring with exactly one carbonyl and two ring nitrogens.
It preserves every bond, heavy atom, formal charge and declared stereocenter.
This does not enumerate lactim states, C-H tautomers or different proton counts.
"""
from dataclasses import dataclass
import hashlib

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

POLICY_VERSION = '2026-10-21-amino-pyrimidinone-ring-protons-v1'
MAX_STATES = 16
_PATH = Chem.MolFromSmarts('[nH1+0D2]:[c+0](-[NH2+0X3]):[nH0+0D2]')


@dataclass(frozen=True)
class RingProtonStates:
    reference: str
    states: tuple[str, ...]
    complete: bool
    reason: str | None = None

    @property
    def applicable(self):
        return len(self.states) > 1 or not self.complete

    @property
    def identity(self):
        return hashlib.sha256(('\n'.join(self.states)).encode()).hexdigest()


def _canonical(molecule):
    molecule = Chem.RemoveHs(Chem.Mol(molecule))
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def enumerate_ring_protons(smiles, charge, *, maximum=MAX_STATES):
    """Return every reachable supported ring-proton state, or an incomplete result.

    The sorted state set/identity is independent of which member was drawn first.
    The cap never licenses a partial partition. No measured equilibrium or
    reaction annotation/label participates in generation or selection.
    """
    if type(maximum) is not int or maximum < 1:
        raise ValueError('Ring-proton state cap must be a positive integer')
    molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError('Ring-proton enumeration requires one valid molecular state')
    if isinstance(charge, bool) or charge != Chem.GetFormalCharge(molecule):
        raise ValueError('Declared ring-proton reference charge disagrees with its SMILES')
    reference = _canonical(molecule)
    if charge != 0 or any(atom.GetIsotope() or atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        return RingProtonStates(reference, (reference,), True)
    formula = rdMolDescriptors.CalcMolFormula(molecule)
    found = {reference}
    pending = [reference]
    while pending:
        current = Chem.MolFromSmiles(pending.pop())
        rings = current.GetRingInfo().AtomRings()
        for donor, center, amino, acceptor in current.GetSubstructMatches(_PATH):
            supported = False
            for ring in rings:
                if len(ring) != 6 or not {donor, center, acceptor} <= set(ring):
                    continue
                if sum(current.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in ring) != 2:
                    continue
                oxo_carbons = [i for i in ring if any(
                    bond.GetBondType() == Chem.BondType.DOUBLE
                    and bond.GetOtherAtom(current.GetAtomWithIdx(i)).GetAtomicNum() == 8
                    for bond in current.GetAtomWithIdx(i).GetBonds())]
                if len(oxo_carbons) == 1 and any(
                    current.GetBondBetweenAtoms(oxo_carbons[0], nitrogen) is not None
                    for nitrogen in (donor, acceptor)):
                    supported = True
                    break
            if not supported:
                continue
            changed = Chem.RWMol(current)
            changed.GetAtomWithIdx(donor).SetNumExplicitHs(0)
            changed.GetAtomWithIdx(donor).SetNoImplicit(True)
            changed.GetAtomWithIdx(acceptor).SetNumExplicitHs(1)
            changed.GetAtomWithIdx(acceptor).SetNoImplicit(True)
            candidate = changed.GetMol()
            try:
                Chem.SanitizeMol(candidate)
            except (ValueError, RuntimeError):
                return RingProtonStates(reference, tuple(sorted(found)), False, 'supported ring-proton construction failed')
            if (Chem.GetFormalCharge(candidate) != charge
                    or rdMolDescriptors.CalcMolFormula(candidate) != formula):
                return RingProtonStates(reference, tuple(sorted(found)), False, 'ring-proton state changed formula or charge')
            canonical = _canonical(candidate)
            if canonical in found:
                continue
            if len(found) >= maximum:
                return RingProtonStates(reference, tuple(sorted(found)), False, 'ring-proton state enumeration cap exceeded')
            found.add(canonical)
            pending.append(canonical)
    return RingProtonStates(reference, tuple(sorted(found)), True)
