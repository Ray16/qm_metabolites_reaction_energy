"""Ring-chain (lactam/hemiaminal) equilibrium of keto/aldehyde species with an intramolecular amide N-H.

A ketone/aldehyde carbonyl that can close a 5- or 6-membered ring with an amide N-H of the same molecule
exists in water as the cyclic hemiaminal (a hydroxy-lactam), e.g. 2-oxoglutaramate is 99.7% cyclic
2-hydroxy-5-oxoproline at pH 7.2 (PMC4020999; ΔG(open->ring) = -14.4 kJ/mol). The biochemical species is
the equilibrium mixture, so its G is the Boltzmann sum over the open form and every ring form:
    G_eff = -RT ln[exp(-G_open/RT) + Σ_ring exp(-G_ring/RT)]
Ring forms are isomers (no water, no proton), so no stoichiometry changes. Both configurations of the new
ring stereocentre are enumerated as separate states (diastereomers in chiral molecules, an equal-G
enantiomer pair otherwise).

EXPERIMENTAL, NOT ADOPTED (2026-10-06): development was abandoned with this code
present. RING_CHAIN remains disabled by default. The neutral QM calculation
gave -13.3 kJ/mol for one ring closure, but that agreement did not survive the
production pKa transformation; it does not validate the aqueous estimator.
Alcohol O-H nucleophiles (hemiacetals: open-chain sugars) are excluded following
the failed population test recorded in analysis/sugar_forms/results.json.
"""
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import rdmolops

# ketone / aldehyde carbon (not acid, ester, amide, thioester: those carbonyl carbons bear a heteroatom),
# ACYCLIC only: a ring carbonyl would close a fused bicycle, outside the validated (open-chain) domain.
_CARBONYL = Chem.MolFromSmarts("[CX3;!R;!$(C[!#6;!#1]);$(C=[OX1])]=[OX1]")
# amide N-H (primary or secondary amide nitrogen bearing at least one H)
_AMIDE_NH = Chem.MolFromSmarts("[NX3;H1,H2;!$(N[a])][CX3]=[OX1]")
RING_SIZES = (5, 6)
MAX_STATES = 4


def _closures(m):
    """(carbonyl C, carbonyl O, amide N) triples whose C-N bond closes a 5/6-membered ring."""
    dm = rdmolops.GetDistanceMatrix(m)
    out = []
    for c, o in m.GetSubstructMatches(_CARBONYL):
        for n, ac, _ in m.GetSubstructMatches(_AMIDE_NH):
            if ac == c:
                continue
            if int(dm[c][n]) + 1 in RING_SIZES:           # ring atoms = path length (bonds) + 1
                out.append((c, o, n))
    return out


@dataclass(frozen=True)
class RingFormEnumeration:
    """Enumerated states and completeness within the supported single-closure model.

    Completeness does not assert that this experimental chemical model covers
    every physically accessible aqueous state.
    """
    forms: tuple[str, ...]
    total_states: int
    state_limit: int
    construction_failures: int = 0

    @property
    def complete(self):
        return self.total_states <= self.state_limit and self.construction_failures == 0


class IncompleteRingChainError(ValueError):
    """The supported ring-state ensemble could not be enumerated completely."""


def enumerate_ring_forms(smi):
    """Return capped forms with explicit state-count/construction coverage.

    The cap limits states passed to QM, not structural enumeration. Callers
    must check ``complete`` before using the forms in an equilibrium sum.
    """
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return RingFormEnumeration((), 0, MAX_STATES, construction_failures=1)
    forms = []
    failures = 0
    for c, o, n in _closures(m):
        h_o, h_n = m.GetAtomWithIdx(o).GetTotalNumHs(), m.GetAtomWithIdx(n).GetTotalNumHs()
        for tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            rw = Chem.RWMol(m)
            rw.GetBondBetweenAtoms(c, o).SetBondType(Chem.BondType.SINGLE)
            rw.AddBond(c, n, Chem.BondType.SINGLE)
            ao, an, acar = rw.GetAtomWithIdx(o), rw.GetAtomWithIdx(n), rw.GetAtomWithIdx(c)
            ao.SetNumExplicitHs(h_o + 1); ao.SetNoImplicit(True)          # C=O -> C-OH
            an.SetNumExplicitHs(h_n - 1); an.SetNoImplicit(True)          # N-H -> ring N
            acar.SetChiralTag(tag)
            try:
                p = rw.GetMol()
                Chem.SanitizeMol(p)
                Chem.AssignStereochemistry(p, cleanIt=True, force=True)
            except Exception:
                failures += 1
                continue
            s = Chem.MolToSmiles(p)
            if s not in forms and s != Chem.MolToSmiles(m):
                forms.append(s)
    return RingFormEnumeration(tuple(forms[:MAX_STATES]), len(forms), MAX_STATES, failures)


def ring_forms(smi):
    """Complete list of supported ring forms, or an explicit coverage error.

    Retains the list API for complete enumerations; a capped partial list must
    never be mistaken for a complete species ensemble.
    """
    result = enumerate_ring_forms(smi)
    if not result.complete:
        raise IncompleteRingChainError(
            f"incomplete ring-chain enumeration: {result.total_states} states, "
            f"limit {result.state_limit}, {result.construction_failures} construction failure(s)")
    return list(result.forms)


def has_ring_chain(smi):
    """True if the species has a lactam ring-chain equilibrium (used to keep it out of truncation:
    truncating away the amide would delete the equilibrium, e.g. rxn00415)."""
    m = Chem.MolFromSmiles(smi)
    return m is not None and bool(_closures(m))
