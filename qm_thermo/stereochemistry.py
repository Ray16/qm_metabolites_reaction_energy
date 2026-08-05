"""Stereochemical integrity of ModelSEED structures, judged thermodynamically.

Two facts drive every decision in this module.

**Not every centre RDKit calls potentially stereogenic is stereogenic.** A
phosphate phosphorus bonded to four oxygens is written in SMILES with a
distinct ``P=O`` and ``P-[O-]``, so ``FindPotentialStereo`` reports it as an
unspecified tetrahedral centre.  The non-bridging oxygens are resonance
equivalent and exchange rapidly, so the "configurations" are one compound.
Every nucleotide trips this: ATP reports three such centres, NAD two, PPi one.
Counting them as missing stereochemistry manufactures a defect that is not
there.  (The exclusion is deliberately narrow -- it requires *all four*
neighbours to be oxygen.  A phosphorothioate, or any phosphorus bearing four
genuinely different substituents, keeps its centre.)

**Undefined stereochemistry only changes a free energy when the alternatives
are diastereomers.**  Enantiomers have identical G in an achiral solvent, so an
undefined centre in a molecule with no other defined centre costs nothing:
1,2-propanediol may be left unresolved without biasing any reaction energy.  An
undefined centre alongside a defined one does not cancel -- alpha- and
beta-D-glucopyranose are different substances with different free energies.

So this module classifies rather than counts, and reports the thermodynamic
consequence instead of a bare "n undefined stereocentres".  Nothing here infers
configuration; it only says what is ambiguous and whether the ambiguity can
move a number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from rdkit import Chem, RDLogger
from rdkit.Chem import FindPotentialStereo
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from rdkit.Chem.rdchem import StereoSpecified

RDLogger.DisableLog("rdApp.*")

#: Roles a potential stereo element can play, ordered from "ignore" to "matters".
ARTIFACT = "resonance_equivalent_oxo_centre"
ANOMERIC = "anomeric_carbon"
CARBON = "carbon_centre"
DOUBLE_BOND = "double_bond"

#: Thermodynamic consequence of the unresolved elements in a structure.
AMBIGUITY_NONE = "none"
AMBIGUITY_ENANTIOMERIC = "enantiomeric"
AMBIGUITY_DIASTEREOMERIC = "diastereomeric"


@dataclass(frozen=True)
class StereoElement:
    """One potential stereo element, with the reason we do or do not count it."""

    index: int
    kind: str
    role: str
    specified: bool
    detail: str = ""

    @property
    def is_real(self) -> bool:
        """True when the element can distinguish two genuinely different species."""
        return self.role != ARTIFACT


@dataclass(frozen=True)
class StereoAssessment:
    """What is stereochemically ambiguous about a structure, and whether it matters.

    ``distinct_states`` is the number of genuinely different structures behind
    the unresolved elements, counted by InChIKey.  It is the final arbiter: the
    rule-based roles below are a cheap, interpretable pre-filter, but a centre
    that enumerates to a single InChIKey is not ambiguous no matter what the
    perception code called it.  ``None`` means the check was not run.
    """

    smiles: str
    elements: tuple[StereoElement, ...] = field(default_factory=tuple)
    parse_error: str = ""
    distinct_states: int | None = None

    @property
    def real(self) -> tuple[StereoElement, ...]:
        return tuple(element for element in self.elements if element.is_real)

    @property
    def artifacts(self) -> tuple[StereoElement, ...]:
        return tuple(element for element in self.elements if not element.is_real)

    @property
    def undefined(self) -> tuple[StereoElement, ...]:
        return tuple(element for element in self.real if not element.specified)

    @property
    def defined(self) -> tuple[StereoElement, ...]:
        return tuple(element for element in self.real if element.specified)

    @property
    def anomeric_undefined(self) -> tuple[StereoElement, ...]:
        return tuple(element for element in self.undefined if element.role == ANOMERIC)

    @property
    def ambiguity(self) -> str:
        """Thermodynamic consequence of whatever is left unresolved.

        ``enantiomeric`` means the unresolved alternatives are mirror images and
        share a free energy, so leaving them unresolved is safe.  It requires
        exactly one unresolved centre and no resolved ones -- with a second
        centre anywhere in the molecule the alternatives become diastereomers.

        A verified count of fewer than two distinct structures overrides the
        rule-based verdict: delocalised systems such as a guanidinium C=N are
        perceived as unspecified but enumerate to one substance.
        """
        if not self.undefined:
            return AMBIGUITY_NONE
        if self.distinct_states is not None and self.distinct_states < 2:
            return AMBIGUITY_NONE
        if len(self.undefined) == 1 and not self.defined:
            return AMBIGUITY_ENANTIOMERIC
        return AMBIGUITY_DIASTEREOMERIC

    @property
    def thermodynamically_ambiguous(self) -> bool:
        return self.ambiguity == AMBIGUITY_DIASTEREOMERIC

    def summary(self) -> dict:
        return {
            "smiles": self.smiles,
            "parse_error": self.parse_error,
            "ambiguity": self.ambiguity,
            "n_real_defined": len(self.defined),
            "n_real_undefined": len(self.undefined),
            "n_artifacts_excluded": len(self.artifacts),
            "anomeric_undefined": len(self.anomeric_undefined),
            "distinct_states": self.distinct_states,
            "undefined_detail": [element.detail for element in self.undefined],
        }


def _is_oxo_artifact(atom: Chem.Atom) -> bool:
    """True for a phosphorus/sulfur written with four oxygens.

    The non-bridging oxygens are resonance equivalent, so the SMILES-level
    distinction between ``=O`` and ``-[O-]`` is notation, not configuration.
    """
    if atom.GetSymbol() not in ("P", "S"):
        return False
    neighbours = atom.GetNeighbors()
    return len(neighbours) == 4 and all(n.GetSymbol() == "O" for n in neighbours)


def _is_anomeric(atom: Chem.Atom) -> bool:
    """True for a ring carbon bearing a ring oxygen and a second oxygen.

    This is the sugar anomeric centre: the alpha/beta pair are diastereomers
    with genuinely different free energies, and it is the centre ModelSEED most
    often leaves open.
    """
    if atom.GetSymbol() != "C" or not atom.IsInRing():
        return False
    ring_oxygen = any(n.GetSymbol() == "O" and n.IsInRing() for n in atom.GetNeighbors())
    oxygens = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "O")
    return ring_oxygen and oxygens >= 2


def _describe(mol: Chem.Mol, element) -> tuple[str, str, str]:
    """Return ``(kind, role, detail)`` for one RDKit potential-stereo element."""
    kind = str(element.type).rsplit(".", 1)[-1]
    if "Bond" in kind:
        bond = mol.GetBondWithIdx(element.centeredOn)
        detail = (f"bond {bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()} "
                  f"{bond.GetBeginAtom().GetSymbol()}={bond.GetEndAtom().GetSymbol()}")
        return kind, DOUBLE_BOND, detail
    atom = mol.GetAtomWithIdx(element.centeredOn)
    neighbours = "".join(sorted(n.GetSymbol() for n in atom.GetNeighbors()))
    detail = f"atom {element.centeredOn} {atom.GetSymbol()} nbrs={neighbours}"
    if _is_oxo_artifact(atom):
        return kind, ARTIFACT, detail + " (resonance-equivalent oxo centre)"
    if _is_anomeric(atom):
        return kind, ANOMERIC, detail + " (anomeric)"
    return kind, CARBON, detail


def assess(smiles: str, verify: bool = True) -> StereoAssessment:
    """Classify every potential stereo element in ``smiles``.

    With ``verify`` the unresolved elements are enumerated and counted by
    InChIKey, so a perceived ambiguity that corresponds to only one substance is
    demoted rather than reported.  Pass ``verify=False`` for the cheap
    rule-based view (and to avoid recursion from the enumerator itself).
    """
    if not smiles:
        return StereoAssessment(smiles=smiles, parse_error="empty SMILES")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return StereoAssessment(smiles=smiles, parse_error="RDKit could not parse SMILES")
    elements = []
    for found in FindPotentialStereo(mol):
        kind, role, detail = _describe(mol, found)
        elements.append(StereoElement(
            index=int(found.centeredOn), kind=kind, role=role,
            specified=found.specified == StereoSpecified.Specified, detail=detail,
        ))
    assessment = StereoAssessment(smiles=smiles, elements=tuple(elements))
    if not verify or not assessment.undefined:
        return assessment
    return StereoAssessment(smiles=smiles, elements=tuple(elements),
                            distinct_states=len(enumerate_resolved(smiles)))


def _clear_artifact_tags(mol: Chem.Mol) -> Chem.Mol:
    """Drop chiral tags an enumerator may have placed on resonance-equivalent centres."""
    editable = Chem.Mol(mol)
    for atom in editable.GetAtoms():
        if _is_oxo_artifact(atom):
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return editable


def enumerate_resolved(smiles: str, max_isomers: int = 16) -> tuple[str, ...]:
    """Enumerate the distinct structures hidden behind the unresolved real centres.

    Artifact centres are stripped from the results and duplicates collapsed by
    InChIKey, so a nucleotide whose only unresolved centres are phosphate
    phosphorus atoms returns a single structure rather than 2^n copies.
    """
    assessment = assess(smiles, verify=False)
    if assessment.parse_error or not assessment.undefined:
        return (smiles,) if not assessment.parse_error else ()
    mol = Chem.MolFromSmiles(smiles)
    options = StereoEnumerationOptions(onlyUnassigned=True, unique=True,
                                       maxIsomers=max(1, max_isomers) * 8)
    seen: dict[str, str] = {}
    for isomer in EnumerateStereoisomers(mol, options=options):
        cleaned = _clear_artifact_tags(isomer)
        try:
            key = Chem.MolToInchiKey(cleaned)
        except Exception:  # pragma: no cover - InChI failure on exotic input
            key = Chem.MolToSmiles(cleaned)
        if key and key not in seen:
            seen[key] = Chem.MolToSmiles(cleaned)
        if len(seen) >= max_isomers:
            break
    return tuple(seen.values())


def structure_key(smiles: str) -> str:
    """Full InChIKey, the arbiter of whether two records are the same substance."""
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:  # pragma: no cover - InChI failure on exotic input
        return ""


def skeleton_key(smiles: str) -> str:
    """First InChIKey block: connectivity only, ignoring stereo and protonation."""
    key = structure_key(smiles)
    return key.split("-")[0] if key else ""


@dataclass(frozen=True)
class StructureCollision:
    """Distinct compound ids that resolve to one and the same structure."""

    inchikey: str
    compound_ids: tuple[str, ...]
    names: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"inchikey": self.inchikey, "compound_ids": list(self.compound_ids),
                "names": list(self.names)}


def find_collisions(structures: dict[str, str],
                    names: dict[str, str] | None = None) -> tuple[StructureCollision, ...]:
    """Group compound ids whose structures are byte-identical after canonicalisation.

    A collision is not automatically an error -- ModelSEED carries genuine
    synonyms -- but two ids used as separate species in one reaction with a
    single structure between them make that reaction's energy zero by
    construction.  ``degenerate_reactions`` is the objective test; this is the
    inventory.
    """
    names = names or {}
    grouped: dict[str, list[str]] = {}
    for compound_id, smiles in structures.items():
        key = structure_key(smiles)
        if key:
            grouped.setdefault(key, []).append(compound_id)
    collisions = []
    for key, ids in grouped.items():
        if len(ids) > 1:
            ordered = tuple(sorted(ids))
            collisions.append(StructureCollision(
                inchikey=key, compound_ids=ordered,
                names=tuple(names.get(i, "") for i in ordered)))
    return tuple(sorted(collisions, key=lambda c: c.compound_ids))


def degenerate_reactions(reactions: dict[str, dict[str, float]],
                         structures: dict[str, str]) -> dict[str, str]:
    """Reactions whose two sides carry the identical multiset of structures.

    Their free energy is exactly zero for any method, so a computed value is an
    artefact of the input rather than a prediction.  ``rxn00266``
    (oxaloacetate = enol-oxaloacetate) is the canonical case: ModelSEED stores
    the keto SMILES for both compounds.
    """
    degenerate = {}
    for reaction_id, stoich in reactions.items():
        left: list[str] = []
        right: list[str] = []
        incomplete = False
        for compound_id, coefficient in stoich.items():
            key = structure_key(structures.get(compound_id, ""))
            if not key:
                incomplete = True
                break
            side = left if coefficient < 0 else right
            side.extend([key] * int(round(abs(coefficient))))
        if incomplete or not left or not right:
            continue
        if sorted(left) == sorted(right):
            degenerate[reaction_id] = sorted(left)[0]
    return degenerate


def assess_many(structures: dict[str, str]) -> dict[str, StereoAssessment]:
    return {compound_id: assess(smiles) for compound_id, smiles in structures.items()}


def ambiguous_compounds(assessments: dict[str, StereoAssessment]) -> tuple[str, ...]:
    """Compound ids whose unresolved stereochemistry can move a free energy."""
    return tuple(sorted(cid for cid, a in assessments.items()
                        if a.thermodynamically_ambiguous))


def affected_reactions(reactions: dict[str, dict[str, float]],
                       ambiguous: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Map each reaction to the ambiguous compounds it contains."""
    flagged = set(ambiguous)
    result = {}
    for reaction_id, stoich in reactions.items():
        hits = tuple(sorted(set(stoich) & flagged))
        if hits:
            result[reaction_id] = hits
    return result
