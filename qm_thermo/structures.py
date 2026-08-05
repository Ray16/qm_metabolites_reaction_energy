"""Load and prepare the central-metabolite structures for QM.

The ModelSEED structures stored in `central_metabolites_in_opentecr.json` are
already the dominant microspecies at pH 7 (explicit charges in the SMILES), so we
trust the given protonation state. This module's job is to parse them into RDKit
molecules with explicit hydrogens, validate against the recorded formula/charge,
and expose a clean dataclass for the rest of the pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from . import config
from . import stereochemistry

#: Escape hatch for reproducing pre-2026-08 numbers, which were computed on
#: stereochemically ambiguous inputs.  It is opt-in because the resulting
#: "conformer ensemble" silently mixes diastereomers.
ALLOW_AMBIGUOUS_STEREO = os.environ.get("ALLOW_AMBIGUOUS_STEREO", "") == "1"

# Ground-state spin multiplicities for the few open-shell metabolites. Everything
# else is a closed-shell singlet; molecular O2 is the notable triplet exception.
SPIN_MULTIPLICITY_OVERRIDES = {
    "cpd00007": 3,   # O2, triplet ground state
}

#: d-block elements whose metabolite-relevant oxidation states are usually
#: open-shell.  An even electron count is *not* evidence of a singlet here:
#: high-spin Fe(II) is a quintet with 24 electrons, and the free Fe and W atoms
#: are 5-D quintets.  The odd-electron screen below catches Fe(III), Cu(II),
#: Co(II), Mn(II) and Cr(III), but it waves Fe(II), Mo(II) and W through to be
#: computed as closed-shell singlets -- an error measured at 200-470 kJ/mol,
#: several times larger than any other in this project.  So these require an
#: explicit multiplicity rather than defaulting to one.
OPEN_SHELL_D_BLOCK = frozenset({
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
})


@dataclass(frozen=True)
class Metabolite:
    """A single metabolite ready for conformer generation."""

    cpd_id: str           # ModelSEED id, e.g. "cpd00002"
    name: str
    smiles: str
    formula: str          # recorded ModelSEED formula
    charge: int           # net charge of the pH-7 microspecies
    inchikey: str
    opentecr_species: str
    mol: Chem.Mol         # RDKit mol, sanitized, with explicit Hs

    @property
    def n_atoms(self) -> int:
        return self.mol.GetNumAtoms()

    @property
    def n_electrons(self) -> int:
        total = sum(a.GetAtomicNum() for a in self.mol.GetAtoms())
        return total - self.charge

    @property
    def spin_multiplicity(self) -> int:
        # Closed-shell singlet by default; open-shell exceptions are overridden.
        return SPIN_MULTIPLICITY_OVERRIDES.get(self.cpd_id, 1)


class StructureError(ValueError):
    """Raised when a recorded structure cannot be parsed or fails validation."""


def _build_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise StructureError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    return mol


def _validate(mol: Chem.Mol, meta: dict) -> None:
    """Cross-check parsed RDKit charge against the recorded ModelSEED charge.

    The formula is *not* hard-validated because ModelSEED's Hill strings and
    RDKit's differ in convention for some hydrogens; the net charge is the
    reliable invariant for QM (it sets the ORCA charge line).
    """
    rd_charge = Chem.GetFormalCharge(mol)
    if rd_charge != meta["charge"]:
        raise StructureError(
            f"{meta['id']}: RDKit charge {rd_charge} != recorded {meta['charge']} "
            f"(SMILES {meta['smiles']!r})"
        )
    n_elec = sum(a.GetAtomicNum() for a in mol.GetAtoms()) - rd_charge
    if n_elec % 2 != 0:
        raise StructureError(
            f"{meta['id']}: odd electron count ({n_elec}); open-shell not supported"
        )
    _validate_spin_state(mol, meta)
    _validate_stereochemistry(meta)


def _validate_spin_state(mol: Chem.Mol, meta: dict) -> None:
    """Refuse open-shell d-block metals rather than assuming a singlet.

    The even-electron check above is necessary but not sufficient. High-spin
    Fe(II) has 24 electrons and is a quintet; the free Fe and W atoms are 5-D
    quintets with 26 and 74. Those all pass an even-parity test and were being
    scored as closed-shell singlets, which is wrong by 200-470 kJ/mol measured
    with g-xTB -- an order of magnitude beyond anything else in the error
    budget, and silent.

    Refusing is the honest outcome, not a temporary gap. A bare aqueous
    transition-metal ion's thermodynamics is dominated by explicit inner-shell
    hydration and ligand-field splitting, neither of which a continuum
    composite represents, so a number here would be wrong even with the right
    multiplicity. Supply an explicit ``SPIN_MULTIPLICITY_OVERRIDES`` entry to
    override, which records the choice instead of hiding it.
    """
    if meta["id"] in SPIN_MULTIPLICITY_OVERRIDES:
        return
    metals = sorted({a.GetSymbol() for a in mol.GetAtoms()
                     if a.GetSymbol() in OPEN_SHELL_D_BLOCK})
    if metals:
        raise StructureError(
            f"{meta['id']}: contains open-shell-capable d-block metal(s) "
            f"{', '.join(metals)}; the closed-shell singlet default is not valid "
            f"(measured 200-470 kJ/mol error on bare ions). Add an explicit "
            f"SPIN_MULTIPLICITY_OVERRIDES entry to compute it deliberately."
        )


def _validate_stereochemistry(meta: dict) -> None:
    """Refuse a structure whose undefined stereochemistry mixes diastereomers.

    ETKDG resolves an undefined centre independently for each embedding, so a
    single compound's "conformer ensemble" ends up containing two different
    substances weighted by embedding luck (measured for D-glucose 6-phosphate:
    14 conformers split 5/9 across both anomers).  Boltzmann-averaging that is
    not a conformer average, and the weights come from exactly the relative
    energies this project has shown to be unreliable.

    Enantiomeric ambiguity is allowed through: mirror images share a free
    energy in an achiral solvent, so leaving one unresolved changes nothing.
    """
    assessment = stereochemistry.assess(meta["smiles"])
    if not assessment.thermodynamically_ambiguous:
        return
    centres = "; ".join(element.detail for element in assessment.undefined)
    message = (
        f"{meta['id']}: undefined stereochemistry that changes the free energy "
        f"({assessment.distinct_states} distinct structures) at {centres}. "
        f"Run pipeline/resolve_stereochemistry.py to enumerate explicit states, "
        f"or set ALLOW_AMBIGUOUS_STEREO=1 to reproduce the older biased numbers."
    )
    if ALLOW_AMBIGUOUS_STEREO:
        print(f"[structures] WARNING {message}")
        return
    raise StructureError(message)


def load_metabolites(
    json_path: str = config.CENTRAL_METABOLITES_JSON,
    *,
    skip_invalid: bool = False,
) -> list[Metabolite]:
    """Parse the metabolite JSON into validated `Metabolite` objects.

    Args:
        json_path: path to the central-metabolites JSON.
        skip_invalid: if True, log and skip entries that fail validation instead
            of raising (useful for a best-effort full run).
    """
    with open(json_path) as fh:
        records = json.load(fh)

    metabolites: list[Metabolite] = []
    for rec in records:
        try:
            mol = _build_mol(rec["smiles"])
            _validate(mol, rec)
        except StructureError as err:
            if skip_invalid:
                print(f"[structures] skipping {rec.get('id')}: {err}")
                continue
            raise
        metabolites.append(
            Metabolite(
                cpd_id=rec["id"],
                name=rec["name"],
                smiles=rec["smiles"],
                formula=rec["formula"],
                charge=int(rec["charge"]),
                inchikey=rec["inchikey"],
                opentecr_species=rec.get("opentecr_species", ""),
                mol=mol,
            )
        )
    return metabolites


def load_by_id(
    cpd_ids: list[str], json_path: str = config.CENTRAL_METABOLITES_JSON
) -> list[Metabolite]:
    """Convenience loader returning only the requested compound ids, in order."""
    by_id = {m.cpd_id: m for m in load_metabolites(json_path)}
    missing = [c for c in cpd_ids if c not in by_id]
    if missing:
        raise KeyError(f"compound ids not in {json_path}: {missing}")
    return [by_id[c] for c in cpd_ids]
