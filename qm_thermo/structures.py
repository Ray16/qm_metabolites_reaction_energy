"""Load and prepare the central-metabolite structures for QM.

The ModelSEED structures stored in `central_metabolites_in_opentecr.json` are
already the dominant microspecies at pH 7 (explicit charges in the SMILES), so we
trust the given protonation state. This module's job is to parse them into RDKit
molecules with explicit hydrogens, validate against the recorded formula/charge,
and expose a clean dataclass for the rest of the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from . import config

# Ground-state spin multiplicities for the few open-shell metabolites. Everything
# else is a closed-shell singlet; molecular O2 is the notable triplet exception.
SPIN_MULTIPLICITY_OVERRIDES = {
    "cpd00007": 3,   # O2, triplet ground state
}


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
