"""Reaction input types, normalization, and validation."""

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

from rdkit import Chem


@dataclass(frozen=True)
class SpeciesRecord:
    """One stoichiometric species in a reaction."""

    coefficient: float
    charge: int
    smiles: str

    def as_tuple(self):
        return (self.coefficient, self.charge, self.smiles)


@dataclass(frozen=True)
class ReactionInput:
    """Validated reaction data independent of execution and QM backends."""

    species: Mapping[str, SpeciesRecord]
    n_hplus: float = 0.0
    note: str = ""

    @classmethod
    def from_mapping(cls, reaction):
        validate_reaction(reaction)
        species = MappingProxyType(
            {
                name: SpeciesRecord(float(value[0]), int(value[1]), value[2])
                for name, value in reaction["species"].items()
            }
        )
        return cls(
            species=species,
            n_hplus=float(reaction.get("n_Hplus", 0)),
            note=str(reaction.get("note", "")),
        )


def validate_reaction(reaction: Mapping[str, Any]) -> None:
    """Reject malformed input before any species-energy work begins."""
    if not isinstance(reaction, dict):
        raise ValueError("reaction must be a mapping")
    species = reaction.get("species")
    if not isinstance(species, dict) or not species:
        raise ValueError("reaction has no species")
    n_hplus = reaction.get("n_Hplus", 0)
    if (
        not isinstance(n_hplus, (int, float))
        or isinstance(n_hplus, bool)
        or not math.isfinite(n_hplus)
    ):
        raise ValueError(f"reaction n_Hplus must be a finite number, got {n_hplus!r}")

    for name, value in species.items():
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(
                f"species {name!r}: expected [coefficient, charge, SMILES], got {value!r}"
            )
        coefficient, charge, smiles = value
        if (
            not isinstance(coefficient, (int, float))
            or isinstance(coefficient, bool)
            or not math.isfinite(coefficient)
            or coefficient == 0
        ):
            raise ValueError(
                f"species {name!r}: coefficient must be a finite nonzero number, "
                f"got {coefficient!r}"
            )
        if (
            not isinstance(charge, (int, float))
            or isinstance(charge, bool)
            or not math.isfinite(charge)
            or not float(charge).is_integer()
        ):
            raise ValueError(
                f"species {name!r}: charge must be a finite integer, got {charge!r}"
            )
        if not isinstance(smiles, str) or not smiles:
            raise ValueError(
                f"species {name!r}: SMILES must be a nonempty string, got {smiles!r}"
            )
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"species {name!r}: unparseable SMILES {smiles!r}")
        if len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError(
                f"species {name!r}: disconnected SMILES is not a supported single "
                f"chemical species ({smiles})"
            )
        formal_charge = Chem.GetFormalCharge(molecule)
        if int(charge) != formal_charge:
            raise ValueError(
                f"species {name!r}: declared charge {charge} != SMILES formal charge "
                f"{formal_charge} ({smiles})"
            )
