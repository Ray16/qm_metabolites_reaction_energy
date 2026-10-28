"""Reaction input types, normalization, and validation."""

from dataclasses import dataclass
import math
import json
import os
from types import MappingProxyType
from typing import Any, Mapping
from fractions import Fraction
from functools import reduce

from rdkit import Chem


def stoichiometric_extent(species):
    """Positive scale relative to the primitive integer reaction equation.

    The calibration panel uses primitive equations. Fractional and multiplied
    equations describe the same event with proportionally scaled energies and
    uncertainties. Include water, since its coefficient also defines extent.
    """
    if not species:
        return 1.0
    coefficients = []
    for value in species.values():
        exact = Fraction(str(value[0]))
        short = exact.limit_denominator(1000000)
        # Recover usual rational inputs (e.g. float(1/3)); never round a small
        # nonzero extent to zero or change an arbitrary decimal coefficient.
        coefficients.append(short if short and math.isclose(float(short), float(exact),
                                                            rel_tol=1e-12, abs_tol=0.0) else exact)
    denominator = math.lcm(*(c.denominator for c in coefficients))
    numerator = reduce(math.gcd, (abs(c.numerator * (denominator // c.denominator))
                                  for c in coefficients))
    if numerator == 0:
        raise ValueError("reaction has no nonzero stoichiometry")
    return numerator / denominator


def unit_stoichiometry(species):
    """Return species at primitive extent, together with the input extent."""
    extent = stoichiometric_extent(species)
    if extent == 1.0:
        return species, extent
    return {name: [c / extent, q, s] for name, (c, q, s) in species.items()}, extent


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
    if 'species_protonation_models' in reaction:
        models = reaction['species_protonation_models']
        if (not isinstance(models, dict) or not models or not set(models) <= set(species)
                or any(not isinstance(model, dict) for model in models.values())):
            raise ValueError('species_protonation_models must identify nonempty input species models')
        if reaction.get('pka_sites'):
            raise ValueError('Flat user pka_sites have no species ownership for coupled replacement')
    from metag.energetics.protonation import normalize_site
    pka_sites = reaction.get("pka_sites", [])
    if not isinstance(pka_sites, (list, tuple)):
        raise ValueError("pka_sites must be a list of pKa terms")
    for site in pka_sites:
        normalize_site(site)
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


def load_reactions(path=None):
    """Load a reactions JSON for the CLI harness. Optional: the score_reaction() API takes a reaction dict
    directly and needs no file, so a missing file is not an error (returns {})."""
    if path is None:
        path = os.environ.get("RXN_FILE", os.path.join(os.path.dirname(__file__), "data", "reactions_opentecr_std.json"))
    if not os.path.exists(path):
        return {}
    with open(path) as stream:
        raw = json.load(stream)
    out = {}
    for key, rx in raw.items():
        d = dict(rx)
        d["species"] = {n: tuple(v) for n, v in d["species"].items()}
        if isinstance(d.get("explicit"), list):
            d["explicit"] = set(d["explicit"])
        if "pka_sites" in d:
            d["pka_sites"] = [tuple(x) for x in d["pka_sites"]]
        out[key] = d
    return out
