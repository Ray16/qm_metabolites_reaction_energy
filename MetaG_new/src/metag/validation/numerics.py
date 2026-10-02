"""Numerical-resolution diagnostics for reaction energies."""

import math

import numpy as np
from rdkit import Chem


EV2KJ = 96.485


def canonical(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule) if molecule is not None else smiles


def float32_ulp_kj(g_kj):
    """Spacing near a float32 electronic energy of the given magnitude."""
    energy_ev = np.float32(abs(float(g_kj)) / EV2KJ)
    return float(np.spacing(energy_ev)) * EV2KJ


def reaction_precision(record, species):
    """Estimate float32 rounding bounds for one assembled reaction."""
    routed = record.get("species_scored") or record.get("species")
    if not isinstance(routed, dict):
        return None

    terms = []
    missing = []
    max_heavy = total_heavy = 0
    for name, value in routed.items():
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            missing.append(name)
            continue
        coefficient, charge, smiles = value[:3]
        molecule = Chem.MolFromSmiles(smiles)
        heavy = molecule.GetNumHeavyAtoms() if molecule is not None else 0
        max_heavy = max(max_heavy, heavy)
        total_heavy += heavy
        if canonical(smiles) == "O" and int(charge) == 0:
            continue
        cached = species.get((canonical(smiles), int(charge)))
        if cached is None:
            missing.append(name)
            continue
        terms.append((float(coefficient), cached["ulp_kj"]))

    if missing:
        return {"missing_species": missing}

    worst = sum(abs(coefficient) * ulp / 2.0 for coefficient, ulp in terms)
    rms = math.sqrt(sum(coefficient**2 * ulp**2 / 12.0 for coefficient, ulp in terms))
    return {
        "worst_bound_kj": worst,
        "rms_scale_kj": rms,
        "n_cached_species": len(terms),
        "max_heavy_atoms": max_heavy,
        "total_heavy_atoms": total_heavy,
    }

