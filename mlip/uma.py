"""Small UMA adapter used by the parallel runner.

Keeping the model-specific import here lets the workflow code share the same
conformer and thermodynamic machinery with other electronic-energy models.
"""
from __future__ import annotations

import os
import sys

EV_TO_KJ = 96.48533212
_CALCULATOR = None


def _calculator():
    global _CALCULATOR
    if _CALCULATOR is None:
        tools_dir = os.environ.get("UMA_TOOLS_DIR")
        if tools_dir:
            sys.path.insert(0, tools_dir)
        try:
            import uma_helper
        except ImportError as exc:
            raise RuntimeError(
                "UMA requires uma_helper. Set UMA_TOOLS_DIR to its directory."
            ) from exc
        _CALCULATOR = uma_helper.get_calculator(
            "omol", os.environ.get("UMA_MODEL", "uma-s-1p2")
        )
    return _CALCULATOR


def gas_energy_kJ(xyz_path: str, charge: int) -> float:
    """Return UMA gas-phase electronic energy in kJ/mol for one geometry."""
    from ase.io import read

    atoms = read(xyz_path)
    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = 1
    atoms.calc = _calculator()
    return float(atoms.get_potential_energy()) * EV_TO_KJ
