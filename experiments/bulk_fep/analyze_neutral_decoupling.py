#!/usr/bin/env python
"""Report MBAR free energy and uncertainty for one neutral decoupling leg."""
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reporter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from openmm import unit
    from openmmtools import multistate

    reporter = multistate.MultiStateReporter(args.reporter, open_mode="r")
    analyzer = multistate.ReplicaExchangeAnalyzer(reporter)
    free_energy, uncertainty = analyzer.get_free_energy()
    kT_kj = (unit.MOLAR_GAS_CONSTANT_R * 298.15 * unit.kelvin).value_in_unit(unit.kilojoule_per_mole)
    result = {"n_states": int(free_energy.shape[0]), "delta_f_kT": float(free_energy[0, -1]),
              "uncertainty_kT": float(uncertainty[0, -1]),
              "delta_G_kJ_mol": float(free_energy[0, -1] * kT_kj),
              "uncertainty_kJ_mol": float(uncertainty[0, -1] * kT_kj),
              "warning": "paired-solute leg only; no pKa interpretation without the complete cycle"}
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
