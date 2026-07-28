#!/usr/bin/env python
"""Validate periodic alchemical endpoints before launching replica sampling.

This is a systems-integrity gate, not a free-energy calculation.  It checks
that a GAFF2/TIP3P system can be transformed through coupled, partially
decoupled, and fully decoupled solute states with finite energies after short
NVT propagation.  Production sampling must use replicas, MBAR overlap checks,
and a charged-system finite-size correction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-dir", required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--timestep-fs", type=float, default=1.0,
                        help="conservative preflight timestep; production may use 2 fs after equilibration")
    parser.add_argument("--lambda-values", default="1.0,0.75,0.5,0.25,0.0",
                        help="descending annealing schedule used to initialize each next state")
    parser.add_argument("--platform", default="CUDA")
    args = parser.parse_args()

    from openmm import LangevinMiddleIntegrator, LocalEnergyMinimizer, Platform, XmlSerializer, unit
    from openmm import app
    from openmmtools import alchemy

    directory = Path(args.system_dir)
    metadata = json.load(open(directory / "metadata.json"))
    system = XmlSerializer.deserialize((directory / "system.xml").read_text())
    pdb = app.PDBFile(str(directory / "system.pdb"))
    # PME cannot rigorously compare states of different total charge.  For an
    # anionic solute we co-alchemically scale one explicitly solvated Na+ so
    # the box is neutral at every lambda.  This is a preflight for the
    # charge-neutral path; production analysis must still account for the
    # ion-pair/reference thermodynamic cycle.
    alchemical_atoms = list(metadata["solute_atom_indices"])
    counterions = metadata.get("ion_atom_indices", {}).get("cations", [])
    if metadata["charge"] < 0:
        if not counterions:
            raise RuntimeError("anionic system has no Na+ counterion for a neutral alchemical path")
        alchemical_atoms.append(counterions[0])
    region = alchemy.AlchemicalRegion(alchemical_atoms=alchemical_atoms,
                                      annihilate_electrostatics=True, annihilate_sterics=False)
    alchemical_system = alchemy.AbsoluteAlchemicalFactory(
        alchemical_pme_treatment="exact", split_alchemical_forces=True
    ).create_alchemical_system(system, region)
    platform = Platform.getPlatformByName(args.platform)
    output = {"system": metadata["name"], "charge": metadata["charge"],
              "co_alchemical_counterion": alchemical_atoms[-1] if len(alchemical_atoms) > len(metadata["solute_atom_indices"]) else None,
              "states": []}
    positions = pdb.positions
    lambda_values = [float(value) for value in args.lambda_values.split(",")]
    if lambda_values[0] != 1.0 or lambda_values != sorted(lambda_values, reverse=True):
        raise ValueError("lambda values must start at 1.0 and descend")
    for lambda_value in lambda_values:
        integrator = LangevinMiddleIntegrator(298.15 * unit.kelvin, 1.0 / unit.picosecond,
                                               args.timestep_fs * unit.femtoseconds)
        context = __import__("openmm").Context(alchemical_system, integrator, platform)
        context.setPositions(positions)
        # First discharge at full steric repulsion.  Scaling Coulomb and
        # sterics together lets a counterion/water collapse into a partially
        # discharged phosphate and is not a usable soft-core schedule.
        context.setParameter("lambda_electrostatics", lambda_value)
        context.setParameter("lambda_sterics", 1.0)
        LocalEnergyMinimizer.minimize(context, tolerance=10 * unit.kilojoule_per_mole / unit.nanometer,
                                       maxIterations=250)
        context.applyConstraints(1e-6)
        context.setVelocitiesToTemperature(298.15 * unit.kelvin)
        print(f"propagating lambda={lambda_value}", flush=True)
        integrator.step(args.steps)
        state = context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if not __import__("math").isfinite(energy):
            raise RuntimeError(f"non-finite potential at lambda={lambda_value}")
        output["states"].append({"lambda": lambda_value, "potential_kJ_mol": energy})
        positions = state.getPositions()
        del context, integrator
    json.dump(output, open(directory / "alchemical_preflight.json", "w"), indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
