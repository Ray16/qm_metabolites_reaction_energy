#!/usr/bin/env python
"""Run resumable charge-neutral solute/Na+ decoupling replica exchange.

This estimates a *paired-solute hydration leg*, not a pKa directly.  It is one
component of the validated thermodynamic cycle and deliberately keeps the PME
box neutral by alchemically scaling one Na+ with the anion.  Use
``analyze_neutral_decoupling.py`` only after sampling is converged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def alchemical_schedule(n_electrostatics: int, n_sterics: int) -> list[tuple[float, float]]:
    """Discharge at full sterics, then remove soft-core sterics at zero charge."""
    electrostatics = [(1.0 - i / (n_electrostatics - 1), 1.0) for i in range(n_electrostatics)]
    sterics = [(0.0, 1.0 - i / (n_sterics - 1)) for i in range(1, n_sterics)]
    return electrostatics + sterics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-dir", required=True)
    parser.add_argument("--out", required=True, help="NetCDF checkpoint/reporter path")
    parser.add_argument("--iterations", type=int, default=100,
                        help="additional iterations; each is --steps MD steps per replica")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--n-electrostatics", type=int, default=12)
    parser.add_argument("--n-sterics", type=int, default=8)
    parser.add_argument("--initialization-steps", type=int, default=200,
                        help="sequential annealing steps used to seed every lambda window")
    parser.add_argument("--seed", type=int, default=20260728,
                        help="independent initialization seed for a new replica")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.n_electrostatics < 2 or args.n_sterics < 2:
        raise ValueError("need at least two electrostatic and steric states")

    from openmm import Context, LangevinMiddleIntegrator, LocalEnergyMinimizer, XmlSerializer, unit
    from openmm import app
    from openmmtools import alchemy, cache, mcmc, multistate, states
    from openmmtools.utils import get_fastest_platform

    directory = Path(args.system_dir)
    metadata = json.load(open(directory / "metadata.json"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    schedule = alchemical_schedule(args.n_electrostatics, args.n_sterics)
    platform = get_fastest_platform(minimum_precision="mixed")

    reporter = multistate.MultiStateReporter(str(out), open_mode="a" if args.resume and out.exists() else None,
                                              checkpoint_interval=10)
    if args.resume and out.exists():
        simulation = multistate.ReplicaExchangeSampler.from_storage(reporter)
    else:
        system = XmlSerializer.deserialize((directory / "system.xml").read_text())
        pdb = app.PDBFile(str(directory / "system.pdb"))
        atoms = list(metadata["solute_atom_indices"])
        if metadata["charge"] < 0:
            cations = metadata.get("ion_atom_indices", {}).get("cations", [])
            if not cations:
                raise RuntimeError("no Na+ available for charge-neutral alchemical path")
            atoms.append(cations[0])
        region = alchemy.AlchemicalRegion(alchemical_atoms=atoms, annihilate_electrostatics=True,
                                          annihilate_sterics=False)
        alchemical_system = alchemy.AbsoluteAlchemicalFactory(
            alchemical_pme_treatment="exact", split_alchemical_forces=True
        ).create_alchemical_system(system, region)
        base = states.ThermodynamicState(alchemical_system, temperature=298.15 * unit.kelvin,
                                         pressure=1.0 * unit.atmosphere)
        thermodynamic_states = []
        for state_index, (lambda_e, lambda_s) in enumerate(schedule):
            state = states.CompoundThermodynamicState(
                base, composable_states=[alchemy.AlchemicalState.from_system(alchemical_system)]
            )
            state.lambda_electrostatics = lambda_e
            state.lambda_sterics = lambda_s
            thermodynamic_states.append(state)
        move = mcmc.LangevinDynamicsMove(timestep=1.0 * unit.femtoseconds,
                                         collision_rate=1.0 / unit.picoseconds,
                                         n_steps=args.steps)
        # Seeding every replica from lambda=1 coordinates makes low-lambda
        # replicas explode before the first swap.  Build a sequentially
        # annealed position for each state instead.
        initializer = LangevinMiddleIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds,
                                                1.0 * unit.femtoseconds)
        context = Context(alchemical_system, initializer, platform)
        context.setPositions(pdb.positions)
        sampler_states = []
        for lambda_e, lambda_s in schedule:
            context.setParameter("lambda_electrostatics", lambda_e)
            context.setParameter("lambda_sterics", lambda_s)
            LocalEnergyMinimizer.minimize(context, tolerance=10 * unit.kilojoule_per_mole / unit.nanometer,
                                           maxIterations=250)
            context.applyConstraints(1e-6)
            context.setVelocitiesToTemperature(298.15 * unit.kelvin, args.seed + state_index)
            initializer.step(args.initialization_steps)
            state = context.getState(getPositions=True)
            sampler_states.append(states.SamplerState(state.getPositions(), box_vectors=state.getPeriodicBoxVectors()))
        del context, initializer
        simulation = multistate.ReplicaExchangeSampler(
            mcmc_moves=move, number_of_iterations=args.iterations,
            online_analysis_interval=10, online_analysis_target_error=0.5
        )
        simulation.create(thermodynamic_states=thermodynamic_states,
                          sampler_states=sampler_states, storage=reporter)
        json.dump({"system": metadata["name"], "charge": metadata["charge"],
                   "co_alchemical_counterion": atoms[-1] if len(atoms) > len(metadata["solute_atom_indices"]) else None,
                   "schedule": [{"lambda_electrostatics": e, "lambda_sterics": s} for e, s in schedule],
                   "steps_per_iteration": args.steps,
                   "initialization_seed": args.seed,
                   "warning": "paired-solute hydration leg; not pKa without reference/gas/finite-size terms"},
                  open(str(out) + ".json", "w"), indent=2)
    simulation.energy_context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)
    simulation.sampler_context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)
    simulation.extend(n_iterations=args.iterations)
    print(f"completed iteration {simulation.iteration}; reporter={out}")


if __name__ == "__main__":
    main()
