#!/usr/bin/env python
"""Prepare periodic GAFF2/TIP3P systems for the phosphate free-energy gate.

This program is intentionally preparation-only.  It records the exact force
field, solute atom indices, charge, box vectors, and input SMILES before any
sampling starts.  A charged-solute hydration calculation needs a finite-size
and electrostatic reference correction, so no number from this script is ever
reported as a pKa result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def prepare(species: dict, output: Path, padding_nm: float, ionic_strength_m: float) -> dict:
    # This script is often invoked by the absolute interpreter path on a
    # cluster.  In that case conda activation has not prepended its bin/ to
    # PATH, and OpenFF cannot discover AmberTools despite it being installed.
    env_bin = str(Path(sys.prefix) / "bin")
    os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")
    from openff.toolkit import Molecule
    from openff.toolkit.utils.ambertools_wrapper import AmberToolsToolkitWrapper
    from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY
    from openmm import XmlSerializer, unit
    from openmm import app
    from openmmforcefields.generators import GAFFTemplateGenerator

    if not AmberToolsToolkitWrapper.is_available():
        raise RuntimeError("AmberTools is required for AM1-BCC charges but is unavailable")
    if not any(isinstance(toolkit, AmberToolsToolkitWrapper)
               for toolkit in GLOBAL_TOOLKIT_REGISTRY.registered_toolkits):
        GLOBAL_TOOLKIT_REGISTRY.register_toolkit(AmberToolsToolkitWrapper())

    molecule = Molecule.from_smiles(species["smiles"], allow_undefined_stereo=True)
    formal_charge = int(round(molecule.total_charge.m))
    if formal_charge != int(species["charge"]):
        raise ValueError(f"{species['name']}: SMILES charge {formal_charge}, expected {species['charge']}")
    molecule.generate_conformers(n_conformers=1)
    off_topology = molecule.to_topology()
    solute_topology = off_topology.to_openmm()
    solute_positions = molecule.conformers[0].to_openmm()
    solute_atoms = list(range(solute_topology.getNumAtoms()))

    forcefield = app.ForceField("amber14/tip3p.xml")
    gaff = GAFFTemplateGenerator(molecules=[molecule], forcefield="gaff-2.11")
    forcefield.registerTemplateGenerator(gaff.generator)
    modeller = app.Modeller(solute_topology, solute_positions)
    modeller.addSolvent(forcefield, model="tip3p", padding=padding_nm * unit.nanometer,
                        ionicStrength=ionic_strength_m * unit.molar, neutralize=True)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                     nonbondedCutoff=1.0 * unit.nanometer,
                                     constraints=app.HBonds, rigidWater=True,
                                     ewaldErrorTolerance=5e-4)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "system.xml", "w") as handle:
        handle.write(XmlSerializer.serialize(system))
    with open(output / "system.pdb", "w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
    ions = {"cations": [], "anions": []}
    for atom in modeller.topology.atoms():
        residue = atom.residue.name.upper()
        if residue in {"NA", "SOD"}:
            ions["cations"].append(atom.index)
        elif residue in {"CL", "CLA"}:
            ions["anions"].append(atom.index)
    metadata = {
        "name": species["name"], "smiles": species["smiles"], "charge": formal_charge,
        "force_field": "GAFF2.11 + Amber14 TIP3P", "water_model": "TIP3P",
        "padding_nm": padding_nm, "ionic_strength_M": ionic_strength_m,
        "solute_atom_indices": solute_atoms, "n_atoms": modeller.topology.getNumAtoms(),
        "ion_atom_indices": ions,
        "box_vectors_nm": [[float(value / unit.nanometer) for value in vector]
                           for vector in modeller.topology.getPeriodicBoxVectors()],
        "status": "prepared; not yet an alchemical free-energy result",
    }
    json.dump(metadata, open(output / "metadata.json", "w"), indent=2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", default="experiments/bulk_fep/phosphate_gate.json")
    parser.add_argument("--out", default="results/bulk_fep/systems")
    parser.add_argument("--padding-nm", type=float, default=1.4)
    parser.add_argument("--ionic-strength-m", type=float, default=0.15)
    args = parser.parse_args()
    gate = json.load(open(args.gate))
    for pair in gate:
        for state in (pair["acid"], pair["base"]):
            directory = Path(args.out) / state["name"]
            meta = prepare(state, directory, args.padding_nm, args.ionic_strength_m)
            print(f"prepared {meta['name']}: q={meta['charge']}, {meta['n_atoms']} atoms", flush=True)


if __name__ == "__main__":
    main()
