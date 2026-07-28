#!/usr/bin/env python
"""Score a charge-adaptive cluster ensemble with the production composite.

For every retained cluster minimum this replaces xTB's gas electronic energy
with MACE-POLAR while retaining the xTB/ALPB free-energy correction:

    G_i = E_MACE-POLAR(cluster)_i + [G_xTB,ALPB(cluster)_i - E_xTB,gas(cluster)_i]

Those ``G_i`` values are then assembled with the grand-canonical water
ensemble.  This is the fair pKa gate for the production reaction pipeline;
the xTB-only ensemble score is only a sampling diagnostic.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from experiments.explicit_water.grand_canonical_clusters import (
    GAS_1ATM_TO_1M_KJ, RT, TEMPERATURE, grand_free_energy, pka_statistics,
    xtb_ohess,
)

HARTREE_TO_KJ = 2625.499639
EV_TO_KJ = 96.48533212


def xtb_gas_energy(xyz: str, charge: int, xtb: str) -> float:
    workdir = tempfile.mkdtemp(prefix="grand_cluster_gas_")
    try:
        local = os.path.join(workdir, "cluster.xyz")
        shutil.copyfile(xyz, local)
        result = subprocess.run([xtb, "cluster.xyz", "--gfn", "2", "--chrg", str(charge),
                                 "--uhf", "0"], cwd=workdir, capture_output=True, text=True,
                                check=False, env={**os.environ, "OMP_NUM_THREADS": "1", "OMP_STACKSIZE": "4G"})
        match = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", result.stdout)
        if match is None:
            raise RuntimeError(f"xTB gas single point failed for {xyz}: {result.stderr[-300:]}")
        return float(match.group(1)) * HARTREE_TO_KJ
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--ensemble", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--xtb", default=os.environ.get("XTB_BIN", "xtb"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", help="resumable per-minimum MACE/xTB cache")
    args = parser.parse_args()

    from ase import Atoms
    from ase.io import read
    from mace.calculators import mace_polar

    ensemble = json.load(open(args.ensemble))
    pairs = json.load(open(args.pairs))
    charges = {pair["acid"]: int(pair["q_acid"]) for pair in pairs}
    charges.update({pair["base"]: int(pair["q_base"]) for pair in pairs})
    cation_control_species = {species for pair in pairs if pair["kind"] == "cationic"
                              for species in (pair["acid"], pair["base"])}
    root = Path(args.ensemble).parent / "geometries"
    cache_path = args.cache or f"{args.out}.cache.json"
    cache = json.load(open(cache_path)) if os.path.isfile(cache_path) else {}
    calculator = mace_polar(model=args.model, device=args.device, default_dtype="float64")

    hybrid: dict[str, dict] = {}
    for species, record in sorted(ensemble["species"].items()):
        if species not in charges:
            continue
        counts = {}
        for n_water, data in record["counts"].items():
            minima = []
            for minimum in data["minima"]:
                key = f"{species}:{n_water}:{minimum['xyz']}"
                xyz = str(root / minimum["xyz"])
                if key not in cache:
                    atoms = read(xyz)
                    atoms.info["charge"] = charges[species]
                    atoms.info["spin"] = 1
                    atoms.calc = calculator
                    e_mace = float(atoms.get_potential_energy()) * EV_TO_KJ
                    e_gas = xtb_gas_energy(xyz, charges[species], args.xtb)
                    cache[key] = {"E_MACE_kJ": e_mace, "E_xTB_gas_kJ": e_gas}
                    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                    json.dump(cache, open(cache_path, "w"), indent=1)
                terms = cache[key]
                minima.append({"G_kJ": terms["E_MACE_kJ"] + minimum["G_kJ"] - terms["E_xTB_gas_kJ"],
                               "xyz": minimum["xyz"]})
            counts[n_water] = {"minima": minima}
        hybrid[species] = {"counts": counts}
        print(f"scored {species}", flush=True)

    # A grand-canonical ensemble changes atom count.  Its water chemical
    # potential must therefore use the *same composite zero* as the clusters;
    # using xTB water with MACE cluster energies leaves an uncancelled MACE
    # atomic reference per water and is catastrophically wrong.
    water_positions = __import__("numpy").array([[0., 0., 0.], [0.96, 0., 0.],
                                                   [-0.24, 0.93, 0.]])
    water = xtb_ohess(["O", "H", "H"], water_positions, 0, args.xtb, 2)
    if water is None:
        raise RuntimeError("xTB water reference failed")
    water_atoms = Atoms(symbols=["O", "H", "H"], positions=water_positions)
    water_atoms.info["charge"] = 0
    water_atoms.info["spin"] = 1
    water_atoms.calc = calculator
    water_mace = float(water_atoms.get_potential_energy()) * EV_TO_KJ
    # Keep an xTB gas calculation here rather than infer it from --ohess:
    # the composite definition uses the gas electronic single point explicitly.
    water_xyz = Path(tempfile.mkdtemp(prefix="water_reference_")) / "water.xyz"
    try:
        from experiments.explicit_water.grand_canonical_clusters import write_xyz
        write_xyz(["O", "H", "H"], water_positions, str(water_xyz))
        water_gas = xtb_gas_energy(str(water_xyz), 0, args.xtb)
    finally:
        shutil.rmtree(str(water_xyz.parent), ignore_errors=True)
    water_composite = water_mace + water[0] - water_gas
    mu_water = water_composite + GAS_1ATM_TO_1M_KJ + RT * __import__("math").log(55.5)
    energy, population = {}, {}
    for species, record in hybrid.items():
        selected = record
        if species in cation_control_species:
            selected = {"counts": {"0": record["counts"].get("0", {"minima": []})}}
        try:
            energy[species], population[species] = grand_free_energy(
                selected, mu_water, cluster_standard_state_kj=GAS_1ATM_TO_1M_KJ)
        except ValueError:
            continue
    output = {"method": "MACE-POLAR(cluster) + G_xTB,ALPB(cluster) - E_xTB,gas(cluster)",
              "ensemble": "grand canonical, charge/site-dependent water ladder",
              "water_composite_G_kJ": water_composite, "mu_water_kJ": mu_water,
              "n_species": len(energy), "statistics": pka_statistics(pairs, energy),
              "occupancy": population, "cache": cache_path}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}; anion MAE={output['statistics']['anion_mae_kJ']:.2f} kJ/mol")


if __name__ == "__main__":
    main()
