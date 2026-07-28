#!/usr/bin/env python
"""Generate and score charge-adaptive explicit-water cluster ensembles.

This is deliberately different from the archived fixed-``n`` experiment.  A
solute is sampled with a ladder of water counts and each state is combined as

    Omega = -RT log sum_i exp(-(G_i - n_i mu_water) / RT).

Consequently an acid and its conjugate base may prefer different first-shell
occupancies.  ``mu_water`` is obtained with the same xTB/ALPB/``--ohess``
protocol and the 55.5 M liquid-water activity convention.  The sampled minima
are a *seeded-cluster approximation* to the configurational integral: they are
deduplicated by free energy, not claimed to be an exhaustive solvent ensemble.

Use this first on pKa reference ions.  It is a validation gate, not yet a
reaction-energy correction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HARTREE_TO_KJ = 2625.499639
R_KJ = 8.314462618e-3
TEMPERATURE = 298.15
RT = R_KJ * TEMPERATURE
RTLN10 = RT * math.log(10.0)
MU_H = -1122.8
# xTB's RRHO free energy is 1 atm.  ALPB's solvation convention is 1 M, so
# each independently translating molecular entity needs this conversion.
GAS_1ATM_TO_1M_KJ = 7.93
COVALENT_R = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07}
BOND_TOL = 1.25


def read_xyz(path: str) -> tuple[list[str], np.ndarray]:
    lines = Path(path).read_text().splitlines()
    n_atoms = int(lines[0].split()[0])
    fields = [line.split() for line in lines[2:2 + n_atoms]]
    return [row[0] for row in fields], np.asarray([[float(x) for x in row[1:4]] for row in fields])


def write_xyz(symbols: list[str], coords: np.ndarray, path: str, comment: str = "") -> None:
    with open(path, "w") as handle:
        handle.write(f"{len(symbols)}\n{comment}\n")
        for symbol, point in zip(symbols, coords):
            handle.write(f"{symbol:<3} {point[0]:18.10f} {point[1]:18.10f} {point[2]:18.10f}\n")


def parent_heavy(symbols: list[str], coords: np.ndarray) -> dict[int, list[int]]:
    """Assign every H to its closest plausible covalent heavy-atom parent."""
    heavy = [i for i, symbol in enumerate(symbols) if symbol != "H"]
    children = {i: [] for i in heavy}
    for hydrogen, symbol in enumerate(symbols):
        if symbol != "H":
            continue
        candidates = []
        for atom in heavy:
            distance = float(np.linalg.norm(coords[hydrogen] - coords[atom]))
            cutoff = BOND_TOL * (COVALENT_R["H"] + COVALENT_R.get(symbols[atom], 0.7))
            if distance < cutoff:
                candidates.append((distance, atom))
        if candidates:
            children[min(candidates)[1]].append(hydrogen)
    return children


def anionic_sites(symbols: list[str], coords: np.ndarray) -> list[int]:
    """Return unprotonated O/S sites, with terminal atoms first."""
    children = parent_heavy(symbols, coords)
    ranked = []
    for i, symbol in enumerate(symbols):
        if symbol not in {"O", "S"} or children.get(i):
            continue
        neighbours = sum(
            symbol_j != "H" and j != i and
            np.linalg.norm(coords[i] - coords[j]) < BOND_TOL *
            (COVALENT_R.get(symbol, 0.7) + COVALENT_R.get(symbol_j, 0.7))
            for j, symbol_j in enumerate(symbols)
        )
        ranked.append((neighbours, i))
    return [i for _, i in sorted(ranked)]


def seed_waters(symbols: list[str], coords: np.ndarray, n_water: int,
                rng: np.random.Generator) -> tuple[list[str], np.ndarray]:
    """Put donor waters on unprotonated O/S sites, avoiding initial clashes."""
    sites = anionic_sites(symbols, coords) or [
        i for i, symbol in enumerate(symbols) if symbol in {"O", "S", "N"}
    ] or list(range(len(symbols)))
    out_symbols, out_coords = list(symbols), [np.array(point) for point in coords]
    for water_index in range(n_water):
        site = sites[water_index % len(sites)]
        heavy = [i for i, symbol in enumerate(symbols) if i != site and symbol != "H"]
        neighbour = min(heavy, key=lambda i: np.linalg.norm(coords[i] - coords[site]))
        axis = coords[site] - coords[neighbour]
        axis /= np.linalg.norm(axis) or 1.0
        best = None
        for _ in range(16):
            direction = axis + 0.35 * rng.normal(size=3)
            direction /= np.linalg.norm(direction) or 1.0
            oxygen = coords[site] + 2.75 * direction
            clearance = min(np.linalg.norm(oxygen - point) for point in out_coords)
            if best is None or clearance > best[0]:
                best = (clearance, oxygen, direction)
        _, oxygen, direction = best
        perpendicular = np.cross(direction, rng.normal(size=3))
        perpendicular /= np.linalg.norm(perpendicular) or 1.0
        out_symbols.extend(["O", "H", "H"])
        out_coords.extend([oxygen, oxygen - 0.96 * direction,
                           oxygen + 0.96 * (-0.33 * direction + 0.94 * perpendicular)])
    return out_symbols, np.asarray(out_coords)


def valid_cluster(symbols: list[str], coords: np.ndarray, n_solute: int,
                  n_water: int, solute_hydrogens: int) -> bool:
    children = parent_heavy(symbols, coords)
    water_oxygen = [n_solute + 3 * i for i in range(n_water)]
    if any(len(children.get(oxygen, [])) != 2 for oxygen in water_oxygen):
        return False
    return sum(len(hydrogens) for atom, hydrogens in children.items() if atom < n_solute) == solute_hydrogens


def xtb_ohess(symbols: list[str], coords: np.ndarray, charge: int, xtb: str,
              threads: int) -> tuple[float, float, list[str], np.ndarray] | None:
    workdir = tempfile.mkdtemp(prefix="grand_cluster_")
    try:
        write_xyz(symbols, coords, os.path.join(workdir, "in.xyz"))
        result = subprocess.run([xtb, "in.xyz", "--gfn", "2", "--alpb", "water",
                                 "--chrg", str(charge), "--uhf", "0", "--ohess"],
                                cwd=workdir, capture_output=True, text=True, check=False,
                                env={**os.environ, "OMP_NUM_THREADS": str(threads), "OMP_STACKSIZE": "4G"})
        free = re.search(r"TOTAL FREE ENERGY\s+(-?\d+\.\d+)", result.stdout)
        energy = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", result.stdout)
        optimized = os.path.join(workdir, "xtbopt.xyz")
        if free is None or energy is None or not os.path.isfile(optimized):
            return None
        opt_symbols, opt_coords = read_xyz(optimized)
        return (float(free.group(1)) * HARTREE_TO_KJ, float(energy.group(1)) * HARTREE_TO_KJ,
                opt_symbols, opt_coords)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _species_record(species: str, xyz: str, charge: int, max_water: int, seeds: int,
                    threads: int, geometry_dir: str, energy_merge_kj: float, xtb: str) -> tuple[str, dict]:
    symbols, coords = read_xyz(xyz)
    n_solute = len(symbols)
    h_reference = sum(symbol == "H" for symbol in symbols)
    stable_seed = int(hashlib.sha256(species.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(stable_seed)
    output = {"charge": charge, "source_xyz": xyz, "max_water": max_water, "counts": {}}
    for n_water in range(max_water + 1):
        minima = []
        attempts = 1 if n_water == 0 else seeds
        for seed in range(attempts):
            initial = (symbols, coords) if n_water == 0 else seed_waters(symbols, coords, n_water, rng)
            result = xtb_ohess(*initial, charge, xtb, threads)
            if result is None:
                continue
            free, electronic, opt_symbols, opt_coords = result
            if not valid_cluster(opt_symbols, opt_coords, n_solute, n_water, h_reference):
                continue
            if any(abs(free - item["G_kJ"]) < energy_merge_kj for item in minima):
                continue
            filename = f"{species}_n{n_water}_s{seed}.xyz"
            write_xyz(opt_symbols, opt_coords, os.path.join(geometry_dir, filename),
                      f"{species}; n_water={n_water}; G={free:.5f} kJ/mol")
            minima.append({"G_kJ": free, "E_kJ": electronic, "xyz": filename, "seed": seed})
        output["counts"][str(n_water)] = {"n_attempted": attempts, "n_minima": len(minima), "minima": minima}
    return species, output


def water_ladder_limit(xyz: str, charge: int, waters_per_anionic_site: int,
                       cap: int | None) -> int:
    """Choose a state-specific shell size from exposed anionic sites and charge.

    Geometry identifies sites where directional water donors can be placed;
    formal charge prevents a delocalised/mis-perceived ion from receiving too
    short a ladder.  A neutral/cationic calibration control gets n=0 unless a
    caller explicitly elects to model it separately.
    """
    if charge >= 0:
        return 0
    symbols, coords = read_xyz(xyz)
    site_count = max(abs(charge), len(anionic_sites(symbols, coords)))
    limit = waters_per_anionic_site * site_count
    return min(limit, cap) if cap is not None else limit


def species_from_pairs(pairs: list[dict], source: dict, requested: set[str] | None) -> dict[str, tuple[str, int]]:
    charges = {pair["acid"]: int(pair["q_acid"]) for pair in pairs}
    charges.update({pair["base"]: int(pair["q_base"]) for pair in pairs})
    output = {}
    for species, charge in charges.items():
        if requested and species not in requested:
            continue
        if species in source:
            output[species] = (source[species][0]["xyz"], charge)
    return output


def grand_free_energy(record: dict, mu_water_kj: float, temperature: float = TEMPERATURE,
                      cluster_standard_state_kj: float = 0.0) -> tuple[float, dict[str, float]]:
    """Grand potential and normalized occupancy by water count."""
    terms = []
    for count, data in record["counts"].items():
        for minimum in data["minima"]:
            terms.append((int(count), minimum["G_kJ"] + cluster_standard_state_kj -
                          int(count) * mu_water_kj))
    if not terms:
        raise ValueError("cluster record contains no valid minima")
    rt = R_KJ * temperature
    floor = min(value for _, value in terms)
    weights = [math.exp(-(value - floor) / rt) for _, value in terms]
    partition = sum(weights)
    populations: dict[str, float] = {}
    for (count, _), weight in zip(terms, weights):
        populations[str(count)] = populations.get(str(count), 0.0) + weight / partition
    return floor - rt * math.log(partition), populations


def pka_statistics(pairs: list[dict], grand_energies: dict[str, float]) -> dict:
    rows = []
    for pair in pairs:
        if pair["acid"] not in grand_energies or pair["base"] not in grand_energies:
            continue
        pka = (grand_energies[pair["base"]] + MU_H - grand_energies[pair["acid"]]) / RTLN10
        rows.append({"key": pair["key"], "kind": pair["kind"], "group": pair["group"],
                     "charge": abs(int(pair["q_base"])), "pka_exp": pair["pKa_exp"], "pka_calc": pka})
    cation_error = [(row["pka_calc"] - row["pka_exp"]) * RTLN10 for row in rows if row["kind"] == "cationic"]
    if not cation_error:
        raise ValueError("no cationic controls available for proton reference")
    shift = float(np.mean(cation_error))
    for row in rows:
        row["error_kJ"] = (row["pka_calc"] - row["pka_exp"]) * RTLN10 - shift
    anion = [row["error_kJ"] for row in rows if row["kind"] == "anionic"]
    phosphate = [row["error_kJ"] for row in rows if row["group"] == "phosphate"]
    return {"n_pairs": len(rows), "n_anions": len(anion), "proton_shift_kJ": shift,
            "anion_mae_kJ": float(np.mean(np.abs(anion))), "anion_bias_kJ": float(np.mean(anion)),
            "phosphate_mae_kJ": float(np.mean(np.abs(phosphate))) if phosphate else None, "rows": rows}


def make_ensemble(args: argparse.Namespace) -> None:
    pairs = json.load(open(args.pairs))
    source = json.load(open(args.source))
    requested = set(args.species.split(",")) if args.species else None
    selected = species_from_pairs(pairs, source, requested)
    geometry_dir = os.path.join(os.path.dirname(args.out), "geometries")
    os.makedirs(geometry_dir, exist_ok=True)
    completed = json.load(open(args.out)) if args.resume and os.path.isfile(args.out) else {"method": "xTB GFN2/ALPB water --ohess seeded cluster ensemble", "species": {}}
    todo = [
        (name, xyz, charge, water_ladder_limit(xyz, charge, args.waters_per_anionic_site, args.max_water))
        for name, (xyz, charge) in selected.items() if name not in completed["species"]
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_species_record, name, xyz, charge, max_water, args.seeds,
                               args.threads, geometry_dir, args.energy_merge_kj, args.xtb): name
                   for name, xyz, charge, max_water in todo}
        for future in as_completed(futures):
            name = futures[future]
            completed["species"][name] = future.result()[1]
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(completed, open(args.out, "w"), indent=2)
            print(f"completed {name}", flush=True)


def score_ensemble(args: argparse.Namespace) -> None:
    data = json.load(open(args.ensemble))
    water_symbols = ["O", "H", "H"]
    water_coords = np.array([[0., 0., 0.], [0.96, 0., 0.], [-0.24, 0.93, 0.]])
    water = xtb_ohess(water_symbols, water_coords, 0, args.xtb, args.threads)
    if water is None:
        raise RuntimeError("water reference xTB calculation failed")
    water_standard = water[0] + GAS_1ATM_TO_1M_KJ
    mu_water = water_standard + RT * math.log(55.5)
    pairs = json.load(open(args.pairs))
    charges = {pair["acid"]: int(pair["q_acid"]) for pair in pairs}
    charges.update({pair["base"]: int(pair["q_base"]) for pair in pairs})
    cation_control_species = {
        species for pair in pairs if pair["kind"] == "cationic"
        for species in (pair["acid"], pair["base"])
    }
    energies, occupancy = {}, {}
    for species, record in data["species"].items():
        try:
            if args.ensemble_cation_controls or species not in cation_control_species:
                energies[species], occupancy[species] = grand_free_energy(
                    record, mu_water, cluster_standard_state_kj=GAS_1ATM_TO_1M_KJ)
            else:
                # The pKa gate uses cations only to remove the proton-reference
                # offset.  They were already well described by continuum and
                # have no anionic acceptor sites for this water-placement model.
                bare = {"counts": {"0": record["counts"].get("0", {"minima": []})}}
                energies[species], occupancy[species] = grand_free_energy(
                    bare, mu_water, cluster_standard_state_kj=GAS_1ATM_TO_1M_KJ)
        except ValueError:
            continue
    result = {"method": "xTB cluster-continuum seeded grand-canonical ensemble",
              "mu_water_kJ": mu_water, "water_G_1M_kJ": water_standard,
              "cluster_standard_state_kJ": GAS_1ATM_TO_1M_KJ,
              "cation_controls": "ensemble" if args.ensemble_cation_controls else "bare n=0",
              "statistics": pka_statistics(pairs, energies), "occupancy": occupancy,
              "grand_energies_kJ": energies}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"anion MAE: {result['statistics']['anion_mae_kJ']:.2f} kJ/mol; "
          f"phosphate MAE: {result['statistics']['phosphate_mae_kJ']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="generate and retain cluster minima")
    build.add_argument("--pairs", required=True)
    build.add_argument("--source", required=True, help="pka_xtb.json source geometries")
    build.add_argument("--out", required=True)
    build.add_argument("--xtb", default=os.environ.get("XTB_BIN", "xtb"))
    build.add_argument("--species", help="comma-separated species subset")
    build.add_argument("--waters-per-anionic-site", type=int, default=2,
                       help="ladder length multiplier; default gives two waters per exposed anionic site")
    build.add_argument("--max-water", type=int,
                       help="optional safety cap on the charge/site-derived water ladder")
    build.add_argument("--seeds", type=int, default=8)
    build.add_argument("--energy-merge-kj", type=float, default=0.5)
    build.add_argument("--workers", type=int, default=4)
    build.add_argument("--threads", type=int, default=2)
    build.add_argument("--resume", action="store_true")
    score = commands.add_parser("score", help="assemble grand potentials and pKa gate")
    score.add_argument("--pairs", required=True)
    score.add_argument("--ensemble", required=True)
    score.add_argument("--out", required=True)
    score.add_argument("--xtb", default=os.environ.get("XTB_BIN", "xtb"))
    score.add_argument("--threads", type=int, default=2)
    score.add_argument("--ensemble-cation-controls", action="store_true",
                       help="also microsolvate cationic calibration controls (off by default)")
    args = parser.parse_args()
    {"build": make_ensemble, "score": score_ensemble}[args.command](args)


if __name__ == "__main__":
    main()
