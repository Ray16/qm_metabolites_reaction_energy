#!/usr/bin/env python
"""Ensemble-consistent openCOSMO-RS pKa gate.

For every selected xTB conformer, this combines the existing UMA gas electronic
energy and xTB RRHO correction with an ORCA r2SCAN-3c/openCOSMO-RS water
solvation free energy.  The final molecular G is Boltzmann-assembled from the
*same* conformers.  This avoids the invalid historical mixture of a single
COSMO-RS conformer with an ensemble-averaged gas/thermal term.
"""
from __future__ import annotations

import argparse
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

from experiments.explicit_water.grand_canonical_clusters import R_KJ, TEMPERATURE, pka_statistics

HARTREE_TO_KJ = 2625.499639
RT = R_KJ * TEMPERATURE


def read_xyz(path: str) -> tuple[list[str], list[list[float]]]:
    lines = Path(path).read_text().splitlines()
    n_atoms = int(lines[0].split()[0])
    rows = [line.split() for line in lines[2:2 + n_atoms]]
    return [row[0] for row in rows], [[float(value) for value in row[1:4]] for row in rows]


def cosmors_dgsolv(xyz: str, charge: int, orca: str, nprocs: int) -> float:
    symbols, coords = read_xyz(xyz)
    workdir = tempfile.mkdtemp(prefix="cosmors_pka_")
    try:
        lines = ["! r2SCAN-3c TightSCF", "%maxcore 3000", f"%pal nprocs {nprocs} end",
                 "%cosmors", '  solvent "water"', "end", f"* xyz {charge} 1"]
        lines.extend(f"  {symbol} {x:.8f} {y:.8f} {z:.8f}" for symbol, (x, y, z) in zip(symbols, coords))
        lines.append("*")
        Path(workdir, "job.inp").write_text("\n".join(lines) + "\n")
        result = subprocess.run([orca, "job.inp"], cwd=workdir, capture_output=True, text=True, check=False,
                                env={**os.environ, "OMP_NUM_THREADS": str(nprocs)})
        match = re.search(r"Free energy of solvation \(dGsolv\)\s*:\s*(-?\d+\.\d+)\s*Eh", result.stdout)
        if match is None:
            raise RuntimeError(f"ORCA COSMO-RS failed for {xyz}: {result.stdout[-500:]}")
        return float(match.group(1)) * HARTREE_TO_KJ
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def select_conformers(ensemble: list[dict], max_confs: int) -> list[dict]:
    """Keep the most populated baseline conformers; reweight after COSMO-RS."""
    return sorted(ensemble, key=lambda record: float(record.get("weight", 0.0)), reverse=True)[:max_confs]


def boltzmann(values: list[float]) -> float:
    minimum = min(values)
    return minimum - RT * math.log(sum(math.exp(-(value - minimum) / RT) for value in values))


def run_one(species: str, charge: int, records: list[dict], geometries: list[dict],
            orca: str, nprocs: int) -> tuple[str, dict]:
    geometry_by_conf = {int(record["conf"]): record["xyz"] for record in geometries}
    out = []
    for record in records:
        conf = int(record["conf"])
        if conf not in geometry_by_conf:
            continue
        dG_solv = cosmors_dgsolv(geometry_by_conf[conf], charge, orca, nprocs)
        total = float(record["E_UMA_kJ"]) + dG_solv + float(record["G_RRHO_kJ"])
        out.append({"conf": conf, "xyz": geometry_by_conf[conf], "E_UMA_kJ": record["E_UMA_kJ"],
                    "G_RRHO_kJ": record["G_RRHO_kJ"], "dGsolv_cosmors_kJ": dG_solv,
                    "G_aq_cosmors_kJ": total})
    if not out:
        raise RuntimeError(f"no matched conformers for {species}")
    return species, {"charge": charge, "conformers": out,
                     "G_aq_cosmors_kJ": boltzmann([record["G_aq_cosmors_kJ"] for record in out])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--gas-ensemble", required=True, help="G_aq_pka.json containing UMA and RRHO terms")
    parser.add_argument("--geometry-ensemble", required=True, help="pka_xtb.json with conformer XYZ paths")
    parser.add_argument("--orca", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--species", help="comma-separated pilot species; default is all pKa species")
    parser.add_argument("--max-confs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--nprocs", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    pairs = json.load(open(args.pairs))
    gas = json.load(open(args.gas_ensemble))
    geometry = json.load(open(args.geometry_ensemble))
    charges = {pair["acid"]: int(pair["q_acid"]) for pair in pairs}
    charges.update({pair["base"]: int(pair["q_base"]) for pair in pairs})
    requested = set(args.species.split(",")) if args.species else set(charges)
    result = json.load(open(args.out)) if args.resume and os.path.isfile(args.out) else {
        "method": "E_UMA + dGsolv_ORCA-r2SCAN-3c-openCOSMO-RS + xTB RRHO; matched conformer ensemble",
        "species": {},
    }
    todo = []
    for species in sorted(requested):
        if species in result["species"] or species not in charges or species not in gas or species not in geometry:
            continue
        todo.append((species, charges[species], select_conformers(gas[species]["conformers"], args.max_confs), geometry[species]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, species, charge, records, geometries, args.orca, args.nprocs): species
                   for species, charge, records, geometries in todo}
        for future in as_completed(futures):
            species, record = future.result()
            result["species"][species] = record
            energies = {name: data["G_aq_cosmors_kJ"] for name, data in result["species"].items()}
            try:
                result["statistics"] = pka_statistics(pairs, energies)
            except ValueError:
                pass
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(result, open(args.out, "w"), indent=2)
            print(f"completed {species}", flush=True)


if __name__ == "__main__":
    main()
