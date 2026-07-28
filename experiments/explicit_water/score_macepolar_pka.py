#!/usr/bin/env python
"""Score archived explicit-water pKa clusters with MACE-POLAR.

The cluster JSON records xTB total and Gibbs energies.  Replacing only their
electronic term yields a controlled MACE-POLAR/xTB-ALPB composite, without
mixing a bare-solute MLIP energy with an xTB water-binding correction.
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
from pathlib import Path

import numpy as np

EV_TO_KJ = 96.48533212
RTLN10 = 8.314462618e-3 * 298.15 * math.log(10.0)
MU_H = -1122.8


def water_count(path: str) -> int:
    name = Path(path).stem
    marker = "microsolv_n"
    if marker not in name:
        raise ValueError(f"cannot infer water count from {path}")
    return int(name.split(marker, 1)[1])


def charges_from_pairs(pairs: list[dict]) -> dict[str, int]:
    charges = {}
    for pair in pairs:
        charges[pair["acid"]] = int(pair["q_acid"])
        charges[pair["base"]] = int(pair["q_base"])
    return charges


def xtb_gas_energy_kJ(xyz: str, charge: int, xtb_bin: str) -> float:
    """Recover the gas electronic energy needed for the ALPB correction."""
    workdir = tempfile.mkdtemp(prefix="cluster_gas_")
    try:
        local_xyz = os.path.join(workdir, "cluster.xyz")
        shutil.copyfile(xyz, local_xyz)
        result = subprocess.run(
            [xtb_bin, "cluster.xyz", "--gfn", "2", "--chrg", str(charge), "--uhf", "0"],
            cwd=workdir, capture_output=True, text=True, check=False,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OMP_STACKSIZE": "4G"},
        )
        match = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", result.stdout)
        if match is None:
            raise RuntimeError(f"xTB gas single point failed for {xyz}: {result.stderr[-400:]}")
        return float(match.group(1)) * 2625.499639
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def pka_statistics(pairs: list[dict], energies: dict[str, float]) -> dict:
    """Return cation-referenced pKa errors, leaving absent/invalid pairs out."""
    rows = []
    for pair in pairs:
        acid, base = pair["acid"], pair["base"]
        if acid not in energies or base not in energies:
            continue
        calculated = (energies[base] + MU_H - energies[acid]) / RTLN10
        rows.append({"key": pair["key"], "kind": pair["kind"], "group": pair["group"],
                     "charge": abs(int(pair["q_base"])), "pka_exp": pair["pKa_exp"],
                     "pka_calc": calculated})
    cation_errors = [(row["pka_calc"] - row["pka_exp"]) * RTLN10
                      for row in rows if row["kind"] == "cationic"]
    if not cation_errors:
        raise ValueError("no cationic control pairs survived")
    shift = float(np.mean(cation_errors))
    for row in rows:
        row["error_kJ"] = (row["pka_calc"] - row["pka_exp"]) * RTLN10 - shift
    anion = [row["error_kJ"] for row in rows if row["kind"] == "anionic"]
    phosphate = [row["error_kJ"] for row in rows
                 if row["kind"] == "anionic" and row["group"] == "phosphate"]
    return {"n_pairs": len(rows), "n_anions": len(anion), "proton_shift_kJ": shift,
            "anion_mae_kJ": float(np.mean(np.abs(anion))),
            "anion_bias_kJ": float(np.mean(anion)),
            "phosphate_mae_kJ": float(np.mean(np.abs(phosphate))) if phosphate else None,
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True, help="curated pKa-pair JSON")
    ap.add_argument("--clusters", required=True, nargs="+", help="microsolv_n*.json files")
    ap.add_argument("--geometry-root", required=True,
                    help="directory holding the cluster XYZ files; stale JSON paths are ignored")
    ap.add_argument("--model", required=True)
    ap.add_argument("--xtb", default=os.environ.get("XTB_BIN", "xtb"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/explicit_water/macepolar_pka_gate.json")
    args = ap.parse_args()

    from ase.io import read
    from mace.calculators import mace_polar

    pairs = json.load(open(args.pairs))
    charge = charges_from_pairs(pairs)
    calculator = mace_polar(model=args.model, device=args.device, default_dtype="float64")
    result = {"method": "E_MACE-POLAR(cluster) + (G_xTB,ALPB - E_xTB,gas)",
              "water_counts": {}}

    for cluster_path in args.clusters:
        count = water_count(cluster_path)
        records = json.load(open(cluster_path))
        energy = {}
        detail = {}
        for species, record in sorted(records.items()):
            if not record or species not in charge:
                continue
            xyz = os.path.join(args.geometry_root, os.path.basename(record["xyz"]))
            if not os.path.isfile(xyz):
                raise FileNotFoundError(f"{species}: {xyz}")
            atoms = read(xyz)
            atoms.info["charge"] = charge[species]
            atoms.info["spin"] = 1
            atoms.calc = calculator
            e_mace = float(atoms.get_potential_energy()) * EV_TO_KJ
            e_xtb_gas = xtb_gas_energy_kJ(xyz, charge[species], args.xtb)
            g_hybrid = e_mace + (float(record["G"]) - e_xtb_gas)
            energy[species] = g_hybrid
            detail[species] = {"E_MACE_kJ": e_mace, "E_xTB_gas_kJ": e_xtb_gas,
                               "G_hybrid_kJ": g_hybrid,
                               "n_valid": record.get("n_valid")}
            print(f"n={count} {species:24} E_MACE={e_mace:12.1f}", flush=True)
        stats = pka_statistics(pairs, energy)
        result["water_counts"][str(count)] = {"statistics": stats, "species": detail}
        print(f"n={count}: anion MAE={stats['anion_mae_kJ']:.1f} kJ/mol "
              f"(n={stats['n_anions']}, phosphate={stats['phosphate_mae_kJ']})", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
