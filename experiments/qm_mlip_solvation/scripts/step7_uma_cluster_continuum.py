#!/usr/bin/env python
"""Step 7: UMA-electronics grand-canonical cluster-continuum for the nucleotidyl
reaction (fix PPi solvation with EXPLICIT waters).

Prior explicit-water work (backup/explicit_water/grand_canonical_clusters.py)
solved the water-BOOKKEEPING problem (grand potential over a water ladder,
Ω = −RT log Σ exp(−(G_i − n_i·μ_water)/RT)) and was validated on pKa ions, but its
xTB electronics left phosphate MAE ~30–40 kJ. We REUSE its machinery (cluster
build, water ladder, μ_water, standard states, grand potential) and swap ONLY the
electronic energy to UMA:

    G_i^UMA = (G_xtb_free − E_xtb_elec) + E_UMA(cluster geometry)

i.e. keep xTB's (RRHO + ALPB) free-energy-of-solvation correction, replace the
electronic energy with UMA. Then grand-canonical per species and reaction ΔG.

  rxn01675/01005 truncated: MeP(-2) + MePPP(-3) -> MePPMe(-2) + PPi(-3), exp ~+2.
Reference: implicit-only UMA gave ΔG ~-28 (ALPB) .. -52 (CPCM-X); the PPi anion is
over-solvated by continuum. Question: do explicit first-shell waters + UMA fix it?

Run (uma env): CUDA_VISIBLE_DEVICES=0 python scripts/step7_uma_cluster_continuum.py
"""
import argparse
import json
import math
import os
import sys
import tempfile

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_THERMO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))   # .../thermodynamic_calc
sys.path.insert(0, os.path.join(_THERMO, "backup", "explicit_water"))
import grand_canonical_clusters as gc   # seed_waters, xtb_ohess, water_ladder_limit, ...
from batched_relax import load_uma, batched_energies, batched_fire
from step4e_targeted import pool_confs

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "artifacts")
GEOM = os.path.join(OUT, "geom_cluster")
os.makedirs(GEOM, exist_ok=True)
EV2KJ = 96.485
RT = gc.RT
XTB = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")

SPECIES = {"MeP": (-2, "COP(=O)([O-])[O-]"),
           "MePPP": (-3, "COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])O"),
           "MePPMe": (-2, "COP(=O)([O-])OP(=O)([O-])OC"),
           "PPi": (-3, "O=P([O-])([O-])OP(=O)([O-])O")}
EXP = 1.85


def best_bare_xyz(pu, name, q, smi, path, pool=48, keep=8):
    """UMA-relaxed lowest-energy bare conformer -> xyz for cluster seeding."""
    cands = pool_confs(smi, q, 1, pool)
    sel = [cands[i] for i in np.argsort(batched_energies(pu, cands))[:keep]]
    rel, E, conv = batched_fire(pu, sel, fmax=0.05, steps=300, stop_frac=0.9, return_converged=True)
    rel = [a for a, c in zip(rel, conv) if c]; E = E[conv]
    a = rel[int(np.argmin(E))]
    gc.write_xyz(a.get_chemical_symbols(), a.get_positions(), path)
    return path


def uma_energy_xyz(pu, xyz, q):
    """UMA single-point energy (kJ) on a cluster geometry file."""
    syms, coords = gc.read_xyz(xyz)
    a = Atoms(symbols=syms, positions=coords, info={"charge": int(q), "spin": 1})
    return float(batched_energies(pu, [a])[0]) * EV2KJ


def uma_grand_energy(pu, record, q, geom_dir, mu_water):
    """Rebuild the cluster record with UMA electronics, then grand potential."""
    rec = {"counts": {}}
    for count, data in record["counts"].items():
        minima = []
        for m in data["minima"]:
            xyzp = os.path.join(geom_dir, m["xyz"])
            if not os.path.isfile(xyzp):
                continue
            e_uma = uma_energy_xyz(pu, xyzp, q)
            g_uma = (m["G_kJ"] - m["E_kJ"]) + e_uma          # swap electronics: xtb(RRHO+ALPB) + UMA elec
            minima.append({"G_kJ": g_uma})
        rec["counts"][count] = {"minima": minima}
    G, occ = gc.grand_free_energy(rec, mu_water, cluster_standard_state_kj=gc.GAS_1ATM_TO_1M_KJ)
    return G, occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waters-per-site", type=int, default=2)
    ap.add_argument("--max-water", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)
    log("loading UMA...")
    pu = load_uma()

    # water reference (mu_water) — xtb ohess for (RRHO+ALPB), UMA for electronics
    wsym = ["O", "H", "H"]; wc = np.array([[0., 0., 0.], [0.96, 0., 0.], [-0.24, 0.93, 0.]])
    wref = gc.xtb_ohess(wsym, wc, 0, XTB, 2)
    if wref is None:
        raise RuntimeError("water xtb ohess failed")
    wfree, welec, wopt_s, wopt_c = wref
    wxyz = os.path.join(GEOM, "water.xyz"); gc.write_xyz(wopt_s, wopt_c, wxyz)
    e_uma_water = uma_energy_xyz(pu, wxyz, 0)
    g_water_uma = (wfree - welec) + e_uma_water
    mu_water = g_water_uma + gc.GAS_1ATM_TO_1M_KJ + RT * math.log(55.5)
    log(f"  mu_water(UMA) = {mu_water:.1f} kJ")

    Omega = {}
    for name, (q, smi) in SPECIES.items():
        bare = best_bare_xyz(pu, name, q, smi, os.path.join(GEOM, f"{name}_bare.xyz"))
        nmax = gc.water_ladder_limit(bare, q, a.waters_per_site, a.max_water)
        log(f"  {name} q{q:+d}: building clusters, water ladder 0..{nmax} ...")
        _, rec = gc._species_record(name, bare, q, nmax, a.seeds, 2, GEOM, 0.5, XTB)
        G, occ = uma_grand_energy(pu, rec, q, GEOM, mu_water)
        Omega[name] = G
        top = sorted(occ.items(), key=lambda kv: -kv[1])[:3]
        log(f"    Ω(UMA) = {G:.1f} kJ   water occupancy {[(k, round(v,2)) for k,v in top]}")

    dG = (Omega["MePPMe"] + Omega["PPi"]) - (Omega["MeP"] + Omega["MePPP"])
    log(f"\n==== nucleotidyl ΔG, UMA cluster-continuum (grand-canonical) ====")
    log(f"  ΔG = {dG:+.1f} kJ/mol   vs exp {EXP:+.1f}  ->  err {dG-EXP:+.1f}")
    log(f"  (implicit-only was -28 (ALPB) .. -52 (CPCM-X); this adds explicit first-shell waters)")
    json.dump(dict(Omega=Omega, dG=dG, exp=EXP, mu_water=mu_water,
                   waters_per_site=a.waters_per_site, max_water=a.max_water, seeds=a.seeds),
              open(os.path.join(OUT, "step7_uma_cluster_continuum.json"), "w"), indent=2)
    log("wrote artifacts/step7_uma_cluster_continuum.json")


if __name__ == "__main__":
    main()
