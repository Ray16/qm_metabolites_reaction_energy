#!/usr/bin/env python
"""Step 4d: BATCHED Boltzmann-ensemble ΔG for the glycosyl transfer (rxn00579).

Combines everything learned:
  - BATCHED UMA relaxation (batched_relax.batched_fire): ALL conformers of ALL
    species relaxed in one forward pass per step → 100 conformers ≈ cost of a few.
  - Per-conformer xTB-ALPB solvation (threaded; xtb is fast CPU) so BOTH gas and
    solvation are averaged (Step-4b showed the solvation term was the larger noise
    source when taken on a single min geometry).
  - Boltzmann ensemble free energy  G_ens = -kT ln Σ exp(-G_i/kT)  (correct
    conformational statistic; min was the fragile one that gave std 12.5).
  - Per-conformer energies CACHED to disk (artifacts/cache_conf/) so re-aggregation
    (min vs Boltzmann, adding conformers) is free — the scalable per-compound design.

Reports min vs Boltzmann ΔG per seed + reproducibility, vs experiment -4.2.
Run (uma env):  CUDA_VISIBLE_DEVICES=0 python scripts/step4d_batched_boltzmann.py \
                    --seeds 1,2,3,4,5 --nconf 48
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batched_relax import load_uma, batched_fire

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "artifacts")
CACHE = os.path.join(OUT, "cache_conf")
os.makedirs(CACHE, exist_ok=True)
EV2KJ = 96.485
HARTREE2KJ = 2625.4996
KT = 2.4789
THERMAL_FIXED = -0.8
EXP = -4.2

SPECIES = {
    "MeUDPGlc": (-2, "OC[C@H]1O[C@@H](OP(=O)([O-])OP(=O)([O-])OC)[C@H](O)[C@@H](O)[C@@H]1O"),
    "Fructose": (0,  "OC[C@H]1OC(O)(CO)[C@@H](O)[C@@H]1O"),
    "MeUDP":    (-2, "COP(=O)([O-])OP(=O)([O-])O"),
    "Suc":      (0,  "OC[C@H]1O[C@@H](O[C@]2(CO)O[C@H](CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"),
}


def boltz(Gs):
    Gs = np.asarray(Gs); ref = Gs.min()
    return float(ref - KT * np.log(np.exp(-(Gs - ref) / KT).sum()))


def gen_confs(smiles, q, seed, nconf):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3(); p.randomSeed = seed; p.pruneRmsThresh = 0.3
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=nconf, params=p))
    if not cids:
        AllChem.EmbedMolecule(m, randomSeed=seed, useRandomCoords=True); cids = [0]
    try:
        AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=300)
    except Exception:
        pass
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    out = []
    for c in cids:
        a = Atoms(symbols=syms, positions=m.GetConformer(c).GetPositions())
        a.info = {"charge": int(q), "spin": 1}
        out.append(a)
    return out


def xtb_dgsolv(atoms, q):
    with tempfile.TemporaryDirectory() as d:
        xyz = os.path.join(d, "m.xyz")
        with open(xyz, "w") as f:
            f.write(f"{len(atoms)}\n\n")
            for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
                f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")

        def e(solv):
            cmd = ["conda", "run", "-n", "xtb", "xtb", xyz, "--gfn", "2",
                   "--chrg", str(int(q)), "--sp"] + (["--alpb", solv] if solv else [])
            r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=400)
            m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", r.stdout)
            return float(m.group(1)) if m else None
        eg, ew = e(None), e("water")
        return (ew - eg) * HARTREE2KJ if (eg is not None and ew is not None) else None


def run_seed(pu, seed, nconf, log):
    """Batched relax all species' conformers together; per-conformer solvation;
    return per-species (E_gas list, dGsolv list)."""
    # 1. generate conformers for all species, tag by species
    tagged = []   # (species_name, ase.Atoms)
    for name, (q, smi) in SPECIES.items():
        for a in gen_confs(smi, q, seed, nconf):
            tagged.append((name, a))
    atoms_list = [a for _, a in tagged]
    log(f"  seed {seed}: {len(atoms_list)} conformers across {len(SPECIES)} species -> ONE batched relax")
    # 2. ONE batched relaxation of everything
    _, E_ev = batched_fire(pu, atoms_list, fmax=0.03, steps=500,
                           verbose=True, log_every=100, label=f"seed{seed}")
    E_gas = E_ev * EV2KJ
    # 3. per-conformer xTB solvation (threaded)
    charges = [SPECIES[nm][0] for nm, _ in tagged]
    with ThreadPoolExecutor(max_workers=8) as ex:
        dGsolv = list(ex.map(lambda ia: xtb_dgsolv(ia[1], charges[ia[0]]),
                             list(enumerate(atoms_list))))
    # 4. group back per species
    agg = {n: {"Eg": [], "Gt": []} for n in SPECIES}
    for (name, _), eg, ds in zip(tagged, E_gas, dGsolv):
        if ds is None or not np.isfinite(eg):
            continue
        agg[name]["Eg"].append(float(eg)); agg[name]["Gt"].append(float(eg + ds))
    # cache per compound+seed
    json.dump(agg, open(os.path.join(CACHE, f"seed{seed}_n{nconf}.json"), "w"))
    return agg


def dG(agg, kind):
    def val(n):
        Gt = np.array(agg[n]["Gt"])
        return float(Gt.min()) if kind == "min" else boltz(Gt)
    return (val("MeUDP") + val("Suc")) - (val("MeUDPGlc") + val("Fructose")) + THERMAL_FIXED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--nconf", type=int, default=48)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    def log(m): print(m, flush=True)
    log(f"loading UMA...  seeds={seeds} nconf={a.nconf}")
    pu = load_uma()

    rows = []
    for seed in seeds:
        agg = run_seed(pu, seed, a.nconf, log)
        r = dict(seed=seed, dG_min=dG(agg, "min"), dG_boltz=dG(agg, "boltz"),
                 nconf={n: len(agg[n]["Gt"]) for n in SPECIES})
        rows.append(r)
        log(f"  seed {seed}:  ΔG_min {r['dG_min']:7.1f}   ΔG_boltz {r['dG_boltz']:7.1f} kJ  (exp {EXP})")

    mn = np.array([r["dG_min"] for r in rows]); bz = np.array([r["dG_boltz"] for r in rows])
    log(f"\n==== BATCHED Boltzmann, rxn00579, {len(seeds)} seeds x {a.nconf} conf ====")
    log(f"  ΔG_min    mean {mn.mean():6.1f}  std {mn.std():5.1f}  range {mn.max()-mn.min():5.1f}  err {mn.mean()-EXP:+.1f}")
    log(f"  ΔG_boltz  mean {bz.mean():6.1f}  std {bz.std():5.1f}  range {bz.max()-bz.min():5.1f}  err {bz.mean()-EXP:+.1f}")
    log(f"  (Step-4b min-only, 24 conf, un-batched: std 12.5, range 37)")
    json.dump(dict(seeds=seeds, nconf=a.nconf, rows=rows, exp=EXP,
                   min=dict(mean=float(mn.mean()), std=float(mn.std())),
                   boltz=dict(mean=float(bz.mean()), std=float(bz.std()))),
              open(os.path.join(OUT, f"step4d_batched_boltzmann_n{a.nconf}.json"), "w"), indent=2)
    log(f"wrote artifacts/step4d_batched_boltzmann_n{a.nconf}.json")


if __name__ == "__main__":
    main()
