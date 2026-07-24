#!/usr/bin/env python
"""Score a conformer ensemble with MACE-POLAR-1 instead of UMA (macepolar env).

MACE-POLAR-1 (Batatia et al., arXiv:2602.19411) extends MACE with explicit
long-range electrostatics and polarisable induction, plus global charge
equilibration via learnable Fukui functions, and is trained on the same OMol25
wB97M-V data as UMA. It therefore swaps in exactly where UMA sits in our
composite -- the GAS-PHASE ELECTRONIC term -- and nowhere else:

    G_aq = E_elec(gas)  <-- UMA or MACE-POLAR-1
         + dGsolv(ALPB)     unchanged (xtb)
         + G_RRHO           unchanged (xtb)

That matters for interpreting the result. Being polarisable does NOT supply an
implicit-solvation model, so the anion-solvation error measured against
experimental pKa lives in the ALPB term and is untouched by this swap. What a
polarisable model can fix is charge localisation and long-range electrostatics
in charged and multi-fragment systems -- which is why it is the right engine for
an explicit-microsolvation route later.

Reuses the identical geometries, sharding and Boltzmann machinery as the UMA
scorer so the two differ in one term only.

Usage:
    /homes/rzhu/miniforge3/envs/macepolar/bin/python run_macepolar_parallel.py \
        --ens ../pipeline/ensemble_deep_xtb.json \
        --tag macepolar_deep [--model ../models/MACE-POLAR-1-L.model] [--per-gpu 2]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
BENCH = os.path.join(THERMO, "pipeline")
sys.path.insert(0, THERMO)

PY = sys.executable
EV_TO_KJ = 96.48533212
RT = 8.314462618e-3 * 298.15

_CALC = None


def calculator(model_path):
    global _CALC
    if _CALC is None:
        # must go through mace_polar: it sets model_type="PolarMACE".
        from mace.calculators import mace_polar
        _CALC = mace_polar(model=model_path, device="cuda",
                           default_dtype="float64")
    return _CALC


def read_xyz_atoms(path):
    from ase import Atoms
    lines = open(path).read().splitlines()
    n = int(lines[0].split()[0])
    syms, pos = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        syms.append(p[0])
        pos.append([float(p[1]), float(p[2]), float(p[3])])
    return Atoms(symbols=syms, positions=pos)


def gas_energy_kJ(xyz_path, chg, model_path):
    """Total charge must reach the model: PolarMACE equilibrates charge globally."""
    atoms = read_xyz_atoms(xyz_path)
    atoms.info["charge"] = int(chg)
    atoms.info["spin"] = 1
    atoms.calc = calculator(model_path)
    return float(atoms.get_potential_energy()) * EV_TO_KJ


def relax_and_score(xyz_path, chg, model_path, fmax=0.05, steps=200):
    """Relax with the ML potential, then re-derive dGsolv at the NEW geometry.

    Two caveats, both real:
      * The incoming geometry is an xtb/ALPB *solution-phase* minimum. These
        models are gas-phase potentials, so relaxing can collapse a polyanion
        into a charge-paired conformation that does not exist in water. Compare
        against the single-point result before trusting it.
      * dGsolv and G_RRHO were computed at the old geometry. dGsolv is
        recomputed here (two cheap xtb single points); G_RRHO is left at its
        shared per-compound value, as in the rest of the pipeline.
    """
    import re
    import shutil
    import tempfile as tf
    from ase.optimize import LBFGS
    from qm_thermo import config

    atoms = read_xyz_atoms(xyz_path)
    atoms.info["charge"] = int(chg)
    atoms.info["spin"] = 1
    atoms.calc = calculator(model_path)
    LBFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    e_kJ = float(atoms.get_potential_energy()) * EV_TO_KJ

    wd = tf.mkdtemp(prefix="mp_relax_")
    try:
        with open(os.path.join(wd, "in.xyz"), "w") as fh:
            fh.write(f"{len(atoms)}\nrelaxed\n")
            for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
                fh.write(f"{s:<3s} {p[0]:>18.10f} {p[1]:>18.10f} {p[2]:>18.10f}\n")
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OMP_STACKSIZE": "4G"}

        def sp(extra):
            r = subprocess.run([config.XTB_BIN, "in.xyz", "--gfn", "2",
                                "--chrg", str(int(chg)), "--uhf", "0", *extra],
                               cwd=wd, capture_output=True, text=True, env=env)
            m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", r.stdout)
            return float(m.group(1)) if m else None

        e_alpb, e_gas = sp(["--alpb", "water"]), sp([])
        dgsolv = None if (e_alpb is None or e_gas is None) else \
            (e_alpb - e_gas) * 2625.499639
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    return e_kJ, dgsolv


def boltzmann(g_list):
    gmin = min(g_list)
    z = [math.exp(-(g - gmin) / RT) for g in g_list]
    Z = sum(z)
    return gmin - RT * math.log(Z), [x / Z for x in z]


def n_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        return max(1, len([l for l in out.stdout.splitlines() if l.strip()]))
    except Exception:
        return 1


def shard_by_cost(items, cost, n):
    bins = [[] for _ in range(n)]
    load = [0] * n
    for it in sorted(items, key=cost, reverse=True):
        k = load.index(min(load))
        bins[k].append(it)
        load[k] += cost(it)
    return [b for b in bins if b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=os.path.join(THERMO, "models",
                                                    "MACE-POLAR-1-L.model"))
    ap.add_argument("--per-gpu", type=int, default=1)
    ap.add_argument("--relax", action="store_true",
                    help="relax each conformer with the ML potential and recompute "
                         "dGsolv at the new geometry (gas-phase relax: see caveats)")
    ap.add_argument("--species", default=os.path.join(BENCH, "species.json"))
    ap.add_argument("--metabolites", help="species-only mode: metabolites json "
                    "(list of {id,name,smiles,charge}); scores G_aq and skips reactions")
    ap.add_argument("--_shard")
    ap.add_argument("--_cpds")
    args = ap.parse_args()

    ensemble = json.load(open(args.ens))
    root = os.path.dirname(os.path.abspath(args.ens))
    if args.metabolites and not args._shard:
        mets = {m["id"]: m for m in json.load(open(args.metabolites))}
        args.species = os.path.join(tempfile.mkdtemp(prefix="mp_spec_"), "species.json")
        json.dump(mets, open(args.species, "w"))
    spec = json.load(open(args.species))

    if args._shard:                                     # worker
        shard = {}
        for c in args._cpds.split(","):
            chg = int(spec[c]["charge"])
            paths = [cf["xyz"] if os.path.isabs(cf["xyz"])
                     else os.path.join(root, cf["xyz"]) for cf in ensemble[c]]
            if args.relax:
                shard[c] = [relax_and_score(p, chg, args.model) for p in paths]
            else:
                shard[c] = [gas_energy_kJ(p, chg, args.model) for p in paths]
            print(f"  [{os.environ.get('CUDA_VISIBLE_DEVICES','?')}] {c} "
                  f"{len(shard[c])} conf", flush=True)
        json.dump(shard, open(args._shard, "w"))
        return

    if args.metabolites:
        reactions, need = None, sorted(ensemble)
    else:
        reactions = json.load(open(os.path.join(BENCH, "reactions.json")))
        need = sorted({c for st in reactions.values() for c in st if c != "cpd00067"})
    missing = [c for c in need if c not in ensemble]
    if missing:
        raise SystemExit(f"ensemble missing: {missing}")

    nw = n_gpus() * max(1, args.per_gpu)
    shards = shard_by_cost(need, lambda c: len(ensemble[c]), nw)
    print(f"=== MACE-POLAR-1 | {os.path.basename(args.model)} | {len(need)} compounds"
          f" / {sum(len(ensemble[c]) for c in need)} conformers | {len(shards)} workers"
          f" on {n_gpus()} GPUs ===", flush=True)

    tmp = tempfile.mkdtemp(prefix="mp_shards_")
    procs, paths = [], []
    _relax_flag = ["--relax"] if args.relax else []
    for i, cpds in enumerate(shards):
        path = os.path.join(tmp, f"shard{i}.json")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i % n_gpus())}
        procs.append(subprocess.Popen(
            [PY, os.path.abspath(__file__), "--ens", args.ens, "--tag", args.tag,
             "--model", args.model, "--species", args.species, "--_shard", path,
             "--_cpds", ",".join(cpds), *_relax_flag], env=env))
        paths.append(path)
    for p in procs:
        p.wait()
    if any(p.returncode for p in procs):
        raise SystemExit(f"{sum(1 for p in procs if p.returncode)} worker(s) failed")

    e_elec = {}
    for path in paths:
        e_elec.update(json.load(open(path)))

    from qm_thermo import config
    from qm_thermo.reactions import Reaction, SpeciesInfo, reaction_dG

    G_aq, breakdown = {}, {}
    for c in need:
        per_conf, g_list = [], []
        for cf, e in zip(ensemble[c], e_elec[c]):
            # --relax returns (energy, dGsolv-at-relaxed-geometry)
            dgs = cf["dGsolv_kJ"]
            if isinstance(e, (list, tuple)):
                e, dgs_new = e
                if dgs_new is not None:
                    dgs = dgs_new
            g = e + dgs + cf["G_RRHO_kJ"]
            g_list.append(g)
            per_conf.append(dict(conf=cf["conf"], E_elec_kJ=e,
                                 dGsolv_kJ=dgs,
                                 G_RRHO_kJ=cf["G_RRHO_kJ"], G_aq_kJ=g))
        g_ens, w = boltzmann(g_list)
        for pc, wi in zip(per_conf, w):
            pc["weight"] = wi
        G_aq[c] = g_ens
        breakdown[c] = dict(name=spec[c]["name"], charge=int(spec[c]["charge"]),
                            n_conf=len(g_list), n_eff=1.0 / sum(x * x for x in w),
                            G_aq_kJ=g_ens, single_conf_G_aq_kJ=min(g_list),
                            conformers=per_conf)
        print(f"{c} q={spec[c]['charge']:+d}: {len(g_list):3d} conf  "
              f"G_aq={g_ens:12.1f}  ({spec[c]['name']})")

    out_bd = os.path.join(HERE, f"G_aq_{args.tag}.json")
    json.dump(breakdown, open(out_bd, "w"), indent=2)
    if reactions is None:
        print(f"\nwrote {out_bd}")
        return

    species = {c: SpeciesInfo(c, n_hydrogens=int(spec[c]["n_hydrogens"]),
                              charge=int(spec[c]["charge"])) for c in spec}
    rows = {rid: reaction_dG(Reaction(rid, st), G_aq, species,
                             conditions=config.DEFAULT_CONDITIONS).dG_transformed_kJ
            for rid, st in reactions.items()}
    bench = os.path.join(THERMO, "results", "benchmark")
    os.makedirs(bench, exist_ok=True)
    json.dump(rows, open(os.path.join(
        bench, f"qm_reaction_dG_{args.tag}.json"), "w"), indent=2)
    print(f"\nwrote {out_bd}")


if __name__ == "__main__":
    main()
