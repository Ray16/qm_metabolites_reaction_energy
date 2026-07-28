#!/usr/bin/env python
"""Multi-GPU Stage B: UMA gas energies over a conformer ensemble, in parallel.

The serial scorer walks conformers one at a time on a single GPU. With the deep
ensembles (~590 conformers instead of ~126) that dominates wall-clock, so this
shards compounds across every visible GPU (optionally several workers per GPU --
uma-s is small and a V100 fits many copies) and merges the results.

Sharding is by conformer count, longest-first, so the GPUs finish together.
Each worker writes its own shard file, so a crashed worker costs only its shard.

Usage:
    python run_uma_ensemble_parallel.py \
        --ens ../pipeline/ensemble_deep_xtb.json \
        --tag ensemble_deep [--per-gpu 2]

Writes: mlip/G_aq_<tag>.json
        results/benchmark/qm_reaction_dG_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
BENCH = os.path.join(THERMO, "pipeline")
sys.path.insert(0, THERMO)

PY = sys.executable


# --------------------------------------------------------------------------
# worker: compute E_UMA for an explicit list of compounds, write a shard
# --------------------------------------------------------------------------
def worker(ens_json, species_json, cpds, out_path):
    from mlip import uma

    ensemble = json.load(open(ens_json))
    spec = json.load(open(species_json))
    # geometry paths may be stored relative to wherever the builder ran; anchor
    # them to the ensemble file so workers can run from any cwd
    root = os.path.dirname(os.path.abspath(ens_json))
    shard = {}
    for c in cpds:
        chg = int(spec[c]["charge"])
        shard[c] = [uma.gas_energy_kJ(
            cf["xyz"] if os.path.isabs(cf["xyz"]) else os.path.join(root, cf["xyz"]),
            chg) for cf in ensemble[c]]
        print(f"  [{os.environ.get('CUDA_VISIBLE_DEVICES','?')}] {c} "
              f"{len(shard[c])} conf", flush=True)
    json.dump(shard, open(out_path, "w"))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def n_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        return max(1, len([l for l in out.stdout.splitlines() if l.strip()]))
    except Exception:
        return 1


def shard_by_cost(items, cost, n):
    """Longest-processing-time-first bin packing, so shards finish together."""
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
    ap.add_argument("--per-gpu", type=int, default=1)
    ap.add_argument("--species", default=os.path.join(BENCH, "species.json"))
    ap.add_argument("--metabolites", help="species-only mode: a metabolites json "
                    "(list of {id,name,smiles,charge}); scores G_aq per species "
                    "and skips the reaction step")
    ap.add_argument("--_shard")          # internal: worker mode
    ap.add_argument("--_cpds")
    args = ap.parse_args()

    if args._shard:      # ---- worker process: --species is already resolved ----
        worker(args.ens, args.species, args._cpds.split(","), args._shard)
        return

    # species-only mode uses the metabolites file as the charge source; workers
    # are handed the resolved path, so they never redo this
    if args.metabolites:
        mets = {m["id"]: m for m in json.load(open(args.metabolites))}
        spec_path = os.path.join(tempfile.mkdtemp(prefix="uma_spec_"), "species.json")
        json.dump(mets, open(spec_path, "w"))
        args.species = spec_path

    ensemble = json.load(open(args.ens))
    if args.metabolites:
        reactions = None
        need = sorted(ensemble)
    else:
        reactions = json.load(open(os.path.join(BENCH, "reactions.json")))
        need = sorted({c for st in reactions.values() for c in st if c != "cpd00067"})
    missing = [c for c in need if c not in ensemble]
    if missing:
        raise SystemExit(f"ensemble missing: {missing}")

    nw = n_gpus() * max(1, args.per_gpu)
    shards = shard_by_cost(need, lambda c: len(ensemble[c]), nw)
    total = sum(len(ensemble[c]) for c in need)
    print(f"=== UMA parallel | {len(need)} compounds / {total} conformers "
          f"| {len(shards)} workers on {n_gpus()} GPUs ===", flush=True)

    tmp = tempfile.mkdtemp(prefix="uma_shards_")
    procs, paths = [], []
    for i, cpds in enumerate(shards):
        path = os.path.join(tmp, f"shard{i}.json")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i % n_gpus())}
        procs.append(subprocess.Popen(
            [PY, os.path.abspath(__file__), "--ens", args.ens, "--tag", args.tag,
             "--species", args.species, "--_shard", path, "--_cpds", ",".join(cpds)],
            env=env))
        paths.append(path)
    for p in procs:
        p.wait()
    failed = [p.returncode for p in procs if p.returncode != 0]
    if failed:
        raise SystemExit(f"{len(failed)} worker(s) failed: {failed}")

    e_uma = {}
    for path in paths:
        e_uma.update(json.load(open(path)))

    # ---- Boltzmann + reactions (cheap, serial) ----
    from qm_thermo.composite import ConformerTerms, boltzmann_ensemble
    from qm_thermo import config
    from qm_thermo.reactions import Reaction, SpeciesInfo, reaction_dG

    spec = json.load(open(args.species))
    G_aq, breakdown = {}, {}
    for c in need:
        chg = int(spec[c]["charge"])
        per_conf, g_list = [], []
        for cf, e in zip(ensemble[c], e_uma[c]):
            g = ConformerTerms(e, cf["dGsolv_kJ"], cf["G_RRHO_kJ"]).aqueous_gibbs_kJ
            g_list.append(g)
            per_conf.append(dict(conf=cf["conf"], E_UMA_kJ=e, dGsolv_kJ=cf["dGsolv_kJ"],
                                 G_RRHO_kJ=cf["G_RRHO_kJ"], G_aq_kJ=g,
                                 n_imag=cf["n_imag"]))
        assembled = boltzmann_ensemble(
            [ConformerTerms(pc["E_UMA_kJ"], pc["dGsolv_kJ"], pc["G_RRHO_kJ"])
             for pc in per_conf], temperature_K=config.DEFAULT_CONDITIONS.temperature_K)
        g_ens, w = assembled.gibbs_kJ, assembled.weights
        for pc, wi in zip(per_conf, w):
            pc["weight"] = wi
        G_aq[c] = g_ens
        breakdown[c] = dict(name=spec[c]["name"], charge=chg, n_conf=len(g_list),
                            n_eff=1.0 / sum(x * x for x in w), G_aq_kJ=g_ens,
                            single_conf_G_aq_kJ=min(g_list), conformers=per_conf)
        print(f"{c} q={chg:+d}: {len(g_list):3d} conf "
              f"(n_eff={breakdown[c]['n_eff']:5.1f})  G_aq={g_ens:12.1f}  "
              f"dG_conf={g_ens - min(g_list):5.1f}  ({spec[c]['name']})")

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
    out_rxn = os.path.join(bench, f"qm_reaction_dG_{args.tag}.json")
    json.dump(rows, open(out_rxn, "w"), indent=2)
    print(f"\nwrote {out_bd}\nwrote {out_rxn}")


if __name__ == "__main__":
    main()
