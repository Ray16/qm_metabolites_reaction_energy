#!/usr/bin/env python
"""Compare frozen and refactored MetaG behavior in isolated interpreters.

Routing mode needs only the input JSON. Warm-cache mode additionally requires
the frozen species cache and liquid-water reference. It replaces backend calls
with hard failures, so a missing cache entry cannot trigger hidden QM work.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
import subprocess
import sys


ROUTING_PROGRAM = r"""
import json, pickle, sys
from metag.pipeline import effective_config, route_reaction

inputs = json.load(open(sys.argv[1]))
results = {}
for reaction_id, raw in inputs.items():
    reaction = dict(raw)
    reaction["species"] = {
        name: tuple(value) for name, value in raw["species"].items()
    }
    routed, routes, truncated = route_reaction(reaction, log=lambda *_: None)
    if isinstance(routed.get("explicit"), set):
        routed["explicit"] = sorted(routed["explicit"])
    results[reaction_id] = (routed, routes, truncated)
pickle.dump((effective_config(), results), sys.stdout.buffer, protocol=4)
"""


WARM_CACHE_PROGRAM = r"""
import json, pickle, sys
import metag.pipeline as pipeline

inputs = json.load(open(sys.argv[1]))
water = json.load(open(sys.argv[2]))["G"]
pipeline.water_ref_G = lambda pu, log=None: water

def forbidden(*args, **kwargs):
    raise AssertionError("warm-cache parity attempted QM work")

pipeline.pool_confs = forbidden
pipeline.batched_energies = forbidden
pipeline.batched_fire = forbidden

results = {}
for reaction_id, raw in inputs.items():
    reaction = dict(raw)
    reaction["species"] = {
        name: tuple(value) for name, value in raw["species"].items()
    }
    results[reaction_id] = pipeline.score_reaction(
        None, reaction, log=lambda *_: None, key=reaction_id
    )
pickle.dump(results, sys.stdout.buffer, protocol=4)
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-package", type=Path, required=True)
    parser.add_argument("--new-package", type=Path, default=Path("src"))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--mode", choices=("routing", "warm-cache"), default="routing")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--water-reference", type=Path)
    return parser.parse_args()


def run_package(package, program, inputs, cache=None, water_reference=None):
    environment = dict(os.environ, PYTHONPATH=str(package.resolve()))
    arguments = [sys.executable, "-c", program, str(inputs.resolve())]
    if cache is not None:
        environment["METAG_CACHE"] = str(cache.resolve())
    if water_reference is not None:
        arguments.append(str(water_reference.resolve()))
    process = subprocess.run(
        arguments,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode:
        sys.stderr.buffer.write(process.stderr)
        raise SystemExit(process.returncode)
    return pickle.loads(process.stdout)


def main():
    args = parse_args()
    if args.mode == "warm-cache" and (args.cache is None or args.water_reference is None):
        raise SystemExit("warm-cache mode requires --cache and --water-reference")

    program = ROUTING_PROGRAM if args.mode == "routing" else WARM_CACHE_PROGRAM
    old = run_package(
        args.old_package, program, args.inputs, args.cache, args.water_reference
    )
    new = run_package(
        args.new_package, program, args.inputs, args.cache, args.water_reference
    )

    if args.mode == "routing":
        old_config, old_results = old
        new_config, new_results = new
        if old_config != new_config:
            raise SystemExit("effective configurations differ")
    else:
        old_results, new_results = old, new

    differences = [
        reaction_id
        for reaction_id in old_results
        if old_results[reaction_id] != new_results.get(reaction_id)
    ]
    print(
        f"mode={args.mode} reactions={len(old_results)} "
        f"exact_differences={len(differences)}"
    )
    if differences:
        print("first differences:", ", ".join(differences[:10]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

