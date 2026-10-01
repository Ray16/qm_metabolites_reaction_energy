"""Reassemble reaction free energies from cached species values, without QM.

This is the fast evaluation path for changes in routing, pKa bookkeeping, and
reaction-level corrections. It deliberately requires an exact species-physics
fingerprint: mixing cache generations can otherwise look like a routing gain.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REACTIONS = ROOT / "experiments/qm_mlip_solvation/scripts/reactions_tecrdb_all.json"
DEFAULT_CACHE = ROOT / "MetaG/.metag_cache/species_cache"
POLICIES = {
    "baseline": {"ZWITTERION_PH0": "0", "NEUTRAL_QM": "0"},
    "zwitterion": {"ZWITTERION_PH0": "1", "NEUTRAL_QM": "0"},
    "neutral": {"ZWITTERION_PH0": "1", "NEUTRAL_QM": "1"},
}


class CacheMiss(Exception):
    pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="append", default=[])
    parser.add_argument("--reactions", default=str(DEFAULT_REACTIONS))
    parser.add_argument("--policy", choices=sorted(POLICIES), default="baseline")
    parser.add_argument("--anchors", choices=("on", "off"), default="off")
    parser.add_argument("--solv-model", choices=("cosmo", "alpb", "cpcmx"), default="cosmo")
    parser.add_argument(
        "--cache-via",
        choices=("primary", "solv_also"),
        default="primary",
        help="Select primary-solvent records or auxiliary values on the primary ensemble.",
    )
    parser.add_argument("--water-ref", help="JSON containing {'G': ...}; never computed by this tool")
    parser.add_argument("--legacy-settings", action="store_true", help="pre-dedup cache key")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def configure_environment(args):
    os.environ["SOLV_MODEL"] = args.solv_model
    os.environ["ANCHOR_CORRECT"] = "1" if args.anchors == "on" else "0"
    os.environ.update(POLICIES[args.policy])
    for assignment in args.env:
        key, value = assignment.split("=", 1)
        os.environ[key] = value


def matching_settings(record_settings, required, cache_via):
    """Require one exact physics generation and requested solvation provenance."""
    actual = dict(record_settings)
    via = actual.pop("via", None)
    expected_via = None if cache_via == "primary" else "solv_also"
    return via == expected_via and actual == required


def load_species_table(cache_dirs, required_settings, cache_via, canonical):
    table = {}
    matched_files = 0
    for directory in cache_dirs:
        for filename in glob.glob(os.path.join(directory, "*.json")):
            try:
                record = json.loads(Path(filename).read_text())
            except (OSError, ValueError):
                continue
            if record.get("method") != "implicit":
                continue
            settings = dict(record.get("settings") or {})
            spin = int(settings.pop("spin", 1))
            required = dict(required_settings)
            required.pop("spin", None)
            if not matching_settings(settings, required, cache_via):
                continue
            key = (canonical(record["smi"]), int(record["q"]), spin)
            value = (float(record["G"]), float(record["sigma"] or 1.0))
            if key in table and not np.allclose(table[key], value, atol=1e-6, rtol=0):
                raise ValueError(f"conflicting cache values for {key}: {table[key]} and {value}")
            table[key] = value
            matched_files += 1
    return table, matched_files


def metrics(results):
    errors = np.asarray([
        row["err"] for row in results.values()
        if row["err"] is not None and row["suspect"] is None
    ], dtype=float)
    if not len(errors):
        return {"n": 0, "mae": None, "median_ae": None, "rmse": None, "bias": None}
    return {
        "n": len(errors),
        "mae": round(float(np.mean(np.abs(errors))), 3),
        "median_ae": round(float(np.median(np.abs(errors))), 3),
        "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 3),
        "bias": round(float(np.mean(errors)), 3),
    }


def run(args):
    configure_environment(args)

    import metag.energetics.uma as uma
    import metag.pipeline as pipeline
    from metag.energetics import species_cache
    from metag.energetics.conformers import spin_multiplicity

    uma.DEV = "cpu"
    settings = dict(pipeline._IMPLICIT_SETTINGS)
    if args.legacy_settings:
        settings.pop("dedup", None)
    caches = args.cache or [str(DEFAULT_CACHE)]
    table, matched_files = load_species_table(
        caches, settings, args.cache_via, species_cache.canonical
    )
    if not table:
        raise RuntimeError(
            f"no cache records match settings={settings!r}, via={args.cache_via!r} in {caches}"
        )

    def cached_implicit(pu, q, smi, *unused_args, **unused_kwargs):
        key = (species_cache.canonical(smi), int(q), spin_multiplicity(smi, q))
        if key not in table:
            raise CacheMiss(f"{smi} q{q}")
        return table[key]

    water_path = (
        Path(args.water_ref) if args.water_ref else
        Path(__file__).with_name(
            "water_ref_G.json" if args.solv_model == "cosmo"
            else f"water_ref_G_{args.solv_model}.json"
        )
    )
    if not water_path.exists():
        raise FileNotFoundError(
            f"missing water reference {water_path}; supply --water-ref (CPU reassembly never runs QM)"
        )
    water_g = float(json.loads(water_path.read_text())["G"])
    pipeline.implicit_G = cached_implicit
    pipeline.water_ref_G = lambda pu, log=None: water_g

    reactions = json.loads(Path(args.reactions).read_text())
    results, misses = {}, {}
    for rid, reaction in reactions.items():
        rxin = dict(reaction, species={k: tuple(v) for k, v in reaction["species"].items()})
        try:
            scored = pipeline.score_reaction(None, rxin, log=lambda *x: None, key=rid)
        except CacheMiss as exc:
            misses[rid] = str(exc)
            continue
        if scored is None:
            misses[rid] = "score_reaction returned None"
            continue
        exp = (reaction.get("exp") or [None])[0]
        results[rid] = {
            "reaction": rid,
            "dG": scored["dG"],
            "dG_raw": scored["dG_raw"],
            "exp": exp,
            "err": None if exp is None or scored["dG"] is None else scored["dG"] - exp,
            "class": scored["sigma_breakdown"].get("class"),
            "anchor": scored["anchor"],
            "U_samp": scored["U_samp"],
            "routes": scored["routes"],
            "suspect": scored["suspect"],
            "note": reaction.get("note", ""),
            "species_scored": scored["species_scored"],
        }

    summary = metrics(results)
    provenance = {
        "reactions": str(Path(args.reactions).resolve()),
        "cache_dirs": [str(Path(path).resolve()) for path in caches],
        "cache_settings": settings,
        "cache_via": args.cache_via,
        "matched_cache_files": matched_files,
        "solv_model": args.solv_model,
        "water_ref": str(water_path.resolve()),
        "policy": args.policy,
        "anchors": args.anchors,
    }
    payload = {
        "summary": summary,
        "coverage": {
            "total": len(reactions),
            "scored": len(results),
            "missed": len(misses),
            "fraction": round(len(results) / len(reactions), 4) if reactions else None,
        },
        "provenance": provenance,
        "results": results,
        "miss": misses,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    print(
        f"species table {len(table)} ({matched_files} files) | scored {len(results)} "
        f"miss {len(misses)} | MAE {summary['mae']} median {summary['median_ae']}"
    )
    return payload


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
