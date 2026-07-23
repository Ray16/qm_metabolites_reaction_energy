"""Command-line entry point for the QM thermodynamics pipeline.

Examples:
    # Verify ORCA/xtb are reachable and run a 1-molecule end-to-end check
    python -m qm_thermo.cli check

    # Compute G(aq) for specific compounds
    python -m qm_thermo.cli compute --ids cpd00001 cpd00020 cpd00159

    # Compute all 83 metabolites (parallel, cached/resumable)
    python -m qm_thermo.cli compute --all

    # Build the openTECR benchmark from cached compound energies
    python -m qm_thermo.cli benchmark
"""

from __future__ import annotations

import argparse
import sys

from . import config
from .benchmark import build_benchmark, error_stats, plot_parity, write_csv
from .compute import compute_batch, compute_compound, load_cached
from .qm_backend import ORCABackend
from .structures import load_by_id, load_metabolites


def _cmd_check(args: argparse.Namespace) -> int:
    import os

    print(f"ORCA binary : {config.ORCA_BIN} ({'ok' if os.path.isfile(config.ORCA_BIN) else 'MISSING'})")
    print(f"OpenMPI root: {config.OPENMPI_ROOT} ({'ok' if os.path.isdir(config.OPENMPI_ROOT) else 'MISSING'})")
    print(f"xtb binary  : {config.XTB_BIN} ({'ok' if os.path.isfile(config.XTB_BIN) else 'MISSING'})")
    config.ensure_dirs()
    meta = load_by_id([args.id])[0]
    print(f"Running end-to-end check on {meta.cpd_id} ({meta.name})...")
    energy = compute_compound(meta, ORCABackend(), use_cache=False)
    print(f"OK: G(aq) = {energy.gibbs_kJ:.2f} kJ/mol")
    return 0


def _cmd_compute(args: argparse.Namespace) -> int:
    config.ensure_dirs()
    if args.all:
        metabolites = load_metabolites(skip_invalid=True)
    else:
        if not args.ids:
            print("error: provide --ids or --all", file=sys.stderr)
            return 2
        metabolites = load_by_id(args.ids)
    compute_batch(metabolites)
    done = sum(load_cached(m.cpd_id) is not None for m in metabolites)
    print(f"compute complete: {done}/{len(metabolites)} compounds cached")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    config.ensure_dirs()
    metabolites = load_metabolites(skip_invalid=True)
    energies = {
        m.cpd_id: ce
        for m in metabolites
        if (ce := load_cached(m.cpd_id)) is not None
    }
    if not energies:
        print("error: no cached compound energies; run `compute` first", file=sys.stderr)
        return 2
    import dataclasses

    from .benchmark import calibrate_proton_reference

    # Fit the aqueous-proton reference to openTECR, then evaluate with it.
    mu_h, n_cal = calibrate_proton_reference(energies, metabolites)
    conditions = dataclasses.replace(
        config.DEFAULT_CONDITIONS, proton_reference_kJ=mu_h
    )
    print(
        f"calibrated proton reference mu_H = {mu_h:.1f} kJ/mol "
        f"(from {n_cal} proton-coupled reactions; "
        f"theoretical ~{config.DEFAULT_CONDITIONS.proton_reference_kJ:.0f})"
    )

    # Report how many covered reactions were dropped as redox-reference cases.
    from .benchmark import is_redox_reaction
    from .references import reactions_within

    covered = reactions_within(set(energies))
    n_redox = sum(is_redox_reaction(r) for r in covered.values())
    if n_redox:
        print(
            f"excluding {n_redox} O2/H2/H2O2 redox-reference reactions "
            "(incompatible with absolute QM energies)"
        )

    rows = build_benchmark(energies, metabolites, conditions=conditions)
    if not rows:
        print("no reactions fully covered by the computed compound set yet.")
        return 0
    csv_path = write_csv(rows)
    plot_path = plot_parity(rows)
    print(f"benchmark: {len(rows)} reactions -> {csv_path}")
    if plot_path:
        print(f"parity plot -> {plot_path}")
    print("\nError vs experimental (openTECR), kJ/mol:")
    for method, st in error_stats(rows).items():
        print(f"  {method:14s} n={st['n']:>3}  MAE={st['MAE']:6.1f}  RMSE={st['RMSE']:6.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qm_thermo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="verify tools and run a 1-molecule check")
    p_check.add_argument("--id", default="cpd00001", help="compound id to test")
    p_check.set_defaults(func=_cmd_check)

    p_compute = sub.add_parser("compute", help="compute G(aq) for compounds")
    p_compute.add_argument("--ids", nargs="*", help="ModelSEED compound ids")
    p_compute.add_argument("--all", action="store_true", help="all 83 metabolites")
    p_compute.set_defaults(func=_cmd_compute)

    p_bench = sub.add_parser("benchmark", help="build openTECR benchmark")
    p_bench.set_defaults(func=_cmd_benchmark)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
