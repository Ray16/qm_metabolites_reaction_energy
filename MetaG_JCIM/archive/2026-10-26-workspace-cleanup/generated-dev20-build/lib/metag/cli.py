"""CLI parsing and batch execution, independent of the QM implementation."""
import argparse
import os
import sys


def parse_args(reactions, argv=None):
    """Parse arguments before importing the optional QM runtime."""
    from metag.tools import command_help
    ap = argparse.ArgumentParser(description='Score reactions, or invoke a maintained CPU tool.',
                                 epilog=command_help(), formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(reactions), default=None)
    ap.add_argument("--seeds", default="1,2", help="Legacy batch argument; the adaptive sampler determines actual seed batches. This does not run independent replicas.")
    ap.add_argument("--sampling-profile", choices=("current", "thorough"), default="current",
                    help="thorough uses stricter plateaus and larger pools/kept sets; costs more and does not guarantee accuracy")
    ap.add_argument("--keep", type=int, default=10)
    ap.add_argument("--pool", type=int, default=48)
    ap.add_argument("--require-consistent-panel", action="store_true",
                    help="Run CPU structural/representation preflight and refuse a failed panel before loading UMA")
    return ap.parse_args(argv)


def panel_preflight(reactions, args, out):
    """Persist a structural gate before importing or loading optional QM code."""
    if not getattr(args, "require_consistent_panel", False):
        return
    from metag.tools.audit_routes import audit_route_panel
    from metag.execution.filesystem import atomic_write_json
    keys = [args.only] if args.only else list(reactions)
    report = audit_route_panel({key:reactions[key] for key in keys})
    path = os.path.join(out, "route_preflight_" + (args.only or "all") + ".json")
    atomic_write_json(path, report)
    if not report['panel_gate']['passed']:
        raise SystemExit("Structural/representation preflight failed; see " + path)


def run_cli(*, reactions, model_name, load_model, score, no_estimate, out,
            args=None, argv=None, _preflight_done=False, _sampling_profile_done=False):
    """Execute a reaction batch with explicitly supplied scoring dependencies."""
    a = args if args is not None else parse_args(reactions, argv)
    if not _sampling_profile_done:
        from metag.sampling_profiles import apply_sampling_profile
        apply_sampling_profile(getattr(a, "sampling_profile", "current"))
    if not _preflight_done:
        panel_preflight(reactions, a, out)
    seeds = [int(s) for s in a.seeds.split(",")]
    log = lambda s: print(s, flush=True)
    keys = [a.only] if a.only else list(reactions)
    log(f"loading UMA... unified pipeline, reactions={keys} seeds={seeds} keep={a.keep}")
    pu = load_model(model_name)          # single source of truth: loaded model == cache-key model (no drift)
    rows = []
    from metag.execution.filesystem import atomic_write_json
    tag = a.only or "all"
    output_path = os.path.join(out, f"unified_pipeline_{tag}.json")
    for key in keys:
        try:
            r = score(pu, key, seeds, a.keep, a.pool, log)
            if r is None:
                r = no_estimate(key, reactions[key], {}, "species calculation failed", log)
        except Exception as exc:
            r = no_estimate(key, reactions[key], {}, f"{type(exc).__name__}: {exc}", log)
        rows.append(r)
        # Persist every outcome before formatting the summary or starting another reaction.
        atomic_write_json(output_path, rows)

    log(f"\n==== UNIFIED PIPELINE — one scheme, three classes ====")
    log(f"  {'reaction':14s} {'ΔG':>7s} {'raw':>7s} {'±σpred':>7s} {'exp':>14s} {'err':>16s} {'note':>12s}")
    for r in rows:
        if r.get("dG") is None:
            log(f"  {r['reaction']:14s} NO ESTIMATE: {r.get('suspect', 'calculation failed')}")
            continue
        note = ("UQ unavailable" if r.get("sign_status")=="uncertainty_unavailable" else
                "UNRESOLVED" if r.get("unresolved") else ("explicit" if r["explicit"] else "implicit"))
        log(f"  {r['reaction']:14s} {r['dG']:7.1f} {r.get('dG_raw', r['dG']):7.1f} "
            f"{str(r.get('sigma_pred','?')):>7s} {str(r['exp']):>14s} {str(r['err']):>16s} {note:>12s}")
    log(f"wrote {output_path}")
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    from metag.tools import COMMANDS, command_help, run_command
    if argv and argv[0] in COMMANDS:
        return run_command(argv[0], argv[1:])
    if argv and argv[0] == 'tools':
        if len(argv) > 1 and argv[1] not in ('-h', '--help'):
            raise SystemExit('Use metag COMMAND --help.\n' + command_help())
        print(command_help())
        return
    if argv and argv[0] == 'score':
        argv = argv[1:]
    from metag.reactions import load_reactions
    reactions = load_reactions()
    args = parse_args(reactions, argv)
    from metag.sampling_profiles import apply_sampling_profile
    apply_sampling_profile(args.sampling_profile)
    panel_preflight(reactions, args, os.environ.get("METAG_OUT", "./metag_out"))
    from metag import pipeline
    return run_cli(reactions=reactions, model_name=pipeline._MODEL,
                   load_model=pipeline.load_uma, score=pipeline.run_reaction,
                   no_estimate=pipeline._no_estimate, out=pipeline.OUT, args=args,
                   _preflight_done=True, _sampling_profile_done=True)


if __name__ == "__main__":
    main()
