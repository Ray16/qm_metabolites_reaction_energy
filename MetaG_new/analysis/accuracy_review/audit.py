"""Replay production inputs from an explicitly supplied cache, with QM forbidden.

Writes a ranked error report against bundled openTECR data, records replay
failures without silently dropping them, and compares new point estimates with
the archived production predictions. Cache misses cannot launch GPU work.

PYTHONPATH=src python analysis/accuracy_review/audit.py --cache DIR --water-reference JSON
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
from contextlib import contextmanager


ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def before_base_fix(pk):
    """Restore only the pre-review base bookkeeping for a controlled comparison."""
    from rdkit import Chem
    saved = pk._neutral_base_pkas, pk._AMINE_N, pk._ALPHA_AMINO_ACID
    try:
        pk._neutral_base_pkas = lambda mol: []
        pk._AMINE_N = Chem.MolFromSmarts("[NX3;H1,H2;!$(N[#6]=[#7,#8,#16]);!$(N-a)]")
        pk._ALPHA_AMINO_ACID = Chem.MolFromSmarts("[NX3,NX4+;H1,H2,H3][CX4][CX3](=O)[OX1,OX2]")
        yield
    finally:
        pk._neutral_base_pkas, pk._AMINE_N, pk._ALPHA_AMINO_ACID = saved


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--water-reference", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, default=ROOT / "src/metag/data/reactions_opentecr_std.json")
    ap.add_argument("--production", type=Path, default=ROOT / "artifacts/results/metag_opentecr_calibrated")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("report.json"))
    ap.add_argument("--route-inputs", type=Path,
                    default=ROOT / "artifacts/results/generality_inputs.json")
    args = ap.parse_args()
    os.environ["METAG_CACHE"] = str(args.cache.resolve())
    from metag import pipeline as p
    from metag.routing import pka_transform as pk

    def no_qm(*args, **kwargs):
        raise RuntimeError("cache miss: replay forbids new QM calculations")

    p.pool_confs = no_qm
    p.batched_energies = no_qm
    p.batched_fire = no_qm
    p.uma_gibbs_corr = no_qm
    p.dgsolv = no_qm
    # Read only: do not let auxiliary writes change the audited cache.
    p._sc.put = lambda *a, **k: None
    water = json.loads(args.water_reference.read_text())["G"]
    p.water_ref_G = lambda pu, log=None: water
    inputs = json.loads(args.inputs.read_text())
    rows, records = [], {}
    for i, (rid, reaction) in enumerate(sorted(inputs.items()), 1):
        archived = json.loads((args.production / f"{rid}.json").read_text())
        reference = statistics.mean(reaction["exp"])
        row = {"reaction": rid, "note": reaction.get("note"), "class": archived.get("class"),
               "reference": reference, "production_dG": archived.get("dG_raw"),
               "production_error": archived["dG_raw"] - reference,
               "species_scored": archived.get("species_scored"), "routes": archived.get("routes")}
        try:
            result = p.score_reaction(None, reaction, key=rid, log=lambda *_: None)
            if result is None or result.get("dG_raw") is None:
                row["replay_failure"] = (result or {}).get("suspect", "no estimate")
            else:
                row.update(replayed_dG=result["dG_raw"], change=result["dG_raw"] - archived["dG_raw"],
                           replayed_error=result["dG_raw"] - reference)
                records[rid] = result
                # Isolate this review's default-route numerical change from differences
                # already present between the checkout and archived production records.
                with before_base_fix(pk):
                    before = p.score_reaction(None, reaction, key=rid, log=lambda *_: None)
                row["before_fix_dG"] = (before or {}).get("dG_raw")
                row["fix_change"] = (result["dG_raw"] - before["dG_raw"]
                                     if before and before.get("dG_raw") is not None else None)
                row["fields_changed_by_fix"] = [
                    field for field in ("stages", "routes", "species_scored", "U_samp", "ci_info",
                                        "class", "suspect", "unresolved")
                    if (before or {}).get(field) != result.get(field)]
        except Exception as exc:
            row["replay_failure"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if i % 50 == 0:
            print(f"replayed {i}/{len(inputs)}; successful {len(records)}", flush=True)

    rows.sort(key=lambda row: -abs(row["production_error"]))
    failures = [r for r in rows if "replay_failure" in r]
    changes = [r for r in rows if abs(r.get("change", 0)) > 1e-8]
    generality = json.loads(args.route_inputs.read_text())
    route_changes = []
    for i, (rid, raw) in enumerate(sorted(generality.items()), 1):
        after = p.route_reaction(raw, log=lambda *_: None)
        with before_base_fix(pk):
            before = p.route_reaction(raw, log=lambda *_: None)
        if before != after:
            route_changes.append({"reaction": rid, "before": before, "after": after})
        if i % 50 == 0:
            print(f"checked ModelSEED routes {i}/{len(generality)}", flush=True)
    report = {
        "inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "water_reference_sha256": hashlib.sha256(args.water_reference.read_bytes()).hexdigest(),
        "cache_path": str(args.cache.resolve()), "config": p.effective_config(),
        "n_production": len(rows), "n_replayed": len(records),
        "n_replay_failures": len(failures), "n_changed_points": len(changes),
        "n_points_changed_by_neutral_base_fix": sum(abs(r.get("fix_change") or 0) > 1e-8 for r in rows),
        "n_records_with_other_fields_changed_by_fix": sum(bool(r.get("fields_changed_by_fix")) for r in rows),
        "modelseed_routing": {"n_reactions": len(generality), "n_changes": len(route_changes),
                              "changes": route_changes},
        "production_MAE": statistics.mean(abs(r["production_error"]) for r in rows),
        "replayed_subset_MAE": statistics.mean(abs(r["replayed_error"]) for r in rows if "replayed_error" in r)
                               if records else None,
        "neutral_base_probe": {s: pk._neutralize_v2(s) for s in ("N", "[NH4+]", "CN", "C[NH3+]")},
        "notes": ["Default production switches; SOLV_RELAX is not enabled by this audit.",
                  "A failed replay is not a new prediction and is never silently substituted.",
                  "No reaction-energy anchors or pKa values were fit to these residuals."],
        "ranked_reactions": rows,
    }
    from metag.tools.cycle_closure import closure_report
    cycle_records = [{"rid": r["reaction"], "species": inputs[r["reaction"]]["species"],
                      "dG": r["replayed_dG"], "sigma": 1.0}
                     for r in rows if "replayed_dG" in r]
    cycle_after = closure_report(cycle_records)
    before_values = {r["reaction"]: r.get("before_fix_dG") for r in rows}
    cycle_before = closure_report([dict(r, dG=before_values[r["rid"]]) for r in cycle_records])
    report["cycle_closure"] = {"weighting": "uniform sigma=1 kJ/mol; rounded point estimates",
                               "before_fix": cycle_before, "after_fix": cycle_after,
                               "identical": cycle_before == cycle_after}
    report["uncertainty_check"] = {
        "n_config_mismatches": sum(bool(r.get("ci_info", {}).get("config_mismatch"))
                                   for r in records.values()),
        "note": "Paired ci_info equality checks interval widths and flags; this is not independent coverage validation."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    records_dir = args.out.parent / "replayed"
    records_dir.mkdir(exist_ok=True)
    for rid, result in records.items():
        (records_dir / f"{rid}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k.startswith("n_") or "MAE" in k}, indent=2))
    if failures:
        print("failed:", ", ".join(r["reaction"] for r in failures))
    if changes:
        print("changed:", ", ".join(r["reaction"] for r in changes))


if __name__ == "__main__":
    main()
