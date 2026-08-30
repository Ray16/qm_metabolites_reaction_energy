"""Adjudicate the GC-vs-eQ divergent ModelSEED reactions with MetaG (first-principles UMA).

For each reaction where group-contribution (gc) and eQuilibrator (eq) disagree by >30 kJ, run
score_reaction and record dG / dG_raw / ci95 / sigma_pred. The question is NOT accuracy vs a ground
truth (there is none) but ADJUDICATION: does UMA land near gc, near eq, split them, or expose both?

Usage (one shard per GPU):
    CUDA_VISIBLE_DEVICES=1 python analysis/adjudicate.py rxn00024,rxn00054
Writes one JSON per reaction to analysis/divergent_results/<rxn>.json (resumable: skips existing).
"""
import os, sys, json, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.environ.get("ADJ_INPUT", os.path.join(HERE, "divergent_inputs.json"))
OUT = os.environ.get("ADJ_OUT", os.path.join(HERE, "divergent_results"))


def main():
    os.makedirs(OUT, exist_ok=True)
    reactions = json.load(open(INPUTS))
    wanted = sys.argv[1].split(",") if len(sys.argv) > 1 else list(reactions)

    from metag.energetics.uma import load_uma
    from metag.pipeline import score_reaction
    pu = load_uma(os.environ.get("METAG_MODEL", "uma-s-1p2p1"))

    for rid in wanted:
        outp = os.path.join(OUT, f"{rid}.json")
        if os.path.exists(outp):
            print(f"[skip] {rid} already done", flush=True)
            continue
        rx = reactions[rid]
        rxin = dict(rx, species={k: tuple(v) for k, v in rx["species"].items()})
        t0 = time.time()
        try:
            r = score_reaction(pu, rxin, seeds=(1, 2), keep=8, pool=48,
                               log=lambda *a: None, key=rid)
            gc, eq = rx["gc"], rx["eq"]
            dG = r["dG"]
            rec = {
                "reaction": rid, "note": rx["note"], "max_heavy": rx["max_heavy"],
                "gc": gc, "eq": eq, "delta_gc_eq": round(gc - eq, 1),
                "dG": dG, "dG_raw": r["dG_raw"], "ci95": r["ci95"],
                "sigma_pred": r["sigma_pred"], "anchor": r["anchor"],
                "suspect": r["suspect"], "unresolved": r["unresolved"],
                "class": r["sigma_breakdown"].get("class"),
                "d_to_gc": round(dG - gc, 1), "d_to_eq": round(dG - eq, 1),
                "secs": round(time.time() - t0, 1),
            }
        except Exception as e:
            rec = {"reaction": rid, "note": rx["note"], "error": str(e),
                   "traceback": traceback.format_exc(), "secs": round(time.time() - t0, 1)}
        json.dump(rec, open(outp, "w"), indent=1)
        tag = rec.get("error", f"dG={rec['dG']:+.1f}  gc={rec['gc']:+.1f} eq={rec['eq']:+.1f}  "
                               f"->gc {rec['d_to_gc']:+.1f} ->eq {rec['d_to_eq']:+.1f}")
        print(f"[done] {rid} ({rec['secs']}s)  {tag}", flush=True)


if __name__ == "__main__":
    main()
