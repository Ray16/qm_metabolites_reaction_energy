"""Re-score TECRDB with the CURRENT MetaG pipeline (reflects this session's fixes: O2 spin, adenylylate
anchor, thermal symmetry, aldehyde, structural sigma-class). The shipped calibration per_reaction was
frozen BEFORE the spin/anchor fixes, so the UMA TECRDB numbers in the figures are stale. This regenerates
them. Cache-accelerated (most species already computed). Resumable (skips existing result files).

    CUDA_VISIBLE_DEVICES=N python analysis/tecrdb_rescore.py rxn01211,rxn00973,...
Writes one JSON per reaction to analysis/tecrdb_rescore_results/<rxn>.json with dG/dG_raw/exp/err/class.
"""
import os, sys, json, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
INP = os.path.join(ROOT, "experiments", "qm_mlip_solvation", "scripts", "reactions_tecrdb_all.json")
OUT = os.environ.get("ADJ_OUT", os.path.join(HERE, "tecrdb_rescore_results"))


def main():
    os.makedirs(OUT, exist_ok=True)
    rx_all = json.load(open(INP))
    wanted = sys.argv[1].split(",") if len(sys.argv) > 1 else list(rx_all)
    from metag.energetics.uma import load_uma
    from metag.pipeline import score_reaction
    pu = load_uma(os.environ.get("METAG_MODEL", "uma-s-1p2p1"))
    for rid in wanted:
        outp = os.path.join(OUT, f"{rid}.json")
        if os.path.exists(outp):
            print(f"[skip] {rid}", flush=True); continue
        rx = rx_all[rid]
        rxin = dict(rx, species={k: tuple(v) for k, v in rx["species"].items()})
        t0 = time.time()
        try:
            r = score_reaction(pu, rxin, seeds=(1, 2), keep=8, pool=48, log=lambda *a: None, key=rid)
            exp = (rx.get("exp") or [None])[0]
            rec = {"reaction": rid, "note": rx.get("note", ""), "ec": rx.get("EC"),
                   "dG": r["dG"], "dG_raw": r["dG_raw"], "exp": exp,
                   "err": (round(r["dG"] - exp, 2) if exp is not None else None),
                   "class": r["sigma_breakdown"].get("class"), "anchor": r["anchor"],
                   "sigma_pred": r["sigma_pred"], "secs": round(time.time() - t0, 1)}
        except Exception as e:
            rec = {"reaction": rid, "error": str(e), "traceback": traceback.format_exc(),
                   "secs": round(time.time() - t0, 1)}
        json.dump(rec, open(outp, "w"), indent=1)
        tag = rec.get("error", f"dG={rec['dG']:+.1f} exp={rec['exp']} err={rec['err']}")
        print(f"[done] {rid} ({rec['secs']}s) {tag}", flush=True)


if __name__ == "__main__":
    main()
