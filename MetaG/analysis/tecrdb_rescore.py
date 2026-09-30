"""Re-score TECRDB with the CURRENT MetaG pipeline. Resumable (skips existing result files) and
multi-node safe: with CLAIMS=<dir> each reaction is claimed by an atomic NFS mkdir before scoring, so
any number of GPU workers can share one queue (one process per GPU, model loaded once).

    # launch via the reservation gate (never set CUDA_VISIBLE_DEVICES by hand):
    gpu_reserve run <idx> -- env ADJ_OUT=... CLAIMS=... python analysis/tecrdb_rescore.py [rxn1,rxn2,...]

Writes one JSON per reaction to $ADJ_OUT/<rxn>.json with everything the nested calibration
(metag.tools.calibrate) and the cycle-closure harness (metag.tools.cycle_closure) need:
dG, dG_raw, exp, err, class, anchor, U_samp, sigma_pred, ci95, calibration scope, routes, suspect.
"""
import os, sys, json, time, socket, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
INP = os.path.join(ROOT, "experiments", "qm_mlip_solvation", "scripts", "reactions_tecrdb_all.json")
OUT = os.environ.get("ADJ_OUT", os.path.join(HERE, "tecrdb_rescore_results"))
CLAIMS = os.environ.get("CLAIMS")
MAX_FAILS = int(os.environ.get("MAX_CONSEC_FAILS", "3"))    # circuit breaker for a broken node


def _claim(rid):
    if not CLAIMS:
        return True
    try:
        os.mkdir(os.path.join(CLAIMS, rid)); return True
    except FileExistsError:
        return False


def _release(rid):
    if CLAIMS:
        try:
            os.rmdir(os.path.join(CLAIMS, rid))
        except OSError:
            pass


def main():
    os.makedirs(OUT, exist_ok=True)
    if CLAIMS:
        os.makedirs(CLAIMS, exist_ok=True)
    rx_all = json.load(open(INP))
    wanted = sys.argv[1].split(",") if len(sys.argv) > 1 else list(rx_all)
    from metag.energetics.uma import load_uma
    from metag.pipeline import score_reaction, _MODEL
    pu = load_uma(_MODEL)                                     # the model the species cache is keyed on (UMA_MODEL)
    host = socket.gethostname(); fails = 0
    for rid in wanted:
        outp = os.path.join(OUT, f"{rid}.json")
        if os.path.exists(outp) or not _claim(rid):
            continue
        if os.path.exists(outp):                              # finished by another worker meanwhile
            continue
        rx = rx_all[rid]
        rxin = dict(rx, species={k: tuple(v) for k, v in rx["species"].items()})
        t0 = time.time()
        try:
            r = score_reaction(pu, rxin, log=lambda *a: None, key=rid)
            if r is None:
                raise RuntimeError("score_reaction returned None (species QM failed)")
            exp = (rx.get("exp") or [None])[0]
            rec = {"reaction": rid, "note": rx.get("note", ""), "ec": rx.get("EC"),
                   "dG": r["dG"], "dG_raw": r["dG_raw"], "exp": exp,
                   "dG_raw_unreliable": r.get("dG_raw_unreliable"),     # set only for suspect results
                   "err": (round(r["dG"] - exp, 2) if exp is not None and r["dG"] is not None else None),
                   "class": r["sigma_breakdown"].get("class"), "anchor": r["anchor"],
                   "water_ref": r["water_ref"], "U_samp": r["U_samp"], "sigma_pred": r["sigma_pred"],
                   "ci95": r["ci95"], "externally_calibrated": r["ci_info"].get("externally_calibrated"),
                   "routes": r["routes"], "suspect": r["suspect"], "std_state_kJ": r["std_state_kJ"],
                   "config": r["config"], "species_scored": r.get("species_scored"),
                   "host": host, "secs": round(time.time() - t0, 1)}
            fails = 0
        except Exception as e:
            rec = {"reaction": rid, "error": str(e), "traceback": traceback.format_exc(),
                   "host": host, "secs": round(time.time() - t0, 1)}
            fails += 1
        tmp = outp + f".tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(tmp, outp)
        tag = rec.get("error", f"dG={rec.get('dG')} exp={rec.get('exp')} err={rec.get('err')}")
        print(f"[done] {rid} ({rec['secs']}s) {tag}", flush=True)
        if fails >= MAX_FAILS:
            print(f"{host} CIRCUIT-BREAKER: {fails} consecutive failures -> stop", flush=True)
            if "error" in rec:                                # let a healthy node retry the last one
                os.remove(outp); _release(rid)
            return


if __name__ == "__main__":
    main()
