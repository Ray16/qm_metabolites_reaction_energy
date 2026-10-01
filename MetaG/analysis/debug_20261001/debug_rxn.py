"""Verbose single-reaction scorer for diagnosis (GPU). Usage: debug_rxn.py rxn1,rxn2 OUTDIR"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INP = os.environ.get("TECRDB_INPUTS", "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/experiments/qm_mlip_solvation/scripts/reactions_tecrdb_std.json")
from metag.energetics.uma import load_uma
from metag.pipeline import score_reaction, _MODEL
rx_all = json.load(open(INP)); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
pu = load_uma(_MODEL)
for rid in sys.argv[1].split(","):
    rx = rx_all[rid]; rxin = dict(rx, species={k: tuple(v) for k, v in rx["species"].items()})
    print(f"===== {rid} {rx.get('note','')}", flush=True); t0 = time.time()
    try:
        r = score_reaction(pu, rxin, log=lambda *a: print(*a, flush=True), key=rid)
    except Exception as e:
        import traceback; traceback.print_exc(); r = None
    if r is None:
        print(f"[RESULT] {rid} None", flush=True); continue
    exp = sorted(rx["exp"])[len(rx["exp"]) // 2]
    rec = {k: r.get(k) for k in ("dG", "dG_raw", "stages", "routes", "species_scored", "anchor", "U_samp", "suspect")}
    rec.update(exp=exp, err=None if r["dG"] is None else r["dG"] - exp, secs=time.time() - t0)
    json.dump(rec, open(os.path.join(out, rid + ".json"), "w"), indent=1)
    print(f"[RESULT] {rid} dG={r['dG']} exp={exp} err={rec['err']}", flush=True)
