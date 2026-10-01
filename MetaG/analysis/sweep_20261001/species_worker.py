"""Claim-based, resumable species worker: compute + cache implicit_G for every 'SMILES qN' in a list.
One process per reserved GPU; any number of nodes share CLAIMS (atomic NFS mkdir) and DONE markers.
    gpu_reserve run <idx> -- env METAG_CACHE=... CLAIMS=... SOLV_MODEL=alpb SOLV_ALSO=cosmo,cpcmx python species_worker.py list.json
Big species (many heavy atoms) are left to the end of the list ordering by the caller."""
import os, sys, json, time, hashlib, socket, traceback
M = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, M)
todo = json.load(open(sys.argv[1]))
CLAIMS = os.environ["CLAIMS"]; DONE = os.environ.get("DONE", CLAIMS + "_done"); os.makedirs(CLAIMS, exist_ok=True); os.makedirs(DONE, exist_ok=True)
from metag.energetics.uma import load_uma
import metag.pipeline as P
pu = load_uma(P._MODEL); host = socket.gethostname()
print(f"[worker] {host} model={P._MODEL} solv={P.SOLV_MODEL} also={P.SOLV_ALSO} n={len(todo)}", flush=True)
for item in todo:
    h = hashlib.md5(item.encode()).hexdigest()
    if os.path.exists(os.path.join(DONE, h + ".json")):
        continue
    try:
        os.mkdir(os.path.join(CLAIMS, h))
    except FileExistsError:
        continue
    smi, q = item.rsplit(" q", 1); t0 = time.time(); warn = []
    try:
        G, s = P.implicit_G(pu, int(q), smi, None, None, None, lambda *a: print(*a, flush=True), "sp", warnings=warn)
        rec = dict(item=item, G=G, sigma=s, warnings=warn)
    except Exception as e:
        rec = dict(item=item, error=repr(e)[:500], tb=traceback.format_exc()[-1500:], warnings=warn)
    rec.update(host=host, secs=round(time.time() - t0, 1))
    json.dump(rec, open(os.path.join(DONE, h + ".json"), "w"))
    print(f"[species] {item} {rec.get('G', rec.get('error'))} ({rec['secs']}s)", flush=True)
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
