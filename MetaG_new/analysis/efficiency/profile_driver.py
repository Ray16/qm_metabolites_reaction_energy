#!/usr/bin/env python
"""Profile one reaction through the PRODUCTION pipeline (metag.pipeline.run_reaction) with METAG_PROFILE=1,
plus extra non-invasive instrumentation via monkeypatching (no src edits; numbers unchanged):
  rdkit.pool (ETKDG+MMFF), dedup (UniqueMinima.add), xtb.wall (solvation ThreadPool blocks),
  per-Hessian records (wall, natoms, caller, geometry hash -> repeated-geometry detection),
  per-batched_fire records (n structures, steps, wall), per-species implicit_G wall.
Usage: python profile_driver.py <rxn_key> <out_json>   (launch via gpu_reserve run)"""
import os, sys, time, json, hashlib, traceback, socket
assert os.environ.get("METAG_PROFILE") == "1"
import numpy as np
import metag.pipeline as P
from metag import profile as prof
from metag.energetics import conformers as C
import metag.energetics.thermal as TH

key, out = sys.argv[1], sys.argv[2]
REC = {"hess": [], "fire": [], "species": [], "dedup_new": 0, "dedup_dup": 0, "dedup_dup_replaced": 0,
       "xtb_pool_calls": 0, "xtb_conformers": 0}

def snap():
    return {k: v[0] for k, v in prof._acc.items()}

# ---- ETKDG pool
_pool = P.pool_confs
def pool_confs(*a, **k):
    with prof.timed("rdkit.pool"):
        return _pool(*a, **k)
P.pool_confs = pool_confs

# ---- dedup
_add = C.UniqueMinima.add
def add(self, atoms, E_kJ, G_kJ):
    G_before = list(self.G)
    with prof.timed("dedup"):
        r = _add(self, atoms, E_kJ, G_kJ)
    if r: REC["dedup_new"] += 1
    else:
        REC["dedup_dup"] += 1
        if self.G != G_before[:len(self.G)] or len(self.G) != len(G_before):
            REC["dedup_dup_replaced"] += 1
    return r
C.UniqueMinima.add = add

# ---- xtb pool wall
from concurrent.futures import ThreadPoolExecutor as _TPE
class TPE(_TPE):
    def __enter__(self):
        self._t0 = time.perf_counter(); REC["xtb_pool_calls"] += 1
        return super().__enter__()
    def __exit__(self, *e):
        r = super().__exit__(*e)
        dt = time.perf_counter() - self._t0
        with prof._lock:
            v = prof._acc.setdefault("xtb.wall(pool)", [0.0, 0]); v[0] += dt; v[1] += 1
        return r
P.ThreadPoolExecutor = TPE
_dgs = P.dgsolv
def dgsolv(*a, **k):
    REC["xtb_conformers"] += 1
    return _dgs(*a, **k)
P.dgsolv = dgsolv

# ---- per-call xtb subprocess timing (thermal._run_xtb)
REC["xtb_calls"] = []
_rx = TH._run_xtb
def _run_xtb(cmd, d, timeout):
    t = time.perf_counter(); r = _rx(cmd, d, timeout)
    REC["xtb_calls"].append({"dt": round(time.perf_counter() - t, 3), "solv": any(c.startswith("--alpb") or c.startswith("--cosmo") for c in cmd),
                             "thread": __import__("threading").current_thread().name[:12], "ok": r is not None})
    return r
TH._run_xtb = _run_xtb

# ---- Hessians
_ugc = P.uma_gibbs_corr
def uma_gibbs_corr(pu, syms, pos, q, *a, **k):
    caller = sys._getframe(1).f_code.co_name
    h = hashlib.sha1(np.round(np.asarray(pos, float), 6).tobytes()).hexdigest()[:12]
    s0 = snap(); t0 = time.perf_counter()
    r = _ugc(pu, syms, pos, q, *a, **k)
    import torch; torch.cuda.synchronize()
    s1 = snap()
    d = {b: s1.get(b, 0) - s0.get(b, 0) for b in ("uma.gpu", "uma.build", "hessian")}
    REC["hess"].append({"caller": caller, "nat": len(syms), "wall": time.perf_counter() - t0, "geom": h,
                        "imag_as_soft": bool(k.get("imag_as_soft", False)), **d})
    return r
P.uma_gibbs_corr = uma_gibbs_corr

# ---- batched_fire
_bf = P.batched_fire
def batched_fire(pu, atoms_list, *a, **k):
    s0 = snap(); n0 = prof._acc.get("uma.writeback", [0, 0])[1]; t0 = time.perf_counter()
    r = _bf(pu, atoms_list, *a, **k)
    s1 = snap(); n1 = prof._acc.get("uma.writeback", [0, 0])[1]
    REC["fire"].append({"label": k.get("label", ""), "n": len(atoms_list), "nat": len(atoms_list[0]),
                        "steps": n1 - n0, "wall": time.perf_counter() - t0,
                        **{b: s1.get(b, 0) - s0.get(b, 0) for b in ("uma.gpu", "uma.build", "uma.writeback")}})
    return r
P.batched_fire = batched_fire

# ---- species
_ig = P.implicit_G
def implicit_G(pu, q, smi, seeds, keep, pool, log, name, warnings=None):
    t0 = time.perf_counter(); s0 = snap(); nh0 = len(REC["hess"]); nf0 = len(REC["fire"])
    lines = []
    def lg(s):
        lines.append(s); log(s)
    try:
        return _ig(pu, q, smi, seeds, keep, pool, lg, name, warnings)
    finally:
        s1 = snap()
        seeds_used = None; minima = None; cached = False
        for s in lines:
            if "CACHED" in s: cached = True
            if "seeds=" in s and "minima=" in s:
                seeds_used = int(s.split("seeds=")[1].split()[0]); minima = s.split("minima=")[1].split()[0]
        REC["species"].append({"name": name, "smi": smi, "q": q, "nat": P.Chem.AddHs(P.Chem.MolFromSmiles(smi)).GetNumAtoms(),
                               "budget": P.sampling_budget(smi), "wall": time.perf_counter() - t0, "cached": cached,
                               "seeds": seeds_used, "minima": minima, "n_hess": len(REC["hess"]) - nh0,
                               "n_fire": len(REC["fire"]) - nf0,
                               "buckets": {b: s1.get(b, 0) - s0.get(b, 0) for b in s1}})
P.implicit_G = implicit_G

import torch
log = lambda s: print(s, flush=True)
t0 = time.perf_counter()
pu = P.load_uma(P._MODEL)
t_load = time.perf_counter() - t0
prof.reset()
t1 = time.perf_counter()
res = None
try:
    res = P.run_reaction(pu, key, [1, 2], 10, 48, log)
except Exception:
    traceback.print_exc()
t_rxn = time.perf_counter() - t1
from metag.energetics import species_cache as SC
summary = {"key": key, "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
           "xtb_workers": P.XTB_WORKERS, "t_load_model": t_load, "t_reaction": t_rxn,
           "buckets": {k: {"s": v[0], "n": v[1]} for k, v in prof._acc.items()},
           "cache": SC.stats(), "dG": None if res is None else res.get("dG"), "dG_raw": None if res is None else res.get("dG_raw"),
           "result": res, **REC}
json.dump(summary, open(out, "w"), indent=1, default=str)
log(f"[driver] {key} reaction wall {t_rxn:.1f}s  dG_raw {summary['dG_raw']}")
