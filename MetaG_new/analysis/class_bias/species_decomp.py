"""Species-level decomposition with the PRODUCTION estimator (no src edits).

For each SMILES: (a) production implicit_G (Boltzmann ensemble, UMA E + ALPB + UMA RRHO; 1 atm gas
std state, as cached) written to a SCRATCH cache; (b) at the lowest gas minimum: E_UMA, RRHO Gcorr
(harmonic, 50 cm-1 floor) and qRRHO-variant, ALPB dGsolv. Output JSON.

Run via gpu_reserve; METAG_CACHE must point to a scratch dir (never the production cache).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
assert "sweep_20261001" not in os.environ.get("METAG_CACHE", ""), "use a scratch cache"
from metag import pipeline as p
from metag.energetics import thermal as th
from metag.energetics.uma import load_uma
from ase import Atoms

def decomp(pu, smi, q):
    mult = p.spin_multiplicity(smi, q)
    cands = p.pool_confs(smi, q, 1, 96, spin=mult)
    sel = [cands[i] for i in np.argsort(p.batched_energies(pu, cands))[:12]]
    rel, E, conv = p.batched_fire(pu, sel, fmax=0.02, steps=600, stop_frac=1.0, return_converged=True)
    best = None
    for a, e, c in zip(rel, E, conv):
        if not c: continue
        s = p.dgsolv(a.get_chemical_symbols(), a.get_positions(), q, "alpb", mult)
        if s is None: continue
        if best is None or e * p.EV2KJ + s < best[0]:
            best = (e * p.EV2KJ + s, a, float(e * p.EV2KJ), s)
    _, a, e, s = best
    g = th.uma_gibbs_corr(pu, a.get_chemical_symbols(), a.get_positions(), q, spin=mult)
    os.environ["QRRHO"] = "1"
    gq = th.uma_gibbs_corr(pu, a.get_chemical_symbols(), a.get_positions(), q, spin=mult)
    os.environ.pop("QRRHO")
    return {"E": e, "gcorr": g, "gcorr_qrrho": gq, "solv_alpb": s}

def main():
    specs = json.load(open(sys.argv[1])); out = sys.argv[2]
    res = json.load(open(out)) if os.path.exists(out) else {}
    pu = load_uma("uma-s-1p2p1")
    for name, (q, smi) in specs.items():
        if name in res: continue
        r = {"smi": smi, "q": q}
        try:
            seeds, keep, pool = p.sampling_budget(smi)
            G, sig = p.implicit_G(pu, q, smi, seeds, keep, pool, lambda *a: None, name, [])
            r.update(G_prod=G, sigma=sig)
            r.update(decomp(pu, smi, q))
        except Exception as ex:
            r["error"] = repr(ex)
        res[name] = r
        print(name, r, flush=True)
        json.dump(res, open(out, "w"), indent=1)
    if "_water_ref" not in res:
        res["_water_ref"] = p.water_ref_G(pu)
        json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    main()
