"""Multi-microstate (sugar ring-chain forms / heteroatom tautomers) study with the PRODUCTION estimator.

Nothing in src/ is modified: the production `implicit_G` scores every state; states are combined as
    G_species = -RT ln sum_i exp(-G_i/RT)          (metag.energetics.microstates.combine)
The input state is ALWAYS retained, every species of a reaction (both sides) is enumerated by the same
rule, no anchors, nothing fitted.

Subcommands
  validate  : free D-glucose / D-fructose / D-ribose, every ring-chain form scored separately ->
              predicted aqueous populations vs experimental NMR populations (run FIRST, blind to residuals)
  reactions : re-score reactions via pipeline.score_reaction with MODE=baseline|taut|sugar|both
              (baseline must reproduce archived production dG_raw exactly from cache)

Caches/logs live in the scratchpad (SCRATCH env, default below). The production species cache is read
(never written) as a fallback layer.

  gpu_reserve run <idx> -- env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      /homes/rzhu/miniforge3/envs/uma/bin/python analysis/sugar_forms/run.py validate
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRATCH = os.environ.get("SCRATCH", "/tmp/claude-21574/-nfs-lambda-stor-01-homes-rzhu-ModelSEED-FAISS-"
                         "thermodynamic-calc/c1661f88-7fcf-4db3-a6b6-a16e1b56c7c4/scratchpad/sugar_forms")
PROD_CACHE = ("/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/MetaG/analysis/"
              "sweep_20261001/cache")
WATER_REF = ("/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/MetaG/analysis/"
             "sweep_20261001/water_ref_G_expt.json")       # sha256 19a70a35... == report.json
os.environ["METAG_CACHE"] = os.path.join(SCRATCH, "cache")
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, HERE)

from metag import pipeline as p                              # noqa: E402
from metag.energetics import species_cache as sc             # noqa: E402
from metag.energetics.microstates import combine             # noqa: E402
import forms                                                 # noqa: E402

RT = 8.314462618e-3 * 298.15
TAUT_SCREEN_KJ = 25.0          # states screened > 25 kJ above the lowest carry < 5e-5 weight
TAUT_CAP = 12                  # max tautomers (incl. input) taken to the screen, by RDKit score

# ---- layered cache: own cache first, then the production cache read-only --------------------------
_orig_get = sc.get
def _layered_get(*a, **k):
    r = _orig_get(*a, **k)
    if r is not None:
        return r
    own = sc.CACHE_DIR
    try:
        sc.CACHE_DIR = PROD_CACHE
        return _orig_get(*a, **k)
    finally:
        sc.CACHE_DIR = own
sc.get = _layered_get
_WATER = json.load(open(WATER_REF))["G"]
p.water_ref_G = lambda pu, log=None: _WATER

MODE = "baseline"
DETAIL = {}                    # (rid, species) -> per-state record
_RID = [None]
_orig_hyd = p._hydrate_all_sites


def screen_G(pu, q, smi, log):
    """Cheap tautomer screen: lowest (E_gas + dGsolv) over a small relaxed conformer set (no RRHO)."""
    import numpy as np
    mult = p.spin_multiplicity(smi, q)
    best = None
    for seed in (1, 2):
        cands = p.pool_confs(smi, q, seed, 16, spin=mult)
        order = np.argsort(p.batched_energies(pu, cands))[:4]
        sel = [cands[i] for i in order]
        rel, E, conv = p.batched_fire(pu, sel, fmax=0.05, steps=300, stop_frac=0.9, return_converged=True,
                                      label="screen")
        ref = p.bond_graph(Chem.AddHs(Chem.MolFromSmiles(smi)))
        for a, e, c in zip(rel, E, conv):
            if not c or not p.same_connectivity(a, ref):
                continue
            s = p.dgsolv(a.get_chemical_symbols(), a.get_positions(), q, p.SOLV_MODEL, mult)
            if s is None:
                continue
            g = float(e) * p.EV2KJ + s
            best = g if best is None else min(best, g)
    return best


from rdkit import Chem                                       # noqa: E402


def score_state(pu, rx, name, q, smi, std, seeds, keep, pool, log, routes):
    """One alternative state through the production species path: implicit_G + 1 M + carbonyl hydration."""
    try:
        g, s = p.implicit_G(pu, q, smi, seeds, keep, pool, log, name, routes["warnings"])
    except p.SpeciesRearranged as e:
        return None, f"rearranged: {e}"
    if g is None:
        return None, "QM failed"
    G = {name: g + std}; sig = {name: s or 0.0}
    _orig_hyd(pu, rx, name, q, smi, G, sig, std, seeds, keep, pool, log, routes)
    return G[name], None


def hooked_hydrate(pu, rx, name, q, smi, G, sig, std, seeds, keep, pool, log, routes):
    _orig_hyd(pu, rx, name, q, smi, G, sig, std, seeds, keep, pool, log, routes)
    if MODE == "baseline" or G.get(name) is None:
        return
    on_ph0 = bool(rx.get("pka_sites")) and q == 0
    t_in = p._acid_transform(smi) if on_ph0 else 0.0
    alts = []
    if MODE in ("sugar", "both"):
        for a in forms.sugar_states(smi)["alternatives"]:
            alts.append((a["smiles"], a["kind"], True))          # own acid transform (distinct anions)
    if MODE in ("taut", "both"):
        tauts = forms.heteroatom_tautomers(smi, cap=TAUT_CAP)
        if len(tauts) > 1:
            gs = {t: screen_G(pu, q, t, log) for t in tauts}
            ok = [g for g in gs.values() if g is not None]
            gmin = min(ok) if ok else None
            for t in tauts[1:]:
                if gs[t] is not None and gs[t] - gmin < TAUT_SCREEN_KJ:
                    alts.append((t, "tautomer", False))       # common anion -> input macroscopic pKa
            DETAIL.setdefault((_RID[0], name), {})["taut_screen"] = {
                t: (None if g is None else round(g - gs[tauts[0]], 1) if gs[tauts[0]] is not None else None)
                for t, g in gs.items()}
            DETAIL[(_RID[0], name)]["n_taut_enumerated"] = len(forms.heteroatom_tautomers(smi))
    if not alts:
        return
    states = [{"name": "input", "G": G[name]}]
    rec = DETAIL.setdefault((_RID[0], name), {})
    rec.update(input=smi, q=q, G_input=G[name], states={}, failed={})
    for i, (s_alt, kind, own_t) in enumerate(alts):
        g, err = score_state(pu, rx, f"{name}[{kind}{i}]", q, s_alt, std, seeds, keep, pool, log, routes)
        if g is None:
            rec["failed"][s_alt] = err
            continue
        dt = (p._acid_transform(s_alt) - t_in) if (on_ph0 and own_t) else 0.0
        states.append({"name": s_alt, "G": g + dt})
        rec["states"][s_alt] = {"kind": kind, "G": g, "dtransform": dt, "rel": g + dt - G[name]}
    ens = combine(states)
    rec.update(G_ens=ens["G"], shift=ens["G"] - G[name], populations=ens["populations"])
    log(f"    [microstates {MODE}: {name} {len(states)} states -> shift {ens['G'] - G[name]:+.2f}]")
    G[name] = ens["G"]


p._hydrate_all_sites = hooked_hydrate


def _pu():
    from metag.energetics.uma import load_uma
    return load_uma(p._MODEL)


# ---------------------------------------------------------------------------------------------------
EXPERIMENT = {   # aqueous (D2O) 25-31 C NMR populations, % ; refs in README of results
    "D-glucose": {"open": "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO",
                  "pop": {"a-p": 38.0, "b-p": 62.0, "a-f": 0.14, "b-f": 0.15, "acyclic": 0.0064}},
    "D-fructose": {"open": "OCC(=O)[C@@H](O)[C@H](O)[C@H](O)CO",
                   "pop": {"a-p": 2.7, "b-p": 68.2, "a-f": 6.2, "b-f": 22.4, "acyclic": 0.5}},
    "D-ribose": {"open": "O=C[C@H](O)[C@H](O)[C@H](O)CO",
                 "pop": {"a-p": 21.5, "b-p": 58.5, "a-f": 6.5, "b-f": 13.5, "acyclic": 0.1}},
}


def validate(out):
    pu = _pu(); log = lambda s: print(s, flush=True)
    std = p.STD_STATE_KJ
    res = {}
    for sugar, d in EXPERIMENT.items():
        rows = {}
        for f in forms.all_sugar_forms(d["open"]):
            if f["kind"] == "open":
                lab = "acyclic"
            else:
                lab = ("b" if forms.anomer_cip(f["smiles"]) == "R" else "a") + ("-p" if f["ring_size"] == 6 else "-f")
            routes = {"warnings": []}
            g, err = score_state(pu, {}, f"{sugar}:{lab}", 0, f["smiles"], std, (1, 2), 10, 48, log, routes)
            rows[lab] = {"smiles": f["smiles"], "G": g, "err": err, "warnings": routes["warnings"]}
            log(f"  {sugar} {lab:8s} {f['smiles']:50s} G={g}")
        ok = [{"name": k, "G": v["G"]} for k, v in rows.items() if v["G"] is not None]
        ens = combine(ok)
        exp = d["pop"]; tot = sum(exp.values())
        cmp = {}
        for k, v in rows.items():
            if v["G"] is None:
                continue
            pe = exp[k] / tot
            cmp[k] = {"pred_pct": round(100 * ens["populations"][k], 4), "exp_pct": exp[k],
                      "G_rel_pred": round(v["G"] - ens["G"], 2),
                      "G_rel_exp": round(-RT * math.log(pe), 2),
                      "error_kJ": round((v["G"] - ens["G"]) - (-RT * math.log(pe)), 2)}
        res[sugar] = {"forms": rows, "comparison": cmp}
        json.dump(res, open(out, "w"), indent=1)
    return res


def reactions(rids, mode, out):
    global MODE
    MODE = mode
    inputs = json.load(open(os.path.join(ROOT, "src/metag/data/reactions_opentecr_std.json")))
    pu = None if mode == "baseline" else _pu()
    if mode == "baseline":
        def no_qm(*a, **k):
            raise RuntimeError("cache miss in baseline replay")
        p.pool_confs = p.batched_energies = p.batched_fire = p.uma_gibbs_corr = p.dgsolv = no_qm
    res = json.load(open(out)) if os.path.exists(out) else {}
    for rid in rids:
        if rid in res and res[rid].get("dG_raw") is not None:
            continue
        _RID[0] = rid; t0 = time.time()
        r = p.score_reaction(pu, inputs[rid], key=rid, log=lambda s: print(s, flush=True))
        det = {sp: v for (rr, sp), v in DETAIL.items() if rr == rid}
        res[rid] = {"dG_raw": None if r is None else r.get("dG_raw"), "states": det,
                    "warnings": None if r is None else r["routes"].get("warnings"),
                    "minutes": round((time.time() - t0) / 60, 1)}
        json.dump(res, open(out, "w"), indent=1, default=str)
        print(f"### {rid} {mode} dG_raw={res[rid]['dG_raw']}", flush=True)
    return res


def _shift(rec, variant):
    """Re-evaluate one species' ensemble shift from its stored states. variant: 'table' (per-form acid
    transform, as run), 'common' (input transform for every form), 'hemiketal_2.6' (ring alpha-carboxyl of a
    hemiketal at the Neu5Ac pKa 2.6 instead of the flat 4.75 -- sensitivity only)."""
    z = 1.0
    for s_alt, st in rec.get("states", {}).items():
        rel = st["rel"]
        if variant == "common":
            rel -= st["dtransform"]
        elif variant == "hemiketal_2.6" and st["dtransform"] > 1e-6 and st["kind"].startswith("ring"):
            rel -= st["dtransform"]
            rel += RT * math.log(10) * (2.6 - 1.8)            # oxo-acid 1.8 (table) vs hemiketal acid 2.6
        z += math.exp(-rel / RT)
    return -RT * math.log(z)


def summarize(modes):
    rep = json.load(open(os.path.join(ROOT, "analysis/accuracy_review/report.json")))
    rows = {r["reaction"]: r for r in rep["ranked_reactions"]}
    inputs = json.load(open(os.path.join(ROOT, "src/metag/data/reactions_opentecr_std.json")))
    out = {}
    for mode in modes:
        f = os.path.join(SCRATCH, f"reactions_{mode}.json")
        if not os.path.exists(f):
            continue
        res = json.load(open(f))
        for rid, r in res.items():
            pr = rows[rid]
            d = out.setdefault(rid, {"class": pr["class"], "reference": round(pr["reference"], 2),
                                     "production_dG": pr["production_dG"],
                                     "production_error": round(pr["production_error"], 2)})
            coeff = {n: v[0] for n, v in pr["species_scored"].items()}
            sp = {}
            for name, rec in r["states"].items():
                if "states" not in rec:
                    continue
                sp[name] = {"coeff": coeff.get(name), "shift": round(rec["shift"], 2),
                            "populations": {k: round(v, 4) for k, v in rec["populations"].items() if v > 1e-3},
                            "failed": rec.get("failed"), "n_taut_enumerated": rec.get("n_taut_enumerated")}
            variants = {"table": None, "common": None, "hemiketal_2.6": None} if mode == "sugar" else {"table": None}
            for v in variants:
                variants[v] = round(sum((coeff.get(n) or 0) * _shift(rec, v)
                                        for n, rec in r["states"].items() if "states" in rec), 2)
            d[mode] = {"dG_raw": r["dG_raw"], "change": round(r["dG_raw"] - pr["production_dG"], 2)
                       if r["dG_raw"] is not None else None, "change_by_variant": variants, "species": sp}
    json.dump(out, open(os.path.join(SCRATCH, "summary.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    os.makedirs(SCRATCH, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "validate":
        validate(os.path.join(SCRATCH, "validate.json"))
    elif cmd == "reactions":
        mode = sys.argv[2]; rids = sys.argv[3].split(",")
        reactions(rids, mode, os.path.join(SCRATCH, f"reactions_{mode}.json"))
    elif cmd == "summarize":
        summarize(sys.argv[2].split(","))
