#!/usr/bin/env python
"""Cluster-continuum (explicit first-shell waters + xtb-ALPB) test of the phosphagen gap.

Species: pH-0 cores of PCr-type hydrolysis (CNC(N)= and CN(C)C(N)= variants), H3PO4, and the ATP-type
control (methyl triphosphate -> methyl diphosphate + Pi). Uses the package machinery of
metag.pipeline.explicit_G (bare_geom -> seed waters -> batched UMA FIRE -> lowest EXPLICIT_KEEP ->
xtb --opt <solv> relaxed ΔGsolv -> UniqueMinima -> Boltzmann; water_ref_G referencing). Per cluster we
store E_UMA, relaxed ΔGsolv, AND the full-cluster UMA RRHO, so analyze.py can evaluate both
  A (pipeline-exact):  G = boltz(E+solv) + thermal(bare solute) - n*water_ref_G
  B (full monomer cycle): G = boltz(E+solv+thermal(cluster)) - n*(water_ref_G + STD_STATE)
Water rules (deterministic):
  R1pkg : package water_count (1 per H on the formal N+; 0 for neutral phosphates), package seed_waters
  R1dir : same count, waters seeded as H-bond acceptors on those N-H
  R2    : 1 per N-H on all guanidinium N (charge is delocalised), directed
  R3    : R2 + 1 per P-OH donor (applied to EVERY species incl. the ATP control), directed
  suffix 'x2' = probe: 2 waters per site (water_count.converged_enough ΔG(n) vs ΔG(n+1/site))
Run: gpu_reserve run <idx> -- python run_microsolv.py --rules R1pkg,R1dir,R2 --reps 0,1,2
Package source is NOT modified; no shared cache is touched.
"""
import argparse, hashlib, json, os, re, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("METAG_CACHE", os.path.join(HERE, "cache"))
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ase import Atoms
from rdkit import Chem

from metag import pipeline as P
from metag.energetics.uma import load_uma, batched_fire
from metag.energetics.explicit_solvation import bare_geom
from metag.energetics import water_clusters as gc
from metag.energetics.thermal import uma_gibbs_corr, XTB, ENV, _write_xyz, _run_xtb
from metag.water_count import water_count

EV2KJ = 96.485
SPECIES = {
    "PCrM": ("CNC(N)=[NH+]P(=O)(O)O", 1),        # PCr-type core, CNC(N)= variant
    "CrM": ("CNC(N)=[NH2+]", 1),
    "PCr": ("CN(C)C(N)=[NH+]P(=O)(O)O", 1),      # creatine core, CN(C)C(N)= variant
    "Cr": ("CN(C)C(N)=[NH2+]", 1),
    "Pi": ("O=P(O)(O)O", 0),
    "MePPP": ("COP(=O)(O)OP(=O)(O)OP(=O)(O)O", 0),   # ATP-type control
    "MePP": ("COP(=O)(O)OP(=O)(O)O", 0),
}
N_SEEDS = P.N_EXPLICIT_SEEDS     # 16
KEEP = P.EXPLICIT_KEEP           # 8
SOLV = P.SOLV_MODEL              # alpb (current physics); explicit_G hard-codes cosmo -- we use the
                                 # production implicit model so implicit and cluster species are consistent


def donor_H(smi, rule):
    """Indices (AddHs atom order = bare_geom order) of donor H that get one water each."""
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    guanC = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "C"
             and sum(n.GetSymbol() == "N" for n in a.GetNeighbors()) == 3]
    out = []
    for h in m.GetAtoms():
        if h.GetSymbol() != "H":
            continue
        x = h.GetNeighbors()[0]
        if rule == "R1":
            ok = x.GetSymbol() == "N" and x.GetFormalCharge() > 0
        else:
            ok = x.GetSymbol() == "N" and any(n.GetIdx() in guanC for n in x.GetNeighbors())
            if rule == "R3":
                ok = ok or (x.GetSymbol() == "O" and any(n.GetSymbol() == "P" for n in x.GetNeighbors()))
        if ok:
            out.append((h.GetIdx(), x.GetIdx()))
    return out


def seed_directed(sym, pos, donors, n_water, rng):
    """Water as H-bond ACCEPTOR on donor X-H: O at 1.85 Å beyond H along X->H (jittered), water H's
    pointing away. Round-robin over donors (probe: 2nd water per donor). Best of 16 trials by clearance."""
    out_s, out_p = list(sym), [np.array(p) for p in pos]
    for w in range(n_water):
        h, x = donors[w % len(donors)]
        u = pos[h] - pos[x]; u /= np.linalg.norm(u)
        jit = 0.25 if w < len(donors) else 0.9            # 2nd-per-site water: wider search
        best = None
        for _ in range(16):
            d = u + jit * rng.normal(size=3); d /= np.linalg.norm(d)
            o = pos[h] + (1.85 if w < len(donors) else 2.6) * d
            clr = min(np.linalg.norm(o - p) for k, p in enumerate(out_p) if k != h)
            if best is None or clr > best[0]:
                best = (clr, o, d)
        _, o, d = best
        p = np.cross(d, rng.normal(size=3)); p /= np.linalg.norm(p)
        c, s = np.cos(np.radians(52.25)), np.sin(np.radians(52.25))
        out_s += ["O", "H", "H"]
        out_p += [o, o + 0.96 * (c * d + s * p), o + 0.96 * (c * d - s * p)]
    return out_s, np.asarray(out_p)


def xtb_relaxed_checked(sym, pos, q, n_solute, n_water, n_solute_H):
    """Same quantity as thermal.xtb_dgsolv_relaxed (E_xtb(--opt solv) - E_xtb(--sp gas)) but also reads
    xtbopt.xyz and reports whether the cluster kept its bonding (no proton transfer to/from water)."""
    with tempfile.TemporaryDirectory() as d:
        xyz = os.path.join(d, "m.xyz"); _write_xyz(xyz, sym, pos)
        eg = _run_xtb([XTB, xyz, "--gfn", "2", "--chrg", str(int(q)), "--sp"], d, 180)
        es = _run_xtb([XTB, "m.xyz", "--gfn", "2", "--chrg", str(int(q)), "--opt", f"--{SOLV}", "water"], d, 900)
        ok = None
        f = os.path.join(d, "xtbopt.xyz")
        if os.path.isfile(f):
            s2, p2 = gc.read_xyz(f)
            ok = bool(gc.valid_cluster(s2, p2, n_solute, n_water, n_solute_H))
        return ((es - eg) if (es is not None and eg is not None) else None), ok


def run_one(pu, name, rule, n_water, rep, bare, log):
    smi, q = SPECIES[name]
    tag = f"{name}_{rule}_n{n_water}_r{rep}"
    path = os.path.join(HERE, "records", tag + ".json")
    if os.path.exists(path):
        log(f"  skip {tag} (done)"); return
    t0 = time.time()
    bsym, bcoord = bare[name]
    n_sol = len(bsym); n_solH = sum(s == "H" for s in bsym)
    seed = int(hashlib.md5(f"{smi}|{q}|{n_water}|{rule}|{rep}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    base = rule.replace("x2", "")
    if n_water == 0:
        clusters = [Atoms(symbols=list(bsym), positions=bcoord, info={"charge": int(q), "spin": 1})]
    else:
        if base == "R1pkg":
            gen = lambda: gc.seed_waters(list(bsym), bcoord, n_water, rng)
        else:
            don = donor_H(smi, "R1" if base == "R1dir" else base)
            gen = lambda: seed_directed(list(bsym), bcoord, don, n_water, rng)
        clusters = [Atoms(symbols=cs, positions=cc, info={"charge": int(q), "spin": 1})
                    for cs, cc in (gen() for _ in range(N_SEEDS))]
    rel, E, conv = batched_fire(pu, clusters, fmax=0.06, steps=350, stop_frac=0.8,
                                return_converged=True, label=f"{tag}")
    E = np.asarray(E) * EV2KJ
    recs = []
    for i, (a, e, c) in enumerate(zip(rel, E, conv)):
        v = bool(gc.valid_cluster(a.get_chemical_symbols(), a.get_positions(), n_sol, n_water, n_solH)) \
            if n_water else True
        recs.append({"i": i, "E": float(e), "conv": bool(c), "valid_uma": v})
    cand = [r for r in recs if r["conv"] and r["valid_uma"]]
    cand = sorted(cand, key=lambda r: r["E"])[:KEEP]
    sel = [rel[r["i"]] for r in cand]
    with ThreadPoolExecutor(max_workers=int(os.environ.get("XTB_WORKERS", "8"))) as ex:
        sv = list(ex.map(lambda a: xtb_relaxed_checked(a.get_chemical_symbols(), a.get_positions(), q,
                                                       n_sol, n_water, n_solH), sel))
    for r, a, (s, ok) in zip(cand, sel, sv):
        r["solv"] = s; r["valid_xtb"] = ok if n_water else True
        g, info = uma_gibbs_corr(pu, a.get_chemical_symbols(), a.get_positions(), q, return_info=True)
        r["th_cluster"] = float(g); r["n_imag"] = info["n_imag"]; r["max_imag_cm"] = info["max_imag_cm"]
        r["kept"] = True
        r["pos"] = a.get_positions().round(4).tolist()
    th_bare = float(uma_gibbs_corr(pu, list(bsym), bcoord, q))
    out = {"name": name, "smi": smi, "q": q, "rule": rule, "n_water": n_water, "rep": rep, "seed": seed,
           "solv": SOLV, "n_seeds": len(clusters), "n_conv": int(np.sum(conv)),
           "n_valid": sum(r["conv"] and r["valid_uma"] for r in recs), "th_bare": th_bare,
           "symbols": rel[0].get_chemical_symbols(), "clusters": recs, "sec": time.time() - t0}
    json.dump(out, open(path, "w"))
    kept = [r for r in recs if r.get("kept") and r["solv"] is not None]
    log(f"  {tag}: conv {out['n_conv']}/{len(clusters)} valid {out['n_valid']} kept {len(kept)} "
        f"xtb-invalid {sum(r['valid_xtb'] is False for r in kept)} {out['sec']:.0f}s")


def counts(name, rule):
    smi, q = SPECIES[name]
    base = rule.replace("x2", ""); mult = 2 if rule.endswith("x2") else 1
    if base in ("R1pkg", "R1dir"):
        n = water_count(smi)[0]
    else:
        n = len(donor_H(smi, base))
    return n * mult


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default="R0,R1pkg,R1pkgx2,R1dir,R2,R2x2")
    ap.add_argument("--reps", default="0,1,2")
    ap.add_argument("--species", default=",".join(SPECIES))
    a = ap.parse_args()
    logf = open(os.path.join(HERE, "logs", f"run_{os.getpid()}.log"), "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log(f"solv={SOLV} seeds={N_SEEDS} keep={KEEP} physics={P.PHYSICS_VERSION}")
    pu = load_uma()
    wref = P.water_ref_G(pu, log)
    json.dump({"water_ref_G": wref, "std": P.STD_STATE_KJ}, open(os.path.join(HERE, "water_ref.json"), "w"))
    names = a.species.split(",")
    bare_path = os.path.join(HERE, "bare_geoms.json")
    bare = json.load(open(bare_path)) if os.path.exists(bare_path) else {}
    for nm in names:
        if nm not in bare:
            s, c = bare_geom(pu, SPECIES[nm][1], SPECIES[nm][0])
            bare[nm] = [list(s), np.asarray(c).tolist()]
            json.dump(bare, open(bare_path, "w"))
    bare = {k: (v[0], np.asarray(v[1])) for k, v in bare.items()}
    for rule in a.rules.split(","):
        for nm in names:
            if rule == "R0":
                if a.reps.split(",")[0] == "0":
                    run_one(pu, nm, "R0", 0, 0, bare, log)
                continue
            n = counts(nm, rule)
            if n == 0:
                continue                                   # identical to R0
            for rep in [int(x) for x in a.reps.split(",")]:
                run_one(pu, nm, rule, n, rep, bare, log)
    log("DONE")


if __name__ == "__main__":
    main()
