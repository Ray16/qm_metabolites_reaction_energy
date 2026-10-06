#!/usr/bin/env python
"""Assemble cluster-continuum reaction ΔG from records/ (see run_microsolv.py for definitions).
Run (CPU): PYTHONPATH=../../src python analyze.py  -> results.json + printed table."""
import glob, json, os, sys
import numpy as np
from ase import Atoms
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metag.energetics.conformers import UniqueMinima, boltz
from metag.water_count import converged_enough

W = json.load(open(os.path.join(HERE, "water_ref.json")))
WREF, STD = W["water_ref_G"], W["std"]
WREF_PIPE = -200682.84                    # sweep_20261001 water_ref_G_expt.json (used for implicit sums)
RXN = {  # name: (reactant, products, pKa transform, implicit ΔG, implicit species_sum, reference)
    "PCr-type (CNC(N)=)": ("PCrM", ["CrM", "Pi"], 9.5, -72.7, -82.2, -43.0),
    "PCr-type (CN(C)C(N)=, creatine)": ("PCr", ["Cr", "Pi"], 9.5, -66.7, -76.2, -43.0),
    "ATP-type control": ("MePPP", ["MePP", "Pi"], 2.5, -38.2, -40.7, -36.0),
}
RULES = ["R1pkg", "R1pkgx2", "R1dir", "R2", "R2x2", "R3", "R3x2"]

recs = {}
for f in glob.glob(os.path.join(HERE, "records", "*.json")):
    d = json.load(open(f))
    recs.setdefault((d["name"], d["rule"]), {})[d["rep"]] = d


def species_G(ds):
    """(G_A, G_B, n_kept, n_dropped_xtb, n_imag_clusters, n) from one or more replicate records (pooled)."""
    uA, uB = UniqueMinima(), UniqueMinima()
    n_kept = n_bad = n_imag = 0
    for d in ds:
        for r in d["clusters"]:
            if not r.get("kept") or r.get("solv") is None:
                continue
            if r.get("valid_xtb") is False:
                n_bad += 1; continue
            a = Atoms(symbols=d["symbols"], positions=np.asarray(r["pos"]))
            uA.add(a, r["E"], r["E"] + r["solv"])
            uB.add(a, r["E"], r["E"] + r["solv"] + r["th_cluster"])
            n_kept += 1; n_imag += r["n_imag"] > 0
    if not uA.G:
        return None
    d0, n = ds[0], ds[0]["n_water"]
    GA = boltz(uA.G) + d0["th_bare"] - n * WREF                  # pipeline explicit_G, exactly
    GB = boltz(uB.G) - n * (WREF + STD)                          # full Bryantsev cycle, 1 M waters
    return GA, GB, n_kept, n_bad, n_imag, n, len(uB.G)


def G_for(name, rule, rep):
    """rep=int -> that replicate; rep='pool' -> all replicates; species without sites -> R0."""
    if (name, rule) not in recs:
        r0 = recs[(name, "R0")][0]
        return species_G([r0])
    reps = recs[(name, rule)]
    if rep == "pool":
        return species_G(list(reps.values()))
    return species_G([reps[rep]]) if rep in reps else None


out = {}
for rx, (R, Ps, pka, dg_imp, ss_imp, ref) in RXN.items():
    sp = [R] + Ps
    G0 = {s: G_for(s, "R0", 0) for s in sp}
    ss0 = {k: sum(G0[p][k] for p in Ps) - G0[R][k] - WREF for k in (0, 1)}   # same-protocol bare baseline
    out[rx] = {"implicit_dG": dg_imp, "reference": ref, "pKa": pka, "bare_same_protocol_dG": ss0[0] + pka,
               "rules": {}}
    for rule in RULES:
        if not any((s, rule) in recs for s in sp):
            continue
        row = {"n_water": {s: G_for(s, rule, "pool")[5] if G_for(s, rule, "pool") else None for s in sp}}
        for k, lab in ((0, "A_pipeline_exact"), (1, "B_full_cycle")):
            per = []
            for rep in (0, 1, 2):
                Gs = [G_for(s, rule, rep) for s in sp]
                if all(g is not None for g in Gs):
                    per.append(sum(Gs[i + 1][k] for i in range(len(Ps))) - Gs[0][k] - WREF + pka)
            Gp = [G_for(s, rule, "pool") for s in sp]
            if any(g is None for g in Gp):
                continue
            pooled = sum(Gp[i + 1][k] for i in range(len(Ps))) - Gp[0][k] - WREF + pka
            corr = pooled - (ss0[k] + pka)                       # water effect vs same-protocol bare
            row[lab] = {"pooled": pooled, "per_rep": per, "rep_std": float(np.std(per, ddof=1)) if len(per) > 1 else None,
                        "rep_range": float(np.ptp(per)) if per else None,
                        "water_effect_vs_bare": corr, "implicit_plus_water_effect": dg_imp + corr}
        row["diag"] = {s: {"kept": g[2], "xtb_proton_transfer_dropped": g[3], "clusters_with_imag": g[4],
                           "unique": g[6]} for s, g in zip(sp, Gp) if g}
        out[rx]["rules"][rule] = row
    for base in ("R1pkg", "R2", "R3"):
        a, b = out[rx]["rules"].get(base), out[rx]["rules"].get(base + "x2")
        if a and b:
            for lab in ("A_pipeline_exact", "B_full_cycle"):
                if lab in a and lab in b:
                    a[lab]["probe_dG_x2"] = b[lab]["pooled"]
                    a[lab]["probe_delta"] = b[lab]["pooled"] - a[lab]["pooled"]
                    a[lab]["converged_enough"] = bool(converged_enough(a[lab]["pooled"], b[lab]["pooled"]))
json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)

for rx, o in out.items():
    print(f"\n== {rx}: implicit {o['implicit_dG']:+.1f}  bare(same protocol) {o['bare_same_protocol_dG']:+.1f}  ref {o['reference']:+.1f}")
    for rule, row in o["rules"].items():
        nw = " ".join(f"{s}:{n}" for s, n in row["n_water"].items())
        for lab in ("A_pipeline_exact", "B_full_cycle"):
            if lab not in row: continue
            v = row[lab]
            rs = f"{v['rep_std']:.1f}" if v["rep_std"] is not None else "-"
            pr = f" probe x2 {v['probe_dG_x2']:+.1f} (Δ{v['probe_delta']:+.1f}, conv={v['converged_enough']})" if "probe_delta" in v else ""
            print(f"  {rule:8s} {lab[:1]} n[{nw}]  ΔG {v['pooled']:+7.1f}  reps {[round(x,1) for x in v['per_rep']]} sd {rs}"
                  f"  water-effect {v['water_effect_vs_bare']:+.1f} -> implicit+effect {v['implicit_plus_water_effect']:+.1f}{pr}")
        print(f"           diag {row['diag']}")
