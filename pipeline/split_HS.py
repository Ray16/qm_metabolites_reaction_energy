#!/usr/bin/env python
"""Channel-2 improvement: split the combined G_RRHO into H and S per compound.

Reproduces the pipeline's thermo exactly (xtb --gfn2 --alpb water --ohess on the
lowest conformer) but also records the enthalpy/entropy decomposition:

    G_RRHO = (TOTAL FREE ENERGY - TOTAL ENERGY)          [matches stored value]
    H_RRHO = (TOTAL ENTHALPY    - TOTAL ENERGY)
    T*S    = (TOTAL ENTHALPY    - TOTAL FREE ENERGY)      # T = 298.15 K
    S      = T*S / 298.15                                 # gas-phase RRHO entropy, 1 atm

Usage:
    python split_HS.py            # all 453
    python split_HS.py cpd00001 cpd00020 cpd00002   # validation subset
"""
import os, re, sys, json, shutil, subprocess, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

TC = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc"
XTB = "/nfs/lambda_stor_01/homes/rzhu/gxtb/xtb-6.7.1/bin/xtb"
ENS = f"{TC}/pipeline/ensemble_tecrdb_full.json"
GAQ = f"{TC}/mlip/G_aq_tecrdb_full.json"
OUT = f"{TC}/mlip/HS_split_tecrdb_full.json"
SCRATCH = os.environ.get("SCRATCH_ROOT", "/tmp/hs_split")
T = 298.15
H2KJ = 2625.499639
os.makedirs(SCRATCH, exist_ok=True)

ens = json.load(open(ENS))
gaq = json.load(open(GAQ))

def lowest_xyz(cpd):
    recs = ens.get(cpd) or []
    recs = [r for r in recs if r.get("xyz") and os.path.exists(r["xyz"])]
    if not recs:
        return None
    recs.sort(key=lambda r: r.get("conf", 0))     # conf 0 = lowest kept
    return recs[0]["xyz"]

def run_one(cpd):
    xyz = lowest_xyz(cpd)
    if xyz is None:
        return cpd, {"error": "no geometry"}
    chg = gaq[cpd]["charge"]
    wd = tempfile.mkdtemp(prefix=f"hs_{cpd}_", dir=SCRATCH)
    try:
        shutil.copy(xyz, os.path.join(wd, "in.xyz"))
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        p = subprocess.run([XTB, "in.xyz", "--gfn", "2", "--alpb", "water", "--ohess",
                            "--chrg", str(chg), "--uhf", "0"],
                           cwd=wd, capture_output=True, text=True, env=env, timeout=3600)
        out = p.stdout
        def grab(tag):
            m = re.search(tag + r"\s+(-?\d+\.\d+)", out)
            return float(m.group(1)) if m else None
        Etot = grab("TOTAL ENERGY")
        Htot = grab("TOTAL ENTHALPY")
        Gtot = grab("TOTAL FREE ENERGY")
        if None in (Etot, Htot, Gtot):
            return cpd, {"error": "parse", "rc": p.returncode}
        g_rrho = (Gtot - Etot) * H2KJ
        h_rrho = (Htot - Etot) * H2KJ
        tS = (Htot - Gtot) * H2KJ
        stored = gaq[cpd]["conformers"][0]["G_RRHO_kJ"]
        return cpd, {"charge": chg, "name": gaq[cpd].get("name", ""),
                     "G_RRHO_kJ": g_rrho, "H_RRHO_kJ": h_rrho,
                     "TS_kJ": tS, "S_kJ_per_K": tS / T,
                     "G_RRHO_stored_kJ": stored, "repro_diff_kJ": g_rrho - stored}
    except subprocess.TimeoutExpired:
        return cpd, {"error": "timeout"}
    finally:
        shutil.rmtree(wd, ignore_errors=True)

def main():
    cpds = sys.argv[1:] or list(gaq.keys())
    results = {}
    if os.path.exists(OUT):
        results = json.load(open(OUT))
    todo = [c for c in cpds if c not in results and c in gaq]
    print(f"running {len(todo)} compounds (already have {len(results)})", flush=True)
    nworkers = int(os.environ.get("HS_WORKERS", "12"))
    done = 0
    with ProcessPoolExecutor(max_workers=nworkers) as ex:
        futs = {ex.submit(run_one, c): c for c in todo}
        for fut in as_completed(futs):
            cpd, res = fut.result()
            results[cpd] = res
            done += 1
            if "error" in res:
                print(f"  [{done}/{len(todo)}] {cpd} ERROR {res['error']}", flush=True)
            elif done <= 6 or done % 50 == 0:
                print(f"  [{done}/{len(todo)}] {cpd} S={res['S_kJ_per_K']*1000:.1f} J/K/mol "
                      f"repro_diff={res['repro_diff_kJ']:+.2f} kJ", flush=True)
            if done % 25 == 0:
                json.dump(results, open(OUT, "w"), indent=0)
    json.dump(results, open(OUT, "w"), indent=0)
    ok = [r for r in results.values() if "error" not in r]
    diffs = [abs(r["repro_diff_kJ"]) for r in ok]
    print(f"\ndone: {len(ok)} ok, {len(results)-len(ok)} errors")
    if diffs:
        import statistics
        print(f"G_RRHO reproduction |diff|: max {max(diffs):.2f}  mean {statistics.mean(diffs):.2f} kJ/mol")
    print("wrote", OUT)

if __name__ == "__main__":
    main()
