"""CoA-class head-to-head: does REACTION-LEVEL frozen-tail sampling fix the floppy-CoA-tail error?

For each CoA reaction with a conserved tail spanning one reactant (CoA-SH / acyl-CoA) and one product
(acyl-CoA), the reaction's electronic ΔE contains the CoA-pair term E(CoA_prod) - E(CoA_reac). Under
INDEPENDENT sampling the 45-atom pantetheine-ADP tail lands in different basins on the two sides and
does not cancel -> the +54/-30 errors. FROZEN-tail sampling (paired_dE_symmetric) freezes the tail at a
shared geometry so it cancels, relaxing only the acyl head (the reaction center).

Because every NON-CoA species is identical between the two methods, the reaction-level correction is
exactly  ΔE_frozen(CoA pair) - ΔE_indep(CoA pair)  -- the other species cancel. So:
    ΔG_frozen  ≈  ΔG_production  +  (F_CoA - I_CoA)
(the production thermal+solv+pH is reused; freezing the tail does not change it -- the tail's
thermal/solv also cancels). One UMA load, loops over the reactions given on argv.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import numpy as np
from batched_relax import load_uma, batched_fire
from matched_sampling import paired_dE, paired_dE_symmetric, EV2KJ, RT
from step4e_targeted import pool_confs

CASES = {c["rid"]: c for c in json.load(open(os.path.join(os.path.dirname(__file__), "..", "scripts", "coa_headtohead_cases.json")))}
POOL = int(os.environ.get("POOL", "6"))          # frozen method is low-variance -> few conformers suffice
FMAX = float(os.environ.get("FMAX", "0.06"))
STEPS = int(os.environ.get("STEPS", "200"))
NBUF = int(os.environ.get("MATCH_NBUF", "3"))    # boundary a few bonds past the junction (colleague's caveat 1)
SYM = os.environ.get("SYM", "0") == "1"          # CoA tail is DECOUPLED -> one-ended suffices; sym only if strain flagged
OUT = os.environ.get("OUT_FILE", os.path.join(os.path.dirname(__file__), "..", "artifacts", "coa_headtohead.json"))


def boltz(Es):
    Es = np.array(Es, float); m = Es.min()
    return float(m - RT * np.log(np.exp(-(Es - m) / RT).mean()))


def indep_dE(pu, smiA, qA, smiB, qB, pool):
    a = pool_confs(smiA, qA, 1, pool); b = pool_confs(smiB, qB, 2, pool)
    _, EA, cA = batched_fire(pu, a, fmax=FMAX, steps=STEPS, stop_frac=0.9, return_converged=True)
    _, EB, cB = batched_fire(pu, b, fmax=FMAX, steps=STEPS, stop_frac=0.9, return_converged=True)
    EA = np.array(EA)[cA] * EV2KJ if cA.any() else np.array(EA) * EV2KJ
    EB = np.array(EB)[cB] * EV2KJ if cB.any() else np.array(EB) * EV2KJ
    return boltz(EB) - boltz(EA)


def main():
    rids = sys.argv[1:] or list(CASES)
    pu = load_uma(os.environ.get("UMA_MODEL", "uma-s-1p2p1"))
    out = []
    for rid in rids:
        c = CASES[rid]; t = time.time()
        try:
            I = indep_dE(pu, c["smiR"], c["qR"], c["smiP"], c["qP"], POOL)
            fn = paired_dE_symmetric if SYM else paired_dE
            pr = fn(pu, c["smiR"], c["qR"], c["smiP"], c["qP"], seed=1, pool=POOL, nbuf=NBUF,
                    fmax=FMAX, steps=STEPS) if not SYM else \
                 fn(pu, c["smiR"], c["qR"], c["smiP"], c["qP"], seed=1, pool=POOL, nbuf=NBUF)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"{rid} FAILED: {e}", flush=True); continue
        F = pr["dE_kJ"] if pr else None
        # reliability flag: symmetric -> strain_residual; one-ended -> per-conformer spread (dE_pair_std)
        strain = (pr.get("strain_residual") if SYM else pr.get("dE_pair_std")) if pr else None
        corr = (F - I) if (F is not None) else None
        dG_frozen = (c["prod_dG"] + corr) if (corr is not None and c["prod_dG"] is not None) else None
        err_indep = (c["prod_dG"] - c["exp"]) if c["prod_dG"] is not None else None
        err_frozen = (dG_frozen - c["exp"]) if dG_frozen is not None else None
        row = dict(rid=rid, exp=c["exp"], prod_dG=c["prod_dG"], err_indep=err_indep,
                   I_CoA=I, F_CoA=F, correction=corr, dG_frozen=dG_frozen, err_frozen=err_frozen,
                   strain_residual=strain, sym=SYM, nbuf=NBUF,
                   scaffold=pr.get("scaffold_heavy") if pr else None, free=pr.get("free_heavy_A") if pr else None)
        out.append(row)
        print(f"{rid:<10} exp={c['exp']:+6.1f}  prodΔG={c['prod_dG']:+6.1f}(err{err_indep:+5.1f})  "
              f"I={I:+6.1f} F={F:+6.1f} corr={corr:+6.1f} -> ΔG_frozen={dG_frozen:+6.1f}(err{err_frozen:+5.1f})  "
              f"strain/σ={strain:.1f} free={row['free']}/{row['scaffold']}  {time.time()-t:.0f}s", flush=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
