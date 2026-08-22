"""GATE for the frozen-tail build: is rxn00171's +67 error tail-SAMPLING or head (solvation/electronic)?

rxn00171 = CoA-SH + acetaldehyde + NAD+ -> acetyl-CoA + NADH (+H+). The CoA tail is conserved Δq=0, so
if the +67 is floppy-tail non-cancellation it shows up as (a) large seed-to-seed SPREAD in the
INDEPENDENT electronic ΔE of the CoA pair (CoA-SH -> acetyl-CoA), and (b) a large gap between that
independent ΔE and the seed-STABLE frozen-tail ΔE. If instead the independent ΔE is already tight and
frozen ≈ independent, the +67 lives in the HEAD (thioester vs thiol local solvation / electronic),
which frozen-tail canNOT fix -> don't build it.

Reports, over several seeds:  independent ΔE(CoA pair) mean±std   vs   frozen ΔE mean±std.
tail-sampling component ≈ |independent − frozen| and the independent std. Decision printed.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import numpy as np
from batched_relax import load_uma, batched_fire
from matched_sampling import paired_dE, EV2KJ, RT
from step4e_targeted import pool_confs

# read the CoA pair directly from the reactions file (self-contained; SMARTS match = SMILES-direction
# independent, unlike a substring which misses a reverse-written CoA).
from rdkit import Chem
_RXN = json.load(open(os.path.join(os.path.dirname(__file__), "..", "scripts", "reactions_tecrdb_all.json")))
_PANT = Chem.MolFromSmarts("SCCNC(=O)CCNC(=O)")   # pantetheine (CoA), direction-independent
def _has_coa(s):
    m = Chem.MolFromSmiles(s)
    return m is not None and m.HasSubstructMatch(_PANT)
_sp = _RXN["rxn00171"]["species"]
_R = [(nm, q, s) for nm, (c, q, s) in _sp.items() if c < 0 and _has_coa(s)][0]
_P = [(nm, q, s) for nm, (c, q, s) in _sp.items() if c > 0 and _has_coa(s)][0]
C = dict(coaR_name=_R[0], qR=_R[1], smiR=_R[2], coaP_name=_P[0], qP=_P[1], smiP=_P[2])
POOL = int(os.environ.get("POOL", "10"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "1,2,3").split(",")]


def boltz(Es):
    Es = np.array(Es, float); m = Es.min()
    return float(m - RT * np.log(np.exp(-(Es - m) / RT).mean()))


def indep_dE(pu, seedA, seedB):
    a = pool_confs(C["smiR"], C["qR"], seedA, POOL); b = pool_confs(C["smiP"], C["qP"], seedB, POOL)
    _, EA, cA = batched_fire(pu, a, fmax=0.05, steps=300, stop_frac=0.9, return_converged=True)
    _, EB, cB = batched_fire(pu, b, fmax=0.05, steps=300, stop_frac=0.9, return_converged=True)
    EA = np.array(EA)[cA] * EV2KJ if cA.any() else np.array(EA) * EV2KJ
    EB = np.array(EB)[cB] * EV2KJ if cB.any() else np.array(EB) * EV2KJ
    return boltz(EB) - boltz(EA)


def main():
    pu = load_uma(os.environ.get("UMA_MODEL", "uma-s-1p2p1"))
    print(f"rxn00171  CoA pair: {C['coaR_name']} (q{C['qR']}) -> {C['coaP_name']} (q{C['qP']})", flush=True)
    ind, frz = [], []
    for s in SEEDS:
        t = time.time()
        di = indep_dE(pu, s, s + 100)                 # independent: different seeds each side (the real pipeline behaviour)
        pr = paired_dE(pu, C["smiR"], C["qR"], C["smiP"], C["qP"], seed=s, pool=POOL, nbuf=2)
        f = pr["dE_kJ"] if pr else float("nan")
        ind.append(di); frz.append(f)
        print(f"  seed {s}: independent ΔE = {di:+7.1f}   frozen ΔE = {f:+7.1f}   ({time.time()-t:.0f}s)", flush=True)
    ind = np.array(ind); frz = np.array(frz)
    print(f"\n  INDEPENDENT ΔE(CoA pair): mean {ind.mean():+.1f}  std {ind.std():.1f}  range {ind.max()-ind.min():.1f}")
    print(f"  FROZEN-TAIL  ΔE(CoA pair): mean {np.nanmean(frz):+.1f}  std {np.nanstd(frz):.1f}")
    print(f"  tail-sampling shift (frozen − independent mean) = {np.nanmean(frz)-ind.mean():+.1f} kJ")
    verdict = ("TAIL-SAMPLING (frozen-tail worth building)" if ind.std() > 12 or abs(np.nanmean(frz)-ind.mean()) > 15
               else "HEAD effect (solvation/electronic) — frozen-tail will NOT fix the +67; don't build")
    print(f"\n  DECISION: {verdict}")
    json.dump(dict(independent=ind.tolist(), frozen=frz.tolist(), ind_std=float(ind.std()),
                   shift=float(np.nanmean(frz)-ind.mean()), verdict=verdict),
              open(os.path.join(os.path.dirname(__file__), "..", "artifacts", "rxn00171_decomp.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
