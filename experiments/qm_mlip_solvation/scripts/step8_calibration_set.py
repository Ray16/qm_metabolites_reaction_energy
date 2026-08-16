#!/usr/bin/env python
"""Step 8: validate the water-count HEURISTIC on a small diverse calibration set.

For 2-3 curated small representatives per functional-group category, run the
cluster-cycle grand-potential ladder (step7c) and read the self-selected occupancy
PEAK. Then compare the peak to the coordination-number RULE prediction:

    n_pred ≈ Σ(hard oxyanion O⁻)·(2–3)  +  Σ(soft S⁻)·(1–2)  +  Σ(cation N–H donor)·1

Passes if (a) the peak lands in the predicted band for NEED groups, and (b) soft /
delocalized anions (thiolate, phenolate) peak LOW (validating they're implicit-ish).
This tells us whether the cheap fixed-count rule transfers across chemistries so
production can skip the ladder.

Run (uma env): CUDA_VISIBLE_DEVICES=1 python scripts/step8_calibration_set.py --nmax 8 --seeds 4
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdkit import Chem
from batched_relax import load_uma
from step7c_cluster_cycle import water_cluster_ladder, species_omega

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")

# curated small reps: name -> (charge, SMILES, category, triage)
CALSET = {
    # carboxylate (localized -1, hard) — NEED
    "acetate":         (-1, "CC(=O)[O-]",          "carboxylate", "NEED"),
    "benzoate":        (-1, "c1ccccc1C(=O)[O-]",   "carboxylate", "NEED"),
    # phosphate monoester -2 (our MeP; hard, high density) — NEED
    "methylphosphate": (-2, "COP(=O)([O-])[O-]",   "phosphate",   "NEED"),
    "phosphate_H":     (-1, "OP(=O)(O)[O-]",       "phosphate",   "NEED"),
    # sulfonate (3 O, -1) — NEED
    "methanesulfonate":(-1, "CS(=O)(=O)[O-]",      "sulfonate",   "NEED"),
    "ethanesulfonate": (-1, "CCS(=O)(=O)[O-]",     "sulfonate",   "NEED"),
    # sulfate ester — NEED
    "methylsulfate":   (-1, "COS(=O)(=O)[O-]",     "sulfate",     "NEED"),
    # thiolate (soft S-, low density) — BORDERLINE (expect LOW peak)
    "methanethiolate": (-1, "C[S-]",               "thiolate",    "BORDERLINE"),
    "ethanethiolate":  (-1, "CC[S-]",              "thiolate",    "BORDERLINE"),
    # phenolate (delocalized -1) — BORDERLINE
    "phenolate":       (-1, "c1ccccc1[O-]",        "phenolate",   "BORDERLINE"),
    "p_cresolate":     (-1, "Cc1ccc([O-])cc1",     "phenolate",   "BORDERLINE"),
    # cations (H-bond donors) — BORDERLINE
    "methylammonium":  (+1, "C[NH3+]",             "ammonium",    "BORDERLINE"),
    "methylguanidinium":(+1,"CNC(N)=[NH2+]",       "guanidinium", "BORDERLINE"),
}


def rule_prediction(smi):
    """Coordination-rule band (n_low, n_high) from H-bond sites."""
    mol = Chem.MolFromSmiles(smi)
    hardO = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[O-]")))          # hard oxyanion O-
    softS = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#16-]")))        # soft S-
    # cationic donors: N-H count on positively charged N
    nplus = mol.GetSubstructMatches(Chem.MolFromSmarts("[N+,n+]"))
    nh = 0
    for (idx,) in nplus:
        nh += mol.GetAtomWithIdx(idx).GetTotalNumHs()
    lo = 2 * hardO + 1 * softS + 1 * nh
    hi = 3 * hardO + 2 * softS + 1 * nh
    return lo, hi, dict(hardO=hardO, softS=softS, cationNH=nh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)
    log(f"loading UMA... calibration set n={len(CALSET)} nmax={a.nmax} seeds={a.seeds}")
    pu = load_uma()
    log("  shared water-cluster reference ladder:")
    Gwc = water_cluster_ladder(pu, a.nmax, a.seeds, log)
    if any(Gwc[n] is None for n in Gwc):
        log("  FATAL: water ladder incomplete"); return

    results = []
    for name, (q, smi, cat, tri) in CALSET.items():
        lo, hi, sites = rule_prediction(smi)
        try:
            om, peak, occ = species_omega(pu, name, q, smi, a.nmax, a.seeds, Gwc, log)
        except Exception as e:
            log(f"    {name}: FAILED ({e})"); continue
        mean_n = sum(k * v for k, v in occ.items())   # Boltzmann-mean occupancy (robust to flat surface)
        ok = lo <= mean_n <= hi                          # judge on mean, not the noisy mode
        results.append(dict(name=name, cat=cat, triage=tri, charge=q, peak=peak,
                            mean_n=round(mean_n, 2), rule_lo=lo, rule_hi=hi, sites=sites,
                            in_band=ok, pinned=(peak == a.nmax)))
        log(f"    -> {name:17s} [{cat:11s}] peak n={peak} <n>={mean_n:.1f}  rule {lo}-{hi}  "
            f"{'OK' if ok else 'OUT'}{'  PINNED@cap!' if peak==a.nmax else ''}")

    log(f"\n==== HEURISTIC VALIDATION (Boltzmann-mean occupancy vs coordination rule) ====")
    log(f"  {'compound':17s} {'category':11s} {'q':>2} peak <n>  rule    verdict")
    for r in results:
        v = "PINNED-raise-cap" if r["pinned"] else ("in-band" if r["in_band"] else "OUT-of-band")
        log(f"  {r['name']:17s} {r['cat']:11s} {r['charge']:+2d}   {r['peak']:2d}  {r['mean_n']:4.1f}  "
            f"{r['rule_lo']}-{r['rule_hi']:<3d}  {v}")
    n_ok = sum(r["in_band"] for r in results)
    log(f"\n  {n_ok}/{len(results)} within predicted band. Soft/delocalized (thiolate,"
        f" phenolate) should peak LOW; hard oxyanions in-band.")
    json.dump(results, open(os.path.join(OUT, "step8_calibration_set.json"), "w"), indent=2)
    log("wrote artifacts/step8_calibration_set.json")


if __name__ == "__main__":
    main()
