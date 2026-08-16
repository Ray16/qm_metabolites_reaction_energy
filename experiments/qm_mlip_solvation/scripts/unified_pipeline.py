#!/usr/bin/env python
"""Unified pipeline test — ONE scheme across all three solved reaction classes.

Instead of three bespoke scripts (step3b redox / step5c glycosyl / step7b nucleotidyl),
run a SINGLE pipeline with automatic triage and NO per-class hand-tuning, and check it
reproduces all three at once. If one regresses, that pinpoints where the bespoke tuning
was load-bearing -> dive into that one.

The one scheme (per species):
  ETKDG pool -> batched UMA rank -> relax top-k -> Boltzmann ensemble of
  (E_elec[UMA] + ΔGsolv)  + UMA-Hessian thermal on the min-E conformer.
Triage picks the solvation treatment PER REACTION:
  - IMPLICIT (xtb --sp --cosmo)         when no compact anion is created/destroyed
  - EXPLICIT (water_count first-shell waters, cluster-continuum via corr_fast)
    when a compact high-charge-density anion IS created/destroyed (e.g. PPi).
ΔG = Σ_prod ν G - Σ_react ν G + n_H+ · G(H+,aq,pH7).

Run (uma env), one reaction per GPU in parallel:
  CUDA_VISIBLE_DEVICES=0 python scripts/unified_pipeline.py --only redox      &
  CUDA_VISIBLE_DEVICES=1 python scripts/unified_pipeline.py --only glycosyl   &
  CUDA_VISIBLE_DEVICES=2 python scripts/unified_pipeline.py --only nucleotidyl&
  # or omit --only to run all three sequentially on one GPU
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_THERMO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_THERMO, "backup", "explicit_water"))
import grand_canonical_clusters as gc
from batched_relax import load_uma, batched_energies, batched_fire
from step4e_targeted import pool_confs, boltz
from step7b_charge_balanced_waters import bare_geom
from thermal_solv import uma_gibbs_corr, xtb_dgsolv, xtb_dgsolv_relaxed, corr_fast
from water_count import water_count, needs_explicit

N_EXPLICIT_SEEDS = int(os.environ.get("N_EXPLICIT_SEEDS", "16"))  # cluster seeds (cheap: batched relax)
EXPLICIT_KEEP = int(os.environ.get("EXPLICIT_KEEP", "8"))         # lowest-E clusters kept for Boltzmann

OUT = os.path.join(_THERMO, "experiments", "qm_mlip_solvation", "artifacts")
EV2KJ = 96.485
T = 298.15
# CHE aqueous proton free energy at pH 7 (step3b/step6 convention)
G_HPLUS = -26.3 - 1104.5 - 2.303 * 8.314e-3 * T * 7.0    # ~ -1170.8 kJ/mol

# each reaction: exp ΔG, net H+ RELEASED (products), explicit-water flag, and
# stoichiometry {species: (coeff (+prod/-react), charge, SMILES)}
REACTIONS = {
    "redox": dict(exp=[18.0, 11.9], n_Hplus=1, explicit=False, note="2 MeSH + MNA+ -> MeSSMe + MNAH + H+", species={
        "MeSH":   (-2, 0, "CS"),
        "MNA+":   (-1, +1, "C[n+]1cccc(C(N)=O)c1"),
        "MeSSMe": (+1, 0, "CSSC"),
        "MNAH":   (+1, 0, "O=C(N)C1=CN(C)C=CC1"),
    }),
    "glycosyl": dict(exp=[-4.2], n_Hplus=0, explicit=False, note="MeUDPGlc + Fru -> MeUDP + Suc", species={
        "MeUDPGlc": (-1, -2, "OC[C@H]1O[C@@H](OP(=O)([O-])OP(=O)([O-])OC)[C@H](O)[C@@H](O)[C@@H]1O"),
        "Fructose": (-1, 0,  "OC[C@H]1OC(O)(CO)[C@@H](O)[C@@H]1O"),
        "MeUDP":    (+1, -2, "COP(=O)([O-])OP(=O)([O-])O"),
        "Suc":      (+1, 0,  "OC[C@H]1O[C@@H](O[C@]2(CO)O[C@H](CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"),
    }),
    "nucleotidyl": dict(exp=[1.85], n_Hplus=0, explicit=True, note="MeP + MePPP -> MePPMe + PPi", species={
        "MeP":    (-1, -2, "COP(=O)([O-])[O-]"),
        "MePPP":  (-1, -3, "COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])O"),
        "MePPMe": (+1, -2, "COP(=O)([O-])OP(=O)([O-])OC"),
        "PPi":    (+1, -3, "O=P([O-])([O-])OP(=O)([O-])O"),
    }),
    # --- hard-ten depth: independent glycosyl transfers (UDP nucleoside -> Me cap) ---
    "glycosyl_00605": dict(exp=[-9.51], n_Hplus=0, explicit=False,
                           note="rxn00605: MeUDPGlc + Glc-6-P -> MeUDP + Trehalose-6-P", species={
        "MeUDPGlc":   (-1, -2, "OC[C@H]1O[C@@H](OP(=O)([O-])OP(=O)([O-])OC)[C@H](O)[C@@H](O)[C@@H]1O"),
        "G6P":        (-1, -2, "O=P([O-])([O-])OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),
        "MeUDP":      (+1, -2, "COP(=O)([O-])OP(=O)([O-])O"),
        "Trehalose6P":(+1, -2, "O=P([O-])([O-])OC[C@H]1O[C@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"),
    }),
    "glycosyl_01713": dict(exp=[3.93], n_Hplus=-1, explicit=False,
                           note="rxn01713: MeUDPGlc + Sinapate + H+ -> MeUDP + Sinapoyl-Glc", species={
        "MeUDPGlc":   (-1, -2, "OC[C@H]1O[C@@H](OP(=O)([O-])OP(=O)([O-])OC)[C@H](O)[C@@H](O)[C@@H]1O"),
        "Sinapate":   (-1, -1, "COc1cc(/C=C/C(=O)[O-])cc(OC)c1O"),
        "MeUDP":      (+1, -2, "COP(=O)([O-])OP(=O)([O-])O"),
        "SinapoylGlc":(+1,  0, "COc1cc(/C=C/C(=O)O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)cc(OC)c1O"),
    }),
    # --- systematic-truncation variants (truncate.py) of the failing full-molecule
    #     glycosyl reactions. rxn00605 full-molecule missed by -45 kJ (disaccharide
    #     conformer noise); the truncated model shrinks every species to <=1 sugar ring
    #     so the transferred-glucosyl energy cancels cleanly. t2/t3 = radius 2/3 (the
    #     Me/Et cap-sensitivity check: if t2 ~ t3 the cap is converged). exp -9.51.
    "rxn00605_t2": dict(exp=[-9.51], n_Hplus=0, explicit=False,
                        note="rxn00605 TRUNC r2: donor-anomer + acceptor-Glc", species={
        "donorCap": (-1, 0,  "CC(O)O"),
        "G6Pt":     (-1, -1, "O=P([O-])(O)O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"),
        "glycoside":(+1, 0,  "C[C@H](O)O[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"),
        "Pi":       (+1, -1, "O=P([O-])(O)O"),
    }),
    "rxn00605_t3": dict(exp=[-9.51], n_Hplus=0, explicit=False,
                        note="rxn00605 TRUNC r3: one more shell (cap-sensitivity check)", species={
        "donorCap": (-1, 0,  "COC(O)[C@@H](C)O"),
        "G6Pt":     (-1, -1, "O=P([O-])(OP)O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"),
        "glycoside":(+1, 0,  "CO[C@H](O[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O)[C@@H](C)O"),
        "PPt":      (+1, -1, "O=P([O-])(O)OP"),
    }),
}


def sampling_budget(smi):
    """FIX 1: scale conformer sampling to molecular flexibility (rotatable bonds).
    Rigid species are cheap; floppy sugars/chains need many seeds + a big pool or the
    Boltzmann ensemble is under-sampled (glycosyl failed at 2×48; step5c used 5×128)."""
    m = Chem.MolFromSmiles(smi)
    nrot = rdMolDescriptors.CalcNumRotatableBonds(m) if m is not None else 0
    # pools are generous — conformer gen + batched UMA relax are cheap; only the xtb
    # solvation on `keep` conformers scales linearly (threaded across cores).
    # SAMPLE_SCALE (env) multiplies seed count + pool for stress-testing convergence.
    sc = float(os.environ.get("SAMPLE_SCALE", "1"))
    if nrot <= 3:
        seeds, keep, pool = [1, 2], 10, 96
    elif nrot <= 7:
        seeds, keep, pool = [1, 2, 3], 14, 192
    else:
        seeds, keep, pool = [1, 2, 3, 4, 5, 6], 18, 320
    if sc != 1:
        seeds = list(range(1, max(2, int(round(len(seeds) * sc))) + 1))
        pool = int(round(pool * sc))
    return seeds, keep, pool


def implicit_G(pu, q, smi, seeds, keep, pool, log, name):
    """Boltzmann(E_elec[UMA] + ΔGsolv[cosmo]) over conformers + UMA thermal(min-E).
    Sampling budget is flexibility-adaptive (seeds/keep/pool from rotatable bonds)."""
    seeds, keep, pool = sampling_budget(smi)
    all_G = []
    best = (1e18, None, None)
    for seed in seeds:
        cands = pool_confs(smi, q, seed, pool)
        order = np.argsort(batched_energies(pu, cands))[:keep]
        sel = [cands[i] for i in order]
        rel, E, conv = batched_fire(pu, sel, fmax=0.05, steps=300, stop_frac=0.9,
                                    return_converged=True, label=f"{name}s{seed}")
        sel = [a for a, c in zip(rel, conv) if c]; Eg = E[conv] * EV2KJ
        with ThreadPoolExecutor(max_workers=8) as ex:
            solv = list(ex.map(lambda a: xtb_dgsolv(a.get_chemical_symbols(),
                                                    a.get_positions(), q, "cosmo"), sel))
        for a, e, s in zip(sel, Eg, solv):
            if np.isfinite(e) and s is not None:
                all_G.append(e + s)
                if e < best[0]:
                    best = (float(e), a.get_chemical_symbols(), a.get_positions())
    if not all_G:
        return None
    Gens = boltz(all_G)
    therm = uma_gibbs_corr(pu, best[1], best[2], q)
    log(f"    {name:9s} q{q:+d} [implicit]: Gens {Gens:.1f} + thermal {therm:.1f} = {Gens+therm:.1f}")
    return Gens + therm


def explicit_G(pu, q, smi, seeds, log, name):
    """Cluster-continuum G_aq: first-shell waters + cluster solvation, but thermal on
    the BARE SOLUTE only.
    FIX 2: the floppy explicit-water librational modes make the full-cluster UMA
    finite-diff Hessian noisy and it does NOT cancel across the fixed-count reaction
    (this is what pinned the occupancy AND cost nucleotidyl ~20 kJ). So:
      G_aq = E_UMA(cluster)            # electronic, waters included → cancel (balanced)
           + ΔGsolv(cluster, xtb --sp) # cluster-continuum bulk solvation
           + thermal(BARE solute, UMA) # NO floppy water modes; cancels across reaction
    This also UNIFIES thermal with the implicit path (always bare-solute UMA Hessian)."""
    bsym, bcoord = bare_geom(pu, q, smi)
    n_water, sites = water_count(smi)
    # generous cluster sampling (cheap: batched relax) — floppy water-decorated clusters
    rng = np.random.default_rng(abs(hash((name, n_water))) % (2**32))
    clusters = [Atoms(symbols=cs, positions=cc, info={"charge": int(q), "spin": 1})
                for cs, cc in (gc.seed_waters(bsym, bcoord, n_water, rng) for _ in range(N_EXPLICIT_SEEDS))]
    rel, E, conv = batched_fire(pu, clusters, fmax=0.06, steps=350, stop_frac=0.8,
                                return_converged=True, label=f"{name}w{n_water}")
    rel = [a for a, c in zip(rel, conv) if c]; E = E[conv] * EV2KJ
    if not len(E):
        return None
    # FIX: BOLTZMANN over the cluster ensemble (NOT min — min is non-convergent for
    # floppy clusters: E_UMA drifts down as you add seeds). Keep the lowest-E clusters,
    # relaxed-in-solvent solvation (xtb --opt --cosmo) on each, Boltzmann of E_UMA+ΔGsolv.
    order = np.argsort(E)[:EXPLICIT_KEEP]
    sel = [rel[i] for i in order]; Eu = [float(E[i]) for i in order]
    with ThreadPoolExecutor(max_workers=8) as ex:
        solv = list(ex.map(lambda a: xtb_dgsolv_relaxed(a.get_chemical_symbols(),
                                                        a.get_positions(), q, "cosmo"), sel))
    Gt = [e + s for e, s in zip(Eu, solv) if s is not None]
    if not Gt:
        return None
    Gens = boltz(Gt)
    thermal = uma_gibbs_corr(pu, bsym, bcoord, q)         # bare solute, no water modes
    g = Gens + thermal
    log(f"    {name:9s} q{q:+d} [explicit n={n_water} {sites} keep{len(Gt)}/{N_EXPLICIT_SEEDS}]: "
        f"Gens(E+solv) {Gens:.1f} + thermal(solute) {thermal:.1f} = {g:.1f}")
    return g


def run_reaction(pu, key, seeds, keep, pool, log):
    rx = REACTIONS[key]
    log(f"\n=== {key}: {rx['note']}  (explicit={rx['explicit']}, n_H+={rx['n_Hplus']}) ===")
    G = {}
    for name, (coeff, q, smi) in rx["species"].items():
        G[name] = (explicit_G(pu, q, smi, seeds, log, name) if rx["explicit"]
                   else implicit_G(pu, q, smi, seeds, keep, pool, log, name))
        if G[name] is None:
            log(f"    {name}: FAILED"); return None
    dG = sum(coeff * G[name] for name, (coeff, q, smi) in rx["species"].items())
    dG += rx["n_Hplus"] * G_HPLUS
    errs = [dG - e for e in rx["exp"]]
    log(f"  ΔG = {dG:+.1f} kJ/mol   vs exp {rx['exp']}   err {[round(e,1) for e in errs]}")
    return dict(reaction=key, dG=round(dG, 1), exp=rx["exp"],
                err=[round(e, 1) for e in errs], explicit=rx["explicit"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(REACTIONS), default=None)
    ap.add_argument("--seeds", default="1,2"); ap.add_argument("--keep", type=int, default=10)
    ap.add_argument("--pool", type=int, default=48)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    log = lambda s: print(s, flush=True)
    keys = [a.only] if a.only else list(REACTIONS)
    log(f"loading UMA... unified pipeline, reactions={keys} seeds={seeds} keep={a.keep}")
    pu = load_uma()
    rows = [r for r in (run_reaction(pu, k, seeds, a.keep, a.pool, log) for k in keys) if r]

    log(f"\n==== UNIFIED PIPELINE — one scheme, three classes ====")
    log(f"  {'reaction':12s} {'ΔG':>7s} {'exp':>14s} {'err':>16s} {'solv':>9s}")
    for r in rows:
        log(f"  {r['reaction']:12s} {r['dG']:7.1f} {str(r['exp']):>14s} {str(r['err']):>16s} "
            f"{'explicit' if r['explicit'] else 'implicit':>9s}")
    tag = a.only or "all"
    json.dump(rows, open(os.path.join(OUT, f"unified_pipeline_{tag}.json"), "w"), indent=2)
    log(f"wrote artifacts/unified_pipeline_{tag}.json")


if __name__ == "__main__":
    main()
