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
PH = 7.0
RT_LN10 = 2.303 * 8.314e-3 * T                            # ~5.71 kJ/mol per pKa unit
# pH-0 route (Jinich/Alberty): compute the NEUTRAL protonated microspecies (well-solvated
# by continuum -> no created/destroyed-anion pathology, no huge G(H+) term), then bridge
# to pH 7 analytically with the EXPERIMENTAL pKa. `pka_sites` = list of (side, pKa) for
# each ionizable group that is deprotonated at pH7 but PROTONATED in the QM microspecies.
#   ΔG'(pH7) = ΔG_QM(neutral) + Σ sign·RT ln10·(pH - pKa),  sign=+1 reactant, -1 product.

# Reaction DEFINITIONS are DATA, deliberately kept OUT of this engine -> reactions.json
# (stoichiometry {species: [coeff(+prod/-react), charge, SMILES]}, exp ΔG, n_Hplus,
# explicit-water flag/list, optional pH-0 pka_sites). This file stays generic: sampling
# heuristics, solvation triage, thermal/electronic backends. Add reactions to the JSON.
_RXN_JSON = os.path.join(os.path.dirname(__file__), "reactions.json")
def _load_reactions(path=_RXN_JSON):
    raw = json.load(open(path))
    out = {}
    for key, rx in raw.items():
        d = dict(rx)
        d["species"] = {n: tuple(v) for n, v in d["species"].items()}
        if isinstance(d.get("explicit"), list):
            d["explicit"] = set(d["explicit"])
        if "pka_sites" in d:
            d["pka_sites"] = [tuple(x) for x in d["pka_sites"]]
        out[key] = d
    return out
REACTIONS = _load_reactions()


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


# auto-convergent sampling knobs (env-overridable for stress tests; NOT per-reaction tuning)
CONV_TOL   = float(os.environ.get("CONV_TOL", "1.5"))    # kJ: Gens & min-E must move < this
CONV_HITS  = int(os.environ.get("CONV_HITS", "2"))       # for this many consecutive seed-batches
CONV_MAX   = int(os.environ.get("CONV_MAX", "16"))       # hard cap on seed-batches (runaway guard)


def implicit_G(pu, q, smi, seeds, keep, pool, log, name):
    """Boltzmann(E_elec[UMA] + ΔGsolv[cosmo]) over conformers + UMA thermal(min-E).

    HEURISTIC (general, self-calibrating -- no fixed per-flexibility tiers, no per-reaction
    tuning): keep adding conformer seed-batches until BOTH the Boltzmann Gens AND the minimum
    energy stop moving (< CONV_TOL for CONV_HITS consecutive batches), capped at CONV_MAX.
    Rigid species converge in ~2-3 batches; floppy sugar-phosphates draw as many as they need.
    Per-batch pool/keep still scale with rotatable bonds (bigger search for floppier molecules).
    Reports the seed count + the last increment so the sampling uncertainty is visible (UQ)."""
    _, keep, pool = sampling_budget(smi)                  # per-batch pool/keep sizing only
    all_G = []
    best = (1e18, None, None)
    prev_Gens = prev_best = None
    hits = 0
    seed = 0
    last_dG = float("nan")
    while seed < CONV_MAX:
        seed += 1
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
            continue
        Gens = boltz(all_G)
        if prev_Gens is not None:
            last_dG = abs(Gens - prev_Gens)
            if last_dG < CONV_TOL and abs(best[0] - prev_best) < CONV_TOL:
                hits += 1
                if hits >= CONV_HITS:
                    prev_Gens = Gens; break
            else:
                hits = 0
        prev_Gens, prev_best = Gens, best[0]
    if not all_G:
        return None
    Gens = boltz(all_G)
    therm = uma_gibbs_corr(pu, best[1], best[2], q)
    tag = "conv" if seed < CONV_MAX else "CAPPED"
    log(f"    {name:9s} q{q:+d} [implicit {tag} seeds={seed} last|ΔGens|={last_dG:.1f}]: "
        f"Gens {Gens:.1f} + thermal {therm:.1f} = {Gens+therm:.1f}")
    return Gens + therm


_WATER_REF = {}
def water_ref_G(pu, log=None):
    """Free energy of ONE liquid-water molecule in the SAME method, for the Bryantsev
    cluster-continuum monomer cycle. Subtracting n_water*this from a cluster G makes the
    explicit waters reference bulk liquid, so they cancel for a SPECTATOR anion (equal n
    both sides) AND stay correct for a CREATED/DESTROYED anion (unequal n). Without it,
    explicit_G leaks n*G(water) (~ -2e5 kJ each) into any reaction that changes anion count.
      G*_liq(H2O) = E_UMA(H2O) + thermal(H2O) + dGsolv(H2O) + RT ln(55.34)   [gas->liquid std state]"""
    if "G" in _WATER_REF:
        return _WATER_REF["G"]
    sym, coord = bare_geom(pu, 0, "O")
    atoms = Atoms(symbols=list(sym), positions=coord, info={"charge": 0, "spin": 1})
    E = float(batched_energies(pu, [atoms])[0]) * EV2KJ
    solv = xtb_dgsolv(list(sym), coord, 0, "cosmo")
    thermal = uma_gibbs_corr(pu, list(sym), coord, 0)
    conc = 8.314e-3 * 298.15 * float(np.log(55.34))          # +9.96 kJ/mol, gas 1M -> liquid 55.3M
    G = E + solv + thermal + conc
    _WATER_REF["G"] = G
    if log:
        log(f"    [water ref] G*_liq(H2O) = E {E:.1f} + solv {solv:.1f} + thermal {thermal:.1f} "
            f"+ conc {conc:.1f} = {G:.1f}")
    return G


def explicit_G(pu, q, smi, seeds, log, name):
    """Cluster-continuum G_aq: first-shell waters + cluster solvation, but thermal on
    the BARE SOLUTE only. The n explicit waters are referenced to bulk liquid via
    water_ref_G (Bryantsev monomer cycle) so the count need NOT cancel across the reaction.
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
    wref = n_water * water_ref_G(pu, log)                 # reference explicit waters to bulk liquid
    g = Gens + thermal - wref
    log(f"    {name:9s} q{q:+d} [explicit n={n_water} {sites} keep{len(Gt)}/{N_EXPLICIT_SEEDS}]: "
        f"Gens(E+solv) {Gens:.1f} + thermal(solute) {thermal:.1f} - {n_water}*Gwater {wref:.1f} = {g:.1f}")
    return g


def run_reaction(pu, key, seeds, keep, pool, log):
    rx = REACTIONS[key]
    log(f"\n=== {key}: {rx['note']}  (explicit={rx['explicit']}, n_H+={rx['n_Hplus']}) ===")
    # `explicit` may be True/False (whole reaction) OR a list/set of species names
    # that need explicit first-shell waters (per-species triage: only the anion that
    # is CREATED/DESTROYED, never a spectator phosphate).
    exp_flag = rx["explicit"]
    def requested_explicit(nm):
        if isinstance(exp_flag, (list, set, tuple)):
            return nm in exp_flag
        return bool(exp_flag)

    def is_spectator_anion(nm):
        """Explicit first-shell water is ONLY valid for a SPECTATOR anion -- one with a
        charge- and site-matched partner on the opposite side, so the waters (and the
        ~125 kJ anion-water binding) cancel in ΔG. For a CREATED/DESTROYED anion there is
        no partner and explicit leaks the binding (demonstrated: acetate explicit err
        -126; rxn01713 +166). Such species must use implicit or the pH-0 route instead."""
        coeff_n, q_n, smi_n = rx["species"][nm]
        if q_n >= 0:
            return True
        nwn = water_count(smi_n)[0]
        side_n = coeff_n > 0
        for other, (c, q, s) in rx["species"].items():
            if other == nm:
                continue
            if (c > 0) != side_n and q == q_n and water_count(s)[0] == nwn:
                return True
        return False

    G = {}
    for name, (coeff, q, smi) in rx["species"].items():
        if smi == "O" and q == 0:                        # liquid-water reactant (hydrolysis)
            G[name] = water_ref_G(pu, log)
            log(f"    {name:9s} q+0 [water ref liquid]: {G[name]:.1f}")
        elif requested_explicit(name):
            if is_spectator_anion(name):
                G[name] = explicit_G(pu, q, smi, seeds, log, name)
            else:                                        # GUARD: explicit invalid here
                log(f"    !! {name}: explicit REFUSED (created/destroyed anion, no "
                    f"cancellation partner) -> implicit. Use pH-0 (pka_sites) for accuracy.")
                G[name] = implicit_G(pu, q, smi, seeds, keep, pool, log, name)
        else:
            G[name] = implicit_G(pu, q, smi, seeds, keep, pool, log, name)
        if G[name] is None:
            log(f"    {name}: FAILED"); return None
    dG = sum(coeff * G[name] for name, (coeff, q, smi) in rx["species"].items())
    dG += rx["n_Hplus"] * G_HPLUS
    # pH-0 route: analytic pKa transform replaces the anion-solvation + explicit-proton terms
    for side, pka in rx.get("pka_sites", []):
        contrib = (1.0 if side == "react" else -1.0) * RT_LN10 * (PH - pka)
        dG += contrib
        log(f"    pKa transform [{side} pKa {pka}] += {contrib:+.1f} kJ/mol")
    errs = [dG - e for e in rx["exp"]]
    log(f"  ΔG = {dG:+.1f} kJ/mol   vs exp {rx['exp']}   err {[round(e,1) for e in errs]}")
    exp_out = sorted(exp_flag) if isinstance(exp_flag, (set, list, tuple)) else exp_flag
    return dict(reaction=key, dG=round(dG, 1), exp=rx["exp"],
                err=[round(e, 1) for e in errs], explicit=exp_out)


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
