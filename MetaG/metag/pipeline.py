#!/usr/bin/env python
"""MetaG scoring pipeline: score_reaction() — ONE scheme across all reaction classes.

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
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from metag.energetics import water_clusters as gc
from metag.energetics.uma import load_uma, batched_energies, batched_fire
from metag.energetics.conformers import pool_confs, boltz, spin_multiplicity
from metag.energetics.explicit_solvation import bare_geom
from metag.energetics.thermal import uma_gibbs_corr, xtb_dgsolv, xtb_dgsolv_relaxed, corr_fast
from metag.water_count import water_count, needs_explicit

N_EXPLICIT_SEEDS = int(os.environ.get("N_EXPLICIT_SEEDS", "16"))  # cluster seeds (cheap: batched relax)
EXPLICIT_KEEP = int(os.environ.get("EXPLICIT_KEEP", "8"))         # lowest-E clusters kept for Boltzmann

OUT = os.environ.get("METAG_OUT", os.path.join(os.getcwd(), "metag_out"))  # CLI output dir (configurable)
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
_RXN_JSON = os.environ.get("RXN_FILE", os.path.join(os.path.dirname(__file__), "reactions.json"))
def _load_reactions(path=_RXN_JSON):
    """Load a reactions JSON for the CLI harness. Optional: the score_reaction() API takes a reaction dict
    directly and needs no file, so a missing file is not an error (returns {})."""
    if not os.path.exists(path):
        return {}
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


# auto-convergent sampling knobs (env-overridable; NOT per-reaction tuning).
# BUDGET: adaptive by design (rigid species stop early); the cap bounds worst-case cost.
# TOL is set at the noise floor -- chasing sub-2 kJ convergence wastes compute for no
# accuracy gain (exp noise is ~2 kJ, method floor higher), so we DON'T.
CONV_TOL   = float(os.environ.get("CONV_TOL", "2.5"))    # kJ: Gens & min-E must move < this
CONV_HITS  = int(os.environ.get("CONV_HITS", "2"))       # for this many consecutive seed-batches
CONV_MAX   = int(os.environ.get("CONV_MAX", "8"))        # cap on seed-batches (budget guard)

# per-species QM cache: key must carry everything that changes the number (see species_cache.py).
from metag.energetics import species_cache as _sc
_MODEL = os.environ.get("UMA_MODEL", "uma-s-1p2p1")   # patch model (batched_relax._ensure_registered); in the cache key. UMA_MODEL overrides for A/B (e.g. uma-s-1p2)
_SAMPLE_SCALE = float(os.environ.get("SAMPLE_SCALE", "1"))
_IMPLICIT_SETTINGS = {"model": _MODEL, "solv": "cosmo", "budget": "nrot-tiered-v1",
                      "conv_tol": CONV_TOL, "conv_hits": CONV_HITS, "conv_max": CONV_MAX,
                      "sample_scale": _SAMPLE_SCALE}
_EXPLICIT_SETTINGS = {"model": _MODEL, "solv": "cosmo", "water": "count-v1",
                      "n_seeds": N_EXPLICIT_SEEDS, "keep": EXPLICIT_KEEP}


def implicit_G(pu, q, smi, seeds, keep, pool, log, name):
    """Boltzmann(E_elec[UMA] + ΔGsolv[cosmo]) over conformers + UMA thermal(min-E).

    HEURISTIC (general, self-calibrating -- no fixed per-flexibility tiers, no per-reaction
    tuning): keep adding conformer seed-batches until BOTH the Boltzmann Gens AND the minimum
    energy stop moving (< CONV_TOL for CONV_HITS consecutive batches), capped at CONV_MAX.
    Rigid species converge in ~2-3 batches; floppy sugar-phosphates draw as many as they need.
    Per-batch pool/keep still scale with rotatable bonds (bigger search for floppier molecules).
    Reports the seed count + the last increment so the sampling uncertainty is visible (UQ)."""
    mult = spin_multiplicity(smi, q)                      # ground-state spin (O2 triplet, radicals doublet)
    # spin is deterministic in (smi,q); fork the cache key ONLY for open-shell species so every
    # closed-shell singlet key is preserved (no cache bust) and pre-fix O2 singlet entries are ignored.
    _settings = _IMPLICIT_SETTINGS if mult == 1 else dict(_IMPLICIT_SETTINGS, spin=mult)
    _cached = _sc.get(smi, q, "implicit", _settings)
    if _cached is not None:
        log(f"    {name:9s} q{q:+d} [implicit CACHED]: {_cached[0]:.1f}")
        return _cached[0], _cached[1]
    if mult != 1:
        log(f"    {name:9s} q{q:+d} [open-shell: spin multiplicity {mult}]")
    _, keep, pool = sampling_budget(smi)                  # per-batch pool/keep sizing only
    all_G = []
    best = (1e18, None, None)
    prev_Gens = prev_best = None
    hits = 0
    seed = 0
    last_dG = float("nan")
    gens_traj = []
    while seed < CONV_MAX:
        seed += 1
        cands = pool_confs(smi, q, seed, pool, spin=mult)
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
        Gens = boltz(all_G); gens_traj.append(Gens)
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
        return None, None
    Gens = boltz(all_G)
    therm = uma_gibbs_corr(pu, best[1], best[2], q, spin=mult)
    # sampling uncertainty: spread of Gens over the last few batches (0 if never moved / capped-tight)
    tail = gens_traj[-3:]
    sigma = float(np.std(tail)) if len(tail) > 1 else (last_dG if np.isfinite(last_dG) else 3.0)
    tag = "conv" if seed < CONV_MAX else "CAPPED"
    log(f"    {name:9s} q{q:+d} [implicit {tag} seeds={seed} σ={sigma:.1f}]: "
        f"Gens {Gens:.1f} + thermal {therm:.1f} = {Gens+therm:.1f}")
    _sc.put(smi, q, "implicit", _settings, Gens + therm, sigma)
    return Gens + therm, sigma


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

    DEAD PATH IN THE DEFAULT CONFIG (2026-08): with PH0_AUTO + AUTO_TRUNCATE default-on, both branches
    set explicit=False, and the per-species triage REFUSES explicit for a created/destroyed anion and it
    is a no-op for a spectator (its G cancels with the partner regardless of method). This only runs as a
    gated fallback when pH-0 refuses on mass-balance. Its water-thermal bookkeeping is self-consistent
    ONLY because it is gated to spectators (waters cancel); do NOT repurpose it for a created/destroyed
    anion without first revisiting water_ref_G (the monomer-cycle reference would otherwise bias the
    unequal-water term). Kept for that fallback and for the explicit-water experiments, not the hot path.
    FIX 2: the floppy explicit-water librational modes make the full-cluster UMA
    finite-diff Hessian noisy and it does NOT cancel across the fixed-count reaction
    (this is what pinned the occupancy AND cost nucleotidyl ~20 kJ). So:
      G_aq = E_UMA(cluster)            # electronic, waters included → cancel (balanced)
           + ΔGsolv(cluster, xtb --sp) # cluster-continuum bulk solvation
           + thermal(BARE solute, UMA) # NO floppy water modes; cancels across reaction
    This also UNIFIES thermal with the implicit path (always bare-solute UMA Hessian)."""
    _cached = _sc.get(smi, q, "explicit", _EXPLICIT_SETTINGS)
    if _cached is not None:
        log(f"    {name:9s} q{q:+d} [explicit CACHED]: {_cached[0]:.1f}")
        return _cached[0], _cached[1]
    bsym, bcoord = bare_geom(pu, q, smi)
    n_water, sites = water_count(smi)
    # generous cluster sampling (cheap: batched relax) — floppy water-decorated clusters.
    # DETERMINISTIC seed: built-in hash() is salted per process (PYTHONHASHSEED) -> non-reproducible
    # cluster geometries across runs. Seed from a stable content hash of everything that defines the
    # cluster (smi, q, n_water), matching the content-addressed cache key so a cache hit and a fresh
    # compute use the SAME seed.
    _seed = int(hashlib.md5(f"{smi}|{q}|{n_water}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(_seed)
    clusters = [Atoms(symbols=cs, positions=cc, info={"charge": int(q), "spin": 1})
                for cs, cc in (gc.seed_waters(bsym, bcoord, n_water, rng) for _ in range(N_EXPLICIT_SEEDS))]
    rel, E, conv = batched_fire(pu, clusters, fmax=0.06, steps=350, stop_frac=0.8,
                                return_converged=True, label=f"{name}w{n_water}")
    rel = [a for a, c in zip(rel, conv) if c]; E = E[conv] * EV2KJ
    if not len(E):
        return None, None
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
        return None, None
    Gens = boltz(Gt)
    thermal = uma_gibbs_corr(pu, bsym, bcoord, q)         # bare solute, no water modes
    wref = n_water * water_ref_G(pu, log)                 # reference explicit waters to bulk liquid
    g = Gens + thermal - wref
    # explicit clusters are noisier than implicit (stochastic seeding; reverse-test showed
    # ~14 kJ run-to-run) -> conservative sampling-σ floor from the kept-cluster spread.
    sigma = max(float(np.std(Gt)) if len(Gt) > 1 else 7.0, 5.0)
    log(f"    {name:9s} q{q:+d} [explicit n={n_water} {sites} keep{len(Gt)}/{N_EXPLICIT_SEEDS} σ={sigma:.1f}]: "
        f"Gens(E+solv) {Gens:.1f} + thermal(solute) {thermal:.1f} - {n_water}*Gwater {wref:.1f} = {g:.1f}")
    _sc.put(smi, q, "explicit", _EXPLICIT_SETTINGS, g, sigma)
    return g, sigma


def _flag(name, default=False):
    """Env flag with a default. Coherent-router gates that are self-gating (gated >= baseline) are
    DEFAULT-ON for production; set NAME=0 (or off/false/no) to disable for an ablation/baseline run."""
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("", "0", "off", "false", "no")


def score_reaction(pu, reaction, seeds=(1, 2), keep=10, pool=48, log=print, allow_truncate=True, key="rxn"):
    """Public entry point: score one reaction's standard transformed Gibbs energy (ΔrG'°) from structure.

    `reaction` = {"species": {name: [coeff, charge, smiles]}, "note": str, "n_Hplus": int,
    and optionally "exp": [float] (experiment, for err reporting), "explicit": bool|list, "pka_sites"}.
    `pu` is a loaded UMA model (metag.energetics.uma.load_uma). Returns the result dict:
    dG (physics+anchor), dG_raw (pure physics), anchor, sigma_pred, ci95, ci_center, U_samp, ...
    """
    orig_species = {k: list(v) for k, v in reaction["species"].items()}   # pre-routing, for the anchor gate
    rx = dict(reaction)
    rx.setdefault("explicit", False)                        # `explicit` is optional per the contract;
    # routing paths (truncate/pH-0) set it, but a non-routing reaction (e.g. neutral oxygenase) must
    # still default it -- line 449 reads rx["explicit"] unconditionally. (list -> set for triage below.)
    if isinstance(rx["explicit"], list):
        rx["explicit"] = set(rx["explicit"])
    truncated = False                                        # did AUTO_TRUNCATE actually fire?
    # COFACTOR RING-TRUNCATION (opt-in COFACTOR_RING=1): replace NAD(P)+/NAD(P)H with their
    # redox-active nicotinamide RING model. The identical ADP-ribose-phosphate tail cancels in
    # ΔG but its floppy-conformer error does NOT in full-molecule QM -- which is why NAD (floppy)
    # carries a spurious ~-49 kJ bias while NADP (rigid) does not. Ring model removes the noise +
    # collapses NAD/NADP to one model. Isodesmic, experiment-free. Runs BEFORE truncation so the
    # small ring survives. Validated on TECRDB redox: NAD MAE 44.6->10.4 (n=4). Gated to a genuine
    # ox/red couple (never mis-fires on NAD biosynthesis). n_H+ preserved (Δq,ΔH of the couple kept).
    if _flag("COFACTOR_RING", default=True):
        try:
            from metag.routing.cofactor_cores import cofactor_ring
            new = cofactor_ring(rx["species"])
            if new is not rx["species"]:
                # Coring changes species CHARGES; n_H+ must move by the negative of the charge change
                # to keep charge closure (Σcoeff·q + n_H+ = 0). This is a NO-OP for symmetric
                # NAD(P)+/NADH swaps (the +2/-2 shifts cancel) but CORRECTS asymmetric double-redox
                # (2 GSH : 1 GSSG) where the old "n_H+ preserved" left a -2 imbalance -> ~+2340 kJ
                # proton leak (glutathione reductase rxn00070/00086 -> +2363; fixed -> ~+22, exp ~+15).
                dq = (sum(c * q for (c, q, s) in new.values())
                      - sum(c * q for (c, q, s) in rx["species"].values()))
                rx = dict(rx, species=new, n_Hplus=rx["n_Hplus"] - dq,
                          note=rx.get("note", "") + " [RINGCOFACTOR]")
                log(f"  [cofactor-ring: NAD(P) -> nicotinamide ring model"
                    + (f"; n_H+ {rx['n_Hplus']+dq:+d}->{rx['n_Hplus']:+d} (charge closure)]" if dq else "]"))
        except Exception as e:
            log(f"  [cofactor-ring error: {e}; unchanged]")
    # CoA CORE-REDUCTION (opt-in COA_CORE=1): DEFAULT-OFF -- proven a no-op on the benchmark.
    # HYPOTHESIS (rejected): the floppy pantetheine-ADP tail's conformer noise doesn't cancel in ΔG,
    # so capping S-CoA -> S-CH3 would denoise it. VERDICT: all 28 TECRDB CoA reactions are SYMMETRIC
    # (CoA on both sides), so the scaffold cancels EXACTLY regardless of truncation; the residual
    # sampling noise is <=1 kJ and net slightly harmful (0/5 matched pairs improved; see
    # tools/collect_fix.py coa_core + the symmetry proof). CoA's +23.5 MAE is ELECTRONIC (thioester
    # C(=O)-S reference), a CBH/DLPNO target like hydratase -- NOT a sampling problem. Kept behind the
    # flag (self-gating, harmless) for the record; do not re-enable without an ASYMMETRIC CoA reaction.
    if _flag("COA_CORE", default=False):
        try:
            from metag.routing.coa_core import coa_core
            new = coa_core(rx["species"])
            if new is not rx["species"]:
                rx = dict(rx, species=new, note=rx.get("note", "") + " [COA-CORE]")
                log("  [coa-core: acyl-S-CoA -> acyl-S-CH3 (pantetheine-ADP scaffold capped)]")
        except Exception as e:
            log(f"  [coa-core error: {e}; unchanged]")
    # NTP CORE-REDUCTION (opt-in NTP_CORE=1): the phosphoryl-transfer analogue. On ATP/ADP/AMP-type
    # species the adenosine-ribose rides along UNCHANGED (ATP->ADP keeps the whole nucleoside; only the
    # gamma-phosphate moves) -> it cancels in ΔG but its huge floppy tail does NOT in full-molecule QM
    # (phosphagen kinases run ATP/ADP conformer-CAPPED, sigma~0 -> +44..+77 err; generic MCS truncation
    # REFUSES the phosphoryl->guanidinium cut). Cap the nucleoside-5'-O with methyl, keep the reactive
    # polyphosphate (isodesmic, experiment-free). SELF-GATING on mass+charge balance. Runs AFTER
    # COA_CORE (which strips CoA's own adenosine first) and BEFORE truncation so the small core survives.
    # DEFAULT-OFF pending kinase validation: NTP-core was NO-GO on phosphagens (isolated the P-N
    # error, didn't fix it -> it's electronic, goes to DLPNO; and the methyl-cap even lost fortuitous
    # cancellation, +9 worse). Must prove it HELPS the kinase class before default-on.
    if _flag("NTP_CORE", default=False):
        try:
            from metag.routing.ntp_core import ntp_core
            new = ntp_core(rx["species"])
            if new is not rx["species"]:
                rx = dict(rx, species=new, note=rx.get("note", "") + " [NTP-CORE]")
                log("  [ntp-core: nucleoside-5'-phosphate -> methyl polyphosphate (adenosine capped)]")
        except Exception as e:
            log(f"  [ntp-core error: {e}; unchanged]")
    # AUTO-TRUNCATION (general heuristic, opt-in AUTO_TRUNCATE=1): replace the reaction with
    # its truncated reactive core (removes conserved backbone -> kills catastrophic cancellation
    # + its conformer noise). Falls back to full molecules if no clean balanced truncation.
    # PHYSICS ROUTING GATE (ROUTE_FULL, default-on): prefer the FULL molecule over truncation when the
    # reaction is compact/ring-embedded, charge-conserved, and has no floppy large-fragment linker
    # (route_full.prefer_full). Validated +1.05 kJ MAE vs baseline on the ring-fix candidate set; every
    # large truncation-regression (disaccharide/bisphosphate/SAH/G6P) is kept out of full by the
    # floppy-linker term. Set ROUTE_FULL=0 to ablate.
    _prefer_full = False
    if allow_truncate and _flag("AUTO_TRUNCATE", default=True) and _flag("ROUTE_FULL", default=True):
        try:
            from metag.routing.truncation_gate import prefer_full
            _prefer_full = prefer_full(rx["species"])
        except Exception:
            _prefer_full = False
        if _prefer_full:
            log("  [route: prefer FULL molecule (compact ring, charge-conserved, no floppy linker) -> skip truncation]")
    if allow_truncate and _flag("AUTO_TRUNCATE", default=True) and not _prefer_full:
        try:
            from metag.routing.truncate import build_truncated_reaction
            _rad = int(os.environ.get("TRUNC_RADIUS", "2"))
            tr = build_truncated_reaction(rx["species"], radius=_rad)
            if tr is None and os.environ.get("TRUNC_V2"):   # v2: global-map truncation for the
                from metag.routing.truncate_global import build_truncated_reaction_v2   # multi-coeff/unequal-side cases
                tr = build_truncated_reaction_v2(rx["species"], radius=_rad)
                if tr is not None:
                    log("  [v2 global-map truncation engaged]")
            # VALIDITY GUARD (general): a truncation removes CONSERVED spectator parts and caps cut C-C
            # bonds FAR from functional groups. It must NEVER cut a bond ON a CATIONIC center: capping a
            # quaternary ammonium [N+](C)(C)(C) with H yields a PRIMARY ammonium CH2-NH3+ -- a different
            # molecule (carnitine betaine -> primary amine; silently wrong ΔG, the failed 3-dehydrocarnitine
            # core just exposed it). GENERAL symptom for ANY cation (N+, S+, guanidinium, sulfonium...):
            # a positively-charged atom in the core carries MORE H than the same atom in the full species.
            # Match the truncation's 'X_t' core name back to the original 'X'. (Root-cause fix would live in
            # truncate.py's cut-site selection; this is the safety guard + the species-failure retry below.)
            def _truncation_mangles_cation(orig, cored):
                # symptom = a positively-charged atom LOST heavy-atom coordination vs the full species
                # (quaternary N+ degree 4 -> bare [N+] degree 1 = an undervalent, chemically-invalid
                # cation). Element-agnostic (N+, S+, guanidinium C...); compares the max cation heavy-degree.
                from rdkit import Chem as _C
                def cation_deg(smi):
                    m = _C.MolFromSmiles(smi)
                    return sorted(a.GetDegree() for a in m.GetAtoms() if a.GetFormalCharge() > 0) if m else []
                for nm, (c, q, s) in cored.items():
                    onm = nm[:-2] if nm.endswith("_t") else nm
                    if onm not in orig:
                        continue
                    dc, do = cation_deg(s), cation_deg(orig[onm][2])
                    if do and (not dc or max(dc) < max(do)):    # a retained cation LOST coordination -> mangled
                        return nm
                return None
            if tr is not None:
                bad = _truncation_mangles_cation(rx["species"], tr[0])
                if bad:
                    log(f"  [auto-truncation REJECTED: mangles a cationic center ({bad}) -> full molecules]")
                    tr = None
            if tr is not None:
                rx = dict(rx, species=tr[0], n_Hplus=tr[1], explicit=False,
                          note=rx.get("note", "") + " [AUTO-TRUNCATED]")
                truncated = True
                log(f"  [auto-truncated -> {len(tr[0])} core species, n_H+={tr[1]}]")
            else:
                log(f"  [auto-truncation fallback: full molecules]")
        except Exception as e:
            log(f"  [auto-truncation error: {e}; full molecules]")
    # pH-0 / pKa AUTO-ROUTING (general heuristic, opt-in PH0_AUTO=1): for the charged-anion class
    # (phosphoryl/NTP/PPi/carboxylate) protonate every anionic site to its NEUTRAL microspecies
    # -- UMA's comfortable regime (no formal charge to delocalise, no diffuse-anion basis ceiling,
    # continuum-solvation valid) -- then bridge to pH7 analytically with textbook pKa's. Runs AFTER
    # truncation so it neutralises the small cores. Falls back (no-op) when no anionic site exists
    # (thioester/glycosyl-anomeric neutral classes) or on any parse failure. Not fitted to the DB.
    if _flag("PH0_AUTO", default=True) and not rx.get("pka_sites"):
        try:
            from metag.routing.pka_transform import build_ph0_reaction, is_isomerization
            if is_isomerization(rx["species"]):
                # ISOMERASE GATE: pH-0 hurts isomerizations (no anion-solvation change to
                # fix; neutralising spectator anions only injects sampling noise). Skip.
                log("  [pH0-auto: isomerization -> gated OFF (pH-0 would only add noise)]")
            else:
                # UNIFIED build: anions ALWAYS neutralized; basic amines additionally when the
                # internal base gate passes (PH0_BASES, default-on) -> covers the net-proton
                # deamination/transaminase/lyase class the old anion-only path refused.
                out = build_ph0_reaction(rx["species"], rx["n_Hplus"],
                                         base=_flag("PH0_BASES", default=True))
                if out is not None:
                    ns, pks, nh = out
                    rx = dict(rx, species=ns, n_Hplus=nh, pka_sites=pks, explicit=False,
                              note=rx.get("note", "") + " [pH0]")
                    log(f"  [pH0 -> {len(pks)} pKa sites, n_H+={nh}]")
                else:
                    log(f"  [pH0: no ionizable site -> unchanged]")
        except Exception as e:
            log(f"  [pH0-auto error: {e}; unchanged]")
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
    sig = {}
    for name, (coeff, q, smi) in rx["species"].items():
        if smi == "O" and q == 0:                        # liquid-water reactant (hydrolysis)
            G[name] = water_ref_G(pu, log); sig[name] = 1.0
            log(f"    {name:9s} q+0 [water ref liquid]: {G[name]:.1f}")
        elif requested_explicit(name):
            if is_spectator_anion(name):
                G[name], sig[name] = explicit_G(pu, q, smi, seeds, log, name)
            else:                                        # GUARD: explicit invalid here
                log(f"    !! {name}: explicit REFUSED (created/destroyed anion, no "
                    f"cancellation partner) -> implicit. Use pH-0 (pka_sites) for accuracy.")
                G[name], sig[name] = implicit_G(pu, q, smi, seeds, keep, pool, log, name)
        else:
            G[name], sig[name] = implicit_G(pu, q, smi, seeds, keep, pool, log, name)
        if G[name] is None:
            log(f"    {name}: FAILED")
            if truncated:                                    # a truncated core failed -> retry FULL molecules
                log(f"  [retry {key} with truncation OFF (full molecules)]")
                return run_reaction(pu, key, seeds, keep, pool, log, allow_truncate=False)
            return None
        try:                                                 # release per-species GPU memory so a
            import torch                                      # later big species doesn't OOM from
            if torch.cuda.is_available():                     # fragmentation left by earlier ones
                torch.cuda.empty_cache()
        except Exception:
            pass
    # ALDEHYDE HYDRATION (ALDEHYDE_HYDRATION, default-on, SELF-GATING): fold each hydratable aldehyde's
    # aqueous carbonyl<->gem-diol equilibrium into its effective G. The gem-diol is a +1-water microspecies
    # (parallel to protonation): G_eff = -RT ln[exp(-G_ald/RT)+exp(-(G_diol-G_water)/RT)]. Strongly hydrated
    # (glyoxylate) -> diol dominates -> G pulled down (fixed); weakly hydrated (GAP) -> unchanged (no harm).
    # No water is added to the stoichiometry (unit water activity, like H+ for protonation). See
    # scripts/aldehyde_hydration.py + memory aldehyde-hydration-signal.
    if _flag("ALDEHYDE_HYDRATION", default=True):
        from metag.routing import aldehyde_hydration as _ah
        _gw = None
        for name, (coeff, q, smi) in list(rx["species"].items()):
            if G.get(name) is None:
                continue
            diol = _ah.gem_diol(smi)
            if diol is None:
                continue
            if _gw is None:
                _gw = water_ref_G(pu)
            Gd, _sd = implicit_G(pu, q, diol, seeds, keep, pool, log, name + "(gem-diol)")
            if Gd is None:
                continue
            Geff = _ah.mixture_G(G[name], Gd, _gw)
            log(f"    [hydration: {name} carbonyl {G[name]:.1f} + gem-diol {Gd:.1f} "
                f"(ΔG_hyd {Gd - _gw - G[name]:+.1f}) -> mixture {Geff:.1f}  shift {Geff - G[name]:+.1f}]")
            G[name] = Geff
            sig[name] = float(np.hypot(sig.get(name, 0.0), _sd or 0.0))

    dG = sum(coeff * G[name] for name, (coeff, q, smi) in rx["species"].items())
    dG += rx["n_Hplus"] * G_HPLUS
    # propagate per-species sampling σ to a reaction sampling-uncertainty (quadrature).
    U_samp = float(np.sqrt(sum((coeff * sig[name])**2
                               for name, (coeff, q, smi) in rx["species"].items())))
    # pH-0 route: analytic pKa transform replaces the anion-solvation + explicit-proton terms.
    # EXACT Alberty form -RT*ln(1+10^(pH-pKa)) per proton (correct near AND above pH~pKa; the
    # linear RT*ln10*(pH-pKa) form wrongly makes a high-pKa site e.g. Pi's 12.35 count -5 kJ).
    pka_total = 0.0
    for site in rx.get("pka_sites", []):
        side, pka = site[0], site[1]
        kind = site[2] if len(site) > 2 else "acid"
        # acid group (neutral=protonated HA): -RT ln(1+10^(pH-pKa)); base group (neutral=deprotonated
        # B, protonates below pKa): the mirror -RT ln(1+10^(pKa-pH)). Sign per side (react +, prod -).
        expo = (PH - pka) if kind == "acid" else (pka - PH)
        contrib = (1.0 if side == "react" else -1.0) * RT_LN10 * math.log10(1.0 + 10.0 ** expo)
        dG += contrib
        pka_total += contrib
    if rx.get("pka_sites"):
        log(f"    pKa transform (exact Alberty, {len(rx['pka_sites'])} protons) += {pka_total:+.1f} kJ/mol")
    # ROBUSTNESS GUARDS (defense-in-depth). A valid balanced ΔrG'° is physically bounded and the
    # computed reaction must conserve charge with its proton bookkeeping. Two independent failure modes
    # we have actually hit -> flag `suspect` so downstream analysis DROPS the point instead of averaging
    # garbage into the MAE (and it prints loudly in the log).
    suspect = None
    #  (1) charge must close: Σ coeff·q(final species) + n_H+ == 0. A broken transform (e.g. a cation
    #      left un-neutralised) leaks ~n·G(H+) ≈ ±1170 kJ per unbalanced proton. pH-0 already refuses
    #      these, but any other path that breaks it is caught here.
    q_imbalance = sum(coeff * q for name, (coeff, q, smi) in rx["species"].items()) + rx["n_Hplus"]
    if q_imbalance != 0:
        suspect = f"charge imbalance {q_imbalance:+d} (Σcoeff·q + n_H+ ≠ 0)"
        log(f"  !! SUSPECT: {suspect} -> ΔG leaks ~{q_imbalance}·G(H+); result unreliable, do not trust.")
    #  (2) magnitude sanity: |ΔrG'°| beyond a physical ceiling => unbound multi-anion, proton leak, or
    #      a loader/QM failure (we have seen -3.6e6 kJ from a stale loader). Bound is generous.
    _sanity = float(os.environ.get("DG_SANITY_KJ", "500"))
    if abs(dG) > _sanity:
        m = f"|ΔG|={abs(dG):.0f} > {_sanity:.0f} kJ (unphysical: proton leak / unbound anion / loader failure)"
        suspect = m if suspect is None else f"{suspect}; {m}"
        log(f"  !! SUSPECT: {m} -> flagged; exclude from statistics.")
    # ANCHOR CORRECTION (ANCHOR_CORRECT, default-on): the SYSTEMATIC bond-type / anion-pattern sub-classes
    # (phosphagen P-N; phosphatase monoester, PPi excluded; thioester acyl-CoA ligase) carry a class-wide,
    # SIGN-CONSISTENT offset -- a bond-type reference error / shared anion-solvation error -- that CANCELS
    # against a per-sub-class anchor pool (isodesmic referencing to measured members). Subtract the
    # calibrated offset and fold the intra-class residual into σ. Detected STRUCTURALLY on the ORIGINAL
    # reaction (pre-truncation/pH-0). Only the LOO-proven-systematic sub-classes fire; SCATTERED classes
    # (kinase phospho-ester, NAD, isomerase, Mg/NTP acceptor-specific) are never touched -- their error has
    # no common term (proven: bias~0, high σ). Skipped when the result is already flagged suspect.
    # Validated leave-anchors-out: phosphagen 66.7->9.5, phosphatase-monoester 13.8->5.0. See route_anchor.py.
    # EMPIRICAL per-class calibration: corrects the systematic anion-solvation-wall offset for the
    # LOO-proven-systematic sub-classes (phosphagen, phosphatase-monoester) against a small anchor pool.
    # It IS empirical (uses anchor reaction ΔrG'°) -- so anchors MUST be INDEPENDENT literature values,
    # NOT the scored benchmark, and results should report the pure-physics number alongside. Standard
    # per-class-calibration practice (Jinich/Alberty), physically justified, leave-anchors-out validated.
    dG_raw = dG                                       # pure-physics ΔG (pre-anchor) -- ALWAYS reported
    anchor_meta = None                                # so the empirical correction is transparent
    if _flag("ANCHOR_CORRECT", default=True) and suspect is None:
        try:
            from metag.routing.anchor import anchor_correct
            ac = anchor_correct(dG, orig_species)
            if ac is not None:
                dG_corr, sig_anchor, sc = ac
                log(f"  [anchor-correct: {sc} -> ΔG {dG:+.1f} -> {dG_corr:+.1f} (offset {dG-dG_corr:+.1f})]")
                dG = dG_corr
                # NOTE: the intra-class residual (sig_anchor) is NOT folded into U_samp here -- it is the
                # class-level predictive error, which sigma_pred below carries via the calibrated
                # sigma_class (avoids double-counting). U_samp stays the conformer-sampling spread only.
                anchor_meta = {"subclass": sc, "offset": round(dG_raw - dG_corr, 1), "sigma": sig_anchor}
        except Exception as e:
            log(f"  [anchor-correct error: {e}; uncorrected]")
    # CALIBRATED prediction uncertainty for downstream flux / TFA. U_samp (conformer-sampling spread,
    # ~1-3 kJ) is NOT the prediction interval -- the SYSTEMATIC method error (per mechanism class) dominates
    # (~5-25 kJ). sigma_pred = sqrt(U_samp^2 + sigma_class^2), class-conditional and calibrated on the
    # benchmark residual (tools/calibrate_uncertainty.py; 5-fold-CV coverage ~78%/94% at 1/2 sigma).
    # Reporting +-U_samp alone would make a TFA solver ~10x overconfident on the hard classes.
    try:
        from metag.uncertainty import reaction_sigma, prediction_interval
        _smis = [s[2] for s in rx["species"].values()]
        # Pass orig_species (pre-routing) so the σ-class is assigned STRUCTURALLY, coherent with the
        # anchor (which also gates on orig_species) and independent of the note. Falls back to the note
        # taxonomy for reactions that match no structural anchor class.
        sigma_pred, sigma_breakdown = reaction_sigma(rx["note"], _smis, U_samp, species=orig_species)
        # DE-BIASED asymmetric 95% interval for the TRUE ΔrG'° (for TFA: a symmetric ±σ is mis-centered
        # on a biased class). exp lies in [ci_lo, ci_hi]; ci_center is the bias-removed point estimate.
        ci_lo, ci_hi, ci_center, ci_info = prediction_interval(rx["note"], _smis, dG, level=95,
                                                               species=orig_species)
    except Exception as e:
        sigma_pred, sigma_breakdown = None, {"error": str(e)}
        ci_lo = ci_hi = ci_center = None; ci_info = {}
    errs = [dG - e for e in rx.get("exp", [])]              # exp is optional (deployment has no experiment)
    # RESOLUTION heuristic: if the CALIBRATED prediction interval is comparable to |ΔG|, the sign is not
    # resolvable -- flag it (near-equilibrium isomerases are concentration-limited, not QM-fixable).
    sig_for_flag = sigma_pred if sigma_pred is not None else U_samp
    unresolved = abs(dG) < sig_for_flag
    flag = "  [UNRESOLVED: |ΔG|<σ_pred]" if unresolved else ""
    log(f"  ΔG = {dG:+.1f} (raw {dG_raw:+.1f}) ± {sigma_pred} kJ/mol  "
        f"[σ_class {sigma_breakdown.get('sigma_class','?')} ({sigma_breakdown.get('class','?')}), U_samp {U_samp:.1f}]"
        f"   vs exp {rx.get('exp')}   err {[round(e,1) for e in errs]}{flag}")
    exp_out = sorted(exp_flag) if isinstance(exp_flag, (set, list, tuple)) else exp_flag
    return dict(reaction=key, dG=round(dG, 1), dG_raw=round(dG_raw, 1), anchor=anchor_meta,
                sigma_pred=sigma_pred, sigma_breakdown=sigma_breakdown, U_samp=round(U_samp, 1),
                ci95=[ci_lo, ci_hi], ci_center=ci_center, ci_info=ci_info,
                unresolved=unresolved, exp=rx.get("exp"), err=[round(e, 1) for e in errs],
                explicit=exp_out, suspect=suspect)


def run_reaction(pu, key, seeds, keep, pool, log, allow_truncate=True):
    """Harness wrapper: look a reaction up by key in the loaded REACTIONS file and score it."""
    return score_reaction(pu, REACTIONS[key], seeds, keep, pool, log, allow_truncate, key=key)


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
    pu = load_uma(_MODEL)          # single source of truth: loaded model == cache-key model (no drift)
    rows = [r for r in (run_reaction(pu, k, seeds, a.keep, a.pool, log) for k in keys) if r]

    log(f"\n==== UNIFIED PIPELINE — one scheme, three classes ====")
    log(f"  {'reaction':14s} {'ΔG':>7s} {'raw':>7s} {'±σpred':>7s} {'exp':>14s} {'err':>16s} {'note':>12s}")
    for r in rows:
        note = "UNRESOLVED" if r.get("unresolved") else ("explicit" if r["explicit"] else "implicit")
        log(f"  {r['reaction']:14s} {r['dG']:7.1f} {r.get('dG_raw', r['dG']):7.1f} "
            f"{str(r.get('sigma_pred','?')):>7s} {str(r['exp']):>14s} {str(r['err']):>16s} {note:>12s}")
    tag = a.only or "all"
    json.dump(rows, open(os.path.join(OUT, f"unified_pipeline_{tag}.json"), "w"), indent=2)
    log(f"wrote artifacts/unified_pipeline_{tag}.json")


if __name__ == "__main__":
    main()
