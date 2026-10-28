#!/usr/bin/env python
"""MetaG scoring pipeline: score_reaction() -- one scheme across all reaction classes.

Per reaction (each step self-gating; flags in parentheses; defaults as of 2026-10-01):
  1. structural routing on the input species: cofactor ring cores (COFACTOR_RING), nucleoside cap (NTP_CORE),
     CoA core (COA_CORE, off), spectator truncation (AUTO_TRUNCATE, ROUTE_FULL; C-C cuts only, TRUNC_FG_CUTS;
     radius 3 at anomeric centres, TRUNC_ANOMERIC_RADIUS), pH-0 neutral microspecies + pKa transform
     (PH0_AUTO, PH0_BASES; also for isomerizations, PH0_ISOMERASE), zwitterion guard (ZWITTERION_PH0).
  2. per species (content-addressed cache): ETKDG pool -> batched UMA rank -> gas relaxation -> unique
     minima; G = Boltzmann over minima of E_elec[UMA] + ΔG_solv[xtb-ALPB] + RRHO[UMA Hessian, per minimum]
     (THERMAL_ENSEMBLE); adaptive seed batches (CONV_*); + 1 atm -> 1 M standard state (STD_STATE_1M);
     liquid water from the experimental hydration free energy (WATER_REF_EXP).
  3. ΔG = Σ ν·G + n_H+·G(H+, pH 7) + Σ pKa-transform terms (pH-7 effective constants) + carbonyl hydration
     microstates (CARBONYL_HYDRATION_ALL, K_hyd-calibrated HYDRATION_CAL).
  4. guards (charge closure, |ΔG| sanity; fail closed) -> sampling and calibration diagnostics.
     Prediction intervals/sign classifications are withheld outside their calibrated configuration/domain.
     No fitted anchors (ANCHOR_CORRECT off) and no water patches.

CLI harness (reactions from RXN_FILE; launch through gpu_reserve, never set CUDA_VISIBLE_DEVICES by hand):
  gpu_reserve run <idx> -- python -m metag.pipeline --only <key>
"""
import functools
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ase import Atoms
from rdkit import Chem

from metag.chem import is_water, fmt_num
from metag.conditions import T, PH, CONDITIONS, RT_LN10
from metag.reactions import validate_reaction
from metag.energetics import water_clusters as gc
from metag.energetics.uma import load_uma, batched_energies, batched_fire
from metag.energetics.conformers import (pool_confs, boltz, spin_multiplicity, UniqueMinima,
                                         bond_graph, same_connectivity, same_stereochemistry)
from metag.energetics.explicit_solvation import bare_geom
import metag.energetics.thermal as thermal_mod
from metag.energetics.thermal import uma_gibbs_corr, xtb_dgsolv, xtb_dgsolv_relaxed, dgsolv
from metag.water_count import water_count
from metag.energetics.protonation import site_free_energy
from metag.energetics.provenance import producer_identity, source_identity, ordered_geometry_record
from metag.energetics.official_uma import trace_backend
from metag.assembly import assemble_reaction

N_EXPLICIT_SEEDS = int(os.environ.get("N_EXPLICIT_SEEDS", "16"))  # cluster seeds (cheap: batched relax)
EXPLICIT_KEEP = int(os.environ.get("EXPLICIT_KEEP", "8"))         # lowest-E clusters kept for Boltzmann
# xtb (1-threaded) solvation pool width. Hardcoding 8 oversubscribed the shared node: NG reaction
# shards x 8 threads could exceed the core count (e.g. 20x8=160 on 80 cores -> thrash). The sweep
# driver sets METAG_XTB_WORKERS = cores/NG so N shards never collectively exceed the cores; result is
# byte-identical (ex.map returns in index order regardless of width).
from metag.energetics.conformers import XTB_WORKERS

OUT = os.environ.get("METAG_OUT", os.path.join(os.getcwd(), "metag_out"))  # CLI output dir (configurable)
EV2KJ = 96.485
# CHE aqueous proton free energy at pH 7 (step3b/step6 convention)
G_HPLUS = -26.3 - 1104.5 - 2.303 * 8.314e-3 * T * 7.0    # ~ -1170.8 kJ/mol
# The thermodynamic conditions every MetaG number refers to (returned with each result and part of the
# configuration fingerprint). Anything compared against MetaG -- experiment or another estimator -- must be
# transformed to these conditions first; pMg=None = no Mg2+ binding (no Mg-complex microspecies).
# STANDARD STATE (package-wide convention: every solute at 1 M, liquid water at 55.34 M, H+ at pH 7).
# Species G = G_gas(RRHO, ASE IdealGasThermo at 1 atm) + ΔG_solv(xtb). xtb --cosmo reports ΔG_solv with
# NO reference-state shift (Gshift = 0; the `bar1M` keyword is a no-op for --cosmo -- verified), i.e. the
# 1 M(gas) -> 1 M(aq) Ben-Naim convention. The gas term is at 1 atm, so each solute needs
# +RT ln(RT/P°V°) = +RT ln 24.46 = +7.93 kJ to reach 1 M. Without it a reaction inherits -7.93·Δn
# (Δn = net number of solute+water molecules). G_HPLUS already includes it: -1104.5 kJ = -264.0 kcal/mol
# is the Tissandier 1 atm(gas) -> 1 M(aq) proton solvation. Same constant as
# water_clusters.GAS_1ATM_TO_1M_KJ. Default ON; STD_STATE_1M=0 restores the old (inconsistent) bookkeeping
# for A/B only. (xtb 6.7.1 prints "Reference state gsolv [1 M gas/solution]" for --cosmo/--alpb water.)
STD_STATE_KJ = 8.314e-3 * T * math.log(0.082057 * T)     # RT ln(24.46 L/mol) = 7.93 kJ/mol
# pH-0 route (Jinich/Alberty): compute the NEUTRAL protonated microspecies (well-solvated
# by continuum -> no created/destroyed-anion pathology, no huge G(H+) term), then bridge
# to pH 7 analytically with the EXPERIMENTAL pKa. `pka_sites` = list of (side, pKa) for
# each ionizable group that is deprotonated at pH7 but PROTONATED in the QM microspecies.
#   ΔG'(pH7) = ΔG_QM(neutral) + Σ sign·RT ln10·(pH - pKa),  sign=+1 reactant, -1 product.

# Reaction DEFINITIONS are DATA, deliberately kept OUT of this engine -> reactions.json
# (stoichiometry {species: [coeff(+prod/-react), charge, SMILES]}, exp ΔG, n_Hplus,
# explicit-water flag/list, optional pH-0 pka_sites). This file stays generic: sampling
# heuristics, solvation triage, thermal/electronic backends. Add reactions to the JSON.
_RXN_JSON = os.environ.get("RXN_FILE", os.path.join(os.path.dirname(__file__), "data", "reactions_opentecr_std.json"))
def _load_reactions(path=_RXN_JSON):
    """Compatibility alias for the backend-independent reaction loader."""
    from metag.reactions import load_reactions
    return load_reactions(path)


REACTIONS = _load_reactions()


def sampling_budget(smi):
    from metag.energetics.sampling import sampling_budget as budget
    return budget(smi, sample_scale=_SAMPLE_SCALE, keep_scale=_KEEP_SCALE)


# auto-convergent sampling knobs (env-overridable; NOT per-reaction tuning).
# BUDGET: adaptive by design (rigid species stop early); the cap bounds worst-case cost.
# These default plateau tolerances are heuristic, not a method-error floor or
# proof of complete sampling. The optional thorough search preset uses stricter
# criteria and larger pools; independent-seed checks still assess repeatability.
CONV_TOL   = float(os.environ.get("CONV_TOL", "2.5"))    # kJ: Gens & min-E must move < this
CONV_HITS  = int(os.environ.get("CONV_HITS", "2"))       # for this many consecutive seed-batches
CONV_MAX   = int(os.environ.get("CONV_MAX", "8"))        # cap on seed-batches (budget guard)

# per-species QM cache: key must carry everything that changes the number (see species_cache.py).
from metag.energetics import species_cache as _sc
from metag.energetics.thermal import qrrho_enabled, hessian_method
from metag.energetics.conformers import DEDUP_E_TOL
_MODEL = os.environ.get("UMA_MODEL", "uma-s-1p2p1")   # registered checkpoint; the SDK/runtime policy is separately fingerprinted
_SAMPLE_SCALE = float(os.environ.get("SAMPLE_SCALE", "1"))
_KEEP_SCALE = float(os.environ.get("KEEP_SCALE", "1"))   # A/B: scale `keep` (conformers relaxed+solvated)
# SOLV_MODEL: implicit solvation for species G (cosmo | alpb | cpcmx). SOLV_ALSO: extra models computed on
# the SAME conformers and cached alongside (for A/B without re-sampling), e.g. SOLV_ALSO=alpb,cpcmx.
# Default ALPB (2026-10-01): xtb --cosmo carries no hydrogen-bond term (Ghb = Gshift = 0) and under-solvates
# every polar group (FreeSolv per-group error: alcohol OH +17.8, COOH +18.8 kJ/mol vs ALPB +4.3/-4.9; carbonyl
# hydration log K MAE 3.6 vs 1.1). SOLV_MODEL=cosmo reproduces the earlier pipeline.
SOLV_MODEL = os.environ.get("SOLV_MODEL", "alpb").strip().lower()
SOLV_ALSO = list(dict.fromkeys(
    m.strip().lower() for m in os.environ.get("SOLV_ALSO", "").split(",")
    if m.strip() and m.strip().lower() != SOLV_MODEL
))
# PHYSICS_VERSION: bump whenever species-level physics changes in a way the settings below do not
# capture (sampling, relaxation, thermal, solvation code). It is part of every species-cache key, so a
# bump forces a clean recompute instead of silently serving numbers from older code.
#   2026-09-30b: float64 energies + dedup e_tol 1.5 kJ, post-relaxation connectivity check, xtb-failure
#                guard, projected external Hessian modes, imaginary-mode rejection, and RRHO
#                symmetry/linearity detection.
THERMAL_RELAX_STEPS = int(os.environ.get("THERMAL_RELAX_STEPS", "400"))
if THERMAL_RELAX_STEPS < 1:
    raise ValueError("THERMAL_RELAX_STEPS must be positive")
THERMAL_RETRY_STEPS = int(os.environ.get("THERMAL_RETRY_STEPS", "1200"))
PHYSICS_VERSION = "2026-10-24-official-uma-v1"   # official SDK semantics; Nitro-preferred optimization with FP32 finals
_IMPLICIT_SETTINGS = {"model": _MODEL, "solv": SOLV_MODEL, "budget": "nrot-tiered-v1",
                      "conv_tol": CONV_TOL, "conv_hits": CONV_HITS, "conv_max": CONV_MAX,
                      "sample_scale": _SAMPLE_SCALE, "physics": PHYSICS_VERSION, "qrrho": qrrho_enabled(),
                      **({"keep_scale": _KEEP_SCALE} if _KEEP_SCALE != 1 else {}),
                      # UMA_HESSIAN=autograd (exact Hessian) keys separately; default 'fd' keeps old keys valid
                      **({"hessian": hessian_method()} if hessian_method() != "fd" else {})}
# CONF_DEDUP (default-on): Boltzmann over unique minima. CONF_DEDUP=0 reproduces the legacy cumulative sum
# (every relaxed copy counted as a state) for A/B against old caches; the cache key tracks the choice.
_DEDUP = os.environ.get("CONF_DEDUP", "1").strip().lower() not in ("", "0", "off", "false", "no")
if _DEDUP:
    _IMPLICIT_SETTINGS["dedup"] = f"rmsd-v2-etol{DEDUP_E_TOL}"
# SOLV_RELAX (A/B, default off): after sampling, the lowest unique minima (within SOLV_RELAX_WIN kJ, at most
# SOLV_RELAX_N) are re-relaxed on the aqueous surface E_UMA + ΔG_solv(SOLV_MODEL) (metag/energetics/solv_relax.py)
# and the ensemble energy is the Boltzmann sum over those solution-phase minima. Thermal stays at the gas minimum.
_SOLV_RELAX = os.environ.get("SOLV_RELAX", "0").strip().lower() not in ("", "0", "off", "false", "no")
SOLV_RELAX_N = int(os.environ.get("SOLV_RELAX_N", "8"))
SOLV_RELAX_WIN = float(os.environ.get("SOLV_RELAX_WIN", "25"))
SOLV_RELAX_STEPS = int(os.environ.get("SOLV_RELAX_STEPS", "200"))
if _SOLV_RELAX:
    if SOLV_RELAX_N < 1 or SOLV_RELAX_STEPS < 1 or not math.isfinite(SOLV_RELAX_WIN) or SOLV_RELAX_WIN <= 0:
        raise ValueError("SOLV_RELAX requires positive candidate/step budgets and a finite positive window")
    if SOLV_MODEL not in {"alpb", "cosmo", "gbsa"}:
        raise ValueError(f"SOLV_RELAX has no implemented gradient for {SOLV_MODEL!r}")
    _IMPLICIT_SETTINGS["solv_relax"] = f"complete-gradient-v2-gas-basin-rrho-n{SOLV_RELAX_N}-w{SOLV_RELAX_WIN:g}-s{SOLV_RELAX_STEPS}"
# ACID_HB_FILTER (diagnostic, default off): for a NEUTRAL species with >= 2 acid groups (each P, each carboxyl
# C) -- i.e. the protonated pH-0 reference of a polyanion -- conformers in which an acidic O-H of one
# group H-bonds an O of ANOTHER acid group are excluded from the ensemble. Those H-bonds cannot exist at
# pH 7 (both groups ionised and mutually repulsive), and the pKa table that maps the neutral reference to
# pH 7 is for non-interacting groups; keeping them over-stabilises the neutral reference (FBP + fructose
# -> F6P + F1P = +18.6 kJ in UMA/ALPB vs ~0 implied by FBP's near-additive pKa's).
# DIAGNOSTIC, DEFAULT OFF (review 2026-10-01): a hard conformer filter removes physically accessible
# conformers of the neutral reference state and is discontinuous (all conformers restored when every one
# has the contact); its TECRDB gain was ~0.1 kJ and its only independent support is one isodesmic cycle
# (FBP + fructose -> F6P + F1P, +18.6 kJ). The FBP artefact is reported as a known limitation instead.
_ACID_HB_FILTER = os.environ.get("ACID_HB_FILTER", "0").strip().lower() not in ("", "0", "off", "false", "no")
ACID_HB_DIST = float(os.environ.get("ACID_HB_DIST", "2.2"))           # H...O (Å)
# THERMAL_ENSEMBLE (DEFAULT ON since review 2026-10-01): RRHO correction for EVERY thermally relevant unique
# minimum (within THERMAL_ENS_WIN kJ of the lowest E+ΔGsolv, at most THERMAL_ENS_N), Boltzmann over
# G_i = E_i + ΔGsolv_i + Gcorr_i. THERMAL_ENSEMBLE=0 = the earlier estimator: Boltzmann over E_i + ΔGsolv_i,
# then ONE RRHO correction from the lowest-G true minimum. Measured on TECRDB: per-reaction shift -0.9 ± 2.2 kJ
# (up to 8 kJ for floppy polyols / PRT; GcorR spread across minima median 3.2 kJ, up to 13), aggregate MAE
# unchanged -> adopted because it is the correct estimator, not for accuracy.
_THERMAL_ENSEMBLE = os.environ.get("THERMAL_ENSEMBLE", "1").strip().lower() not in ("", "0", "off", "false", "no")
THERMAL_ENS_WIN = float(os.environ.get("THERMAL_ENS_WIN", "15"))
THERMAL_ENS_N = int(os.environ.get("THERMAL_ENS_N", "10"))
THERMAL_VALIDATION_MAX = int(os.environ.get("THERMAL_VALIDATION_MAX", "64"))
if _THERMAL_ENSEMBLE:
    _IMPLICIT_SETTINGS["thermal_ensemble"] = f"v3-w{THERMAL_ENS_WIN:g}-n{THERMAL_ENS_N}"   # v3: validated RRHO on the accepted minimum   # v2: RRHO inside convergence + U_samp
_ACID_HB_KEY = f"v2-{ACID_HB_DIST:g}"   # v2: a P-O-P chain is one acid group                    # cache-key tag, added only where the filter can act
def _explicit_settings():
    """Cache key of an explicit cluster. Built at CALL time from the EFFECTIVE values: the cached cluster G
    contains n*water_ref_G, which depends on the solvation model, the WATER_REF_EXP flag (read through
    _flag, i.e. including its FLAG_DEFAULTS default) and the experimental ΔG_hyd(H2O). Reading the raw
    environment variable here (as before 2026-10-01) gave the default run and the WATER_REF_EXP=0
    ablation the same key."""
    return {"model": _MODEL, "solv": "cosmo", "water": "count-v1",
            "n_seeds": N_EXPLICIT_SEEDS, "keep": EXPLICIT_KEEP, "dedup": "erot-v1",   # clusters: moments fallback
            "physics": PHYSICS_VERSION, "qrrho": qrrho_enabled(),
            "water_solv": SOLV_MODEL, "water_ref_exp": bool(_flag("WATER_REF_EXP")),
            "water_dgsolv_kj": float(os.environ.get("WATER_DGSOLV_KJ", "-26.4")),
            **({"hessian": hessian_method()} if hessian_method() != "fd" else {})}
# Fraction of a species' relaxed conformers whose xtb solvation may fail (timeout / non-zero exit) before
# its G is considered degraded: a degraded G is still used for this reaction but NOT cached, so a
# transient overload during a sweep cannot poison every later reaction that reuses the species.
MAX_XTB_FAIL_FRAC = 0.25
# SOLVATION SUPPORT: every relaxed conformer within this gas-phase energy window of the lowest one must have a
# solvation value. Failures are not random (compact / highly charged conformers time out more), so an ensemble
# that silently drops low-energy members is biased; such a species FAILS instead (not cached). 30 kJ (~12 RT)
# also covers conformers whose solvation differs strongly from the gas-phase ordering.
SOLV_SUPPORT_KJ = 30.0
# thermal term: lowest-G unique minima tried in order until one is a true minimum (no imaginary mode above
# thermal.IMAG_TOL_CM, after one tighter re-optimisation); none -> the species fails.
N_THERMAL_CANDIDATES = 3


from metag.energetics.errors import SpeciesRearranged


def _model_name(pu):
    """The UMA model actually loaded (load_uma tags the predict unit). Refuses a mismatch with UMA_MODEL,
    which keys the species cache: scoring with one model under another model's key poisons the cache."""
    name = getattr(pu, "metag_model", None)
    if name is not None and name != _MODEL:
        raise ValueError(f"loaded UMA model {name!r} != UMA_MODEL {_MODEL!r} (the species-cache key); "
                         f"set UMA_MODEL={name} or load_uma({_MODEL!r})")
    return name or _MODEL


def _producer_identity(pu):
    extra = {"xtbcpx": thermal_mod.XTBCPX} if "cpcmx" in [SOLV_MODEL, *SOLV_ALSO] else {}
    return producer_identity(pu, thermal_mod.XTB, additional_executables=extra)


def _last_match(uniq, atoms, E_kJ):
    """Index of the unique minimum that `atoms` duplicates (same test UniqueMinima.add used), or None."""
    rep = uniq._mol_at(atoms) if uniq.template is not None else None
    for j in range(len(uniq.E)):
        if rep is not None and uniq._same(E_kJ, rep, j):
            return j
    return None


def implicit_G(pu, q, smi, seeds, keep, pool, log, name, warnings=None, diagnostics=None):
    """Boltzmann over unique minima of G_i = E_elec[UMA]_i + ΔGsolv[SOLV_MODEL]_i + Gcorr[UMA RRHO]_i
    (THERMAL_ENSEMBLE, default; every sampled minimum validated before final assembly). With THERMAL_ENSEMBLE=0 only
    E + ΔGsolv is averaged and one RRHO correction (lowest-G true minimum) is added.

    Plateau heuristic: keep adding conformer seed-batches until BOTH the Boltzmann Gens AND the minimum
    energy stop moving (< CONV_TOL for CONV_HITS consecutive batches), capped at CONV_MAX.
    Meeting this criterion does not establish that all relevant basins were sampled.
    Per-batch pool/keep still scale with rotatable bonds (bigger search for floppier molecules).
    Reports the seed count + the last increment so the sampling uncertainty is visible (UQ).

    Species-level problems (bonding changed during relaxation, xtb failures) are appended to
    `warnings` and stored with the cached value, so a cache hit reports them too.
    Optional `diagnostics` receives sampling and thermal metadata; historical caches
    without a trajectory explicitly report its unavailability."""
    # Cache identity and the finite conformer search must use the same atom
    # ordering. Otherwise the first spelling encountered can choose the result
    # subsequently served for every equivalent SMILES under one cache key.
    smi = _sc.canonical(smi)
    from metag.energetics.sampling import sampling_diagnostics, species_diagnostics
    if diagnostics is not None:
        diagnostics.update(method="implicit", source="fresh", sampling={"status": "incomplete"})
    warnings = warnings if warnings is not None else []
    if _SOLV_RELAX and (not _THERMAL_ENSEMBLE or not _DEDUP):
        message = f"{name}: SOLV_RELAX requires per-basin gas thermal validation and conformer deduplication"
        warnings.append(message)
        if diagnostics is not None:
            diagnostics["solv_relax"] = {"status": "unsupported_configuration", "reason": message}
        return None, None
    mult = spin_multiplicity(smi, q)                      # ground-state spin (O2 triplet, radicals doublet)
    # spin is deterministic in (smi,q); fork the cache key ONLY for open-shell species so every
    # closed-shell singlet key is preserved (no cache bust) and pre-fix O2 singlet entries are ignored.
    seed_offset = int(os.environ.get("CONFORMER_SEED_OFFSET", "0"))
    if seed_offset < 0:
        raise ValueError("CONFORMER_SEED_OFFSET must be nonnegative")
    _settings = dict(_IMPLICIT_SETTINGS, seed_offset=seed_offset, model=_model_name(pu),
                     thermal_validation_max=THERMAL_VALIDATION_MAX, thermal_retry_steps=THERMAL_RETRY_STEPS,
                     producer=_producer_identity(pu))
    if THERMAL_RELAX_STEPS != 400:
        _settings["thermal_relax_steps"] = THERMAL_RELAX_STEPS
    if mult != 1:
        _settings["spin"] = mult
    template = Chem.AddHs(Chem.MolFromSmiles(smi))
    hb_groups = _acid_groups(template) if (_ACID_HB_FILTER and q == 0) else None
    if hb_groups is not None and len(set(hb_groups[0].values())) < 2:
        hb_groups = None                                  # fewer than two acid groups: nothing to filter
    if hb_groups is not None:
        _settings["acid_hb_filter"] = _ACID_HB_KEY
    _cached = _sc.get(smi, q, "implicit", _settings, with_meta=True)
    if _cached is not None:
        G_c, s_c, meta_c = _cached
        if diagnostics is not None:
            diagnostics.update(species_diagnostics(meta_c, "cache"))
        warnings.extend(meta_c.get("warnings", []))
        log(f"    {name:9s} q{q:+d} [implicit CACHED]: {G_c:.1f}")
        return G_c, s_c
    if mult != 1:
        log(f"    {name:9s} q{q:+d} [open-shell: spin multiplicity {mult}]")
    _, keep, pool = sampling_budget(smi)                  # per-batch pool/keep sizing only
    ref_graph = bond_graph(template)
    uniq = (UniqueMinima(template=template) if _DEDUP
            else UniqueMinima(e_tol=-1.0))                 # e_tol<0: never merge (legacy A/B)
    all_G = uniq.G
    # CONNECTIVITY GUARD: a conformer whose bonding changed during gas-phase relaxation (a zwitterion's
    # N-H proton moving to its O-, an acid proton hopping) is a different species from the SMILES being
    # scored. Such conformers are set aside; if EVERY conformer rearranged the species is not a gas-phase
    # minimum and SpeciesRearranged is raised (never scored or cached under this SMILES).
    rearranged = []                                       # (atoms, e, solv-dict) set aside
    hb_set_aside = []                                     # (atoms, e, sd) with an inter-acid H-bond
    thermal_track = (_ThermalTrack(pu, template, q, mult)
                     if _THERMAL_ENSEMBLE and _DEDUP else None)
    n_relaxed = n_xtb_fail = 0
    failed_E = []                                         # gas energies of conformers whose solvation failed
    # secondary solvation models on the same unique minima: {model: [G per unique minimum]}
    also = {m: [] for m in SOLV_ALSO}
    best = (1e18, None, None)
    prev_Gens = prev_best = None
    hits = 0
    seed = 0
    last_dG = float("nan")
    gens_traj = []
    successful_seeds, minimum_traj = [], []
    while seed < CONV_MAX:
        seed += 1
        cands = pool_confs(smi, q, seed + seed_offset, pool, spin=mult)
        order = np.argsort(batched_energies(pu, cands))[:keep]
        sel = [cands[i] for i in order]
        rel, E, conv = batched_fire(pu, sel, fmax=0.05, steps=300, stop_frac=0.9,
                                    return_converged=True, label=f"{name}s{seed}")
        sel = [a for a, c in zip(rel, conv) if c]; Eg = E[conv] * EV2KJ
        models = [SOLV_MODEL] + SOLV_ALSO
        with ThreadPoolExecutor(max_workers=XTB_WORKERS) as ex:
            solv = list(ex.map(lambda am: dgsolv(am[0].get_chemical_symbols(), am[0].get_positions(), q,
                                                 am[1], mult), [(a, m) for a in sel for m in models]))
        solv = [dict(zip(models, solv[i * len(models):(i + 1) * len(models)])) for i in range(len(sel))]
        for a, sd in zip(sel, solv):                      # one retry: most xtb failures are transient
            # Nonfinite energies are failed solvent evaluations, never populations.
            # Normalize auxiliary failures too so their coverage remains explicit.
            for model, value in sd.items():
                if value is not None and not np.isfinite(value):
                    sd[model] = None
            if sd[SOLV_MODEL] is None:
                retry = dgsolv(a.get_chemical_symbols(), a.get_positions(), q, SOLV_MODEL, mult)
                sd[SOLV_MODEL] = retry if retry is not None and np.isfinite(retry) else None
        eligible_in_batch = 0
        for a, e, sd in zip(sel, Eg, solv):
            n_relaxed += 1
            s = sd[SOLV_MODEL]
            if s is None:
                n_xtb_fail += 1
                if np.isfinite(e):
                    failed_E.append(float(e))
            if not np.isfinite(e) or s is None:
                continue
            if not same_connectivity(a, ref_graph) or not same_stereochemistry(a, template):
                rearranged.append((a, float(e), sd))
                continue
            if hb_groups is not None and _has_interacid_hbond(a, hb_groups):
                hb_set_aside.append((a, float(e), sd))
                continue
            _add_minimum(uniq, also, a, float(e), sd)
            # Rediscovering an existing valid minimum is real search evidence.
            # A batch with no eligible conformer is not a repeated measurement
            # of the old ensemble and cannot contribute a plateau hit.
            eligible_in_batch += 1
            if e < best[0]:
                best = (float(e), a.get_chemical_symbols(), a.get_positions())
        if not all_G or eligible_in_batch == 0:
            hits = 0
            continue
        Gens = boltz(all_G)
        if thermal_track is not None:                      # convergence / U_samp on E + ΔGsolv + RRHO_i
            Gens = thermal_track.update(uniq)
        gens_traj.append(Gens)
        successful_seeds.append(seed)
        minimum_traj.append(best[0])
        if prev_Gens is not None:
            last_dG = abs(Gens - prev_Gens)
            if last_dG < CONV_TOL and abs(best[0] - prev_best) < CONV_TOL:
                hits += 1
                if hits >= CONV_HITS:
                    prev_Gens = Gens; break
            else:
                hits = 0
        prev_Gens, prev_best = Gens, best[0]
    sp_warn = []
    if hb_set_aside:
        if not all_G:                                     # every conformer H-bonds across groups: keep them
            for a, e, sd in hb_set_aside:                 # (no filtered ensemble exists) and say so
                _add_minimum(uniq, also, a, e, sd)
                if e < best[0]:
                    best = (e, a.get_chemical_symbols(), a.get_positions())
            sp_warn.append(f"{name} ({smi}): every conformer has an inter-acid H-bond; filter not applied")
        log(f"    {name}: {len(hb_set_aside)} conformer(s) with an inter-acid-group H-bond excluded")
    if not all_G:
        if rearranged:
            raise SpeciesRearranged(name, smi, len(rearranged))
        return None, None
    if rearranged:
        log(f"    {name}: {len(rearranged)} conformer(s) changed bonding during relaxation -> excluded")
    missing = [e for e in failed_E if e - min(uniq.E) < SOLV_SUPPORT_KJ]
    if missing:
        sp_warn.append(f"{name} ({smi}): solvation failed for {len(missing)} conformer(s) within "
                       f"{SOLV_SUPPORT_KJ:.0f} kJ of the minimum -> ensemble incomplete, species failed")
        log(f"    !! {sp_warn[-1]}")
        warnings.extend(sp_warn)
        return None, None
    Gens = boltz(all_G)
    relax_info = None
    if thermal_track is not None:
        validation_diagnostics = {}
        validated = _validated_thermal_minima(pu, uniq, also, template, ref_graph, q, mult, name, log,
                                              track=thermal_track, diagnostics=validation_diagnostics)
        if validated is None:
            message = f"{name}: sampled thermal ensemble failed ({validation_diagnostics.get('reason', 'unresolved')})"
            warnings.append(message)
            if diagnostics is not None:
                diagnostics["thermal"] = {"status": "incomplete", "n_sampled": len(uniq.G),
                                          "validation_maximum": THERMAL_VALIDATION_MAX, **validation_diagnostics}
            log(f"    !! {message}")
            return None, None
        uniq, also, gas_thermal, validation_info = validated
        ref_j = min(range(len(uniq.G)), key=lambda j: uniq.G[j])
        therm = gas_thermal[ref_j]
        Gens_th, also_th, ens_info = thermal_track.final(
            uniq, also, therm, ref_j, validated=dict(enumerate(gas_thermal)))
        ens_info.update(validation_info)
        t_info = {"ensemble": ens_info}
        Gens, also, therm = Gens_th, also_th, 0.0
    else:
        therm, t_info, t_res = _thermal_at_minimum(pu, uniq, best, template, ref_graph, q, mult, name, log)
        if therm is not None and uniq.template is not None and uniq.ref:
            _apply_thermal_resolution(pu, uniq, also, t_res, template, q, mult, name, log)
            Gens = boltz(uniq.G)
            t_info = dict(t_info, resolution=t_res["kind"], n_removed=len(t_res["rejected"]))
        gas_thermal = [therm] * len(uniq.G)
    if therm is None:
        sp_warn.append(f"{name} ({smi}): no true minimum among the lowest {N_THERMAL_CANDIDATES} "
                       f"(imaginary modes {t_info}) -> species failed")
        log(f"    !! {sp_warn[-1]}")
        warnings.extend(sp_warn)
        return None, None
    if _SOLV_RELAX:
        # Gas-minimum validation may replace/drop geometries. Only after it is complete
        # can the aqueous ensemble be constructed without subsequently overwriting it.
        # Carry each starting basin's gas RRHO as an explicit approximation; a gas
        # Hessian at an aqueous stationary point is not a validated gas minimum.
        if uniq.template is None or not uniq.ref:
            warnings.append(f"{name}: SOLV_RELAX requires conformer deduplication")
            return None, None
        relax_diagnostics = {}
        relaxed = _solvent_relaxed_ensemble(
            pu, uniq, q, mult, ref_graph, template, name, log, gas_thermal,
            diagnostics=relax_diagnostics)
        if diagnostics is not None:
            diagnostics["solv_relax"] = relax_diagnostics
        if relaxed is None:
            warnings.append(f"{name}: no converged solution-phase minimum; species failed")
            return None, None
        G_aq, also, relax_info = relaxed
        if any(relax_info.get(k, 0) for k in ("n_unconverged", "n_rearranged", "n_failed", "n_unattempted")):
            warnings.append(f"{name}: incomplete solution-phase ensemble; species failed")
            if diagnostics is not None:
                diagnostics["solv_relax"] = dict(relax_info, status="incomplete")
            return None, None
        log(f"    {name}: solution-phase ensemble {Gens + therm:.1f} -> {boltz(G_aq):.1f} "
            f"({relax_info['n_relaxed']} minima, gas-basin RRHO approximation)")
        Gens, therm = boltz(G_aq), 0.0
        t_info = dict(t_info, reference_geometry_scope="validated gas minima; inherited gas RRHO",
                      destination_hessian_validation="not_performed")
    # sampling uncertainty: spread of Gens over the last few batches (0 if never moved / capped-tight)
    tail = gens_traj[-3:]
    sigma = float(np.std(tail)) if len(tail) > 1 else (last_dG if np.isfinite(last_dG) else 3.0)
    tag = "conv" if hits >= CONV_HITS else "CAPPED"
    log(f"    {name:9s} q{q:+d} [implicit {tag} seeds={seed} minima={len(uniq)}/{uniq.n_seen} σ={sigma:.1f}]: "
        f"Gens {Gens:.1f} + thermal {therm:.1f} = {Gens+therm:.1f}")
    sampling_info = sampling_diagnostics(gens_traj, successful_seeds, minimum_traj, seeds=seed,
                                         hits=hits, required_hits=CONV_HITS, maximum=CONV_MAX, tolerance=CONV_TOL)
    sampling_info["ensemble_energy_scope"] = ("gas minima plus vertical solvation and per-minimum thermal approximations"
                                               if thermal_track is not None else "gas minima plus vertical solvation; thermal added after sampling")
    sampling_info["seed_offset"] = seed_offset
    sampling_info["solution_relaxation_in_trajectory"] = False
    sampling_info["final_validation_in_trajectory"] = False
    sampling_info["final_minus_last_provisional_kj"] = (
        float(Gens + therm - gens_traj[-1]) if gens_traj else None)
    if diagnostics is not None:
        diagnostics.update(species_diagnostics({"sampling": sampling_info, "thermal": t_info}, "fresh"))
    fail_frac = n_xtb_fail / max(n_relaxed, 1)
    incomplete_also = {
        m: sum(v is None for v in vals)
        for m, vals in also.items()
        if vals and any(v is None for v in vals)
    }
    for m, n_failed in incomplete_also.items():
        sp_warn.append(f"{name} ({smi}): auxiliary {m} solvation failed on "
                       f"{n_failed}/{len(also[m])} accepted minima -> auxiliary G not cached")
        log(f"    !! {sp_warn[-1]}")
    incomplete_relax = relax_info is not None and any(
        relax_info.get(k, 0) for k in ("n_unconverged", "n_rearranged", "n_failed", "n_unattempted"))
    if fail_frac > MAX_XTB_FAIL_FRAC:                     # degraded ensemble: use it, but never cache it
        sp_warn.append(f"{name} ({smi}): xtb solvation failed on {n_xtb_fail}/{n_relaxed} conformers "
                       f"-> G not cached")
        log(f"    !! {sp_warn[-1]}")
    elif not incomplete_relax:
        meta = {"warnings": sp_warn, "n_minima": len(uniq), "n_seen": uniq.n_seen,
                "n_xtb_fail": n_xtb_fail, "n_rearranged": len(rearranged), "seeds": seed, "thermal": t_info,
                "sampling": sampling_info,
                **({"solv_relax": relax_info} if relax_info else {})}
        _sc.put(smi, q, "implicit", _settings, Gens + therm, sigma, meta=meta)
        for m in SOLV_ALSO:                               # other models on the same conformers + thermal; a
            vals = _complete_auxiliary_values(also[m])    # separate key ("via") -- NOT the value a primary
            if vals is not None:                          # run would compute; never renormalize a partial
                _sc.put(smi, q, "implicit", dict(_settings, solv=m, via="solv_also"),
                        boltz(vals) + therm, sigma, meta=meta)
    warnings.extend(sp_warn)
    return Gens + therm, sigma


def _acid_transform(smi_neutral):
    """-RT ln Π(1+10^(pH-pKa)) over the acid sites of a NEUTRAL pH-0 species (its max-anion form
    classified with the pKa table), i.e. G'(pH) - G(neutral) for the acid groups."""
    from metag.routing import pka_transform as _pk
    _, pkas, _ = _pk._neutralize(_pk._canonicalize_maxanion(smi_neutral))
    return math.fsum(site_free_energy(p, "acid", PH, RT_LN10) for p in (pkas or []))


def _hydrate_all_sites(pu, rx, name, q, smi, G, sig, std, seeds, keep, pool, log, routes):
    from metag.energetics.aqueous_states import hydrate_all_sites
    return hydrate_all_sites(pu, rx, name, q, smi, G, sig, std, seeds, keep, pool, log, routes,
                             implicit=implicit_G, water_reference=water_ref_G,
                             acid_transform=_acid_transform, calibrate=_flag("HYDRATION_CAL"))


def _fold_ring_chain(pu, rx, name, q, smi, G, sig, seeds, keep, pool, log, routes):
    from metag.energetics.aqueous_states import fold_ring_chain
    return fold_ring_chain(pu, rx, name, q, smi, G, sig, seeds, keep, pool, log, routes,
                           implicit=implicit_G, acid_transform=_acid_transform,
                           standard_state=STD_STATE_KJ if _flag("STD_STATE_1M") else 0.0,
                           temperature=T)


from metag.energetics.acid_contacts import acid_groups as _acid_groups, is_carboxyl as _is_carboxyl


def _has_interacid_hbond(atoms, groups):
    from metag.energetics.acid_contacts import has_interacid_hbond
    return has_interacid_hbond(atoms, groups, ACID_HB_DIST)


def _solvent_relaxed_ensemble(pu, uniq, q, mult, ref_graph, template, name, log, gas_thermal, *, diagnostics=None):
    """Re-relax the lowest unique minima on E_UMA + ΔG_solv(SOLV_MODEL); returns (G_aq list of unique
    stationary candidates, {aux model: values}, info) or None if no relaxed structure survives (the caller
    fails the species). Only converged structures with unchanged bonding are accepted. Each energy
    includes the starting basin's gas RRHO approximation; duplicate aqueous basins are merged AFTER
    that term is included. This is not a solution-phase Hessian or a solvent conformer search."""
    from metag.energetics.solv_relax import make_extra_forces
    if len(gas_thermal) != len(uniq.G) or not all(np.isfinite(g) for g in gas_thermal):
        raise ValueError("one finite gas RRHO correction is required per starting minimum")
    order = sorted(range(len(uniq.G)), key=lambda j: uniq.G[j])
    pick = [j for j in order if uniq.G[j] - uniq.G[order[0]] < SOLV_RELAX_WIN][:SOLV_RELAX_N]
    syms = [a.GetSymbol() for a in template.GetAtoms()]
    ats = [Atoms(symbols=syms, positions=uniq.ref[j].GetConformer().GetPositions(),
                 info={"charge": int(q), "spin": int(mult)}) for j in pick]
    extra = make_extra_forces(q, mult, SOLV_MODEL, workers=XTB_WORKERS)
    rel, E, conv = batched_fire(pu, ats, fmax=0.05, steps=SOLV_RELAX_STEPS, stop_frac=1.0,
                                return_converged=True, extra_forces=extra, label=f"{name}-aq")
    models = [SOLV_MODEL] + SOLV_ALSO
    with ThreadPoolExecutor(max_workers=XTB_WORKERS) as ex:
        solv = list(ex.map(lambda am: dgsolv(am[0].get_chemical_symbols(), am[0].get_positions(), q,
                                             am[1], mult), [(a, m) for a in rel for m in models]))
    solv = [dict(zip(models, solv[i * len(models):(i + 1) * len(models)])) for i in range(len(rel))]
    uq = UniqueMinima(template=template)
    also = {m: [] for m in SOLV_ALSO}
    n_bond = n_fail = 0
    for i, (a, e, sd) in enumerate(zip(rel, E, solv)):
        if not conv[i]:
            continue
        if i in extra.failed or sd[SOLV_MODEL] is None or not np.isfinite(e) \
                or not np.isfinite(sd[SOLV_MODEL]):
            n_fail += 1; continue
        if not same_connectivity(a, ref_graph) or not same_stereochemistry(a, template):
            n_bond += 1; continue
        # Keep electronic energies for the geometric dedup criterion; add RRHO to
        # each model's correction so its population weight follows the same basin.
        corrected = {m: None if value is None else value + gas_thermal[pick[i]]
                     for m, value in sd.items()}
        _add_minimum(uq, also, a, float(e) * EV2KJ, corrected)
    info = {"n_start": len(pick), "n_relaxed": len(uq.G), "n_unconverged": int((~np.asarray(conv)).sum()),
            "n_rearranged": n_bond, "n_failed": n_fail,
            "n_unattempted": len(uniq.G) - len(pick),
            "thermal_model": "starting-gas-basin RRHO; no aqueous Hessian",
            "destination_hessian_validation": "not_performed",
            "sampling_sigma_scope": "gas-ensemble convergence only",
            "gradient_failures": dict(getattr(extra, "failures", {})),
            "status": "complete" if len(uq.G) and not any((n_bond, n_fail, int((~np.asarray(conv)).sum()), len(uniq.G)-len(pick))) else "incomplete"}
    if diagnostics is not None:
        diagnostics.update(info)
    if not uq.G:
        return None
    return list(uq.G), also, info


from metag.energetics.thermal_ensemble import ThermalTrack


class _ThermalTrack(ThermalTrack):
    """Compatibility adapter binding the pipeline's configured thermal backend."""

    def __init__(self, pu, template, q, mult):
        super().__init__(pu, template, q, mult, correction=uma_gibbs_corr, combine=boltz,
                         window=THERMAL_ENS_WIN, maximum=THERMAL_ENS_N)


def _thermal_ensemble(pu, uniq, also, therm_ref, q, mult, template, name, log):
    from metag.energetics.thermal_ensemble import thermal_ensemble
    return thermal_ensemble(pu, uniq, also, therm_ref, q, mult, template,
                            correction=uma_gibbs_corr, combine=boltz, window=THERMAL_ENS_WIN,
                            maximum=THERMAL_ENS_N, imaginary_tolerance=thermal_mod.IMAG_TOL_CM)


def _validated_thermal_minima(pu, uniq, also, template, ref_graph, q, mult, name, log, *, track=None, diagnostics=None):
    """Resolve every sampled candidate, then deduplicate the resulting geometries.

    A window cannot bound thermally reordered populations. Exceeding the explicit
    candidate budget or an unresolved candidate fails the species, rather than
    silently discarding its population. Work on singleton copies so index changes
    during saddle resolution cannot invalidate another candidate's bookkeeping.
    """
    diagnostics = diagnostics if diagnostics is not None else {}
    if not uniq.G or len(uniq.G) > THERMAL_VALIDATION_MAX:
        diagnostics.update(reason="candidate validation budget exceeded" if uniq.G else "empty ensemble",
                           n_sampled=len(uniq.G), validation_maximum=THERMAL_VALIDATION_MAX)
        return None
    output = UniqueMinima(template=template)
    aux = {m: [] for m in also}
    corrections, kinds = [], []
    optimizer_trials = []
    diagnostics["optimizer_trials"] = optimizer_trials
    n_reused = n_retried = 0
    syms = [a.GetSymbol() for a in template.GetAtoms()]
    for j in range(len(uniq.G)):
        one = UniqueMinima(template=template)
        at = Atoms(syms, positions=uniq.ref[j].GetConformer().GetPositions())
        one.add(at, uniq.E[j], uniq.G[j])
        one_aux = {m: [values[j]] for m, values in also.items()}
        strict = track.validated.get(track._key(uniq, j)) if track is not None else None
        if strict is None:
            g, info, resolution = _thermal_at_minimum(
                pu, one, (uniq.E[j], syms, at.positions), template, ref_graph, q, mult, name, log)
        else:
            g, info = strict
            resolution = {"index": 0, "kind": "as_is", "pos": at.positions, "rejected": []}
            n_reused += 1
        if (g is None or not np.isfinite(g)) and THERMAL_RETRY_STEPS > THERMAL_RELAX_STEPS:
            # Retry the SAME candidate with more optimization work, not a looser
            # force threshold or a smaller ensemble. Already validated candidates
            # never take this path.
            n_retried += 1
            log(f"    {name}: retry unresolved minimum with {THERMAL_RETRY_STEPS} optimization steps")
            g, info, resolution = _thermal_at_minimum(
                pu, one, (uniq.E[j], syms, at.positions), template, ref_graph, q, mult, name, log,
                optimization_steps=THERMAL_RETRY_STEPS)
        if g is None or not np.isfinite(g):
            diagnostics.update(reason="unresolved minimum or nonfinite thermal correction", candidate_index=j,
                               thermal_attempts=info,
                               failure_geometry={
                                   "scope": "failed sampled gas-state candidate; not an accepted thermochemical minimum",
                                   "symbols": syms, "charge": q, "spin_multiplicity": mult,
                                   "reference_smiles": Chem.MolToSmiles(Chem.RemoveHs(template)),
                                   "initial_positions_angstrom": at.positions.tolist(),
                                   "initial_electronic_kj": float(uniq.E[j]),
                                   "initial_vertical_kj": float(uniq.G[j]),
                                   "thermal_trials": resolution.get("thermal_trials", []),
                                   "optimization_trials": resolution.get("optimization_trials", []),
                                   "seed_offset": int(os.environ.get("CONFORMER_SEED_OFFSET", "0")),
                                   "producer": _producer_identity(pu)})
            return None
        optimizer_trials.extend(dict(trial, candidate_index=j)
                                for trial in resolution.get("optimization_trials", []))
        try:
            idx = _apply_thermal_resolution(pu, one, one_aux, resolution, template, q, mult, name, log)
        except RuntimeError as exc:
            log(f"    {name}: thermal geometry update failed: {exc}")
            diagnostics.update(reason="validated geometry energy update failed", candidate_index=j, error=str(exc))
            return None
        if idx is None or not np.isfinite(one.E[idx]) or not np.isfinite(one.G[idx]):
            diagnostics.update(reason="invalid resolved geometry energy", candidate_index=j)
            return None
        at.positions = one.ref[idx].GetConformer().GetPositions()
        added = output.add(at, one.E[idx], one.G[idx])
        k = output.last_index
        if added:
            corrections.append(float(g))
            for m in aux:
                aux[m].append(one_aux[m][idx])
        elif output.last_replaced:
            corrections[k] = float(g)
            for m in aux:
                aux[m][k] = one_aux[m][idx]
        kinds.append(resolution["kind"])
    output.n_seen = uniq.n_seen
    return output, aux, corrections, {
        "n_candidates_validated": len(uniq.G), "n_strict_results_reused": n_reused,
        "n_candidates_retried": n_retried, "retry_optimization_steps": THERMAL_RETRY_STEPS,
        "n_merged_after_resolution": len(uniq.G) - len(output.G),
        "validation_maximum": THERMAL_VALIDATION_MAX, "resolution_kinds": kinds,
        "optimizer_trials": optimizer_trials,
        "validation_scope": "all sampled candidates; heuristic mode-following artefact test; no unsampled-basin bound",
        "validated_minima": [
            {"electronic_kj": e, "solvation_kj": vertical - e, "thermal_kj": g,
             "total_kj": vertical + g,
             "reference_smiles": Chem.MolToSmiles(Chem.RemoveHs(output.ref[j])),
             "geometry": ordered_geometry_record(
                 syms, output.ref[j].GetConformer().GetPositions(), q, mult),
             "geometry_scope": "Accepted gas minimum; these component energies refer to the exported ordered geometry."}
            for j, (e, vertical, g) in enumerate(zip(output.E, output.G, corrections))]}


def _thermal_at_minimum(pu, uniq, best, template, ref_graph, q, mult, name, log, *, optimization_steps=None):
    """RRHO correction at the lowest-G unique minimum that is a TRUE minimum. Candidates are the unique
    minima in order of G; one with an imaginary mode above thermal.IMAG_TOL_CM is re-optimised once more
    tightly (fmax 0.01 eV/Å); if still not a minimum, the strongest imaginary mode is followed; else the next
    candidate is tried. Returns (Gcorr, info, resolution) or (None, per-candidate info, resolution).

    resolution = {"index": accepted minimum index or None, "kind": "as_is" | "artefact" | "retightened" |
    "mode_followed" | None, "pos": geometry the correction belongs to, "rejected": [indices that are not
    minima]}. The caller (_apply_thermal_resolution) makes E, ΔG_solv, geometry and Gcorr refer to the SAME
    structure: retightened -> the minimum's representative is replaced by the tightened geometry;
    mode_followed -> the saddle is removed and the lower structure enters the ensemble; rejected -> removed."""
    optimization_steps = THERMAL_RELAX_STEPS if optimization_steps is None else optimization_steps
    syms = [a.GetSymbol() for a in template.GetAtoms()]
    if uniq.template is not None and uniq.ref:
        order = sorted(range(len(uniq.G)), key=lambda j: uniq.G[j])[:N_THERMAL_CANDIDATES]
        cands = [(j, uniq.ref[j].GetConformer().GetPositions()) for j in order]
    else:                                                 # legacy never-merge / moments mode
        cands = [(None, best[2])]
    tried, rejected = [], []
    thermal_trials, optimization_trials = [], []

    def record_thermal(positions, correction, info):
        thermal_trials.append({
            "positions_angstrom": np.asarray(positions, dtype=float).tolist(),
            "correction_kj": float(correction) if correction is not None and np.isfinite(correction) else None,
            "n_imag": int(info.get("n_imag", 0)),
            "max_imag_cm": float(info.get("max_imag_cm", 0))})
    for j, pos0 in cands:
        pos = pos0
        for attempt in (0, 1):
            g, info = uma_gibbs_corr(pu, syms, pos, q, spin=mult, return_info=True)
            record_thermal(pos, g, info)
            vecs = info.pop("_imag_vecs")
            if info["n_imag"] == 0:
                info["retightened"] = bool(attempt)
                return g, info, {"index": j, "kind": "retightened" if attempt else "as_is", "pos": pos,
                                 "rejected": rejected}
            if attempt == 1:
                # A harmless strongest mode does not establish that every other
                # negative mode is harmless. Probe all reported strong modes.
                all_artefacts = bool(vecs) and len(vecs) == info["n_imag"]
                for mode in vecs:
                    verdict, new_pos = _follow_imaginary_mode(
                        pu, syms, pos, mode, q, mult, ref_graph, name,
                        template=template, log=log, optimization_steps=optimization_steps)
                    if verdict != "artefact":
                        all_artefacts = False
                    if verdict == "saddle":
                        g2, info2 = uma_gibbs_corr(pu, syms, new_pos, q, spin=mult, return_info=True)
                        record_thermal(new_pos, g2, info2)
                        info2.pop("_imag_vecs")
                        if info2["n_imag"] == 0:
                            info2.update(retightened=True, mode_followed=True)
                            return g2, info2, {"index": j, "kind": "mode_followed", "pos": new_pos,
                                              "rejected": rejected}
                if all_artefacts:
                    g_soft, info_soft = uma_gibbs_corr(pu, syms, pos, q, spin=mult, return_info=True,
                                                       imag_as_soft=True)
                    info_soft.pop("_imag_vecs")
                    info_soft.update(retightened=True, imag_artefact_cm=info["max_imag_cm"],
                                     n_imag_probed=len(vecs))
                    kind = "retightened" if pos is not pos0 else "artefact"
                    return g_soft, info_soft, {"index": j, "kind": kind, "pos": pos, "rejected": rejected}
            tried.append(info["max_imag_cm"])
            if attempt == 0:                              # re-optimise tightly once, keep only if unrearranged
                import time
                at = Atoms(symbols=syms, positions=pos, info={"charge": int(q), "spin": int(mult)})
                fire_started = time.monotonic()
                (at,), _, conv = batched_fire(pu, [at], fmax=0.01, steps=optimization_steps, return_converged=True,
                                              label=f"{name}-tight")
                identity_ok = same_connectivity(at, ref_graph) and same_stereochemistry(at, template)
                optimization_trials.append({"positions_angstrom": at.positions.tolist(),
                                            "converged": bool(conv[0]), "identity_preserved": bool(identity_ok),
                                            "optimizer": "batched_FIRE", "elapsed_seconds": time.monotonic()-fire_started,
                                            "step_budget": optimization_steps, "fmax_eV_A": 0.01})
                if not identity_ok:
                    break
                if (not conv[0] and THERMAL_RETRY_STEPS > THERMAL_RELAX_STEPS and
                        optimization_steps >= THERMAL_RETRY_STEPS):
                    from metag.energetics.minimum_optimizer import recover_force_minimum
                    # The controlled AMP experiment compared these optimizers
                    # from this same pre-FIRE geometry, not from a later trial.
                    fallback_start = Atoms(symbols=syms, positions=pos,
                                           info={"charge": int(q), "spin": int(mult)})
                    recovered, trial = recover_force_minimum(
                        pu, fallback_start, template, ref_graph, steps=optimization_steps)
                    optimization_trials.append(trial)
                    if trial["force_accepted"]:
                        g_recovered, info_recovered = uma_gibbs_corr(
                            pu, syms, recovered.positions, q, spin=mult, return_info=True)
                        record_thermal(recovered.positions, g_recovered, info_recovered)
                        info_recovered.pop("_imag_vecs", None)
                        trial["thermal_n_imag"] = int(info_recovered["n_imag"])
                        trial["thermal_max_imag_cm"] = float(info_recovered["max_imag_cm"])
                        trial["strict_thermal_accepted"] = bool(
                            info_recovered["n_imag"] == 0 and g_recovered is not None and np.isfinite(g_recovered))
                        if trial["strict_thermal_accepted"]:
                            info_recovered.update(retightened=True, optimizer="ASE_LBFGS")
                            log(f"    {name}: strict minimum recovered by ASE_LBFGS "
                                f"({trial['steps']} steps, {trial['model_evaluations']} model evaluations, "
                                f"{trial['elapsed_seconds']:.2f} s)")
                            return g_recovered, info_recovered, {
                                "index": j, "kind": "retightened", "pos": recovered.positions,
                                "rejected": rejected, "optimization_trials": optimization_trials,
                                "thermal_trials": thermal_trials}
                    log(f"    {name}: LBFGS fallback did not establish a strict minimum: {trial}")
                if conv[0]:                               # an unconverged tight re-opt (flat torsions) still
                    pos = at.get_positions()              # gets the mode-following test from the original pos
        if j is not None:
            rejected.append(j)                            # not a minimum: excluded from the ensemble
        log(f"    {name}: minimum candidate has an imaginary mode ({tried[-1]} cm-1) -> next candidate")
    return None, tried, {"index": None, "kind": None, "pos": None, "rejected": rejected,
                         "thermal_trials": thermal_trials, "optimization_trials": optimization_trials}


def _apply_thermal_resolution(pu, uniq, also, res, template, q, mult, name, log):
    """Make the ensemble consistent with the thermal validation (see _thermal_at_minimum). Returns the index
    (after the update) of the minimum the validated correction belongs to."""
    syms = [a.GetSymbol() for a in template.GetAtoms()]
    models = [SOLV_MODEL] + SOLV_ALSO
    j, kind, pos = res["index"], res["kind"], res["pos"]
    drop = set(res["rejected"])
    if kind in ("retightened", "mode_followed"):
        at = Atoms(symbols=syms, positions=pos, info={"charge": int(q), "spin": int(mult)})
        e = float(batched_energies(pu, [at])[0]) * EV2KJ
        sd = {m: dgsolv(syms, pos, q, m, mult) for m in models}
        if not np.isfinite(e) or sd[SOLV_MODEL] is None or not np.isfinite(sd[SOLV_MODEL]):
            raise RuntimeError(f"{name}: solvation failed on the validated thermal geometry")
        if kind == "retightened":                         # same basin: the representative becomes this geometry
            uniq.E[j] = e; uniq.G[j] = e + sd[SOLV_MODEL]; uniq.ref[j] = uniq._mol_at(at)
            for m in also:
                also[m][j] = None if sd[m] is None else e + sd[m]
        else:                                             # saddle j removed; the lower structure enters
            drop.add(j)
            uniq.E.append(e); uniq.G.append(e + sd[SOLV_MODEL]); uniq.ref.append(uniq._mol_at(at))
            for m in also:
                also[m].append(None if sd[m] is None else e + sd[m])
            j = len(uniq.G) - 1
        log(f"    {name}: thermal validation -> {kind} geometry carries E, ΔGsolv and RRHO")
    if drop:
        keep = uniq.drop(drop)
        for m in also:
            also[m] = [also[m][k] for k in keep]
        j = keep.index(j) if j in keep else None
        log(f"    {name}: {len(drop)} non-minimum candidate(s) removed from the ensemble")
    return j


def _follow_imaginary_mode(pu, syms, pos, mode, q, mult, ref_graph, name, step=0.1, drop_kJ=1.0,
                           template=None, log=None, optimization_steps=None):
    """('saddle', lower_pos) if relaxing from pos +- step·mode reaches an energy > drop_kJ below pos (same
    connectivity), else ('artefact', None)."""
    at0 = Atoms(symbols=syms, positions=pos, info={"charge": int(q), "spin": int(mult)})
    e0 = float(batched_energies(pu, [at0])[0]) * EV2KJ
    disp = [Atoms(symbols=syms, positions=pos + sgn * step * mode, info={"charge": int(q), "spin": int(mult)})
            for sgn in (+1.0, -1.0)]
    steps = THERMAL_RELAX_STEPS if optimization_steps is None else optimization_steps
    rel, E, conv = batched_fire(pu, disp, fmax=0.01, steps=steps, return_converged=True, label=f"{name}-mode")
    best = None
    n_valid = 0
    probes = []
    for a, e, c in zip(rel, E, conv):
        e = float(e) * EV2KJ
        identity = same_connectivity(a, ref_graph) and (template is None or same_stereochemistry(a, template))
        probes.append({"converged": bool(c), "identity": bool(identity),
                       "delta_kj": float(e-e0) if np.isfinite(e) and np.isfinite(e0) else None})
        if c and np.isfinite(e) and np.isfinite(e0) and identity:
            n_valid += 1
            if e < e0 - drop_kJ and (best is None or e < best[0]):
                best = (e, a.get_positions())
    if log is not None:
        log(f"    {name}: imaginary-mode probes {probes}")
    if best is not None:
        return "saddle", best[1]
    # A failed/rearranged/nonfinite probe cannot establish that negative curvature
    # is numerical noise. Both directions must provide usable evidence.
    return ("artefact", None) if n_valid == 2 else ("inconclusive", None)


def _add_minimum(uniq, also, atoms, e, sd):
    """Register one relaxed, solvated conformer in the unique-minima set (primary SOLV_MODEL) and keep the
    secondary SOLV_ALSO values aligned per minimum. A secondary model that failed on this conformer leaves
    None -- it never removes the conformer from the PRIMARY ensemble."""
    if uniq.add(atoms, e, e + sd[SOLV_MODEL]):            # new minimum
        for m in also:
            also[m].append(None if sd[m] is None else e + sd[m])
        return
    j = uniq.last_index
    if j is None or not uniq.last_replaced:
        return
    for m in also:
        # Auxiliary energy must belong to the primary representative's geometry.
        # Independently keeping its minimum mixes coordinates with the later RRHO.
        also[m][j] = None if sd[m] is None else e + sd[m]


def _complete_auxiliary_values(values):
    """Return a complete auxiliary conformer ensemble, or None if any accepted minimum is missing."""
    return values if values and all(v is not None and np.isfinite(v) for v in values) else None


_WATER_REF = {}
def water_ref_G(pu, log=None):
    """Free energy of ONE liquid-water molecule in the SAME method, for the Bryantsev
    cluster-continuum monomer cycle. Subtracting n_water*this from a cluster G makes the
    explicit waters reference bulk liquid, so they cancel for a SPECTATOR anion (equal n
    both sides) AND stay correct for a CREATED/DESTROYED anion (unequal n). Without it,
    explicit_G leaks n*G(water) (~ -2e5 kJ each) into any reaction that changes anion count.
      G*_liq(H2O) = E_UMA(H2O) + thermal(H2O, 1 atm) + dGsolv(H2O) + RT ln(55.34)
    NOTE: this value (like every cached species G) is on the 1 atm gas standard state; score_reaction adds
    the 1 atm -> 1 M term (STD_STATE_KJ) uniformly to every species incl. water, so the RT ln 55.34
    (1 M -> 55.34 M pure liquid) term below is then correct."""
    _WSOLV_EXP = float(os.environ.get("WATER_DGSOLV_KJ", "-26.4"))    # exp ΔGhyd(H2O), -6.3 kcal/mol
    use_exp = _flag("WATER_REF_EXP")
    producer = _producer_identity(pu)
    model_key = json.dumps(producer, sort_keys=True)
    memo_key = (_model_name(pu), model_key, SOLV_MODEL, use_exp, _WSOLV_EXP, qrrho_enabled(), hessian_method())
    if producer["cacheable"] and memo_key in _WATER_REF:
        return _WATER_REF[memo_key]
    sym, coord = bare_geom(pu, 0, "O")
    atoms = Atoms(symbols=list(sym), positions=coord, info={"charge": 0, "spin": 1})
    E = float(batched_energies(pu, [atoms])[0]) * EV2KJ
    # WATER REFERENCE (WATER_REF_EXP, default ON since 2026-10-01): the chemical potential of liquid water is an
    # experimental constant, so the water molecule's solvation is taken from experiment (ΔG_hyd = -26.4 kJ/mol)
    # rather than from a continuum model of water in water (xtb-COSMO -2.8, xtb-ALPB -38). The class anchors
    # that had absorbed the COSMO water error are no longer used. WATER_REF_EXP=0 restores the model value.
    solv = _WSOLV_EXP if use_exp else dgsolv(list(sym), coord, 0, SOLV_MODEL)
    thermal = uma_gibbs_corr(pu, list(sym), coord, 0)
    conc = 8.314e-3 * 298.15 * float(np.log(55.34))          # +9.96 kJ/mol, 1 M -> liquid 55.34 M
    G = E + solv + thermal + conc
    if producer["cacheable"]:
        _WATER_REF[memo_key] = G
    if log:
        log(f"    [water ref] G*_liq(H2O) = E {E:.1f} + solv {solv:.1f} + thermal {thermal:.1f} "
            f"+ conc {conc:.1f} = {G:.1f}")
    return G


def _explicit_cluster_identity(atoms, template, n_water):
    """Require the specified solute plus intact waters, with no interfragment covalent bonds."""
    n = template.GetNumAtoms()
    expected = [a.GetSymbol() for a in template.GetAtoms()] + ["O", "H", "H"] * n_water
    if atoms.get_chemical_symbols() != expected or not np.isfinite(atoms.positions).all():
        return False
    graph = bond_graph(template)
    for w in range(n_water):
        o = n + 3 * w
        graph.update({(o, o + 1), (o, o + 2)})
    return same_connectivity(atoms, graph) and same_stereochemistry(atoms[:n], template)


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
    from metag.chem import require_supported_isotopes
    require_supported_isotopes(smi)
    smi = _sc.canonical(smi)
    _settings = dict(_explicit_settings(), model=_model_name(pu),
                     producer=_producer_identity(pu))
    _cached = _sc.get(smi, q, "explicit", _settings)
    if _cached is not None:
        log(f"    {name:9s} q{q:+d} [explicit CACHED]: {_cached[0]:.1f}")
        return _cached[0], _cached[1]
    try:
        bsym, bcoord = bare_geom(pu, q, smi)
    except ValueError as exc:
        log(f"    {name}: {exc}")
        return None, None
    template = Chem.AddHs(Chem.MolFromSmiles(smi))
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
    valid = np.array([bool(c) and np.isfinite(e) and _explicit_cluster_identity(a, template, n_water)
                      for a, e, c in zip(rel, E, conv)], dtype=bool)
    rel = [a for a, ok in zip(rel, valid) if ok]; E = np.asarray(E)[valid] * EV2KJ
    if not len(E):
        return None, None
    # FIX: BOLTZMANN over the cluster ensemble (NOT min — min is non-convergent for
    # floppy clusters: E_UMA drifts down as you add seeds). Keep the lowest-E clusters,
    # vertical solvation on each validated geometry, Boltzmann of E_UMA+ΔGsolv.
    order = np.argsort(E)[:EXPLICIT_KEEP]
    sel = [rel[i] for i in order]; Eu = [float(E[i]) for i in order]
    with ThreadPoolExecutor(max_workers=XTB_WORKERS) as ex:
        solv = list(ex.map(lambda a: dgsolv(a.get_chemical_symbols(),
                                            a.get_positions(), q, "cosmo", 1), sel))
    _uq = UniqueMinima()                                  # merge seeds that relaxed to the same cluster
    for a, e, s in zip(sel, Eu, solv):
        if s is not None and np.isfinite(s):
            _uq.add(a, e, e + s)
    Gt = _uq.G
    if not Gt:
        return None, None
    Gens = boltz(Gt)
    thermal, thermal_info = uma_gibbs_corr(pu, bsym, bcoord, q, return_info=True)
    if thermal_info.get("n_imag", 0) or not np.isfinite(thermal):
        log(f"    {name}: bare-solute thermal minimum unresolved")
        return None, None
    wref = n_water * water_ref_G(pu, log)                 # reference explicit waters to bulk liquid
    g = Gens + thermal - wref
    # explicit clusters are noisier than implicit (stochastic seeding; reverse-test showed
    # ~14 kJ run-to-run) -> conservative sampling-σ floor from the kept-cluster spread.
    sigma = max(float(np.std(Gt)) if len(Gt) > 1 else 7.0, 5.0)
    log(f"    {name:9s} q{q:+d} [explicit n={n_water} {sites} keep{len(Gt)}/{N_EXPLICIT_SEEDS} σ={sigma:.1f}]: "
        f"Gens(E+solv) {Gens:.1f} + thermal(solute) {thermal:.1f} - {n_water}*Gwater {wref:.1f} = {g:.1f}")
    _sc.put(smi, q, "explicit", _settings, g, sigma)
    return g, sigma


from metag.config import FLAG_DEFAULTS, flag as _flag


def _ah_mod():
    from metag.routing import aldehyde_hydration
    return aldehyde_hydration


def _constants_hash(module, names):
    """sha256 (12 hex) of the named module-level constants (missing names recorded as None)."""
    payload = json.dumps({n: getattr(module, n, None) for n in names}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def effective_config(*, allow_truncate=True, trunc_radius=None, force_neutral=False, pu=None):
    """Normalized fingerprint of everything that changes a reported number: model, physics version,
    solvation, sampling, every routing/correction switch, pKa model and the deployed anchor offsets. Stored
    in each result and in the calibration artifact; metag.uncertainty refuses `coverage_calibrated` when
    the runtime fingerprint differs from the calibrated one. Includes content hashes of the pKa and hydration
    constant tables and the full species-cache settings, so estimator or constant changes cannot pass silently."""
    from dataclasses import asdict
    from metag.routing.policy import RoutingPolicy
    policy = RoutingPolicy.from_runtime(_flag, os.environ, trunc_radius)
    from metag.routing import pka_transform as _pk
    from metag.routing.anchor import ANCHORS
    anchors = json.dumps({k: [v["offset"], v.get("ref", "tecrdb")] for k, v in sorted(ANCHORS.items())},
                         sort_keys=True)
    cfg = {"model": _MODEL, "physics": PHYSICS_VERSION,
           "estimator_source_sha256": source_identity(),
           "aqueous_sampling_uncertainty": "2026-10-09-partition-derivatives",
           "reaction_sampling_uncertainty": "2026-10-09-shared-estimate-groups",
           "aqueous_state_coverage": "2026-10-07-unique-hydrates", "solv_model": SOLV_MODEL, "dedup": _DEDUP,
           "conv": [CONV_TOL, CONV_HITS, CONV_MAX], "sample_scale": _SAMPLE_SCALE, "qrrho": qrrho_enabled(),
           **({"hessian": hessian_method()} if hessian_method() != "fd" else {}),
           "keep_scale": _KEEP_SCALE,
           "explicit_sampling": [N_EXPLICIT_SEEDS, EXPLICIT_KEEP],
           "trunc_radius": policy.trunc_radius,
           "routing_policy": asdict(policy), "allow_truncate": bool(allow_truncate),
           "force_neutral": bool(force_neutral),
           "producer": _producer_identity(pu),
           **({"uma_runtime_profile": pu.metag_runtime_profile}
              if getattr(pu, "_metag_official_calculator", False) is True else {}),
           "trunc_validate_tol": float(os.environ.get("TRUNC_VALIDATE_TOL", "5")),
           "trunc_validation_policy": "2026-10-07-distinct-models-no-noise-inflation",
           "conformer_seed_offset": int(os.environ.get("CONFORMER_SEED_OFFSET", "0")),
           "dg_sanity_kj": float(os.environ.get("DG_SANITY_KJ", "500")),
           "smd_threshold": float(os.environ.get("SMD_THRESHOLD", "5")),
           "water_ref_delta": float(os.environ.get("WATER_REF_DELTA", "-23.2")),
           "water_dgsolv_kj": float(os.environ.get("WATER_DGSOLV_KJ", "-26.4")),
           "pka_env": _pk._pka_env_enabled(), "free_ppi_pka": _pk._free_ppi_pka_enabled(),
           "anhydride_pka": _pk._anhydride_pka_enabled(),
           **_pk.pka_model_config(),
           "pka_table": _pk.PKA_TABLE_VERSION, "ph0_redox_proton": _pk._redox_proton_enabled(),
           "polyacid_pka": _pk._polyacid_pka_enabled(), "arylamine_nonbasic": _pk._arylamine_nonbasic_enabled(),
           "carboxyl_pairs": _pk._carboxyl_pairs_enabled(),
           # CONTENT hashes: any edit of a constant changes the fingerprint even without a version bump
           "pka_constants": _constants_hash(_pk, ("P_LADDER", "ANHYDRIDE_P_LADDER", "P_N_LADDER", "ACYL_P_LADDER",
                                                  "CARBONATE_LADDER", "PPI_LADDER", "SULFATE_LADDER",
                                                  "CARBOXYL_PKA", "CARBOXYL_PKA_ALPHA", "SULFONATE_PKA",
                                                  "THIOL_PKA", "PHENOL_PKA", "POLYACID_PKA", "CARBOXYL_PAIR_LADDER",
                                                  "AMMONIA_PKA", "AAA_AMINE_PKA", "PRIMARY_AMINE_PKA",
                                                  "IMIDAZOLE_PKA", "GUANIDINIUM_PKA")),
           "hydration_constants": _constants_hash(_ah_mod(), ("HYDRATION_CAL", "MAX_HYDRATION_SITES")),
           # species-level estimator (thermal ensemble, solvent relaxation, dedup, sampling, model ...)
           "implicit_settings": json.dumps(_IMPLICIT_SETTINGS, sort_keys=True),
           "acid_hb_filter": [_ACID_HB_FILTER, ACID_HB_DIST],
           "thermal_ensemble": [_THERMAL_ENSEMBLE, THERMAL_ENS_WIN, THERMAL_ENS_N],
           "thermal_validation_max": THERMAL_VALIDATION_MAX,
           "thermal_relax_steps": THERMAL_RELAX_STEPS,
           "thermal_retry_steps": THERMAL_RETRY_STEPS,
           "solv_relax": [_SOLV_RELAX, SOLV_RELAX_N, SOLV_RELAX_WIN, SOLV_RELAX_STEPS],
           "anchors": hashlib.sha256(anchors.encode()).hexdigest()[:12]}
    cfg.update({k.lower(): _flag(k) for k in sorted(FLAG_DEFAULTS)})
    if _flag('SOURCE_PKA_REGISTRY'):
        from metag.protonation_registry import registry_identity, POLICY_VERSION as registry_policy
        from metag.energetics.protonation_models import POLICY_VERSION as model_policy
        cfg.update(source_protonation_registry_sha256=registry_identity(),
                   source_protonation_registry_policy=registry_policy,
                   species_protonation_model_policy=model_policy)
    if _flag('RING_PROTON_ENSEMBLE'):
        from metag.routing.ring_protons import POLICY_VERSION as ring_policy, MAX_STATES
        cfg.update(ring_proton_policy=ring_policy, ring_proton_max_states=MAX_STATES)
    cfg["conditions"] = json.dumps(CONDITIONS, sort_keys=True)
    from metag.energetics.minimum_optimizer import policy as minimum_optimizer_policy
    cfg["minimum_optimizer_fallback"] = minimum_optimizer_policy()
    from metag.validation.stereochemistry import POLICY_VERSION as stereo_policy
    cfg["stereochemistry_policy"] = stereo_policy
    cfg["software"] = _software_versions()
    return cfg


@functools.lru_cache(maxsize=1)
def _software_versions():
    """Versions of the external codes whose numerics enter the result (fairchem/torch for UMA, RDKit for
    conformers + SMILES handling, xtb for solvation); part of the configuration fingerprint."""
    import importlib.metadata as md
    import subprocess
    out = {}
    for pkg in ("fairchem-core", "torch", "rdkit", "ase", "numpy"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    try:
        from metag.energetics.thermal import XTB
        r = subprocess.run([XTB, "--version"], capture_output=True, text=True, timeout=30)
        m = [ln for ln in (r.stdout + r.stderr).splitlines() if "xtb version" in ln]
        out["xtb"] = m[0].split("version")[1].split()[0] if m else None
    except Exception:
        out["xtb"] = None
    return json.dumps(out, sort_keys=True)


def route_reaction(reaction, allow_truncate=True, trunc_radius=None, log=print,
                   force_neutral=False, policy=None):
    """Compatibility facade for :func:`metag.routing.planner.route_reaction`."""
    from metag.routing.planner import route_reaction as plan
    from metag.routing.policy import RoutingPolicy

    if policy is None:
        policy = RoutingPolicy.from_runtime(_flag, os.environ, trunc_radius)
        trunc_radius = None
    return plan(reaction, allow_truncate=allow_truncate, trunc_radius=trunc_radius,
                log=log, force_neutral=force_neutral, policy=policy)


def _no_estimate(key, reaction, routes, reason, log, *, config=None):
    """Structured failed estimate, retaining routing and species diagnostics without a point energy."""
    log(f"  ΔG = NONE ({reason})")
    return dict(reaction=key, dG=None, dG_raw=None, dG_raw_unreliable=None, suspect=reason, routes=routes,
                original_species={name: list(value) for name, value in reaction["species"].items()},
                energy_units="kJ/mol", quantity="standard_transformed_reaction_gibbs",
                anchor=None, smd=None, water_ref=None, stages={}, conditions=CONDITIONS, config=effective_config() if config is None else config,
                U_samp=None, sigma_pred=None, sigma_breakdown={}, ci95=[None, None], ci_center=None,
                uncertainty_diagnostic=None, sign_status="no_estimate",
                ci_info={"coverage_calibrated": False, "externally_calibrated": False,
                         "calibration_scope": f"no estimate: {reason}"},
                unresolved=None, exp=reaction.get("exp"), err=[], note=reaction.get("note", ""),
                experimental_reference={k: v for k, v in reaction.items() if k.startswith("exp_")},
                explicit=reaction.get("explicit", False), std_state_kJ=None, species_scored=None)



def _species_free_energies(pu, rx, routes, G, sig, std, seeds, keep, pool, log, requested_explicit,
                           is_spectator_anion):
    """Fill G/sig (kJ/mol, 1 M standard state) for every species of the routed reaction; stops at the first
    species that fails (G[name] = None). Raises SpeciesRearranged (see implicit_G)."""
    species_info = routes.setdefault("species_diagnostics", {})
    for name, (coeff, q, smi) in rx["species"].items():
        diagnostic = species_info[name] = {"sampling": {"status": "unavailable", "reason": "backend did not provide diagnostics"}}
        if is_water(smi, q):                             # liquid water (hydrolysis / hydration)
            diagnostic.update(method="liquid_water_reference", sampling={"status": "not_applicable"})
            G[name] = water_ref_G(pu, log); sig[name] = 0.0     # deterministic reference: no sampling σ
            log(f"    {name:9s} q+0 [water ref liquid]: {G[name]:.1f}")
        elif requested_explicit(name):
            if is_spectator_anion(name):
                diagnostic.update(method="explicit", sampling={"status": "unavailable", "reason": "explicit-cluster sampling is not covered by implicit diagnostics"})
                G[name], sig[name] = explicit_G(pu, q, smi, seeds, log, name)
            else:                                        # GUARD: explicit invalid here
                log(f"    !! {name}: explicit REFUSED (created/destroyed anion, no "
                    f"cancellation partner) -> implicit. Use pH-0 (pka_sites) for accuracy.")
                G[name], sig[name] = implicit_G(pu, q, smi, seeds, keep, pool, log, name, routes["warnings"], diagnostics=diagnostic)
        else:
            G[name], sig[name] = implicit_G(pu, q, smi, seeds, keep, pool, log, name, routes["warnings"], diagnostics=diagnostic)
        if G[name] is None:
            log(f"    {name}: FAILED")
            return                                           # caller sees the None and retries / fails
        # 1 atm -> 1 M standard state, applied OUTSIDE the species cache (cached G stays on 1 atm). An
        # explicit cluster is ONE solute minus n liquid waters: Gcl + c - n(Gw + c) -> extra -n·c.
        G[name] += std
        if requested_explicit(name) and is_spectator_anion(name) and not is_water(smi, q):
            G[name] -= water_count(smi)[0] * std
        try:                                                 # release per-species GPU memory so a
            import torch                                      # later big species doesn't OOM from
            if torch.cuda.is_available():                     # fragmentation left by earlier ones
                torch.cuda.empty_cache()
        except Exception:
            pass


@trace_backend
def score_reaction(pu, reaction, seeds=(1, 2), keep=10, pool=48, log=print, allow_truncate=True, key="rxn",
                   trunc_radius=None, _validating=False, _force_neutral=False):
    """Public entry point: score one reaction's standard transformed Gibbs energy (ΔrG'°) from structure.

    `reaction` = {"species": {name: [coeff, charge, smiles]}, "note": str, "n_Hplus": int,
    and optionally "exp": [float] (experiment, for err reporting), "explicit": bool|list, "pka_sites"}.
    `pu` is a loaded UMA model (metag.energetics.uma.load_uma). Returns the result dict:
    dG and dG_raw (routed physical estimate), sigma_pred, ci95, U_samp, routes, ...
    `seeds`/`keep`/`pool` are accepted for backward compatibility; the sampling budget is set per species
    by sampling_budget() + the adaptive convergence loop (CONV_*), not by these arguments.

    TRUNC_VALIDATE=1 (opt-in): radius-sensitivity guard for spectator truncation. A true spectator cut
    leaves ΔG invariant to the cut radius, so the reaction is scored at radius R and R+1; the truncation
    is kept only when the radii produce distinct molecular models, the energy difference <= tolerance,
    and 2*sqrt(U_R² + U_R+1²) <= tolerance. Otherwise generic truncation is disabled;
    any remaining cofactor-core approximations are reported separately. Recorded in result["trunc_validation"].
    """
    if _flag("TRUNC_VALIDATE") and not _validating and allow_truncate:
        return _score_trunc_validated(pu, reaction, seeds, keep, pool, log, key, trunc_radius)
    validate_reaction(reaction)
    cfg = effective_config(allow_truncate=allow_truncate, trunc_radius=trunc_radius,
                           force_neutral=_force_neutral, pu=pu)
    if reaction.get('species_state_pools'):
        from metag.energetics.provenance import content_hash
        from metag.routing.sugar_states import POLICY
        cfg['declared_sugar_pool_policy'] = POLICY
        cfg['caller_species_state_pools_sha256'] = content_hash(reaction['species_state_pools'])
    no_estimate = functools.partial(_no_estimate, config=cfg)
    orig_species = {k: list(v) for k, v in reaction["species"].items()}   # pre-routing identity
    from metag.chem import isotope_labels, is_balanced
    isotopic = {name: labels for name, (_, _, smi) in orig_species.items() if (labels := isotope_labels(smi))}
    if isotopic:
        reason = "isotope-specific thermochemistry is unsupported; mass and solvent/reservoir conventions are absent"
        return no_estimate(key, reaction, {"errors": [], "warnings": [reason], "truncated": False,
                            "routing_status": "not_run_unsupported_input",
                            "unsupported_isotopes": isotopic,
                            "input_balanced": is_balanced(orig_species, reaction.get("n_Hplus", 0))}, reason, log)
    rx, routes, truncated = route_reaction(reaction, allow_truncate=allow_truncate, trunc_radius=trunc_radius,
                                           log=log, force_neutral=_force_neutral)
    routes["rearrangement_reroute"] = bool(_force_neutral)
    if routes["errors"]:
        # an UNEXPECTED exception inside a routing transform (expected misses return "not applicable" or
        # are reverted by the balance guard): the reaction would silently be scored by a different
        # estimator than the configured one -> fail closed, before spending any QM.
        return no_estimate(key, reaction, routes, "routing error: " + "; ".join(routes["errors"]), log)
    if not routes["input_balanced"]:
        # Elemental or charge imbalance makes a reaction free energy undefined. route_reaction has already
        # recorded the exact residual; stop before loading any species energies, including cache hits.
        return no_estimate(key, reaction, routes, routes["warnings"][-1], log)
    from metag.routing.sugar_states import pool_aware_stereochemistry
    stereo = pool_aware_stereochemistry(orig_species, rx['species'],
                                       routes.get('declared_sugar_pools', {}))
    if stereo.get('covered_by_declared_sugar_pools'):
        routes['stereochemistry'] = stereo
    if stereo["input"] or stereo["routed"]:
        routes["stereochemistry"] = stereo
        reason = ("unresolved stereochemical identity: specify each stereogenic atom/bond; "
                  "an unspecified SMILES is not a declared equilibrating stereoisomer pool")
        routes["warnings"].append(reason)
        return no_estimate(key, reaction, routes, reason, log)
    effective_pka_sites = rx.get('pka_sites', [])
    coupled_protonation = 0.0
    if reaction.get('species_protonation_models') or _flag('SOURCE_PKA_REGISTRY'):
        from metag.protonation_registry import prepare_reaction_protonation
        from metag.energetics.protonation_models import POLICY_VERSION
        try:
            effective_pka_sites, coupled_protonation, model_diagnostics, registry_report, _ = prepare_reaction_protonation(
                reaction, rx['species'], effective_pka_sites,
                routes.get('protonation_sites_by_species'), CONDITIONS,
                enabled=_flag('SOURCE_PKA_REGISTRY'), rt_ln10=RT_LN10,
                expected_registry_sha256=cfg.get('source_protonation_registry_sha256'))
        except (ValueError, TypeError, KeyError) as exc:
            reason = 'unsupported source-backed protonation model: ' + str(exc)
            routes['protonation_model_error'] = reason
            return no_estimate(key, reaction, routes, reason, log)
        routes['source_protonation_registry'] = registry_report
        if model_diagnostics is not None:
            routes['coupled_protonation_models'] = model_diagnostics
            cfg['species_protonation_model_policy'] = POLICY_VERSION
            if reaction.get('species_protonation_models'):
                cfg['species_protonation_models_sha256'] = model_diagnostics['models_sha256']
    if reaction.get('pka_sites'):
        from metag.energetics.provenance import content_hash
        cfg['caller_pka_sites_sha256'] = content_hash(reaction['pka_sites'])
    sugar_plans = {}
    if routes.get('declared_sugar_pools'):
        from metag.state_pools import prepare_sugar_pools
        try:
            sugar_plans = prepare_sugar_pools(
                rx, routes, explicit=rx['explicit'],
                caller_pka_sites=bool(reaction.get('pka_sites')),
                ring_chain=_flag('RING_CHAIN'), ring_protons=_flag('RING_PROTON_ENSEMBLE'))
            if _flag('ALDEHYDE_HYDRATION') and not _flag('CARBONYL_HYDRATION_ALL'):
                raise ValueError('Sugar pools require complete carbonyl hydration or hydration disabled')
        except (ValueError, TypeError, KeyError) as exc:
            reason = 'unsupported declared sugar ensemble: ' + str(exc)
            routes['sugar_pool_error'] = reason
            return no_estimate(key, reaction, routes, reason, log)
        routes['sugar_state_plan'] = sugar_plans
    ring_plans = {}
    if _flag('RING_PROTON_ENSEMBLE'):
        from metag.state_pools import prepare_ring_pools
        try:
            ring_plans = prepare_ring_pools(rx, routes, explicit=rx['explicit'], ring_chain=_flag('RING_CHAIN'))
            if ring_plans and _flag('ALDEHYDE_HYDRATION') and not _flag('CARBONYL_HYDRATION_ALL'):
                raise ValueError('Ring-proton pools require complete carbonyl hydration or hydration disabled')
        except (ValueError, TypeError, KeyError) as exc:
            reason = 'unsupported ring-proton ensemble: ' + str(exc)
            routes['ring_proton_error'] = reason
            return no_estimate(key, reaction, routes, reason, log)
        routes['ring_proton_state_plan'] = ring_plans
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
    # STD_STATE_1M default ON (2026-09-30): thermodynamically REQUIRED -- the gas RRHO term is at 1 atm while
    # xtb reports ΔG_solv on the 1 M(gas) -> 1 M(aq) reference and G(H+) is on 1 M, so without it every
    # reaction carries -7.93·Δn. It was default-off from 2026-09-26 because enabling it raised TECRDB MAE
    # 10.94 -> 11.26 (Δn=+1 bias -2.8 -> +4.2, Δn=-1 +0.4 -> -6.4); that residual is reported as a known
    # limitation (a missing opposite-signed per-molecule solvation term), not hidden by dropping the term.
    std = STD_STATE_KJ if _flag("STD_STATE_1M") else 0.0
    try:
        _species_free_energies(pu, rx, routes, G, sig, std, seeds, keep, pool, log, requested_explicit,
                               is_spectator_anion)
    except SpeciesRearranged as e:
        # Sampled gas relaxations could not retain the requested connectivity.
        # This is not proof that no physical minimum exists. A permitted neutral
        # reference may be retried once; its second failure is terminal.
        if _force_neutral:
            return no_estimate(key, reaction, routes,
                                f"species identity unresolved on neutral-reference route: {e}", log)
        # A requested ionic calculation must never acquire an undeclared pKa
        # model because its geometry failed. Honor the resolved route policy,
        # including explicit disabling of every neutral-reference switch.
        reference_policy = cfg["routing_policy"]
        from metag.routing.pka_transform import has_zwitterion
        # PH0_AUTO has already run with its isomerase/base gates. A failed
        # geometry is no authority to override those gates with force_base.
        # The zwitterion guard applies only to a remaining zwitterion; it
        # cannot authorize an unrelated ionic species to change reference.
        permitted = (reference_policy["neutral_qm"] or
                     (reference_policy["zwitterion_ph0"] and
                      has_zwitterion(rx["species"])))
        attempt = {"reason": str(e), "requested_species": dict(rx["species"]),
                   "neutral_reference_permitted": permitted,
                   "requested_pka_sites": rx.get("pka_sites", [])}
        routes["species_rearrangement"] = attempt
        if not permitted or rx.get("pka_sites"):
            return no_estimate(key, reaction, routes,
                              f"species identity unresolved; neutral-reference fallback disabled "
                              f"or incompatible with supplied pKa sites: {e}", log)
        log(f"  [{e} -> re-route through the neutral-microspecies path]")
        result = score_reaction(pu, reaction, seeds, keep, pool, log, allow_truncate=allow_truncate, key=key,
                                trunc_radius=trunc_radius, _validating=_validating, _force_neutral=True)
        result["routes"]["species_rearrangement"] = attempt
        return result
    if any(v is None for v in G.values()) or len(G) < len(rx["species"]):
        if truncated:                                        # a truncated core failed -> retry FULL molecules
            log(f"  [retry {key} with truncation OFF (full molecules)]")
            # re-score the ORIGINAL input dict (not a registry lookup: dict-API callers are not in REACTIONS)
            return score_reaction(pu, reaction, seeds, keep, pool, log, allow_truncate=False, key=key,
                                  trunc_radius=trunc_radius, _validating=_validating,
                                  _force_neutral=_force_neutral)
        failed = [name for name in rx["species"] if G.get(name) is None]
        routes["failed_species"] = failed
        return no_estimate(key, reaction, routes,
                            "species energy or minimum validation failed: " + ", ".join(failed), log)
    # ALDEHYDE HYDRATION (ALDEHYDE_HYDRATION, default-on, SELF-GATING): fold each hydratable aldehyde's
    # aqueous carbonyl<->gem-diol equilibrium into its effective G. The gem-diol is a +1-water microspecies
    # (parallel to protonation): G_eff = -RT ln[exp(-G_ald/RT)+exp(-(G_diol-G_water)/RT)]. Strongly hydrated
    # (glyoxylate) -> diol dominates -> G pulled down (fixed); weakly hydrated (GAP) -> unchanged (no harm).
    # No water is added to the stoichiometry (unit water activity, like H+ for protonation). See
    # scripts/aldehyde_hydration.py + memory aldehyde-hydration-signal.
    if _flag("ALDEHYDE_HYDRATION"):
        from metag.routing import aldehyde_hydration as _ah
        _gw = None
        _all_carbonyls = _flag("CARBONYL_HYDRATION_ALL")
        for name, (coeff, q, smi) in list(rx["species"].items()):
            if G.get(name) is None:
                continue
            if _all_carbonyls:
                _hydrate_all_sites(pu, rx, name, q, smi, G, sig, std, seeds, keep, pool, log, routes)
                continue
            diol = _ah.gem_diol(smi)
            if diol is None:
                continue
            if _gw is None:
                _gw = water_ref_G(pu)
            try:
                Gd, _sd = implicit_G(pu, q, diol, seeds, keep, pool, log, name + "(gem-diol)", routes["warnings"])
            except SpeciesRearranged as e:
                routes["warnings"].append(f"hydration skipped: {e}")
                routes.setdefault("incomplete_hydration", []).append(name)
                continue
            if Gd is None:
                routes.setdefault("incomplete_hydration", []).append(name)
                continue
            Gd += std                                     # G[name] already carries std; put diol + water on 1 M too
            Geff = _ah.mixture_G(G[name], Gd, _gw + std)
            log(f"    [hydration: {name} carbonyl {G[name]:.1f} + gem-diol {Gd:.1f} "
                f"(ΔG_hyd {Gd - (_gw + std) - G[name]:+.1f}) -> mixture {Geff:.1f}  shift {Geff - G[name]:+.1f}]")
            from metag.energetics.aqueous_states import mixture_sampling_uncertainty
            sig[name], uncertainty = mixture_sampling_uncertainty(
                [G[name], Gd - (_gw + std)], [sig[name], _sd], _ah.RT)
            routes.setdefault("hydration_diagnostics", {})[name] = {"sampling_uncertainty": uncertainty}
            G[name] = Geff

    # RING-CHAIN (experimental, default off; routing/ring_chain.py): fold the lactam/hemiaminal ring
    # forms of a keto/aldehyde species with an intramolecular amide N-H into its effective G (isomers: plain
    # Boltzmann sum, no water/proton). Runs AFTER hydration (HYDRATION_CAL is relative to the carbonyl form).
    if _flag("RING_CHAIN"):
        for name, (coeff, q, smi) in list(rx["species"].items()):
            if G.get(name) is not None:
                _fold_ring_chain(pu, rx, name, q, smi, G, sig, seeds, keep, pool, log, routes)

    if routes.get("incomplete_ring_chain"):
        return no_estimate(key, reaction, routes,
                            "incomplete ring-chain ensemble: " + ", ".join(sorted(set(routes["incomplete_ring_chain"]))), log)
    if routes.get("incomplete_hydration"):
        return no_estimate(key, reaction, routes,
                            "incomplete hydration ensemble: " + ", ".join(sorted(set(routes["incomplete_hydration"]))), log)
    if ring_plans:
        from metag.state_pools import apply_ring_pools
        def evaluate_ring_state(name, charge, smiles):
            detail = {'warnings': [], 'species': {}}
            routes.setdefault('ring_proton_attempts', {}).setdefault(name, []).append(
                {'smiles': smiles, 'diagnostics': detail})
            try:
                value, sigma = implicit_G(pu, charge, smiles, seeds, keep, pool, log,
                                           name + '(ring-proton)', detail['warnings'], diagnostics=detail['species'])
            except SpeciesRearranged as exc:
                raise ValueError('Ring-proton state identity unresolved: ' + str(exc)) from exc
            if value is None or sigma is None:
                return value, sigma, detail
            alt_G, alt_sig = {name: value + std}, {name: sigma}
            alt_routes = {'warnings': detail['warnings']}
            alt_rx = dict(rx, species={name: [rx['species'][name][0], charge, smiles]})
            if _flag('ALDEHYDE_HYDRATION'):
                _hydrate_all_sites(pu, alt_rx, name, charge, smiles, alt_G, alt_sig, std,
                                   seeds, keep, pool, log, alt_routes)
            detail['aqueous_states'] = alt_routes
            routes['warnings'].extend(detail['warnings'])
            if alt_routes.get('incomplete_hydration'):
                raise ValueError('Incomplete joint ring-proton/hydration state set: ' + name)
            return alt_G[name], alt_sig[name], detail
        try:
            G, sig, pool_diagnostics = apply_ring_pools(ring_plans, rx['species'], G, sig,
                                                       evaluate_state=evaluate_ring_state)
        except (ValueError, TypeError, KeyError) as exc:
            reason = 'incomplete ring-proton ensemble: ' + str(exc)
            routes['ring_proton_error'] = reason
            return no_estimate(key, reaction, routes, reason, log)
        routes['ring_proton_ensembles'] = pool_diagnostics
    if sugar_plans:
        from metag.state_pools import apply_sugar_pools
        def evaluate_sugar_state(name, charge, smiles):
            detail = {'warnings': [], 'species': {}}
            routes.setdefault('sugar_state_attempts', {}).setdefault(name, []).append(
                {'smiles':smiles, 'diagnostics':detail})
            try:
                value, sigma = implicit_G(pu, charge, smiles, seeds, keep, pool, log,
                                           name + '(sugar-state)', detail['warnings'],
                                           diagnostics=detail['species'])
            except SpeciesRearranged as exc:
                raise ValueError('Sugar state identity unresolved: ' + str(exc)) from exc
            if value is None or sigma is None:
                return value, sigma, detail
            alt_G, alt_sig = {name:value + std}, {name:sigma}
            alt_routes = {'warnings':detail['warnings']}
            alt_rx = dict(rx, species={name:[rx['species'][name][0], charge, smiles]})
            if _flag('ALDEHYDE_HYDRATION'):
                _hydrate_all_sites(pu, alt_rx, name, charge, smiles, alt_G, alt_sig, std,
                                   seeds, keep, pool, log, alt_routes)
            detail['aqueous_states'] = alt_routes
            routes['warnings'].extend(detail['warnings'])
            if alt_routes.get('incomplete_hydration'):
                raise ValueError('Incomplete joint sugar/hydration state set: ' + name)
            return alt_G[name], alt_sig[name], detail
        try:
            G, sig, sugar_diagnostics = apply_sugar_pools(
                sugar_plans, rx['species'], G, sig, evaluate_state=evaluate_sugar_state)
        except (ValueError, TypeError, KeyError) as exc:
            reason = 'incomplete declared sugar ensemble: ' + str(exc)
            routes['sugar_pool_error'] = reason
            return no_estimate(key, reaction, routes, reason, log)
        routes['sugar_ensembles'] = sugar_diagnostics
    sampling_groups = {}
    for name, (_, q, smi) in rx["species"].items():
        if name in routes.get('sugar_ensembles', {}):
            sampling_groups[name] = ('sugar_ensemble', q,
                                     routes['sugar_ensembles'][name]['state_set_sha256'])
            continue
        if name in routes.get('ring_proton_ensembles', {}):
            sampling_groups[name] = ('ring_proton_ensemble', q,
                                     routes['ring_proton_ensembles'][name]['state_set_sha256'])
            continue
        method = routes.get("species_diagnostics", {}).get(name, {}).get("method")
        # Unknown callback provenance remains independent; do not invent
        # covariance merely because molecular drawings happen to match.
        sampling_groups[name] = (method, q, _sc.canonical(smi)) if method else ("unverified", name)
    routes["sampling_uncertainty"] = {
        "policy": "shared_estimates_combined_before_quadrature",
        "n_scored_labels": len(sampling_groups), "n_sampling_groups": len(set(sampling_groups.values())),
        "scope": "sampling only; distinct species/method estimates independent; no fitted network covariance"}
    energy = assemble_reaction(rx["species"], G, sig, n_hplus=rx["n_Hplus"],
                               proton_energy=G_HPLUS, pka_sites=effective_pka_sites,
                               ph=PH, rt_ln10=RT_LN10, sampling_groups=sampling_groups,
                               coupled_protonation=coupled_protonation,
                               sampling_offsets={name:row['reference_owned_factor_kj']
                                                 for name,row in routes.get('sugar_ensembles', {}).items()})
    dG, U_samp, pka_total = energy.total, energy.sampling_sigma, energy.protonation
    if rx.get("pka_sites"):
        from metag.energetics.protonation import reaction_transform_diagnostics
        from metag.validation.protonation import inspect_species
        routes["protonation_diagnostics"] = reaction_transform_diagnostics(effective_pka_sites, PH, RT_LN10)
        if routes.get('coupled_protonation_models'):
            diagnostic = routes['protonation_diagnostics']
            diagnostic.update(model='mixed_independent_and_sequential_macroscopic',
                              independent_correction_kj=diagnostic['correction_kj'],
                              coupled_correction_kj=coupled_protonation, correction_kj=pka_total)
            diagnostic['dG_d_pH_kj_per_unit'] += model_diagnostics['reaction_dG_d_pH_kj_per_unit']
        coverage = {}
        for name, (_, _, smi) in rx["species"].items():
            if name in (routes.get('coupled_protonation_models') or {}).get('species', {}):
                coverage[name] = {'status':'source_backed_macroscopic_reference',
                    'issues':['computed_reference_ensemble_and_source_transfer_require_validation'],
                    'scope':'Whole fully protonated family; missing-state and solvent/thermal errors remain.'}
                continue
            try:
                report = inspect_species(smi)
            except (ValueError, RuntimeError) as exc:
                report = {"status": "unsupported", "issues": ["structural_diagnostic_unavailable"],
                          "error": str(exc)}
            if report["issues"] or report["status"] == "unsupported":
                coverage[name] = report
        routes["protonation_diagnostics"]["structural_coverage"] = coverage
    stages = {"species_sum": energy.species_sum, "proton": energy.proton}
    if rx.get("pka_sites"):
        model_label = 'mixed independent and macroscopic models' if routes.get('coupled_protonation_models') else 'independent sites'
        log(f"    pKa transform ({model_label}, {len(effective_pka_sites)} independent terms) += {pka_total:+.1f} kJ/mol")
    # ROBUSTNESS GUARDS (defense-in-depth). A valid balanced ΔrG'° is physically bounded and the
    # computed reaction must conserve charge with its proton bookkeeping. Two independent failure modes
    # we have actually hit -> flag `suspect` so downstream analysis DROPS the point instead of averaging
    # garbage into the MAE (and it prints loudly in the log).
    suspect = None
    #  (1) charge must close: Σ coeff·q(final species) + n_H+ == 0. A broken transform (e.g. a cation
    #      left un-neutralised) leaks ~n·G(H+) ≈ ±1170 kJ per unbalanced proton. pH-0 already refuses
    #      these, but any other path that breaks it is caught here.
    q_imbalance = sum(coeff * q for name, (coeff, q, smi) in rx["species"].items()) + rx["n_Hplus"]
    if abs(q_imbalance) > 1e-9:
        suspect = f"charge imbalance {fmt_num(q_imbalance)} (Σcoeff·q + n_H+ ≠ 0)"
        log(f"  !! SUSPECT: {suspect} -> ΔG leaks ~{fmt_num(q_imbalance)}·G(H+); result unreliable, do not trust.")
    if not routes["input_balanced"]:
        m = "input reaction not element/charge balanced"
        suspect = m if suspect is None else f"{suspect}; {m}"
    #  (2) finiteness: a NaN/inf (failed species leaking through) is never a result.
    if not math.isfinite(dG):
        m = "non-finite ΔG"
        suspect = m if suspect is None else f"{suspect}; {m}"
    #  (3) magnitude diagnostic -- a WARNING, not a rejection. ΔG is extensive (a reaction written with
    #      doubled coefficients has twice the ΔG; 4 aminophenol + 3 O2 -> 2 phenoxazinone + 6 H2O is ~ -900
    #      kJ and physically right), so a fixed ceiling rejected valid reactions. The bound scales with the
    #      reaction extent (largest non-water |coefficient|); the leaks it used to catch (unbalanced protons,
    #      stale loaders) are now caught by the balance guard and charge closure.
    extent = max([abs(c) for (c, q, s_) in rx["species"].values() if not is_water(s_, q)] or [1.0])
    _sanity = float(os.environ.get("DG_SANITY_KJ", "500"))
    if math.isfinite(dG) and abs(dG) > _sanity * extent:
        m = f"|ΔG|={abs(dG):.0f} kJ > {_sanity:.0f} kJ x extent {fmt_num(extent)}: check (not rejected)"
        routes["warnings"].append(m)
        log(f"  !! {m}")
    # ANCHOR CORRECTION (ANCHOR_CORRECT, default OFF since 2026-10-01 -- the anchors absorbed xtb-COSMO's missing
    # H-bond term; with ALPB + the experimental water reference no fitted offsets are used): the SYSTEMATIC bond-type / anion-pattern sub-classes
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
    stages["pka_transform"] = pka_total
    # dG_raw = the UNANCHORED estimate of the ROUTED reaction: Σν·G (incl. microspecies mixtures such as
    # aldehyde hydration) + n_H+·G(H+) + pKa-transform terms. It contains no parameter fitted to reaction free
    # energies, but it is not "raw QM of the input": routing (cores, truncation, neutral microspecies) has
    # already been applied. No benchmark-fitted reaction-energy correction follows.
    dG_raw = dG
    stages["unanchored"] = dG_raw
    # Historical benchmark offsets and the legacy solute-only SMD/water patches
    # are rejected by validate_production_policy before any molecular work.
    # Their archived implementations remain in frozen experiment snapshots.
    anchor_meta = smd_meta = water_meta = None
    from metag.reporting import uncertainty_diagnostics
    sigma_pred, sigma_breakdown, ci_lo, ci_hi, ci_center, ci_info = uncertainty_diagnostics(
        reaction.get('note', ''), orig_species, rx['species'], routes, stages, dG, U_samp, cfg)
    exp_out = sorted(exp_flag) if isinstance(exp_flag, (set, list, tuple)) else exp_flag
    stages.update(reported=dG)
    stages = {k: round(v, 2) for k, v in stages.items()}
    common = dict(reaction=key, anchor=anchor_meta, smd=smd_meta, water_ref=water_meta, stages=stages,
                  original_species={name: list(value) for name, value in orig_species.items()},
                  energy_units="kJ/mol", quantity="standard_transformed_reaction_gibbs", reported_rounding_kj=0.1,
                  conditions=CONDITIONS,
                  U_samp=round(U_samp, 1), exp=rx.get("exp"), explicit=exp_out, suspect=suspect,
                  experimental_reference={k: v for k, v in rx.items() if k.startswith("exp_")},
                  note=rx["note"], routes=routes, std_state_kJ=round(std, 2), config=cfg,
                  species_scored={k: list(v) for k, v in rx["species"].items()})
    if suspect is not None:
        # FAIL CLOSED: an invalid reaction (charge leak, unbalanced input, nonfinite energy) returns NO
        # estimate -- dG / dG_raw / σ / interval are None and calibration is explicitly off -- so a TFA/MDF
        # consumer cannot mistake it for a usable number. The raw value is kept only for diagnosis.
        log(f"  ΔG = NONE (suspect: {suspect}); raw value {dG_raw:+.1f} kept as dG_raw_unreliable")
        return dict(common, dG=None, dG_raw=None, dG_raw_unreliable=round(dG_raw, 1), sigma_pred=None,
                    sigma_breakdown=sigma_breakdown, ci95=[None, None], ci_center=None,
                    ci_info=dict(ci_info, coverage_calibrated=False, externally_calibrated=False,
                                 calibration_scope=f"no estimate: suspect ({suspect})"),
                    uncertainty_diagnostic=None, sign_status="no_estimate", unresolved=None, err=[])
    errs = [dG - e for e in rx.get("exp", [])]              # exp is optional (deployment has no experiment)
    # RESOLUTION heuristic: if the CALIBRATED prediction interval is comparable to |ΔG|, the sign is not
    # resolvable -- flag it (near-equilibrium isomerases are concentration-limited, not QM-fixable).
    # sign unresolved at the REPORTED confidence level exactly when the 95% interval contains zero
    from metag.validation.uncertainty_contract import prediction_uncertainty
    published = prediction_uncertainty(sigma_pred,ci_lo,ci_hi,ci_center,ci_info)
    if published['ci_info']['coverage_calibrated']:
        detail=f" ± {sigma_pred} kJ/mol"
        if published['unresolved']:
            detail+=" [calibrated interval includes zero]"
    else:
        detail=f" kJ/mol [prediction interval unavailable; diagnostic scale {sigma_pred}, sampling spread {U_samp:.1f}]"
    log(f"  ΔG = {dG:+.1f} (raw {dG_raw:+.1f}){detail}"
        f"   vs exp {rx.get('exp')}   err {[round(e,1) for e in errs]}")
    return dict(common,dG=round(dG,1),dG_raw=round(dG_raw,1),sigma_breakdown=sigma_breakdown,
                **published,err=[round(e,1) for e in errs])

def _scored_structure_signature(result):
    """Canonical net species equation, independent of labels and SMILES spelling."""
    from collections import defaultdict
    species = result.get("species_scored")
    if not species:
        return None
    net = defaultdict(float)
    for coefficient, charge, smiles in species.values():
        key = (int(charge), Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
        net[key] += coefficient
    return tuple(sorted((q, smi, c) for (q, smi), c in net.items() if c))


def _score_trunc_validated(pu, reaction, seeds, keep, pool, log, key, trunc_radius):
    """TRUNC_VALIDATE guard (see score_reaction): accept a truncation only if ΔG is radius-invariant."""
    R = int(trunc_radius if trunc_radius is not None else os.environ.get("TRUNC_RADIUS", "2"))
    tol = float(os.environ.get("TRUNC_VALIDATE_TOL", "5"))
    kw = dict(seeds=seeds, keep=keep, pool=pool, log=log, key=key, _validating=True)
    rR = score_reaction(pu, reaction, trunc_radius=R, **kw)
    if rR is None or not rR["routes"]["truncated"]:
        return rR                                            # no truncation fired: nothing to validate
    rR1 = score_reaction(pu, reaction, trunc_radius=R + 1, **kw)
    info = {"radius": R, "dG_raw_R": rR["dG_raw"], "tol": tol}
    if rR1 is not None and rR1["routes"]["truncated"] and rR1["suspect"] is None and rR["suspect"] is None:
        d = abs(rR["dG_raw"] - rR1["dG_raw"])
        noise = 2.0 * math.hypot(rR["U_samp"], rR1["U_samp"])
        thr = tol
        info.update(sampling_noise_2sigma=noise, criterion="delta <= tolerance and sampling noise <= tolerance")
        info.update(dG_raw_R1=rR1["dG_raw"], delta=round(d, 1), threshold=round(thr, 1))
        signature = _scored_structure_signature(rR)
        other_signature = _scored_structure_signature(rR1)
        distinct = signature is not None and other_signature is not None and signature != other_signature
        info["distinct_radius_models"] = distinct
        if distinct and np.isfinite(d) and np.isfinite(noise) and d <= thr and noise <= tol:
            log(f"  [trunc-validate: radius {R} vs {R+1} agree (|Δ|={d:.1f} <= {thr:.1f}) -> keep truncation]")
            rR["trunc_validation"] = dict(info, verdict="accepted")
            return rR
        if not np.isfinite(noise) or noise > tol:
            verdict = "rejected: sampling unresolved"
        elif not distinct:
            verdict = "rejected: radius comparison did not change the molecular model"
        else:
            verdict = "rejected: radius-sensitive"
    else:
        verdict = "rejected: radius R+1 truncation unavailable"
    log(f"  [trunc-validate: {verdict} -> generic truncation disabled]")
    rF = score_reaction(pu, reaction, allow_truncate=False, **kw)
    if rF is not None:
        core_routes = [k for k in ("cofactor_ring", "ntp_core", "coa_core") if rF["routes"].get(k)]
        rF["trunc_validation"] = dict(info, verdict=verdict, remaining_core_routes=core_routes,
                                      fallback_scope="generic truncation disabled; core routes reported separately")
    return rF


def run_reaction(pu, key, seeds, keep, pool, log, allow_truncate=True):
    """Harness wrapper: look a reaction up by key in the loaded REACTIONS file and score it."""
    from metag import profile
    profile.reset()
    r = score_reaction(pu, REACTIONS[key], seeds, keep, pool, log, allow_truncate, key=key)
    profile.report(log=log, header=f"reaction {key}")   # Step-0 GPU-vs-CPU split (METAG_PROFILE=1)
    return r


def main():
    """Compatibility CLI, preserving monkeypatchable pipeline entry points."""
    from metag.cli import run_cli
    return run_cli(reactions=REACTIONS, model_name=_MODEL, load_model=load_uma,
                   score=run_reaction, no_estimate=_no_estimate, out=OUT)


if __name__ == "__main__":
    main()
