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
  4. guards (charge closure, |ΔG| sanity; fail closed) -> σ and 95% interval (coverage-calibrated on TECRDB
     nested CV, or nominal). No fitted anchors (ANCHOR_CORRECT off) and no water patches.

CLI harness (reactions from RXN_FILE; launch through gpu_reserve, never set CUDA_VISIBLE_DEVICES by hand):
  gpu_reserve run <idx> -- python -m metag.pipeline --only <key>
"""
import argparse
import functools
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

from metag.chem import is_water, is_balanced, reaction_residual, fmt_num
from metag.reactions import validate_reaction
from metag.energetics import water_clusters as gc
from metag.energetics.uma import load_uma, batched_energies, batched_fire
from metag.energetics.conformers import (pool_confs, boltz, spin_multiplicity, UniqueMinima,
                                         bond_graph, same_connectivity)
from metag.energetics.explicit_solvation import bare_geom
import metag.energetics.thermal as thermal_mod
from metag.energetics.thermal import uma_gibbs_corr, xtb_dgsolv, xtb_dgsolv_relaxed, corr_fast, dgsolv
from metag.water_count import water_count, needs_explicit

N_EXPLICIT_SEEDS = int(os.environ.get("N_EXPLICIT_SEEDS", "16"))  # cluster seeds (cheap: batched relax)
EXPLICIT_KEEP = int(os.environ.get("EXPLICIT_KEEP", "8"))         # lowest-E clusters kept for Boltzmann

OUT = os.environ.get("METAG_OUT", os.path.join(os.getcwd(), "metag_out"))  # CLI output dir (configurable)
EV2KJ = 96.485
T = 298.15
# CHE aqueous proton free energy at pH 7 (step3b/step6 convention)
G_HPLUS = -26.3 - 1104.5 - 2.303 * 8.314e-3 * T * 7.0    # ~ -1170.8 kJ/mol
PH = 7.0
# The thermodynamic conditions every MetaG number refers to (returned with each result and part of the
# configuration fingerprint). Anything compared against MetaG -- experiment or another estimator -- must be
# transformed to these conditions first; pMg=None = no Mg2+ binding (no Mg-complex microspecies).
CONDITIONS = {"T_K": 298.15, "pH": PH, "ionic_strength_M": 0.0, "pMg": None, "water_activity": 1.0,
              "standard_state": "solutes 1 M; pure liquid water; H+ at the stated pH (transformed ΔrG'°)"}
RT_LN10 = 2.303 * 8.314e-3 * T                            # ~5.71 kJ/mol per pKa unit
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
    sc = _SAMPLE_SCALE                                    # read once at import, same value as the cache key
    if nrot <= 3:
        seeds, keep, pool = [1, 2], 10, 96
    elif nrot <= 7:
        seeds, keep, pool = [1, 2, 3], 14, 192
    else:
        seeds, keep, pool = [1, 2, 3, 4, 5, 6], 18, 320
    if sc != 1:
        seeds = list(range(1, max(2, int(round(len(seeds) * sc))) + 1))
        pool = int(round(pool * sc))
    # KEEP_SCALE (A/B, default 1): the physics review found `keep` too small for floppy sugars/cofactors, so
    # the Boltzmann ensemble under-counts conformational entropy (G too high). SAMPLE_SCALE scaled seeds+pool
    # but NOT keep; this scales how many low-E conformers are relaxed + solvated. Part of the cache key.
    if _KEEP_SCALE != 1:
        keep = int(round(keep * _KEEP_SCALE)); pool = max(pool, keep * 8)
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
from metag.energetics.thermal import qrrho_enabled
from metag.energetics.conformers import DEDUP_E_TOL
_MODEL = os.environ.get("UMA_MODEL", "uma-s-1p2p1")   # patch model (batched_relax._ensure_registered); in the cache key. UMA_MODEL overrides for A/B (e.g. uma-s-1p2)
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
PHYSICS_VERSION = "2026-10-01c"   # 10-01c: dedup representative = geometry of its E/G; thermal resolution applied to the ensemble
_IMPLICIT_SETTINGS = {"model": _MODEL, "solv": SOLV_MODEL, "budget": "nrot-tiered-v1",
                      "conv_tol": CONV_TOL, "conv_hits": CONV_HITS, "conv_max": CONV_MAX,
                      "sample_scale": _SAMPLE_SCALE, "physics": PHYSICS_VERSION, "qrrho": qrrho_enabled(),
                      **({"keep_scale": _KEEP_SCALE} if _KEEP_SCALE != 1 else {})}
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
    _IMPLICIT_SETTINGS["solv_relax"] = f"v1-n{SOLV_RELAX_N}-w{SOLV_RELAX_WIN:g}-s{SOLV_RELAX_STEPS}"
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
            "water_dgsolv_kj": float(os.environ.get("WATER_DGSOLV_KJ", "-26.4"))}
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


class SpeciesRearranged(Exception):
    """Every relaxed conformer of a species changed covalent connectivity (e.g. zwitterion proton transfer):
    the requested species is not a gas-phase minimum. Never scored or cached under its SMILES; score_reaction
    re-routes the reaction through the neutral-microspecies path (the transfer is then accounted for by pKa
    terms) or fails it."""
    def __init__(self, name, smi, n):
        super().__init__(f"{name} ({smi}): bonding changed during relaxation in all {n} conformers")
        self.smi = smi


def _model_name(pu):
    """The UMA model actually loaded (load_uma tags the predict unit). Refuses a mismatch with UMA_MODEL,
    which keys the species cache: scoring with one model under another model's key poisons the cache."""
    name = getattr(pu, "metag_model", None)
    if name is not None and name != _MODEL:
        raise ValueError(f"loaded UMA model {name!r} != UMA_MODEL {_MODEL!r} (the species-cache key); "
                         f"set UMA_MODEL={name} or load_uma({_MODEL!r})")
    return name or _MODEL


def _last_match(uniq, atoms, E_kJ):
    """Index of the unique minimum that `atoms` duplicates (same test UniqueMinima.add used), or None."""
    rep = uniq._mol_at(atoms) if uniq.template is not None else None
    for j in range(len(uniq.E)):
        if rep is not None and uniq._same(E_kJ, rep, j):
            return j
    return None


def implicit_G(pu, q, smi, seeds, keep, pool, log, name, warnings=None):
    """Boltzmann over unique minima of G_i = E_elec[UMA]_i + ΔGsolv[SOLV_MODEL]_i + Gcorr[UMA RRHO]_i
    (THERMAL_ENSEMBLE, default; Gcorr per minimum within THERMAL_ENS_WIN). With THERMAL_ENSEMBLE=0 only
    E + ΔGsolv is averaged and one RRHO correction (lowest-G true minimum) is added.

    HEURISTIC (general, self-calibrating -- no fixed per-flexibility tiers, no per-reaction
    tuning): keep adding conformer seed-batches until BOTH the Boltzmann Gens AND the minimum
    energy stop moving (< CONV_TOL for CONV_HITS consecutive batches), capped at CONV_MAX.
    Rigid species converge in ~2-3 batches; floppy sugar-phosphates draw as many as they need.
    Per-batch pool/keep still scale with rotatable bonds (bigger search for floppier molecules).
    Reports the seed count + the last increment so the sampling uncertainty is visible (UQ).

    Species-level problems (bonding changed during relaxation, xtb failures) are appended to
    `warnings` and stored with the cached value, so a cache hit reports them too."""
    warnings = warnings if warnings is not None else []
    mult = spin_multiplicity(smi, q)                      # ground-state spin (O2 triplet, radicals doublet)
    # spin is deterministic in (smi,q); fork the cache key ONLY for open-shell species so every
    # closed-shell singlet key is preserved (no cache bust) and pre-fix O2 singlet entries are ignored.
    _settings = dict(_IMPLICIT_SETTINGS, model=_model_name(pu))
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
    while seed < CONV_MAX:
        seed += 1
        cands = pool_confs(smi, q, seed, pool, spin=mult)
        order = np.argsort(batched_energies(pu, cands))[:keep]
        sel = [cands[i] for i in order]
        rel, E, conv = batched_fire(pu, sel, fmax=0.05, steps=300, stop_frac=0.9,
                                    return_converged=True, label=f"{name}s{seed}")
        sel = [a for a, c in zip(rel, conv) if c]; Eg = E[conv] * EV2KJ
        models = [SOLV_MODEL] + SOLV_ALSO
        with ThreadPoolExecutor(max_workers=8) as ex:
            solv = list(ex.map(lambda am: dgsolv(am[0].get_chemical_symbols(), am[0].get_positions(), q,
                                                 am[1], mult), [(a, m) for a in sel for m in models]))
        solv = [dict(zip(models, solv[i * len(models):(i + 1) * len(models)])) for i in range(len(sel))]
        for a, sd in zip(sel, solv):                      # one retry: most xtb failures are transient
            if sd[SOLV_MODEL] is None:
                sd[SOLV_MODEL] = dgsolv(a.get_chemical_symbols(), a.get_positions(), q, SOLV_MODEL, mult)
        for a, e, sd in zip(sel, Eg, solv):
            n_relaxed += 1
            s = sd[SOLV_MODEL]
            if s is None:
                n_xtb_fail += 1
                if np.isfinite(e):
                    failed_E.append(float(e))
            if not np.isfinite(e) or s is None:
                continue
            if not same_connectivity(a, ref_graph):
                rearranged.append((a, float(e), sd))
                continue
            if hb_groups is not None and _has_interacid_hbond(a, hb_groups):
                hb_set_aside.append((a, float(e), sd))
                continue
            _add_minimum(uniq, also, a, float(e), sd)
            if e < best[0]:
                best = (float(e), a.get_chemical_symbols(), a.get_positions())
        if not all_G:
            continue
        Gens = boltz(all_G)
        if thermal_track is not None:                      # convergence / U_samp on E + ΔGsolv + RRHO_i
            Gens = thermal_track.update(uniq)
        gens_traj.append(Gens)
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
    if _SOLV_RELAX and uniq.template is not None and uniq.ref:
        relaxed = _solvent_relaxed_ensemble(pu, uniq, q, mult, ref_graph, template, name, log)
        if relaxed is not None:
            G_aq, also_aq, relax_info = relaxed
            log(f"    {name}: solution-phase relaxation Gens {Gens:.1f} -> {boltz(G_aq):.1f} "
                f"({relax_info['n_relaxed']} minima)")
            Gens = boltz(G_aq)
            also = also_aq
    therm, t_info, t_res = _thermal_at_minimum(pu, uniq, best, template, ref_graph, q, mult, name, log)
    ref_j = None
    if therm is not None and uniq.template is not None and uniq.ref:
        ref_j = _apply_thermal_resolution(pu, uniq, also, t_res, template, q, mult, name, log)
        Gens = boltz(uniq.G)                              # ensemble after removing non-minima / updating geometry
        t_info = dict(t_info, resolution=t_res["kind"], n_removed=len(t_res["rejected"]))
    if therm is not None and thermal_track is not None and ref_j is not None:
        Gens_th, also_th, ens_info = thermal_track.final(uniq, also, therm, ref_j)
        log(f"    {name}: thermal ensemble (Gens+thermal) {Gens + therm:.1f} -> {Gens_th:.1f} "
            f"({ens_info['n_hessians']} minima)")
        t_info = dict(t_info, ensemble=ens_info, single_minimum_G=round(Gens + therm, 3))
        Gens, also, therm = Gens_th, also_th, 0.0         # thermal now inside every ensemble member
    if therm is None:
        sp_warn.append(f"{name} ({smi}): no true minimum among the lowest {N_THERMAL_CANDIDATES} "
                       f"(imaginary modes {t_info}) -> species failed")
        log(f"    !! {sp_warn[-1]}")
        warnings.extend(sp_warn)
        return None, None
    # sampling uncertainty: spread of Gens over the last few batches (0 if never moved / capped-tight)
    tail = gens_traj[-3:]
    sigma = float(np.std(tail)) if len(tail) > 1 else (last_dG if np.isfinite(last_dG) else 3.0)
    tag = "conv" if hits >= CONV_HITS else "CAPPED"
    log(f"    {name:9s} q{q:+d} [implicit {tag} seeds={seed} minima={len(uniq)}/{uniq.n_seen} σ={sigma:.1f}]: "
        f"Gens {Gens:.1f} + thermal {therm:.1f} = {Gens+therm:.1f}")
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
    if fail_frac > MAX_XTB_FAIL_FRAC:                     # degraded ensemble: use it, but never cache it
        sp_warn.append(f"{name} ({smi}): xtb solvation failed on {n_xtb_fail}/{n_relaxed} conformers "
                       f"-> G not cached")
        log(f"    !! {sp_warn[-1]}")
    else:
        meta = {"warnings": sp_warn, "n_minima": len(uniq), "n_seen": uniq.n_seen,
                "n_xtb_fail": n_xtb_fail, "n_rearranged": len(rearranged), "seeds": seed, "thermal": t_info,
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
    return -sum(RT_LN10 * math.log10(1.0 + 10.0 ** (PH - p)) for p in (pkas or []))


def _hydrate_all_sites(pu, rx, name, q, smi, G, sig, std, seeds, keep, pool, log, routes):
    """CARBONYL_HYDRATION_ALL: fold the carbonyl <-> gem-diol equilibria of species `name` into G[name] by
    EXACT enumeration of its hydration microstates (every subset of its hydratable sites, each computed).
    On the pH-0 route each state carries its own acid transform (a gem-diol of an alpha-oxo acid is a weaker
    acid) before mixing; the carbonyl form's transform, which the reaction-level pKa sites already apply,
    is removed again. HYDRATION_CAL maps each hydration event with the K_hyd calibration."""
    from metag.routing import aldehyde_hydration as _ah
    states = _ah.hydration_states(smi)
    n_expected = 2 ** min(_ah.n_hydration_sites(smi), _ah.MAX_HYDRATION_SITES) - 1
    if len(states) < n_expected:
        routes["warnings"].append(f"{name}: {n_expected - len(states)} hydration state(s) not constructible")
    if not states:
        return
    if _ah.n_hydration_sites(smi) > _ah.MAX_HYDRATION_SITES:
        routes["warnings"].append(f"{name}: >{_ah.MAX_HYDRATION_SITES} hydratable sites, first "
                                  f"{_ah.MAX_HYDRATION_SITES} (canonical order) enumerated")
    on_ph0 = bool(rx.get("pka_sites")) and q == 0
    t_c = _acid_transform(smi) if on_ph0 else 0.0
    g_w = water_ref_G(pu) + std
    folded, sds = [], []
    for n, hyd in states:
        try:
            Gd, sd = implicit_G(pu, q, hyd, seeds, keep, pool, log, f"{name}(hydrate{n})", routes["warnings"])
        except SpeciesRearranged as e:
            routes["warnings"].append(f"hydration state omitted (not a minimum): {e}")
            continue
        if Gd is None:
            routes["warnings"].append(f"hydration state omitted (QM failed): {name} {hyd}")
            continue
        dg = Gd + std - n * g_w - G[name]                 # hydration free energy of this state (neutral form)
        if _flag("HYDRATION_CAL"):
            a, b = _ah.HYDRATION_CAL
            dg = a * dg + n * b                           # calibration is per hydration event
        t_d = _acid_transform(hyd) if on_ph0 else 0.0
        folded.append((n, G[name] + n * g_w + dg + t_d - t_c))
        sds.append(sd or 0.0)
    if not folded:
        return
    Geff = _ah.mixture_G_states(G[name], folded, g_w)
    log(f"    [hydration: {name} {len(folded)} state(s) -> shift {Geff - G[name]:+.1f}]")
    G[name] = Geff
    sig[name] = float(np.sqrt(sig.get(name, 0.0) ** 2 + sum(x * x for x in sds)))


def _acid_groups(template):
    """{atom idx: group id} for the acidic O-H hydrogens' oxygens and all oxygens of every acid group
    (each P atom and each carboxyl C is its own group) of an H-explicit RDKit molecule."""
    group_of_o, donors = {}, []
    # a P-O-P chain (pyro/triphosphate) is ONE acid group: the P-OH...O=P contacts between neighbouring
    # phosphoryls of a chain are intrinsic to it; only contacts between DISTINCT acid units are excluded.
    chain = {}
    for a in template.GetAtoms():
        if a.GetSymbol() == "P" and a.GetIdx() not in chain:
            stack, comp = [a.GetIdx()], []
            while stack:
                x = stack.pop()
                if x in chain:
                    continue
                chain[x] = a.GetIdx(); comp.append(x)
                for o in template.GetAtomWithIdx(x).GetNeighbors():
                    for y in o.GetNeighbors():
                        if y.GetSymbol() == "P" and y.GetIdx() not in chain:
                            stack.append(y.GetIdx())
    for a in template.GetAtoms():
        if a.GetSymbol() == "P" or (a.GetSymbol() == "C" and _is_carboxyl(a)):
            gid = chain.get(a.GetIdx(), a.GetIdx())
            for o in a.GetNeighbors():
                if o.GetSymbol() != "O":
                    continue
                if a.GetSymbol() == "P" and any(n.GetSymbol() == "P" and n.GetIdx() != a.GetIdx()
                                                 for n in o.GetNeighbors()):
                    continue                                 # bridging P-O-P oxygen: shared, skip
                group_of_o[o.GetIdx()] = gid
                for h in o.GetNeighbors():
                    if h.GetSymbol() == "H":
                        donors.append((h.GetIdx(), gid))
    return group_of_o, donors


def _is_carboxyl(c):
    """Carboxylic acid carbon: one C=O and one O-H oxygen (not a hemiacetal/ester carbon)."""
    os_ = [n for n in c.GetNeighbors() if n.GetSymbol() == "O"]
    if len(os_) != 2:
        return False
    mol = c.GetOwningMol()
    dbl = [o for o in os_ if mol.GetBondBetweenAtoms(c.GetIdx(), o.GetIdx()).GetBondTypeAsDouble() == 2]
    oh = [o for o in os_ if any(h.GetSymbol() == "H" for h in o.GetNeighbors())]
    return len(dbl) == 1 and len(oh) == 1


def _has_interacid_hbond(atoms, groups):
    group_of_o, donors = groups
    pos = atoms.get_positions()
    for h, g in donors:
        for o, go in group_of_o.items():
            if go != g and np.linalg.norm(pos[h] - pos[o]) < ACID_HB_DIST:
                return True
    return False


def _solvent_relaxed_ensemble(pu, uniq, q, mult, ref_graph, template, name, log):
    """Re-relax the lowest unique minima on E_UMA + ΔG_solv(SOLV_MODEL); returns (G_aq list of unique
    solution-phase minima, {aux model: values}, info) or None if no relaxed structure survives (the caller
    then keeps the vertical ensemble). Relaxed structures that change bonding are dropped; two starting
    minima that fall into the same solution-phase basin are merged (UniqueMinima on the relaxed set)."""
    from metag.energetics.solv_relax import make_extra_forces
    order = sorted(range(len(uniq.G)), key=lambda j: uniq.G[j])
    pick = [j for j in order if uniq.G[j] - uniq.G[order[0]] < SOLV_RELAX_WIN][:SOLV_RELAX_N]
    syms = [a.GetSymbol() for a in template.GetAtoms()]
    ats = [Atoms(symbols=syms, positions=uniq.ref[j].GetConformer().GetPositions(),
                 info={"charge": int(q), "spin": int(mult)}) for j in pick]
    extra = make_extra_forces(q, mult, SOLV_MODEL)
    rel, E, conv = batched_fire(pu, ats, fmax=0.05, steps=SOLV_RELAX_STEPS, stop_frac=1.0,
                                return_converged=True, extra_forces=extra, label=f"{name}-aq")
    models = [SOLV_MODEL] + SOLV_ALSO
    with ThreadPoolExecutor(max_workers=8) as ex:
        solv = list(ex.map(lambda am: dgsolv(am[0].get_chemical_symbols(), am[0].get_positions(), q,
                                             am[1], mult), [(a, m) for a in rel for m in models]))
    solv = [dict(zip(models, solv[i * len(models):(i + 1) * len(models)])) for i in range(len(rel))]
    uq = UniqueMinima(template=template)
    also = {m: [] for m in SOLV_ALSO}
    n_bond = n_fail = 0
    for i, (a, e, sd) in enumerate(zip(rel, E, solv)):
        if i in extra.failed or sd[SOLV_MODEL] is None or not np.isfinite(e):
            n_fail += 1; continue
        if not same_connectivity(a, ref_graph):
            n_bond += 1; continue
        _add_minimum(uq, also, a, float(e) * EV2KJ, sd)
    if not uq.G:
        return None
    info = {"n_start": len(pick), "n_relaxed": len(uq.G), "n_unconverged": int((~np.asarray(conv)).sum()),
            "n_rearranged": n_bond, "n_failed": n_fail}
    return list(uq.G), also, info


class _ThermalTrack:
    """Per-minimum RRHO bookkeeping DURING sampling (THERMAL_ENSEMBLE): after every seed batch the unique
    minima within THERMAL_ENS_WIN kJ of the lowest E+ΔGsolv (at most THERMAL_ENS_N) get a UMA-Hessian RRHO
    correction, so the convergence test and the sampling-uncertainty trajectory see G = E + ΔGsolv + Gcorr.
    Corrections are keyed by the representative GEOMETRY (not the list index), so a representative replaced
    by a lower-G duplicate (UniqueMinima.add updates E, G and geometry together) gets a new Hessian, and
    removing / adding minima cannot misassign a correction. Minima outside the window carry the lowest
    window member's correction (negligible Boltzmann weight). Imaginary modes are floored as soft here; the
    VALIDATED correction from _thermal_at_minimum replaces the value of the minimum it belongs to in final()."""

    def __init__(self, pu, template, q, mult):
        self.pu, self.q, self.mult = pu, q, mult
        self.syms = [a.GetSymbol() for a in template.GetAtoms()]
        self.corr = {}                                     # geometry key -> Gcorr
        self.n_hessians = 0

    @staticmethod
    def _key(uniq, j):
        return hash(np.round(uniq.ref[j].GetConformer().GetPositions(), 4).tobytes())

    def _ensure(self, uniq, j):
        k = self._key(uniq, j)
        if k not in self.corr:
            pos = uniq.ref[j].GetConformer().GetPositions()
            g, _ = uma_gibbs_corr(self.pu, self.syms, pos, self.q, spin=self.mult, return_info=True,
                                  imag_as_soft=True)
            self.corr[k] = g; self.n_hessians += 1
        return self.corr[k]

    def _totals(self, uniq, override=None):
        order = sorted(range(len(uniq.G)), key=lambda j: uniq.G[j])
        win = [j for j in order if uniq.G[j] - uniq.G[order[0]] < THERMAL_ENS_WIN][:THERMAL_ENS_N]
        if override is not None and override[0] not in win:
            win.append(override[0])
        c = {}
        for j in win:
            c[j] = override[1] if (override is not None and j == override[0]) else self._ensure(uniq, j)
        ref = c[order[0]]
        return [uniq.G[j] + c.get(j, ref) for j in range(len(uniq.G))], c, order, win

    def update(self, uniq):
        tot, _, _, _ = self._totals(uniq)
        return boltz(tot)

    def final(self, uniq, also, therm_valid, j_valid):
        """therm_valid = validated correction of minimum j_valid (the structure _thermal_at_minimum accepted)."""
        tot, c, order, win = self._totals(uniq, override=(j_valid, therm_valid))
        ref = c[order[0]]
        also_tot = {m: [None if v is None else v + c.get(j, ref) for j, v in enumerate(vals)]
                    for m, vals in also.items()}
        spread = [c[j] - c[order[0]] for j in win]
        info = {"n_window": len(win), "n_hessians": self.n_hessians, "tracked_in_convergence": True,
                "validated_index_is_lowest": bool(j_valid == order[0]),
                "gcorr_spread_kj": round(float(max(spread) - min(spread)), 2) if spread else 0.0}
        return boltz(tot), also_tot, info


def _thermal_ensemble(pu, uniq, also, therm_ref, q, mult, template, name, log):
    """Per-minimum RRHO: Gcorr_i for the unique minima within THERMAL_ENS_WIN of the lowest G (at most
    THERMAL_ENS_N); minima outside the window keep the reference correction therm_ref (negligible weight).
    Imaginary modes above the tolerance are floored as soft modes here (a per-minimum saddle search would
    exceed the purpose of this test); their number is recorded."""
    syms = [a.GetSymbol() for a in template.GetAtoms()]
    order = sorted(range(len(uniq.G)), key=lambda j: uniq.G[j])
    win = [j for j in order if uniq.G[j] - uniq.G[order[0]] < THERMAL_ENS_WIN][:THERMAL_ENS_N]
    corr = {j: therm_ref for j in range(len(uniq.G))}
    n_imag = 0
    for j in win:
        pos = uniq.ref[j].GetConformer().GetPositions()
        g, info = uma_gibbs_corr(pu, syms, pos, q, spin=mult, return_info=True, imag_as_soft=True)
        corr[j] = g
        n_imag += int(info.get("max_imag_cm", 0.0) > thermal_mod.IMAG_TOL_CM)
    G_tot = [uniq.G[j] + corr[j] for j in range(len(uniq.G))]
    also_tot = {m: [None if v is None else v + corr[j] for j, v in enumerate(vals)] for m, vals in also.items()}
    spread = [corr[j] - corr[order[0]] for j in win]
    info = {"n_hessians": len(win), "n_imag_floored": n_imag,
            "gcorr_spread_kj": round(float(max(spread) - min(spread)), 2) if spread else 0.0}
    return boltz(G_tot), also_tot, info


def _thermal_at_minimum(pu, uniq, best, template, ref_graph, q, mult, name, log):
    """RRHO correction at the lowest-G unique minimum that is a TRUE minimum. Candidates are the unique
    minima in order of G; one with an imaginary mode above thermal.IMAG_TOL_CM is re-optimised once more
    tightly (fmax 0.01 eV/Å); if still not a minimum, the strongest imaginary mode is followed; else the next
    candidate is tried. Returns (Gcorr, info, resolution) or (None, per-candidate info, resolution).

    resolution = {"index": accepted minimum index or None, "kind": "as_is" | "artefact" | "retightened" |
    "mode_followed" | None, "pos": geometry the correction belongs to, "rejected": [indices that are not
    minima]}. The caller (_apply_thermal_resolution) makes E, ΔG_solv, geometry and Gcorr refer to the SAME
    structure: retightened -> the minimum's representative is replaced by the tightened geometry;
    mode_followed -> the saddle is removed and the lower structure enters the ensemble; rejected -> removed."""
    syms = [a.GetSymbol() for a in template.GetAtoms()]
    if uniq.template is not None and uniq.ref:
        order = sorted(range(len(uniq.G)), key=lambda j: uniq.G[j])[:N_THERMAL_CANDIDATES]
        cands = [(j, uniq.ref[j].GetConformer().GetPositions()) for j in order]
    else:                                                 # legacy never-merge / moments mode
        cands = [(None, best[2])]
    tried, rejected = [], []
    for j, pos0 in cands:
        pos = pos0
        for attempt in (0, 1):
            g, info = uma_gibbs_corr(pu, syms, pos, q, spin=mult, return_info=True)
            vecs = info.pop("_imag_vecs")
            if info["n_imag"] == 0:
                info["retightened"] = bool(attempt)
                return g, info, {"index": j, "kind": "retightened" if attempt else "as_is", "pos": pos,
                                 "rejected": rejected}
            if attempt == 1:
                verdict, new_pos = _follow_imaginary_mode(pu, syms, pos, vecs[0], q, mult, ref_graph, name)
                if verdict == "artefact":
                    g_soft, info_soft = uma_gibbs_corr(pu, syms, pos, q, spin=mult, return_info=True,
                                                       imag_as_soft=True)
                    info_soft.pop("_imag_vecs")
                    info_soft.update(retightened=True, imag_artefact_cm=info["max_imag_cm"])
                    kind = "retightened" if pos is not pos0 else "artefact"
                    return g_soft, info_soft, {"index": j, "kind": kind, "pos": pos, "rejected": rejected}
                if verdict == "saddle":
                    g2, info2 = uma_gibbs_corr(pu, syms, new_pos, q, spin=mult, return_info=True)
                    info2.pop("_imag_vecs")
                    if info2["n_imag"] == 0:
                        info2.update(retightened=True, mode_followed=True)
                        return g2, info2, {"index": j, "kind": "mode_followed", "pos": new_pos,
                                           "rejected": rejected}
            tried.append(info["max_imag_cm"])
            if attempt == 0:                              # re-optimise tightly once, keep only if unrearranged
                at = Atoms(symbols=syms, positions=pos, info={"charge": int(q), "spin": int(mult)})
                (at,), _, conv = batched_fire(pu, [at], fmax=0.01, steps=400, return_converged=True,
                                              label=f"{name}-tight")
                if not same_connectivity(at, ref_graph):
                    break
                if conv[0]:                               # an unconverged tight re-opt (flat torsions) still
                    pos = at.get_positions()              # gets the mode-following test from the original pos
        if j is not None:
            rejected.append(j)                            # not a minimum: excluded from the ensemble
        log(f"    {name}: minimum candidate has an imaginary mode ({tried[-1]} cm-1) -> next candidate")
    return None, tried, {"index": None, "kind": None, "pos": None, "rejected": rejected}


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
        if sd[SOLV_MODEL] is None:
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


def _follow_imaginary_mode(pu, syms, pos, mode, q, mult, ref_graph, name, step=0.1, drop_kJ=1.0):
    """('saddle', lower_pos) if relaxing from pos +- step·mode reaches an energy > drop_kJ below pos (same
    connectivity), else ('artefact', None)."""
    at0 = Atoms(symbols=syms, positions=pos, info={"charge": int(q), "spin": int(mult)})
    e0 = float(batched_energies(pu, [at0])[0]) * EV2KJ
    disp = [Atoms(symbols=syms, positions=pos + sgn * step * mode, info={"charge": int(q), "spin": int(mult)})
            for sgn in (+1.0, -1.0)]
    rel, E, conv = batched_fire(pu, disp, fmax=0.01, steps=400, return_converged=True, label=f"{name}-mode")
    best = None
    for a, e, c in zip(rel, E, conv):
        e = float(e) * EV2KJ
        if c and same_connectivity(a, ref_graph) and e < e0 - drop_kJ and (best is None or e < best[0]):
            best = (e, a.get_positions())
    return ("saddle", best[1]) if best else ("artefact", None)


def _add_minimum(uniq, also, atoms, e, sd):
    """Register one relaxed, solvated conformer in the unique-minima set (primary SOLV_MODEL) and keep the
    secondary SOLV_ALSO values aligned per minimum. A secondary model that failed on this conformer leaves
    None -- it never removes the conformer from the PRIMARY ensemble."""
    if uniq.add(atoms, e, e + sd[SOLV_MODEL]):            # new minimum
        for m in also:
            also[m].append(None if sd[m] is None else e + sd[m])
        return
    j = _last_match(uniq, atoms, e)                       # duplicate: keep the lower G per model too
    if j is None:
        return
    for m in also:
        if sd[m] is not None:
            also[m][j] = e + sd[m] if also[m][j] is None else min(also[m][j], e + sd[m])


def _complete_auxiliary_values(values):
    """Return a complete auxiliary conformer ensemble, or None if any accepted minimum is missing."""
    return values if values and all(v is not None for v in values) else None


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
    memo_key = (_model_name(pu), SOLV_MODEL, use_exp, _WSOLV_EXP, qrrho_enabled())
    if memo_key in _WATER_REF:
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
    _WATER_REF[memo_key] = G
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
    _settings = dict(_explicit_settings(), model=_model_name(pu))
    _cached = _sc.get(smi, q, "explicit", _settings)
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
    _uq = UniqueMinima()                                  # merge seeds that relaxed to the same cluster
    for a, e, s in zip(sel, Eu, solv):
        if s is not None:
            _uq.add(a, e, e + s)
    Gt = _uq.G
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
    _sc.put(smi, q, "explicit", _settings, g, sigma)
    return g, sigma


# ONE table of pipeline switches and their production defaults (the calibrated configuration). Every
# switch is read through _flag(), so the default lives here only and effective_config() can report it.
FLAG_DEFAULTS = {
    "COFACTOR_RING": True, "COA_CORE": False,
    # NTP_CORE ADOPTED 2026-10-01: the nucleoside of a spectator NTP/NDP/NMP is capped to a methyl
    # polyphosphate (isodesmic, like COFACTOR_RING). TECRDB 10.35 -> 10.18 (adenylate kinase -26 -> +4,
    # nucleoside-diphosphate kinase exactly isodesmic).
    "NTP_CORE": True,
    "AUTO_TRUNCATE": True, "ROUTE_FULL": True, "TRUNC_V2": False, "TRUNC_VALIDATE": False,
    "PH0_AUTO": True, "PH0_BASES": True, "ZWITTERION_PH0": True, "NEUTRAL_QM": False,
    # PH0_ISOMERASE ADOPTED 2026-10-01: isomerizations also take the pH-0 route. The gate dated from the
    # COSMO baseline; under ALPB charged sugar-phosphate ring isomers do not cancel (G6P isomerase -25,
    # mannose-6-P isomerase -34) while their neutral forms do: isomerase MAE 8.4 -> 5.6, TECRDB 10.08 -> 9.64.
    "PH0_ISOMERASE": True,
    "CARBONYL_HYDRATION_ALL": True,    # every aldehyde/ketone except alpha-keto acids (validated vs K_hyd, ALPB)
    "HYDRATION_CAL": True,             # K_hyd-calibrated ΔG_hyd (independent data; only with CARBONYL_HYDRATION_ALL)
    # 2026-10-01 physics revision (analysis/sweep_20261001/NOTES.md): ALPB solvation + the experimental
    # liquid-water reference replace the COSMO-era compensations. The class anchors and the hydro-lyase
    # water patch were absorbing COSMO's missing H-bond term; with ALPB + exact water they are not used
    # (TECRDB 364: COSMO+anchors MAE 12.01 -> ALPB, no anchors 10.73 -> + hydration 10.35).
    # ZWITTERION_PH0: zwitterions are not gas-phase minima (proton transfer on relaxation) -> neutral route.
    "STD_STATE_1M": True, "ALDEHYDE_HYDRATION": True, "ANCHOR_CORRECT": False, "SMD_SOLV": False,
    "WATER_REF_HYDROLYASE": False, "WATER_REF_EXP": True,
    # UNDER A/B (default = the validated old behaviour; flip only after the subset A/B shows an improvement):
    "TRUNC_SPECTATOR_CATIONS": False,   # a cation removed WITH the spectator is not a "mangled" cation
    "TRUNC_MAXANION_RETRY": False,      # retry truncation on max-anion forms (protonation-consistent spectators)
    # ADOPTED 2026-09-30 (correctness): the old rule cut C-O/P-O bonds and H-capped them, producing invalid
    # cores (aspartate->isopropylamine, phosphoester->free H3PO4, P(III)); this cuts only single C-C bonds at
    # an sp3 kept carbon. A/B on its affected set: dG_raw MAE 13.1->12.5, RMS 17.1->15.3; where it looks worse
    # it exposes a real error the broken core had cancelled by luck. Set 0 only to reproduce the old cores.
    "TRUNC_FG_CUTS": True,
    # ADOPTED 2026-09-30: radius 2 is not a converged local model for reactions at an anomeric
    # sugar carbon: it can amputate ribose substituents (including a conserved 5'-phosphate)
    # that remain electrostatically coupled to the reactive center. Radius 3 was identical to
    # radius 4/full on the completed validation panel; severe MAE 47.11->29.64 (n=6; 5 improved,
    # 0 worsened), while four controls were unchanged. Set 0 only to reproduce radius-2 results.
    "TRUNC_ANOMERIC_RADIUS": True,
}


def _flag(name):
    """Pipeline switch: env NAME (0/off/false/no disables) overriding FLAG_DEFAULTS[name]. The defaults are
    the production configuration; set NAME=0/1 only for an ablation/baseline run."""
    v = os.environ.get(name)
    if v is None:
        return FLAG_DEFAULTS[name]
    return v.strip().lower() not in ("", "0", "off", "false", "no")


def _ah_mod():
    from metag.routing import aldehyde_hydration
    return aldehyde_hydration


def _constants_hash(module, names):
    """sha256 (12 hex) of the named module-level constants (missing names recorded as None)."""
    payload = json.dumps({n: getattr(module, n, None) for n in names}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def effective_config():
    """Normalized fingerprint of everything that changes a reported number: model, physics version,
    solvation, sampling, every routing/correction switch, pKa model and the deployed anchor offsets. Stored
    in each result and in the calibration artifact; metag.uncertainty refuses `coverage_calibrated` when
    the runtime fingerprint differs from the calibrated one. Includes content hashes of the pKa and hydration
    constant tables and the full species-cache settings, so estimator or constant changes cannot pass silently."""
    from metag.routing import pka_transform as _pk
    from metag.routing.anchor import ANCHORS
    anchors = json.dumps({k: [v["offset"], v.get("ref", "tecrdb")] for k, v in sorted(ANCHORS.items())},
                         sort_keys=True)
    cfg = {"model": _MODEL, "physics": PHYSICS_VERSION, "solv_model": SOLV_MODEL, "dedup": _DEDUP,
           "conv": [CONV_TOL, CONV_HITS, CONV_MAX], "sample_scale": _SAMPLE_SCALE, "qrrho": qrrho_enabled(),
           "keep_scale": _KEEP_SCALE,
           "explicit_sampling": [N_EXPLICIT_SEEDS, EXPLICIT_KEEP],
           "trunc_radius": int(os.environ.get("TRUNC_RADIUS", "2")),
           "trunc_validate_tol": float(os.environ.get("TRUNC_VALIDATE_TOL", "5")),
           "dg_sanity_kj": float(os.environ.get("DG_SANITY_KJ", "500")),
           "smd_threshold": float(os.environ.get("SMD_THRESHOLD", "5")),
           "water_ref_delta": float(os.environ.get("WATER_REF_DELTA", "-23.2")),
           "water_dgsolv_kj": float(os.environ.get("WATER_DGSOLV_KJ", "-26.4")),
           "pka_env": _pk._pka_env_enabled(), "free_ppi_pka": _pk._free_ppi_pka_enabled(),
           "anhydride_pka": _pk._anhydride_pka_enabled(),
           "pka_model": os.environ.get("PKA_MODEL", "table").strip().lower(),
           "pka_table": _pk.PKA_TABLE_VERSION, "ph0_redox_proton": _pk._redox_proton_enabled(),
           "polyacid_pka": _pk._polyacid_pka_enabled(), "arylamine_nonbasic": _pk._arylamine_nonbasic_enabled(),
           "carboxyl_pairs": _pk._carboxyl_pairs_enabled(),
           # CONTENT hashes: any edit of a constant changes the fingerprint even without a version bump
           "pka_constants": _constants_hash(_pk, ("P_LADDER", "ANHYDRIDE_P_LADDER", "P_N_LADDER", "ACYL_P_LADDER",
                                                  "CARBONATE_LADDER", "PPI_LADDER", "SULFATE_LADDER",
                                                  "CARBOXYL_PKA", "CARBOXYL_PKA_ALPHA", "SULFONATE_PKA",
                                                  "THIOL_PKA", "PHENOL_PKA", "POLYACID_PKA", "CARBOXYL_PAIR_LADDER")),
           "hydration_constants": _constants_hash(_ah_mod(), ("HYDRATION_CAL", "MAX_HYDRATION_SITES")),
           # species-level estimator (thermal ensemble, solvent relaxation, dedup, sampling, model ...)
           "implicit_settings": json.dumps(_IMPLICIT_SETTINGS, sort_keys=True),
           "acid_hb_filter": [_ACID_HB_FILTER, ACID_HB_DIST],
           "thermal_ensemble": [_THERMAL_ENSEMBLE, THERMAL_ENS_WIN, THERMAL_ENS_N],
           "solv_relax": [_SOLV_RELAX, SOLV_RELAX_N, SOLV_RELAX_WIN, SOLV_RELAX_STEPS],
           "anchors": hashlib.sha256(anchors.encode()).hexdigest()[:12]}
    cfg.update({k.lower(): _flag(k) for k in sorted(FLAG_DEFAULTS)})
    cfg["conditions"] = json.dumps(CONDITIONS, sort_keys=True)
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


def route_reaction(
    reaction,
    allow_truncate=True,
    trunc_radius=None,
    log=print,
    force_neutral=False,
    policy=None,
):
    """Structural routing -- pure logic, no QM: cofactor ring cores -> CoA / NTP cores -> full-vs-truncate
    gate -> spectator truncation -> pH-0 transform (-> optional zwitterion / neutral-QM invariant).

    Returns (rx, routes, truncated): `rx` is the rewritten reaction score_reaction scores (the input dict
    is never mutated), `routes` records which transforms fired plus any errors/warnings.

    BALANCE GUARD: every rewrite must keep the reaction element- and charge-balanced with its proton
    count in n_H+ -- the one invariant all of them share. A step whose output breaks a balance its input
    had is REVERTED and recorded in routes["errors"]. Without it an element leak (e.g. a mixed disulfide
    replaced by a symmetric core, dropping the CoA half) reached QM and was caught, if at all, only by
    the |ΔG| sanity bound."""
    from metag.routing.policy import RoutingPolicy

    validate_reaction(reaction)
    if policy is None:
        policy = RoutingPolicy.from_runtime(_flag, os.environ, trunc_radius)
    elif trunc_radius is not None:
        raise ValueError("trunc_radius cannot be combined with an explicit routing policy")
    elif not isinstance(policy, RoutingPolicy):
        raise TypeError("policy must be a RoutingPolicy")
    rx = dict(reaction)
    rx.setdefault("n_Hplus", 0)
    rx.setdefault("explicit", False)                        # `explicit` is optional per the contract;
    # routing paths (truncate/pH-0) set it, but a non-routing reaction (e.g. neutral oxygenase) must
    # still default it -- line 449 reads rx["explicit"] unconditionally. (list -> set for triage below.)
    if isinstance(rx["explicit"], list):
        rx["explicit"] = set(rx["explicit"])
    truncated = False                                        # did AUTO_TRUNCATE actually fire?
    routes = {"cofactor_ring": False, "coa_core": False, "ntp_core": False, "prefer_full": False,
              "truncated": False, "trunc_radius": None, "ph0": False, "zwitterion_ph0": False,
              "errors": [], "warnings": [], "reverted": [], "input_balanced": True}
    if not is_balanced(rx["species"], rx["n_Hplus"]):
        res, dq = reaction_residual(rx["species"], rx["n_Hplus"])
        routes["input_balanced"] = False
        routes["warnings"].append(f"input reaction not element/charge balanced (residual {res}, charge "
                                  f"{fmt_num(dq) if dq is not None else '?'}): ΔG is not meaningful")
        log(f"  !! {routes['warnings'][-1]}")
    # COFACTOR RING-TRUNCATION (opt-in COFACTOR_RING=1): replace NAD(P)+/NAD(P)H with their
    # redox-active nicotinamide RING model. The identical ADP-ribose-phosphate tail cancels in
    # ΔG but its floppy-conformer error does NOT in full-molecule QM -- which is why NAD (floppy)
    # carries a spurious ~-49 kJ bias while NADP (rigid) does not. Ring model removes the noise +
    # collapses NAD/NADP to one model. Isodesmic, experiment-free. Runs BEFORE truncation so the
    # small ring survives. Validated on TECRDB redox: NAD MAE 44.6->10.4 (n=4). Gated to a genuine
    # ox/red couple (never mis-fires on NAD biosynthesis). n_H+ preserved (Δq,ΔH of the couple kept).
    _pre = rx
    if policy.cofactor_ring:
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
                routes["cofactor_ring"] = True
                log(f"  [cofactor-ring: NAD(P) -> nicotinamide ring model"
                    + (f"; n_H+ {fmt_num(rx['n_Hplus']+dq)}->{fmt_num(rx['n_Hplus'])} (charge closure)]" if dq else "]"))
        except Exception as e:
            log(f"  [cofactor-ring error: {e}; unchanged]"); routes["errors"].append(f"cofactor_ring: {e}")
    rx = _balance_guard("cofactor_ring", _pre, rx, routes, log, reset={"cofactor_ring": False})
    # CoA CORE-REDUCTION (opt-in COA_CORE=1): DEFAULT-OFF -- proven a no-op on the benchmark.
    # HYPOTHESIS (rejected): the floppy pantetheine-ADP tail's conformer noise doesn't cancel in ΔG,
    # so capping S-CoA -> S-CH3 would denoise it. VERDICT: all 28 TECRDB CoA reactions are SYMMETRIC
    # (CoA on both sides), so the scaffold cancels EXACTLY regardless of truncation; the residual
    # sampling noise is <=1 kJ and net slightly harmful (0/5 matched pairs improved; see
    # tools/collect_fix.py coa_core + the symmetry proof). CoA's +23.5 MAE is ELECTRONIC (thioester
    # C(=O)-S reference), a CBH/DLPNO target like hydratase -- NOT a sampling problem. Kept behind the
    # flag (self-gating, harmless) for the record; do not re-enable without an ASYMMETRIC CoA reaction.
    _pre = rx
    if policy.coa_core:
        try:
            from metag.routing.coa_core import coa_core
            new = coa_core(rx["species"])
            if new is not rx["species"]:
                rx = dict(rx, species=new, note=rx.get("note", "") + " [COA-CORE]"); routes["coa_core"] = True
                log("  [coa-core: acyl-S-CoA -> acyl-S-CH3 (pantetheine-ADP scaffold capped)]")
        except Exception as e:
            log(f"  [coa-core error: {e}; unchanged]"); routes["errors"].append(f"coa_core: {e}")
    rx = _balance_guard("coa_core", _pre, rx, routes, log, reset={"coa_core": False})
    # NTP CORE-REDUCTION (opt-in NTP_CORE=1): the phosphoryl-transfer analogue. On ATP/ADP/AMP-type
    # species the adenosine-ribose rides along UNCHANGED (ATP->ADP keeps the whole nucleoside; only the
    # gamma-phosphate moves) -> it cancels in ΔG but its huge floppy tail does NOT in full-molecule QM
    # (phosphagen kinases run ATP/ADP conformer-CAPPED, sigma~0 -> +44..+77 err; generic MCS truncation
    # REFUSES the phosphoryl->guanidinium cut). Cap the nucleoside-5'-O with methyl, keep the reactive
    # polyphosphate (isodesmic, experiment-free). SELF-GATING on mass+charge balance. Runs AFTER
    # COA_CORE (which strips CoA's own adenosine first) and BEFORE truncation so the small core survives.
    # DEFAULT ON since 2026-10-01 (validated on the kinase class under ALPB: adenylate kinase -26 -> +4,
    # nucleoside-diphosphate kinase exactly isodesmic; TECRDB ablation +0.21 kJ if removed).
    _pre = rx
    if policy.ntp_core:
        try:
            from metag.routing.ntp_core import ntp_core
            new = ntp_core(rx["species"])
            if new is not rx["species"]:
                rx = dict(rx, species=new, note=rx.get("note", "") + " [NTP-CORE]"); routes["ntp_core"] = True
                log("  [ntp-core: nucleoside-5'-phosphate -> methyl polyphosphate (adenosine capped)]")
        except Exception as e:
            log(f"  [ntp-core error: {e}; unchanged]"); routes["errors"].append(f"ntp_core: {e}")
    rx = _balance_guard("ntp_core", _pre, rx, routes, log, reset={"ntp_core": False})
    # AUTO-TRUNCATION (general heuristic, opt-in AUTO_TRUNCATE=1): replace the reaction with
    # its truncated reactive core (removes conserved backbone -> kills catastrophic cancellation
    # + its conformer noise). Falls back to full molecules if no clean balanced truncation.
    # PHYSICS ROUTING GATE (ROUTE_FULL, default-on): prefer the FULL molecule over truncation when the
    # reaction is compact/ring-embedded, charge-conserved, and has no floppy large-fragment linker
    # (truncation_gate.prefer_full). Validated +1.05 kJ MAE vs baseline on the ring-fix candidate set; every
    # large truncation-regression (disaccharide/bisphosphate/SAH/G6P) is kept out of full by the
    # floppy-linker term. Set ROUTE_FULL=0 to ablate.
    _prefer_full = False
    if allow_truncate and policy.auto_truncate and policy.route_full:
        try:
            from metag.routing.truncation_gate import prefer_full
            _prefer_full = prefer_full(rx["species"], anchor_correct=policy.anchor_correct)
        except Exception as e:
            _prefer_full = False; routes["errors"].append(f"prefer_full: {e}")
        routes["prefer_full"] = bool(_prefer_full)
        if _prefer_full:
            log("  [route: prefer FULL molecule (compact ring, charge-conserved, no floppy linker) -> skip truncation]")
    _pre = rx
    if allow_truncate and policy.auto_truncate and not _prefer_full:
        try:
            from metag.routing.truncate import build_truncated_reaction, truncation_radius
            _radius_override = policy.trunc_radius_explicit
            _rad = policy.trunc_radius
            if policy.trunc_anomeric_radius and not _radius_override:
                _adaptive_rad = truncation_radius(rx["species"], default=_rad)
                if _adaptive_rad != _rad:
                    _rad = _adaptive_rad
                    routes["trunc_radius_reason"] = "anomeric_ring_reaction_center"
                    log("  [truncation radius 3: preserve the complete reacting sugar environment]")
            tr = build_truncated_reaction(rx["species"], radius=_rad, fg_cuts=policy.trunc_fg_cuts)
            if tr is None and policy.trunc_v2:   # v2: global-map truncation for the
                from metag.routing.truncate_global import build_truncated_reaction_v2   # multi-coeff/unequal-side cases
                tr = build_truncated_reaction_v2(rx["species"], radius=_rad)
                if tr is not None:
                    log("  [v2 global-map truncation engaged]")
            if tr is None and policy.trunc_maxanion_retry:
                # PROTONATION-CONSISTENT RETRY: the removed spectator must be identical on both sides, but sources
                # draw the same moiety with different protons (ModelSEED: ATP's AMP part as [O-], ADP's as O), which
                # fails the removed-fragment consistency guard. Retry on the max-anion (fully deprotonated) forms --
                # only when pH-0 will re-neutralize the cores afterwards, so the scored microspecies is unchanged in
                # kind; the core's n_H+ comes from its own H/charge and the balance guard checks the result.
                from metag.routing.pka_transform import _canonicalize_maxanion, is_isomerization
                if policy.ph0_auto and not rx.get("pka_sites") and not is_isomerization(rx["species"]):
                    canon = {}
                    for nm, (c, q, s_) in rx["species"].items():
                        cs = _canonicalize_maxanion(s_)
                        canon[nm] = [c, Chem.GetFormalCharge(Chem.MolFromSmiles(cs)), cs]
                    if canon != {k: list(v) for k, v in rx["species"].items()}:
                        tr = build_truncated_reaction(canon, radius=_rad, fg_cuts=policy.trunc_fg_cuts)
                        if tr is not None:
                            routes["truncated_maxanion"] = True
                            log("  [truncation on max-anion forms (protonation-consistent spectators)]")
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
                    # mangled = a RETAINED cation lost coordination. A cation absent from the core (dc empty) was
                    # removed WITH the spectator -- identical on both sides, so it cancels: not a mangle (this
                    # branch used to reject ~89% of the truncations the guard blocked, e.g. SAM/SAH, carnitine)
                    removed_with_spectator = not dc and policy.trunc_spectator_cations
                    if do and not removed_with_spectator and (not dc or max(dc) < max(do)):
                        return nm
                return None
            if tr is not None:
                bad = _truncation_mangles_cation(rx["species"], tr[0])
                if bad:
                    log(f"  [auto-truncation REJECTED: mangles a cationic center ({bad}) -> full molecules]")
                    routes["warnings"].append(f"truncation rejected: mangles a cationic center ({bad})")
                    tr = None
            if tr is not None:
                rx = dict(rx, species=tr[0], n_Hplus=tr[1], explicit=False,
                          note=rx.get("note", "") + " [AUTO-TRUNCATED]")
                truncated = True
                routes.update(truncated=True, trunc_radius=_rad)
                log(f"  [auto-truncated -> {len(tr[0])} core species, n_H+={tr[1]}]")
            else:
                log(f"  [auto-truncation fallback: full molecules]")
        except Exception as e:
            log(f"  [auto-truncation error: {e}; full molecules]"); routes["errors"].append(f"truncation: {e}")
    rx = _balance_guard("truncation", _pre, rx, routes, log, reset={"truncated": False, "trunc_radius": None})
    truncated = routes["truncated"]
    # pH-0 / pKa AUTO-ROUTING (general heuristic, opt-in PH0_AUTO=1): for the charged-anion class
    # (phosphoryl/NTP/PPi/carboxylate) protonate every anionic site to its NEUTRAL microspecies
    # -- UMA's comfortable regime (no formal charge to delocalise, no diffuse-anion basis ceiling,
    # continuum-solvation valid) -- then bridge to pH7 analytically with textbook pKa's. Runs AFTER
    # truncation so it neutralises the small cores. Falls back (no-op) when no anionic site exists
    # (thioester/glycosyl-anomeric neutral classes) or on any parse failure. Not fitted to the DB.
    _pre = rx
    if policy.ph0_auto and not rx.get("pka_sites"):
        try:
            from metag.routing.pka_transform import build_ph0_reaction, is_isomerization
            if is_isomerization(rx["species"]) and not policy.ph0_isomerase:
                # ISOMERASE GATE (PH0_ISOMERASE=1 lifts it: A/B under ALPB, where the neutral species are the
                # better-described ones -- the gate was set on the COSMO baseline): pH-0 hurts isomerizations (no anion-solvation change to
                # fix; neutralising spectator anions only injects sampling noise). Skip.
                log("  [pH0-auto: isomerization -> gated OFF (pH-0 would only add noise)]")
            else:
                # UNIFIED build: anions ALWAYS neutralized; basic amines additionally when the
                # internal base gate passes (PH0_BASES, default-on) -> covers the net-proton
                # deamination/transaminase/lyase class the old anion-only path refused.
                why = []
                out = build_ph0_reaction(rx["species"], rx["n_Hplus"],
                                         base=policy.ph0_bases, why=why)
                if out is not None:
                    ns, pks, nh = out
                    rx = dict(rx, species=ns, n_Hplus=nh, pka_sites=pks, explicit=False,
                              note=rx.get("note", "") + " [pH0]")
                    routes["ph0"] = True
                    log(f"  [pH0 -> {len(pks)} pKa sites, n_H+={fmt_num(nh)}]")
                else:
                    reason = why[0] if why else "no ionizable site"
                    routes["ph0_skipped"] = reason
                    log(f"  [pH0: {reason} -> unchanged]")
        except Exception as e:
            log(f"  [pH0-auto error: {e}; unchanged]"); routes["errors"].append(f"ph0: {e}")
    rx = _balance_guard("ph0", _pre, rx, routes, log, reset={"ph0": False})
    # ZWITTERION INVARIANT (ZWITTERION_PH0, default-on): NO zwitterion may reach gas-phase QM, whatever
    # route produced the species list. A zwitterion (protonated N + O- in one molecule) is not a gas-phase
    # minimum: UMA relaxation transfers the N-H proton to the O- in every conformer (verified Gly/Ala/Ser/
    # Glu/Asp/phosphoserine/ethanolamine-P), so the scored species silently becomes the neutral tautomer
    # (~30 kJ above the aqueous zwitterion for glycine). If one is still present here (pH-0 gated off for an
    # isomerization, refused on mass balance, or disabled), take the full neutral-microspecies path: acids
    # AND amines neutralized, the aqueous zwitterion rebuilt analytically by the acid + base pKa terms.
    # TECRDB: 23 reactions reached QM with a zwitterion, MAE 16.0 vs 10.9 for the rest.
    # NEUTRAL-QM INVARIANT (NEUTRAL_QM, generalizes the zwitterion rule): ANY ionizable charge left in a
    # QM species (anionic O, or protonated N-H) -- whether pH-0 was gated off (isomerization), refused (net
    # redox proton), or never applied -- is routed through the full neutral-microspecies path: the implicit-
    # continuum anion/zwitterion solvation failure is the largest measured error source (TECRDB: reactions
    # with bare anions in QM MAE 14.8, zwitterions 16.0, vs ~8-11 otherwise). Only permanent cations stay
    # charged. The base path carries any net (redox) proton as n_H+ with the charge-closure guard below.
    # STATUS: default ON since 2026-10-01 (ALPB baseline). Physical invariant: a zwitterion is not a gas-phase
    # minimum. With the current routing no TECRDB reaction depends on it (ablation 0.00 kJ); kept as a guard.
    _neutral_all = policy.neutral_qm or force_neutral     # force_neutral: re-route after a rearranged species
    _pre = rx
    if policy.zwitterion_ph0 or _neutral_all:
        try:
            from metag.routing.pka_transform import build_ph0_reaction, has_zwitterion, has_ionized
            _needs = (lambda sp: has_ionized(sp)) if _neutral_all else (lambda sp: has_zwitterion(sp))
            if _needs(rx["species"]) and rx.get("pka_sites"):
                routes["warnings"].append("charged species reaches QM (user-supplied pka_sites; not re-routed)")
            elif _needs(rx["species"]):
                out = build_ph0_reaction(rx["species"], rx["n_Hplus"], base=True, force_base=True)
                if out is not None and not _needs(out[0]):
                    ns, pks, nh = out
                    rx = dict(rx, species=ns, n_Hplus=nh, pka_sites=pks,
                              explicit=False, note=rx.get("note", "") + " [pH0-zwitterion]")
                    routes["ph0"] = True; routes["zwitterion_ph0"] = True
                    log(f"  [zwitterion invariant -> neutral-microspecies path, {len(pks)} pKa sites, n_H+={fmt_num(nh)}]")
                else:
                    routes["warnings"].append("zwitterion reaches QM (neutral-microspecies build refused)")
                    log("  !! zwitterion reaches QM: neutral-microspecies build refused")
        except Exception as e:
            routes["errors"].append(f"zwitterion_ph0: {e}")
    rx = _balance_guard("zwitterion_ph0", _pre, rx, routes, log, reset={"zwitterion_ph0": False})
    return rx, routes, truncated


def _no_estimate(key, reaction, routes, reason, log):
    """Fail-closed result with no number, used before any QM when the estimator cannot be applied."""
    log(f"  ΔG = NONE ({reason})")
    return dict(reaction=key, dG=None, dG_raw=None, dG_raw_unreliable=None, suspect=reason, routes=routes,
                anchor=None, smd=None, water_ref=None, stages={}, conditions=CONDITIONS, config=effective_config(),
                U_samp=None, sigma_pred=None, sigma_breakdown={}, ci95=[None, None], ci_center=None,
                ci_info={"externally_calibrated": False, "calibration_scope": f"no estimate: {reason}"},
                unresolved=None, exp=reaction.get("exp"), err=[], note=reaction.get("note", ""),
                explicit=reaction.get("explicit", False), std_state_kJ=None, species_scored=None)


def _balance_guard(step, before, after, routes, log, reset=None):
    """Keep `after` if it is balanced (or if `before` already was not); otherwise revert to `before`,
    record the error, and reset the step's route flags."""
    if after is before or not is_balanced(before["species"], before["n_Hplus"]):
        return after
    if is_balanced(after["species"], after["n_Hplus"]):
        return after
    res, dq = reaction_residual(after["species"], after["n_Hplus"])
    msg = (f"{step}: rewritten reaction not balanced (residual {res}, charge "
           f"{fmt_num(dq) if dq is not None else '?'}) -> step reverted")
    routes["reverted"].append(msg)                        # designed "not applicable" outcome, not an error
    log(f"  !! {msg}")
    for k, v in (reset or {}).items():
        routes[k] = v
    return before


def _species_free_energies(pu, rx, routes, G, sig, std, seeds, keep, pool, log, requested_explicit,
                           is_spectator_anion):
    """Fill G/sig (kJ/mol, 1 M standard state) for every species of the routed reaction; stops at the first
    species that fails (G[name] = None). Raises SpeciesRearranged (see implicit_G)."""
    for name, (coeff, q, smi) in rx["species"].items():
        if is_water(smi, q):                             # liquid water (hydrolysis / hydration)
            G[name] = water_ref_G(pu, log); sig[name] = 0.0     # deterministic reference: no sampling σ
            log(f"    {name:9s} q+0 [water ref liquid]: {G[name]:.1f}")
        elif requested_explicit(name):
            if is_spectator_anion(name):
                G[name], sig[name] = explicit_G(pu, q, smi, seeds, log, name)
            else:                                        # GUARD: explicit invalid here
                log(f"    !! {name}: explicit REFUSED (created/destroyed anion, no "
                    f"cancellation partner) -> implicit. Use pH-0 (pka_sites) for accuracy.")
                G[name], sig[name] = implicit_G(pu, q, smi, seeds, keep, pool, log, name, routes["warnings"])
        else:
            G[name], sig[name] = implicit_G(pu, q, smi, seeds, keep, pool, log, name, routes["warnings"])
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


def score_reaction(pu, reaction, seeds=(1, 2), keep=10, pool=48, log=print, allow_truncate=True, key="rxn",
                   trunc_radius=None, _validating=False, _force_neutral=False):
    """Public entry point: score one reaction's standard transformed Gibbs energy (ΔrG'°) from structure.

    `reaction` = {"species": {name: [coeff, charge, smiles]}, "note": str, "n_Hplus": int,
    and optionally "exp": [float] (experiment, for err reporting), "explicit": bool|list, "pka_sites"}.
    `pu` is a loaded UMA model (metag.energetics.uma.load_uma). Returns the result dict:
    dG (physics+anchor), dG_raw (pure physics), anchor, sigma_pred, ci95, ci_center, U_samp, routes, ...
    `seeds`/`keep`/`pool` are accepted for backward compatibility; the sampling budget is set per species
    by sampling_budget() + the adaptive convergence loop (CONV_*), not by these arguments.

    TRUNC_VALIDATE=1 (opt-in): radius-sensitivity guard for spectator truncation. A true spectator cut
    leaves ΔG invariant to the cut radius, so the reaction is scored at radius R and R+1; the truncation
    is kept only if |ΔG_raw(R) - ΔG_raw(R+1)| <= max(TRUNC_VALIDATE_TOL, 2·sqrt(U_R² + U_R+1²)),
    otherwise the reaction is re-scored with full molecules. Recorded in result["trunc_validation"].
    """
    if _flag("TRUNC_VALIDATE") and not _validating and allow_truncate:
        return _score_trunc_validated(pu, reaction, seeds, keep, pool, log, key, trunc_radius)
    validate_reaction(reaction)
    orig_species = {k: list(v) for k, v in reaction["species"].items()}   # pre-routing, for the anchor gate
    rx, routes, truncated = route_reaction(reaction, allow_truncate=allow_truncate, trunc_radius=trunc_radius,
                                           log=log, force_neutral=_force_neutral)
    routes["rearrangement_reroute"] = bool(_force_neutral)
    if routes["errors"]:
        # an UNEXPECTED exception inside a routing transform (expected misses return "not applicable" or
        # are reverted by the balance guard): the reaction would silently be scored by a different
        # estimator than the configured one -> fail closed, before spending any QM.
        return _no_estimate(key, reaction, routes, "routing error: " + "; ".join(routes["errors"]), log)
    if not routes["input_balanced"]:
        # Elemental or charge imbalance makes a reaction free energy undefined. route_reaction has already
        # recorded the exact residual; stop before loading any species energies, including cache hits.
        return _no_estimate(key, reaction, routes, routes["warnings"][-1], log)
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
        # the input microspecies is not a gas-phase minimum (all conformers transferred a proton): score the
        # reaction through the NEUTRAL-microspecies path instead, where that proton is accounted for by pKa
        # terms. If it happens again there is no valid species to score -> no estimate.
        if _force_neutral:
            log(f"  !! {e} (already on the neutral-microspecies path) -> no estimate")
            return None
        log(f"  [{e} -> re-route through the neutral-microspecies path]")
        return score_reaction(pu, reaction, seeds, keep, pool, log, allow_truncate=allow_truncate, key=key,
                              trunc_radius=trunc_radius, _validating=_validating, _force_neutral=True)
    if any(v is None for v in G.values()) or len(G) < len(rx["species"]):
        if truncated:                                        # a truncated core failed -> retry FULL molecules
            log(f"  [retry {key} with truncation OFF (full molecules)]")
            # re-score the ORIGINAL input dict (not a registry lookup: dict-API callers are not in REACTIONS)
            return score_reaction(pu, reaction, seeds, keep, pool, log, allow_truncate=False, key=key,
                                  _force_neutral=_force_neutral)
        return None
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
                continue
            if Gd is None:
                continue
            Gd += std                                     # G[name] already carries std; put diol + water on 1 M too
            Geff = _ah.mixture_G(G[name], Gd, _gw + std)
            log(f"    [hydration: {name} carbonyl {G[name]:.1f} + gem-diol {Gd:.1f} "
                f"(ΔG_hyd {Gd - (_gw + std) - G[name]:+.1f}) -> mixture {Geff:.1f}  shift {Geff - G[name]:+.1f}]")
            G[name] = Geff
            sig[name] = float(np.hypot(sig.get(name, 0.0), _sd or 0.0))

    dG = sum(coeff * G[name] for name, (coeff, q, smi) in rx["species"].items())
    stages = {"species_sum": dG, "proton": rx["n_Hplus"] * G_HPLUS}
    dG += rx["n_Hplus"] * G_HPLUS
    # propagate per-species sampling σ to a reaction sampling-uncertainty (quadrature).
    U_samp = float(np.sqrt(sum((coeff * sig[name])**2
                               for name, (coeff, q, smi) in rx["species"].items())))
    # pH-0 route: analytic pKa transform replaces the anion-solvation + explicit-proton terms.
    # EXACT Alberty form -RT*ln(1+10^(pH-pKa)) per proton (correct near AND above pH~pKa; the
    # linear RT*ln10*(pH-pKa) form wrongly makes a high-pKa site e.g. Pi's 12.35 count -5 kJ).
    pka_total = 0.0                                      # stages["pka_transform"] after the loop
    for site in rx.get("pka_sites", []):
        side, pka = site[0], site[1]
        kind = site[2] if len(site) > 2 else "acid"
        mult = site[3] if len(site) > 3 else 1.0             # |coeff| of the species (fractional-safe)
        # acid group (neutral=protonated HA): -RT ln(1+10^(pH-pKa)); base group (neutral=deprotonated
        # B, protonates below pKa): the mirror -RT ln(1+10^(pKa-pH)). Sign per side (react +, prod -).
        expo = (PH - pka) if kind == "acid" else (pka - PH)
        contrib = (mult if side == "react" else -mult) * RT_LN10 * math.log10(1.0 + 10.0 ** expo)
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
    # already been applied. The empirical terms after it (hydro-lyase water constant, anchors) are reported
    # separately in `stages`.
    dG_raw = dG
    stages["unanchored"] = dG_raw
    anchor_meta = None                                # so the empirical correction is transparent
    if _flag("ANCHOR_CORRECT") and suspect is None:
        # NOT wrapped in a broad except: an anchor failure must not silently return the uncorrected dG_raw
        # as if the reaction were simply un-anchored (a caller/return-shape mismatch did exactly that).
        from metag.routing.anchor import anchor_correct, subclass_extent
        ac = anchor_correct(dG, orig_species)
        if ac is not None:
            dG_corr, sig_anchor, sc, direction = ac
            log(f"  [anchor-correct: {sc} ({'forward' if direction > 0 else 'reverse'}) -> ΔG {dG:+.1f} -> "
                f"{dG_corr:+.1f} (offset {dG-dG_corr:+.1f})]")
            dG = dG_corr
            # NOTE: the intra-class residual (sig_anchor) is NOT folded into U_samp here -- it is the
            # class-level predictive error, which sigma_pred below carries via the calibrated
            # sigma_class (avoids double-counting). U_samp stays the conformer-sampling spread only.
            # "offset" is the signed correction actually applied (dG_raw - dG): +offset forward, -offset reverse.
            stages["anchor"] = dG_corr - dG_raw
            anchor_meta = {"subclass": sc, "offset": round(dG_raw - dG_corr, 1), "direction": direction,
                           "extent": subclass_extent(orig_species)[2],
                           "sigma": sig_anchor}
    # SMD SOLUTE-SOLVATION CORRECTION (SMD_SOLV, default OFF -- tested and rejected): xtb-COSMO systematically UNDER-SOLVATES a
    # CREATED/DESTROYED compact polar/charged group (the solvation wall -- measured: a hydratase -OH is
    # -3 kJ in COSMO vs -17 in SMD≈exp). Where the structural gate detects such a NON-CANCELLING solvation
    # change, recompute the SOLUTES' solvation with SMD (calibrated neutral continuum, gpu4pyscf via the
    # redox env) and apply the SOLUTE-ONLY correction Σν(ΔGsolv_SMD − ΔGsolv_COSMO). Water is NOT corrected
    # (it stays on the calibrated water_ref_G -- correcting it re-introduces a ~-31 kJ/water screening
    # artifact). GATED (skips group-conserving isomerase/transaminase, where SMD only adds scatter) and
    # MAGNITUDE-FLOORED. Physics, not a fitted offset. Not applied when an anchor already fired (no double
    # correction) or when the result is suspect. Validated full-367 solute-only (analysis/smd_measure.py).
    # DEFAULT-OFF (2026): the solute-only form was DISPROVEN -- excluding water (which is load-bearing for
    # net-water reactions: xtb-COSMO under-solvates water -3 vs exp -26) makes hydratase WORSE (rxn00799
    # 24->37.5). Kept behind the flag while the correct water-consistent form is designed. Do NOT re-enable
    # solute-only. See analysis/smd_measure.py + memory red-wall-is-solvation-not-electronic.
    smd_meta = None
    if _flag("SMD_SOLV") and suspect is None and anchor_meta is None:
        try:
            from metag.routing.solv_gate import needs_smd
            fires, changed = needs_smd(orig_species)      # detect on the ORIGINAL reaction (pre-routing)
            if fires:
                from metag.energetics.smd_solv import smd_dgsolv
                from metag.energetics.thermal import xtb_dgsolv
                from metag.energetics.explicit_solvation import bare_geom
                corr = 0.0
                for name, (coeff, q, smi) in rx["species"].items():
                    if is_water(smi, q):                  # water stays on water_ref_G
                        continue
                    if spin_multiplicity(smi, q) != 1:    # the SMD path is closed-shell RKS only
                        corr = None; break
                    sym, crd = bare_geom(pu, q, smi)      # representative UMA-relaxed geometry
                    gc = xtb_dgsolv(sym, crd, q, "cosmo"); gs = smd_dgsolv(sym, crd, q)
                    if gc is None or gs is None:
                        corr = None; break
                    corr += coeff * (gs - gc)
                thr = float(os.environ.get("SMD_THRESHOLD", "5"))
                if corr is not None and abs(corr) >= thr:
                    log(f"  [smd-solv: gate {changed} -> correction {corr:+.1f} kJ; ΔG {dG:+.1f} -> {dG+corr:+.1f}]")
                    dG += corr
                    smd_meta = {"correction": round(corr, 1), "groups": changed}
                elif corr is not None:
                    log(f"  [smd-solv: gate fired, |Δ|={abs(corr):.1f} < {thr} kJ -> not applied (cancels)]")
        except Exception as e:
            log(f"  [smd-solv error: {e}]"); routes["errors"].append(f"smd_solv: {e}")
            suspect = f"smd_solv error: {e}" if suspect is None else suspect
    # WATER-REFERENCE CORRECTION for HYDRO-LYASES (WATER_REF_HYDROLYASE, default OFF since 2026-10-01: superseded
    # by the global experimental water reference WATER_REF_EXP). ROOT CAUSE of the
    # hydratase wall: water_ref_G uses xtb-COSMO for the water molecule's ΔGsolv (-3.2 kJ) but experiment
    # is -26.4 (xtb-ALPB independently -25.4), so G_liq(water) is ~+23 kJ too high -> every net-water
    # reaction is biased net_water*(-23.2). A GLOBAL fix (WATER_REF_EXP) regresses because the anchors +
    # sigma were calibrated ON the buggy baseline and absorbed it (phosphatase 2->23) and un-anchored
    # net-water classes hide a competing ~23 kJ solute-under-solvation error that currently cancels it.
    # HYDRO-LYASES (neutral C=C+H2O<->C-OH, no anion/phosphate, un-anchored) are the ONE class where the
    # water error is the sole solvation error -> apply the deterministic correction only there. Physics (a
    # known constant), class-gated as a SAFE staged rollout (global fix + full recalibration is the eventual
    # correct state). Validated end-to-end: rxn00799 err +24.3 -> +0.7; hydratase class MAE 19.1 -> 4.8;
    # zero leakage into other classes. Skipped if WATER_REF_EXP already fixed water globally (no double).
    water_meta = None
    # The constant (-23.2 = exp - COSMO water ΔGsolv) corrects the xtb-COSMO water only; a parametrized
    # model (alpb/cpcmx) solvates water correctly (ALPB -25.4 vs exp -26.4), so it is not applied there.
    if (_flag("WATER_REF_HYDROLYASE") and suspect is None and SOLV_MODEL == "cosmo"
            and not _flag("WATER_REF_EXP")):
        from metag.routing.solv_gate import is_hydrolyase
        if is_hydrolyase(orig_species):
            dwref = float(os.environ.get("WATER_REF_DELTA", "-23.2"))   # exp(-26.4) - cosmo(-3.2)
            nwat = sum(c for n, (c, q, s) in rx["species"].items() if is_water(s, q))
            corr = nwat * dwref
            if corr != 0:
                log(f"  [water-ref hydro-lyase: net_water={nwat} -> {corr:+.1f} kJ; ΔG {dG:+.1f} -> {dG+corr:+.1f}]")
                dG += corr
                water_meta = {"correction": round(corr, 1), "net_water": nwat}
                stages["water_ref"] = corr
    # CALIBRATED prediction uncertainty for downstream flux / TFA. U_samp (conformer-sampling spread,
    # ~1-3 kJ) is NOT the prediction interval -- the SYSTEMATIC method error (per mechanism class) dominates
    # (~5-25 kJ). sigma_pred = sqrt(U_samp^2 + sigma_class^2), class-conditional and calibrated on the
    # benchmark residual (metag/tools/calibrate.py; fully nested 5-fold CV, see the artifact).
    # Reporting +-U_samp alone would make a TFA solver ~10x overconfident on the hard classes.
    # No broad except here: σ and the interval are pure lookups, so a failure is a bug to surface, not a
    # reason to return a point estimate without its uncertainty.
    from metag.uncertainty import reaction_sigma, prediction_interval
    cfg = effective_config()
    # σ-class from the ORIGINAL input (structure + note), never from the routed species / tagged note: the
    # class must not depend on which transform fired (e.g. "[COA-CORE]" matching the CoA keyword, or a
    # truncated core losing the pantetheine SMILES tag).
    _smis = [v[2] for v in orig_species.values()]
    _note = reaction.get("note", "")
    sigma_pred, sigma_breakdown = reaction_sigma(_note, _smis, U_samp, species=orig_species, config=cfg)
    # Symmetric nested-CV 95% interval on the SAME σ_total as sigma_pred (class σ ⊕ U_samp), centred on dG.
    # ci_info carries externally_calibrated / calibration_scope: False for OOD-flagged or uncalibrated-class
    # reactions AND whenever the runtime configuration differs from the one the artifact was calibrated on.
    ci_lo, ci_hi, ci_center, ci_info = prediction_interval(_note, _smis, dG, level=95, species=orig_species,
                                                           U_samp=U_samp, config=cfg)
    # UQ_MODEL=features: per-reaction, annotation-free width (structure + route; metag.uq_features). The
    # class-model numbers are kept under sigma_breakdown["class_model"] for comparison. Not part of
    # effective_config(): it does not change dG, and the class artifact's fingerprint must stay valid.
    from metag.uncertainty import uq_model, feature_interval
    if uq_model() == "features":
        from metag.uq_features import reaction_features
        _raw = reaction_features(orig_species, rx["species"], routes, stages, U_samp)
        _cls_sigma = sigma_pred
        sigma_pred, ci_lo, ci_hi, ci_info = feature_interval(_raw, dG, level=95, species=orig_species,
                                                             config=cfg)
        ci_center = round(dG, 1)
        sigma_breakdown = {"uq_model": "features", "features": _raw,
                           "class_model": dict(sigma_breakdown, sigma=_cls_sigma)}
    exp_out = sorted(exp_flag) if isinstance(exp_flag, (set, list, tuple)) else exp_flag
    stages.update(reported=dG)
    stages = {k: round(v, 2) for k, v in stages.items()}
    common = dict(reaction=key, anchor=anchor_meta, smd=smd_meta, water_ref=water_meta, stages=stages,
                  conditions=CONDITIONS,
                  U_samp=round(U_samp, 1), exp=rx.get("exp"), explicit=exp_out, suspect=suspect,
                  note=rx["note"], routes=routes, std_state_kJ=round(std, 2), config=cfg,
                  species_scored={k: list(v) for k, v in rx["species"].items()})
    if suspect is not None:
        # FAIL CLOSED: an invalid reaction (charge leak, unbalanced input, unphysical magnitude) returns NO
        # estimate -- dG / dG_raw / σ / interval are None and calibration is explicitly off -- so a TFA/MDF
        # consumer cannot mistake it for a usable number. The raw value is kept only for diagnosis.
        log(f"  ΔG = NONE (suspect: {suspect}); raw value {dG_raw:+.1f} kept as dG_raw_unreliable")
        return dict(common, dG=None, dG_raw=None, dG_raw_unreliable=round(dG_raw, 1), sigma_pred=None,
                    sigma_breakdown=sigma_breakdown, ci95=[None, None], ci_center=None,
                    ci_info=dict(ci_info, externally_calibrated=False,
                                 calibration_scope=f"no estimate: suspect ({suspect})"),
                    unresolved=None, err=[])
    errs = [dG - e for e in rx.get("exp", [])]              # exp is optional (deployment has no experiment)
    # RESOLUTION heuristic: if the CALIBRATED prediction interval is comparable to |ΔG|, the sign is not
    # resolvable -- flag it (near-equilibrium isomerases are concentration-limited, not QM-fixable).
    # sign unresolved at the REPORTED confidence level exactly when the 95% interval contains zero
    unresolved = ci_lo <= 0.0 <= ci_hi
    flag = "  [UNRESOLVED: ci95 contains 0]" if unresolved else ""
    log(f"  ΔG = {dG:+.1f} (raw {dG_raw:+.1f}) ± {sigma_pred} kJ/mol  "
        f"[σ_class {sigma_breakdown.get('sigma_class','?')} ({sigma_breakdown.get('class','?')}), U_samp {U_samp:.1f}]"
        f"   vs exp {rx.get('exp')}   err {[round(e,1) for e in errs]}{flag}")
    return dict(common, dG=round(dG, 1), dG_raw=round(dG_raw, 1), sigma_pred=sigma_pred,
                sigma_breakdown=sigma_breakdown, ci95=[ci_lo, ci_hi], ci_center=ci_center, ci_info=ci_info,
                unresolved=unresolved, err=[round(e, 1) for e in errs])

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
        thr = max(tol, 2.0 * math.hypot(rR["U_samp"], rR1["U_samp"]))
        info.update(dG_raw_R1=rR1["dG_raw"], delta=round(d, 1), threshold=round(thr, 1))
        if d <= thr:
            log(f"  [trunc-validate: radius {R} vs {R+1} agree (|Δ|={d:.1f} <= {thr:.1f}) -> keep truncation]")
            rR["trunc_validation"] = dict(info, verdict="accepted")
            return rR
        verdict = "rejected: radius-sensitive"
    else:
        verdict = "rejected: radius R+1 truncation unavailable"
    log(f"  [trunc-validate: {verdict} -> full molecules]")
    rF = score_reaction(pu, reaction, allow_truncate=False, **kw)
    if rF is not None:
        rF["trunc_validation"] = dict(info, verdict=verdict)
    return rF


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
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, f"unified_pipeline_{tag}.json"), "w"), indent=2)
    log(f"wrote {os.path.join(OUT, f'unified_pipeline_{tag}.json')}")


if __name__ == "__main__":
    main()
