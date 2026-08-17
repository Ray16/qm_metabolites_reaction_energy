# Module map — QM reaction-ΔG pipeline

The `scripts/` directory now holds **only production modules**. The historical
exploration one-offs (step1–step6 experiments, `diag_*`, `probe_*`, `test_*`,
`validate_*`, `verify_*`) were moved to `../../backup/qm_exploration_scripts/`
(gitignored; still in git history). Their scientific conclusions are baked into
the pipeline + `EXPLORATION_LOG.md` + the memory index; regenerate a diagnostic
if a specific one is needed again.

## Production modules (`scripts/`)

### Entry point
- **`unified_pipeline.py`** — the one scheme across all reaction classes. Auto-routes
  each reaction: truncate → gated pH-0 (skip isomerase) → sample → UMA ΔE + thermal +
  solvation + pKa transform → UNRESOLVED flag + σ. Run: `python scripts/unified_pipeline.py --only rxnXXXXX`.

### Reaction preparation
- **`truncate.py`** — reaction-level spectator truncation (MCS reaction-center → grow → cap).
- **`truncate_v2.py`** — global-map truncation for the ~20% multi-coeff / unequal-side reactions v1 refuses (gated by radius-sensitivity).
- **`truncate_rxnmapper.py`** — RXNMapper-based truncation variant; in-progress confidence-gated hybrid (see HANDOFF open item).
- **`ph0_auto.py`** — pH-0 / pKa-transform builder. Neutralises anionic sites, emits exact-Alberty pKa ladders. Self-gating: `is_isomerization` skip + H-mass-balance guard (refuses net-proton-exchange redox/deamination → falls back to baseline).
- **`microspecies.py`** — protonation / microspecies from ModelSEED ChemAxon pH-7 assignments (used by the reaction-build tools).
- **`water_count.py`** — deterministic first-shell water-count rule for explicit-solvation clusters.

### QM backends
- **`batched_relax.py`** — batched UMA relaxation (many structures / forward pass) + energies + FIRE.
- **`step4e_targeted.py`** — conformer pooling + Boltzmann averaging (`pool_confs`, `boltz`). *[name is exploration-era; rename → `conformers.py` post-sweep]*
- **`step7b_charge_balanced_waters.py`** — bare-solute geometry for cluster-continuum (`bare_geom`). *[rename → `explicit_clusters.py` post-sweep]*
- **`thermal_solv.py`** — fast GPU-efficient thermal (UMA-Hessian) + xtb solvation single point (`corr_fast`).
- `grand_canonical_clusters` — lives in `../../backup/explicit_water/` (on the pipeline's sys.path); explicit-water seeding, only exercised on `explicit=True` reactions.

### Downstream
- **`uncertainty.py`** — calibrated total σ (sampling + class + motif) for TFA / flux. Needs final σ_class calibration.

## Tools (`tools/`)
Analysis + build harness (not imported by the pipeline): `build_tecrdb_reactions.py`,
`build_modelseed_reactions.py` (→ `scripts/reactions_modelseed.json`, 20,802 reactions),
`ph0_final_analysis.py` (gated coherent MAE), `full367_table.py`, `analyze_sweep.py`,
`ph0_before_after.py`, plus the multi-node worker/launch `.sh` scripts.
