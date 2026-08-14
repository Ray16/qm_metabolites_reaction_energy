# Foundation-MLIP + explicit solvation for biochemical reaction ΔG

**Hypothesis.** The reason first-principles QM (xTB/GFN2 + ALPB/CPCM-X) failed on
biochemical ΔG is a specific, known wall: **implicit-continuum solvation
under-solvates polyanions** (ATP⁴⁻, phosphates, CoA) by ~20–40 kJ/mol, and xTB's
gas-phase electronics are shaky for large charged species. Two 2025–2026
developments remove both bottlenecks:

1. **Foundation MLIPs trained on charged molecules** — `facebook/UMA`
   (OMol25-trained) takes **total charge + spin as inputs** and is near-DFT
   accurate across the element/charge range we need, ~10⁴× faster than DFT.
2. **Cluster-continuum explicit solvation** — a few explicit inner-shell waters +
   implicit outer environment reaches ~4 kJ/mol on ions (ML-MD literature); this
   is the automated version of the microsolvation fix that already cut our anion
   pKa error 44→10 kJ (see memory `microsolvation-fixes-anion-solvation`).

**Claim to test:** can UMA (OMol25) + cluster-continuum explicit solvation get a
*polyanionic* biochemical reaction's ΔG right where xTB-ALPB failed?

**Scope, honest.** This is a **frontier-audit tool, not a scalable predictor** —
explicit-solvent sampling can't cover 30k compounds, but it can adjudicate a few
dozen high-value reactions (the parked "Phase 2 QM audit"), now with a credible
engine instead of the xTB composite that scored R²≈−15 on pH-7 TECRDB.

## Why this is different from the prior QM attempts (what NOT to repeat)
- Prior work: xTB/GFN2 geometries + ALPB/CPCM-X implicit solvation, absolute
  `ΔG = S·G_aq`. Failed: anion under-solvation floor (~23 kJ), conformer noise
  ±50 kJ on big cofactors (memory `ph0-che-reimplementation-attempt`).
- AIMNet2 gas-phase opt: didn't robustly help (memory `aimnet2-optimizer-test`).
- **This**: replace the QM engine with a charge-aware foundation MLIP AND replace
  implicit continuum with explicit micro-solvation. Small curated set only.

## Plan
- **Step 0 (feasibility, this commit).** Confirm UMA runs on the charged/
  polyanionic species we need: neutral → −1 → −2 → −3 (PO₄³⁻), elements C/H/O/N/P/S.
  Finite energies + usable forces + stable geometry opt = green light.
  → `scripts/probe_uma_charge.py`
- **Step 1.** Gas-phase reaction ΔE on a handful of balanced small reactions with
  known ΔG (subset of the ten + clean anion cases). Sanity vs experiment sign.
- **Step 2.** Add cluster-continuum solvation (N explicit waters + implicit) and
  measure ΔG vs experiment on the curated set. Compare to the xTB-ALPB floor.
- **Step 3.** If it works: a scoped frontier-audit protocol (which reactions,
  water count, compute budget, expected accuracy).

## Environment
`uma` conda env (fairchem-core 2.21.0, torch 2.8+cu128, ase 3.29). UMA checkpoint
`uma-s-1p2` cached at `~/.cache/fairchem/models--facebook--UMA`. Loader pattern
mirrors `PALM/benchmarks/omol25/smoke_uma.py`.

## Status
- [x] **Step 0 feasibility probe — GREEN.** All 9 species (neutral → PO₄³⁻ →
  pyrophosphate, C/H/O/N/P/S) ran with finite UMA energies + forces and optimized
  to fmax≤0.05; 6/6 anions including the trianions. UMA covers the polyanion
  regime xTB-ALPB mangled. `artifacts/probe_uma_charge.json`. **Caveat: this is
  coverage/stability, NOT accuracy — gas-phase electronic energies only. Accuracy
  on solvated ΔG is Steps 1–2.**
- [x] **Step 1 gas-phase ΔE on 6 isomerizations — NEGATIVE (diagnostic).**
  `scripts/step1_isomerization_dE.py`. MAE 23.7 kJ vs experiment on reactions
  whose true ΔG is ≤8 kJ; sign 33%; **median conformer spread 49 kJ**. This
  REPRODUCES the prior xTB conformer-noise wall (`ph0-che-reimplementation-attempt`)
  with UMA → **electronics were never the bottleneck; conformer sampling +
  solvation are, and they're engine-independent.** Sampling each species
  independently and subtracting cannot work (two ±50 kJ noisy absolutes don't
  cancel); charged isomerizations additionally off by gas-vs-solution electrostatics
  (q−2 phosphate −51 vs exp +3). UMA solved COVERAGE (Step 0) not THERMODYNAMICS.
  → **The only viable path is cancellation via MATCHED geometries** (shared
  scaffold so spectator conformer noise cancels). Isomerizations are a worst case
  for this (whole molecule reorganizes); the ten transfers keep a big identical
  spectator, so cancellation applies there.
- [ ] Step 2 MATCHED-scaffold difference on the redox pair (rxn00086/70) — the
  real test of whether cancellation tames the noise
- [ ] Step 3 (conditional) cluster-continuum solvation + CHE for the ten
- [ ] Step 4 audit protocol
