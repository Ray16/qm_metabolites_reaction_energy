# TODO — QM (UMA) reaction-ΔG pipeline

Living checklist. Authoritative state = `PIPELINE_REFERENCE.md`; reasoning = `EXPLORATION_LOG.md`.

## ▶ IN PROGRESS (running now)
- [ ] **Full-367 best-pipeline sweep** (AUTO_TRUNCATE, 8-GPU strict queue, `run_full367.sh`)
      -> per-reaction accuracy table across all TECRDB. Analyze with `analyze_sweep.py` on
      `logs/full367/` when done.
- [ ] **DFT C-X floor test** (`tests/dft_cx_floor.py`): wB97M-V/def2-TZVPD (OMol25's own level)
      on the truncated rxn00605 core -> does DFT ΔE_elec fix the +12-16 glycosyl residual?
      Decides whether truncation+DFT is the C-X-floor fix.

## ▶ NEXT (priority order)
- [ ] If DFT test works: **truncation + DFT-electronics hybrid** (ΔE_elec DFT on truncated core
      + UMA thermal + xtb solv) as the C-X-floor fix. Validate on nucleotidyl core too. NOT fitting
      (anchors to DFT theory, not experiment).
- [ ] **AUTO_TRUNCATE hardening for DB-wide use**: real atom-mapper (RXNMapper) for atom-splitting
      cases the MCS bijection can't pair (NTP->NDP-sugar+PPi); cleaner methyl-only caps
      (avoid hemiacetal/bare-P). n_H+-conservation guard already makes it SAFE (falls back).
- [ ] **Mg regime (24% of TECRDB)**: beyond viability -> Mg-phosphate binding (ligand-substitution
      keeps Mg 6-coord so hydration cancels); the free -3 phosphate needs pH-0/explicit. Then a real
      Mg reaction at reported pMg.
- [ ] **Update figures/qm_vs_dgpredictor_top10.png** with current-best-pipeline predictions on the 10
      (use full-367 AUTO_TRUNCATE results once done; reverses = -forward).

## DONE (this session) — 6 general heuristics, NO hard-coding
- [x] **AUTO_TRUNCATE** preprocessing (`truncate.build_truncated_reaction`) + n_H+-conservation
      SAFETY GUARD. Validated: rxn00605 full -45 -> +16.5 automatic; bad truncations (rxn00545/00216)
      rejected -> full fallback (no harm).
- [x] **Auto-convergent sampling** (bounded CONV_MAX=8, TOL=2.5): self-calibrating, no fixed tiers.
- [x] **Per-reaction UQ + resolution flag**: sampling-σ -> U_samp; |ΔG|<U_samp -> UNRESOLVED
      (near-eq/isomerase). CAVEAT: U_samp is a LOWER bound (misses cancellation/electronic error).
- [x] **Spectator-anion guard**: explicit water only for spectator anions; created/destroyed -> refuse.
- [x] **pH-0 + pKa transform** for created/destroyed anions (acetylcholine err -2.9, validated).
- [x] **Systematic spectator truncation** (`truncate.py`, MCS atom-map) reproduces hand caps.
- [x] **TECRDB loader** (`tools/build_tecrdb_reactions.py`): 367/367 reactions, proton-balanced.
- [x] **Failure map**: structural (huge/floppy 55%, Mg 24%, anion 16%, isomerase 15%) +
      empirical MAE (huge/floppy 49, Mg 43, anion 38, CLEAN 24, isomerase 13). Every category -> a fix.

## Hard-10 scorecard (best current)
redox +3.5 | ester -1.9 | glycosyl->sucrose +8.8 | rxn00605 +16.5 (auto-trunc) | rxn01713 +33 (floor) |
glyoxalase +20 | nucleotidyl real -77 (electronic floor, converged) | isomerase UNRESOLVED |
Mg hydration within 6% (viable).

## Frontier limit (honest)
The C-X electronic floor (glycosyl/nucleotidyl/thioester reactive bond, +16..+25 after truncation)
is UMA's electronic error; AIMNet2≈UMA; needs DFT/CC -> the DFT test now running is the attack.

## Settled decisions (do not relitigate) — see CLAUDE.md + PIPELINE_REFERENCE.md
Fast split (UMA-Hessian thermal + xtb --sp --cosmo). Boltzmann-not-min. Bare-solute thermal.
ModelSEED ChemAxon protonation (not dimorphite). Occupancy self-selection ABANDONED.
Nothing fitted to the experimental database (TECRDB = validation only).
