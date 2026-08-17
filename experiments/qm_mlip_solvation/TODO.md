# TODO — QM (UMA) reaction-ΔG pipeline

Living checklist. Authoritative state = `PIPELINE_REFERENCE.md`; reasoning = `EXPLORATION_LOG.md`.

## ▶ IN PROGRESS (running now)
- [ ] **Full-367 baseline sweep** now MULTI-NODE (~37 GPUs: lambda0/1/5/6, claim-based
      `tools/full367_worker.sh` + `launch_workers.sh`; NOT lambda13). ~110/367 done, MAE 28.9
      (baseline = implicit-anion, no pH-0/v2). Refresh `tools/full367_table.py`; monitor for
      NEW error categories (so far none: worst-12 = 8x Mg/NTP/PPi + 3x huge/floppy).

## ✅ DONE THIS SESSION (validated vs TECRDB ground truth, committed)
- [x] **pH-0 auto-routing fix** (`ph0_auto.py` + `PH0_AUTO=1`): neutral-species QM + exact Alberty
      pKa transform for the Mg/NTP/PPi/phosphate anion class. **MAE 91.5 -> 18.0 kJ** on 5 phosphate
      rxns (rxn00695 -96->+0.2, rxn10427 +61->+4.8). Refinements: max-anion canonicalization (fixes
      the 61 FRAGILE charge-state cases), full per-group pKa ladders, n_H+=0 (bug the test caught:
      +1170 kJ). Also helps phosphatases (rxn00132 25->17.6). See memory [[ph0-auto-fix]].
- [x] **truncate_v2** (`truncate_v2.py` + `TRUNC_V2=1`): global-MCS truncation for the 20% of TECRDB
      (multi-coeff + unequal-side) that v1 refused. Validated: rxn03643 -10.1->+1.4, rxn00336 +37->+26.
- [x] **DFT C-X floor test**: DFT wB97M-V ~= UMA (+0.7 kJ) -> glycosyl floor is the REFERENCE method,
      not UMA. Truncation+DFT hybrid DEAD (deferred). See memory [[dft-cx-floor-is-reference-ceiling]].

## ▶ NEXT (priority order)
- [ ] After baseline done: **apply pH-0 (+v2 truncation) second pass** to the anion/huge classes ->
      the high-quality prediction set for the majority. Confirm pH-0 HARMLESS on spectator-anion
      redox first (scope test was inconclusive on contended T4s -- redo on dedicated GPUs).
- [ ] **Harden truncate_v2 for the huge cases it still guard-rejects** (rxn01211 folate -145, the
      NTP-split rxn10427/00070): n_H+ guard rejects them; needs better global-map cap placement.
- [ ] **Integrate**: v2 as the default AUTO_TRUNCATE fallback (v1->v2->full) + PH0_AUTO for anions.
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
