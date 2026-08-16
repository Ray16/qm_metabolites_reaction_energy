# QM Reaction-ΔG Pipeline — Authoritative Reference

Single source of truth for the first-principles QM pipeline that predicts metabolic reaction
ΔG′° (transformed, pH 7) for TMFA/thermodynamic flux. **Nothing is fitted to the experimental
database** — TECRDB is used for VALIDATION only. Updated 2026-08-16.

## Method (one scheme, per species)
`ΔG = Σ coeff·G_aq(species) + n_H⁺·G(H⁺,aq,pH7) [+ pKa transform]`, where per species:
`G_aq = E_elec(UMA gas) + thermal(UMA Hessian, bare solute) + ΔΔG_solv(xtb COSMO)`.
- Engine: UMA `uma-s-1p2` (OMol25), `uma` env; charge/spin `int` in `atoms.info`.
- Boltzmann ensemble over conformers (not min).

## General heuristics (NO per-reaction hard-coding)
1. **Auto-convergent sampling** (`implicit_G`): add conformer seed-batches until BOTH Boltzmann
   Gens AND min-E stop moving (<CONV_TOL=2.5 kJ for CONV_HITS=2 batches, cap CONV_MAX=8).
   Self-calibrating: rigid→2-3 batches, floppy→more. Reports σ (sampling uncertainty).
2. **Per-reaction UQ + resolution flag**: σ propagated in quadrature → `U_samp`; if |ΔG|<U_samp
   the reaction is flagged UNRESOLVED (regime-2 near-equilibrium; still reported for comparison).
3. **Spectator-anion guard** (`is_spectator_anion`): explicit first-shell water ONLY for anions
   with a charge/site-matched partner on the other side (waters cancel). Created/destroyed anions
   → REFUSED (explicit leaks ~125 kJ anion-water binding; demonstrated acetate −126, rxn01713 +166).
4. **pH-0 + empirical-pKa transform** (`pka_sites`): compute the NEUTRAL protonated microspecies
   (continuum-solvates well) + `RT ln10·(pH−pKa)` bridge. Handles created/destroyed ionizable
   groups without the solvation wall. VALIDATED: acetylcholine hydrolysis err −2.9.
5. **Systematic spectator truncation** (`truncate.py`): MCS atom-map → reaction center by bond
   change → radius-R keep → methyl cap. Removes conserved backbone (nucleotide/peptide) so its
   energy AND conformer noise cancel. Reproduces hand caps (nucleotidyl auto-derived at r5).
6. **Explicit-water reference** (`water_ref_G`): Bryantsev monomer cycle so waters reference bulk.

## Reaction data
- `reactions.json` — curated test/hard-case reactions (species=[coeff,charge,SMILES]).
- `reactions_tecrdb_all.json` — 367 TECRDB benchmark reactions (full molecules; n_H⁺ computed
  from H/charge balance; carries exp_sd = across-conditions spread = experimental noise floor).
- `RXN_FILE` env points the pipeline at any reaction file.

## TECRDB-367 structural failure-mode map (cheminformatics, no QM)
`tecrdb367_failure_flags.json`. huge/floppy 55% · CLEAN 26% · Mg-prone 24% · anion-change 16% ·
isomerase 15% · open-shell ~0% (rare in this mapped set). **Dominant challenge = huge/floppy
conformer regime — same as the hard-10 glycosyl/nucleotidyl.**

## Results by reaction type (validated)
| type | best err (kJ) | status | limiter |
|------|------|--------|---------|
| redox (NAD/NADP, rigid) | +3.5 (±3.1 UQ) | SOLVED | — |
| simple ester (acetylcholine) | −1.9 / −2.9 | SOLVED | — |
| glycosyl→sucrose/trehalose | +9 / +13 | electronic floor | C–X bond electronics |
| real nucleotidyl (01675/01005) | −77 / −42 | HARD | C–X electronic floor (NOT sampling: convergent=−76.9) |
| glyoxalase (thioester) | +20 | electronic-ish | thioester electronics |
| Mg²⁺ hydration | within 6% of −1830 | VIABLE | outer shells / +2 continuum |
| Mg-phosphate binding | +262 (broken) | needs pH-0 on −3 PPP | anion solvation of free MePPP³⁻ |
| isomerase (G6P→F6P) | −34 on +3 signal | UNRESOLVED (correct) | below QM noise → concentration-limited |
| quinone (closed-shell couple) | UMA≈AIMNet2 (Δ7) | OK | (radical semiquinone = real wall) |

## The 3 hard regimes beyond the hard-10 (see memory tecrdb-hard-regimes)
1. **Mg²⁺ (33%)** — BUILD. Viability proven (UMA runs, 2.09 Å geometry, hydration 6%). Next:
   fix free-anion (MePPP³⁻) solvation via pH-0, then a real Mg reaction at reported pMg.
2. **near-eq isomerases (11%)** — SCOPE OUT. Confirmed below noise; concentration-limited. Flag+report.
3. **open-shell cofactors** — DEMONSTRATE+DEFER. Closed-shell couples OK (UMA≈AIMNet2); only
   RADICAL/multireference species (semiquinone, Fe-S) need a theory tier we don't have.

## THE central open problem: the C–X electronic floor
glycosyl + real-nucleotidyl + glyoxalase all bottleneck on UMA's electronic error at the reactive
C–O/C–S bond (anomeric/glycosidic/thioester). CONFIRMED electronic (converged sampling still −77;
UMA vs AIMNet2 disagree +86 on glycosyl gas ΔE). AIMNet2≈UMA won't fix it (both DFT-MLIP). Needs
either a genuinely higher-theory anchor (DFT/CC on the small reactive core — not installed) or a
matched-geometry isodesmic scheme. This is the frontier for lifting ~55% of TECRDB.

## UQ caveat (IMPORTANT for TMFA) — U_samp is a LOWER BOUND
The reported `U_samp` captures **conformer sampling noise only**. The TECRDB sweep proved it is
NOT the total uncertainty: full-molecule reactions show errors of 30-64 kJ while U_samp is only
±2-5 kJ, because the dominant error is **catastrophic cancellation + C-X electronic floor**, which
conformer-σ cannot see. So DO NOT feed U_samp to TMFA as-is for untruncated large reactions — it is
overconfident. The honest total-U needs either (a) TRUNCATION first (shrinks species -> cancellation
error drops -> U_samp becomes representative), or (b) a cross-method electronic-U (UMA vs AIMNet2).
For now: treat U_samp as a floor; flag reactions with any species >~20 heavy atoms as high
method-uncertainty regardless of U_samp.

## Central lever for the huge/floppy 55%: AUTO-TRUNCATION (next general heuristic)
The sweep confirms the untruncated pipeline fails on the huge/floppy majority via catastrophic
cancellation. `truncate.py` (systematic spectator truncation) is the general, non-hard-coded fix:
integrate it as a pipeline preprocessing step so every reaction is scored on its truncated reactive
core. Caveats to harden first: MCS-bijection pairing can't represent one reactant splitting across
two products (needs a real atom-mapper for those); cap-quality (methyl-only, avoid hemiacetal/bare-P).

## Correctness / verification log
- TECRDB builder: 367/367 reactions parse + proton-balance (0 skipped); spot-checked rxn00605/579/1675.
- Reverses exactly antisymmetric for deterministic (implicit) reactions (code-correctness ✓);
  explicit path shows ~21 kJ non-closure = genuine stochastic sampling noise (expected).
- Redox reproducible across runs (+21.5±); acetylcholine 3-way isolates solvation cleanly.

## File map
`scripts/unified_pipeline.py` (engine) · `scripts/reactions*.json` (data) · `scripts/truncate.py`
(truncation) · `tests/` (diagnostics: mg_hydration, mg_phosphate_binding, quinone_*, check_water*,
gly_floor_*) · `tools/build_tecrdb_reactions.py` · `pipeline/tecrdb367_failure_flags.json`.
EXPLORATION_LOG.md = running experiment log. This file = current-state reference.
