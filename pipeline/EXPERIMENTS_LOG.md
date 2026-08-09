# What works / what doesn't — tested on the 367-reaction TECRDB set

Physics-first attempts to reduce QC ΔrG′° error (raw MAE 36.1). Each row was run,
not argued. Companion to `IMPROVEMENT_ROADMAP.md`.

## Doesn't work

| idea | result | why |
|---|---|---|
| **Reaction-difference cancellation** (cancel a shared cofactor, no fitting) | NAD/NADH 35.4→33.8; NADP/NADPH 49.8→33.8; **ATP/ADP 46.4→52.8 (worse)** | Error is not in the shared cofactor — it's in the *substrate* centres that change. Removing the cofactor leaves it; two substrate-phosphate errors add. Relative alchemy gets no free cancellation. |
| **Cheap explicit-water clusters for phosphate** | H2PO4⁻→HPO4²⁻ −13, but AMP²⁻ −36, PPi³⁻ −31, **PPi⁴⁻ −76, PO4³⁻ −66** | Clusters over-stabilise high-charge anions (opposite sign to continuum). Water count/placement not converged for multiply-charged. |
| **CPCM-X for phosphate** | phosphate pKa MAE 150.6, wants to remove 329 kJ/mol from ATP | Continuum solvation model collapses on phosphate; cation control also fails. |
| **Empirical linear correction on QC error** | LRO 22–25, LCO 29; dominated by empirical-direct (7.8) | QC injects per-species solvation noise ~orthogonal to ΔG; correcting it is worse than regressing ΔG directly. See `FINDINGS_correction_layer.md`. |
| **Cheap transferable descriptors** (charge/Born z²/r, groups) | remove only ~7 of 36 | Per-species solvation error is not an analytic function of charge/size. |

## Works (partially) — deploy the cheap ones

| idea | result | note |
|---|---|---|
| **Chemistry-routed solvation, at the pKa (single-species) level** | phenol ALPB 53→CPCM-X **8**; thiol 35→CPCM-X **18**; carboxylate 31→explicit **16** | Real per-species gains. But see the reaction-level result below — they do not propagate. |
| **per-species additive model (oracle)** | explains 71% of variance, floor MAE 15.9 | Confirms the error is per-species → cache per species. But a ~29% non-additive floor (conformer/RRHO/speciation) remains even with perfect solvation. |

### CPCM-X solvation routing — RAN at reaction level, does NOT help (2026-08-08)

Recomputed CPCM-X dGsolv on the existing geometries for all 299 non-phosphate
TECRDB-full species (2130 conformers, `cpcmx_dgsolv_tecrdb_full.json`), swapped
per policy, re-scored all 367 reactions (`score_solvation_routing.py`;
baseline reproduces 36.1 exactly):

| policy | MAE | \|err\|>50 |
|---|--:|--:|
| ALPB baseline | 36.1 | 99 |
| CPCM-X all non-P | **35.3** | 90 |
| CPCM-X carboxylate/phenol/thiol, no amine | 38.4 | 106 |
| CPCM-X phenol/thiol only | 36.3 | 100 |

Best case −0.8 kJ/mol; some subsets worse. **Why:** ~13 of the top-15
error-leverage species are phosphate-bearing (NAD, NADP, ATP, ADP, Pi, PPi,
acetyl-CoA, GAP, DHAP, PRPP) and are skipped because CPCM-X collapses on
phosphate. CPCM-X can only touch low-leverage carboxylates, whose per-species
gain cancels at the reaction level. The species that drive reaction error are
exactly the ones no continuum method can fix. **Cheap solvation-routing win is
not available; the phosphate wall is unavoidable.**

## The bottleneck, cleanly isolated

Best achievable per-group solvation with methods **already in hand**:

| group | best method | MAE |
|---|---|--:|
| phenol | CPCM-X | 8.1 |
| carboxylate | explicit cluster | 15.9 |
| thiol | CPCM-X | 17.9 |
| **phosphate** | ALPB (least bad) | **23.9** |

**Phosphate is the wall.** It is the most frequent charged group and no available
method — continuum or cheap explicit cluster — gets it under ~24 kJ/mol. This is
the documented method limitation. Only two ways past it:
1. **Rigorous periodic explicit-solvent FEP** (Ewald + finite-size correction)
   for the ~15 recurring phosphate species, cached one-time. Untested here; the
   only physics that might reach <10 on di-anions.
2. A **physically-justified phosphate-charge-state correction** — empirical, but
   warranted precisely because the physics methods have a documented, systematic
   phosphate failure (the user's stated criterion for allowing correction).

## Orthogonal cheap wins (the non-solvation ~12 floor)
- Tier-0 speciation: dominant tautomer/anomer/hydrate + pH-7 microstates
  (11/23 metabolites flagged; worth 12–23 kJ/mol on sensitive reactions).
- Deeper conformer sampling / quasi-RRHO for floppy sugars, nucleotides, CoA.
These are independent of the phosphate problem and cheap.
