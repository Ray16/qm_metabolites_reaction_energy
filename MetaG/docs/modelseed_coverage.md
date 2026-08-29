# ModelSEED thermodynamics: what exists, and where MetaG (UMA) adds value

*Measured 2026-08-29 from `ModelSEEDDatabase/Biochemistry/reaction_*.json` (the `deltag`, `status`, and
`thermodynamics` fields). Numbers are non-obsolete, non-transport reactions.*

## How ModelSEED's ΔG estimates are produced

ModelSEED's `deltag` comes from **two estimation methods, both stored** in each reaction's
`thermodynamics` field. **These are the only two methods used** (verified: the sole keys across the whole
DB are `"Group contribution"` and `"eQuilibrator"`).

| method | citation | reactions with a value | role |
|---|---|---|---|
| **Group contribution** | Jankowski et al. 2008, *Biophys J* — doi:10.1529/biophysj.107.124784 | 48,399 | computed for ~all; the fallback `deltag` |
| **eQuilibrator (component contribution)** | Noor et al. 2013, *PLoS Comp Biol* — doi:10.1371/journal.pcbi.1003098 | 15,540 | the **preferred** `deltag` where available |

Order (per ModelSEED's `Scripts/Thermodynamics/README.md`): GC stored first, then eQuilibrator overrides
"in most structures." Of the **18,992** reactions with a real (non-sentinel) `deltag`:
**eQuilibrator = 13,081 (69%)**, **Jankowski GC = 5,911 (31%)**.

**dGPredictor is NOT used by ModelSEED.** It is a separate third-party tool (Wang et al., Morgan
fingerprints + component-contribution-style regression) that we independently retrained/benchmarked. It is
the same *family* as the two above — all three are additivity/fingerprint regressions fit to **TECRDB**,
and all three share the same failure mode (they give up on the same undecomposable structures).

## Coverage crosstab (43,669 reactions)

| | count | share | meaning for MetaG |
|---|---|---|---|
| balanced + has estimate | 12,770 | 29% | **validation overlap** — compare UMA vs eQ/GC (in-distribution check) |
| **balanced + NO estimate** | **10,862** | **25%** | **UMA's unique target** — balanced but the additivity family failed |
| imbalanced + has estimate | 3,815 | 9% | estimate suspect (shouldn't be scoreable) |
| imbalanced + no estimate | 16,222 | 37% | **unscoreable by ANY physics method (incl. UMA)** |

`status == "OK"` = mass/charge balanced. Imbalanced codes: `CI:±n` (charge), `MI:*` (mass),
`CPDFORMERROR` (compound formula error).

## Implications for the generalization goal

1. **~46% of ModelSEED is imbalanced** → a **data-quality** problem, unscoreable by UMA too. Not a
   coverage opportunity (separate mass/charge-balance cleanup effort).
2. **UMA's real unique target is the ~10,862 balanced, estimate-free reactions** — where eQuilibrator *and*
   Jankowski GC both failed. "Something vs nothing," ~11k reactions.
3. **No independent reference exists on that target.** The only methods that could give one (eQ/GC/dGP) are
   exactly the ones that failed there. So validation of UMA's target must be **internal**: thermodynamic
   **cycle-closure consistency** (state-function check across reactions sharing compounds — non-trivial
   because MetaG's routing is reaction-context-dependent), **physical sanity** (sign, bounds), and
   **calibrated OOD-flagged uncertainty**. Comparison to `deltag` only works on the 12,770 overlap, and
   there it just re-confirms the in-distribution TECRDB story (UMA ~11.6 vs the eQ/GC family ~6.5).

**One-line takeaway:** ModelSEED's ΔG *is* eQuilibrator + Jankowski GC (both TECRDB-bounded); the reactions
it can't score are precisely where a first-principles method is the *only* option — and validating there
means proving self-consistency, because the incumbents left no answer to check against.

## Would GC / eQuilibrator actually *generalize* to ModelSEED?

Two different questions — separate them:

- **Coverage generalization: YES.** Group contribution is *compositional* — it decomposes a molecule into
  functional groups and sums calibrated group energies, so it returns a number for any molecule whose
  groups are all known (hence a `deltag` for ~all balanced reactions). eQuilibrator adds reactant
  contribution (direct regression) where TECRDB is dense and falls back to GC where sparse.
- **Accuracy generalization: DOUBTFUL, and unverifiable.** Group additivity is *known* to break exactly on
  the non-additive cases (conjugation, ring strain, adjacent-group and long-range electronic effects,
  unusual oxidation states) — which are over-represented in the novel ModelSEED chemistry that TECRDB
  (curated common metabolism) lacks. GC returns a number regardless, with no self-diagnosis; eQuilibrator's
  `deltagerr` grows on extrapolation, which is a partial honesty signal but not validation. There is no
  ModelSEED experiment to check either against.

**The physically-motivated expectation (unproven, but grounded):** GC/eQ error *grows with structural
novelty* (additivity degrades); UMA error is ~*novelty-independent* — measured on TECRDB, UMA's error does
NOT correlate with size/rings/aromaticity (ρ≈0.1); it fails on solvation/charge magnitude, not on novel
scaffolds (a new scaffold is just another electronic-structure calculation). So as reactions get more novel
the two error curves should **cross**: GC/eQ start better on the familiar (their TECRDB home), UMA ends
better on the frontier. The compound-disjoint hint (eQ/GC-family degrades 6.5→7.5 just holding out
*compounds*) is the visible near-edge of that curve. This crossover is the value proposition — and it can
only be *demonstrated*, not on TECRDB, but by scoring the novel ModelSEED set with internal-consistency +
OOD-flagged validation.

## Empirical test: do GC and eQ agree on ModelSEED? (ground-truth-free, `analysis/gc_vs_eq_modelseed.py`)

ModelSEED stores BOTH the Jankowski-GC and the eQuilibrator estimate for 14,621 reactions. Their mutual
disagreement is a no-experiment probe of additivity reliability:
- mean |GC−eQ| = **20 kJ/mol**, median 4.2, RMS 80.8 (a heavy tail).
- **36% disagree by >10 kJ; 13% by >30 kJ; 9% by >50 kJ**; catastrophic outliers (thousands of kJ) on
  large lipid/CoA/polyketide molecules.
- Divergence GROWS with size: mean |Δ| 8.8 (heavy<10) → 16.4 → 22.3 → 21.1 (heavy 35+). Small/common
  reactions agree (~median 4 kJ, additivity robust); large multi-group molecules diverge 2–2.5×.

**Verdict:** GC/eQ generalize on common metabolism, NOT on the rest — on 36% of ModelSEED the two
TECRDB-fit additivity methods can't agree within 10 kJ, worst where additivity is physically expected to
break (big multi-group molecules). Two consequences: (1) `deltag` is an unreliable reference on the
frontier — comparing UMA to it only validates where GC/eQ already agree (the easy set); where they diverge
is open, a place for UMA to adjudicate. (2) This is the quantified case for first-principles: not "UMA is
more accurate" (unprovable) but "the incumbents are demonstrably inconsistent on a third of the target and
worse with size, with nothing to check them against" — and UMA's novelty-independent error doesn't blow up
where GC/eQ diverge by thousands.
