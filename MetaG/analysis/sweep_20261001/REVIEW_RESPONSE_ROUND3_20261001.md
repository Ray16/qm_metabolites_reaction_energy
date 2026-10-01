# Response to the third-round review (2026-10-01)

This answers the four findings of round 3. It supersedes conflicting statements in rounds 1–2.

**Status:**
- Code fixes are committed, with tests passing.
- The full species recompute was **stopped and restarted** after finding 1, because the bug it reports
  affected every species of the running sweep.
- Production numbers, calibration and figures will be regenerated when the recompute finishes.

## Finding 1 (High) — validated RRHO applied to the wrong conformer; stale representative geometry → fixed

**1. Accepted candidate is now returned and used.**
- `_thermal_at_minimum` returns, besides `(Gcorr, info)`, a resolution
  `{index, kind, pos, rejected}`, where kind is one of `as_is`, `artefact`, `retightened`, `mode_followed`.
- `_apply_thermal_resolution` then makes E_UMA, ΔG_solv (all solvent models), the geometry and G_corr refer to
  the same structure:
  - **as_is / artefact:** the validated correction is assigned to *that* minimum.
  - **retightened** (tight re-optimisation within the same basin): the minimum's representative is replaced by
    the tightened geometry, with E_UMA and ΔG_solv recomputed there.
  - **mode_followed:** the saddle is a first-order saddle, not a thermodynamic state, so it is **removed** from
    the ensemble. The lower structure reached by following the imaginary mode **enters as a minimum** with its
    own E_UMA, ΔG_solv and validated RRHO.
  - **rejected candidates** (imaginary mode above tolerance, no minimum reached) are removed from the ensemble.
- `_ThermalTrack.final(uniq, also, therm, j_valid)` applies the validated value to `j_valid`, not to the
  lowest-G minimum. `validated_index_is_lowest` is recorded in the species metadata.
- The same ensemble cleanup applies to the single-minimum estimator (`THERMAL_ENSEMBLE=0`).

**2. Representative geometry is updated with its energy.**
- `UniqueMinima.add` replaces E, G **and** the geometry when a lower-G duplicate arrives.
- RDKit's `GetBestRMS` aligns the stored probe rigidly onto the old frame. This is the same structure, which the
  test checks via interatomic distances.
- `_ThermalTrack` keys corrections by representative geometry, so a replaced representative gets a new Hessian
  and dropped or added minima cannot misassign corrections.
- `PHYSICS_VERSION` is bumped to `2026-10-01c` and the thermal cache key to `v3`.

**Tests:**
- `test_dedup_replaces_geometry_with_lower_g_duplicate`
- `test_thermal_track_applies_validated_correction_to_accepted_minimum` (validated value on a non-lowest
  minimum)
- `test_unique_minima_drop`

**GPU smoke test** (`list_smoke.json`, scratch cache):
- thioester core → `retightened`;
- dimethylmaleic acid (previously "no true minimum") → `artefact`;
- glucose, pyruvic acid → `as_is`;
- in all of them `validated_index_is_lowest = True`, no removals.
- The mode-followed and rejected branches did not occur in this sample; their ensemble bookkeeping is covered
  by the unit tests.

## Finding 2 (High) — POLYACID_PKA off reopens round-2 point 5 → stated explicitly; value errors remain in production

- **Policy decision by the PI:** for transferability, the production pKa layer uses **generic rules only**.
  `POLYACID_PKA` (named-compound lookup) is **off by default**.
- A topology-based carboxyl-pair rule was also implemented. It is kept **optional (`CARBOXYL_PAIRS`, default
  off)**, because its five constants come from five parent acids.
- **Consequence, stated without qualification:** the rule-table **value errors are unresolved in production**.
  They are a known limitation; the planned general fix is a site-pKa model for the QM microstate.

  | Parent acid at pH 7 | Error of the default rules (kJ/mol) |
  |---|---|
  | oxalate | −22.7 |
  | malonate | −5.6 |
  | maleate | +3.8 |
  | succinate / adipate | +1.9 |
  | citrate | +5.1 (central α-hydroxy carboxyl interacting with both outer ones) |
  | tartrate | −1.1 |
  | fumarate | +0.3 |
  | malate | ≈ 0 |

- One correction is kept because it removes a misclassification rather than adding constants: a carboxyl
  whose α-atom is itself a carboxyl carbon is no longer typed as "α-oxo" (pKa 1.8). Oxalate's two sites
  previously got 1.8 each through that misclassification.
- The statement "nine acids use cited constants by default" (round 1) is **withdrawn**.

## Finding 3 (Medium) — three-way figure framing → corrected

- The MetaG label now reads **"no TECRDB-fitted parameters; development"**. The hydration term uses a
  calibration fitted to external hydration constants, so "no fitted parameters" was false.
- The docstring describes the actual inputs and states the regimes:
  - standardized reference;
  - MetaG ΔG from the production results;
  - eQuilibrator / GC at matching conditions;
  - eQuilibrator partly circular on this reference.
- The "within ~3 kJ" claim is removed; the printed MAEs are the only quantitative statement.
- The unused absolute `DB` path is removed.
- Font rule (18 pt minimum) is enforced in all four figure scripts, tick labels included.

## Finding 4 (Low) — hydration-domain predicate → positive eligibility test

- `khyd_validation.py` selects the fit domain with the pipeline's own positive predicate,
  `len(aldehyde_hydration.hydration_sites(smi)) > 0`.
- It asserts that every member matches `_HYDRATABLE`, so a non-hydratable control can never enter the fit.
- The result is unchanged: the same 11 members, a = 0.639, b = 2.347 kJ, LOO 0.46 log units.

## Still pending (regenerated after the running recompute)

1. TECRDB production run (development MAE), recalibration, production rerun with matching intervals.
2. Cycle closure and common-reference comparison recomputed.
3. Generality report: 300 unseen ModelSEED reactions (failure rate, OOD flags, rule firing).
4. Figures regenerated.
