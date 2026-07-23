# QM composite for metabolic ΔrG′° — findings

Status as of 2026-07-23. Read this before running anything.

## Bottom line for the ten reactions

| method | MAE vs TECRDB (kJ/mol) |
|---|---|
| **eQuilibrator** | **3.6** — within the 6.2 experimental sd |
| predict zero | 10.5 |
| QM, isodesmic (best formulation found) | 24.6 |
| GroupContribution | 35.4 |
| QM, absolute composite | 38.3 |
| dGPredictor | 61.2 |

**The ten are already solved, by eQuilibrator, and QM cannot improve on it.**
The disagreements being adjudicated are 16–100 kJ/mol; our best QM error is 24.6,
so QM cannot referee them. eQuilibrator agrees with TECRDB to within experimental
noise on exactly the reactions dGPredictor fails — so the adjudication answer is
"dGPredictor is the outlier", and it needs no quantum chemistry.

Database-wide, on 1550 TECRDB↔ModelSEED matched reactions:
eQuilibrator covers 84% at MAE 5.5; dGPredictor covers 100% at 13.2.


## What QM *can* deliver for this project (added 2026-07-23)

The repo's goal is validating Freiburger's retrained dGPredictor against TECRDB,
and its headline weakness is "confident, localised regression on disulfide/thiol
redox" (glutathione-disulfide reductase). That is a narrower job than predicting
ΔG accurately, and QM can do it.

Mean |ΔrG'°| on the four glutathione redox reactions:

| method | mean abs |
|---|---|
| TECRDB (experiment) | 14.9 |
| eQuilibrator | 15.5 |
| **QM composite** | **35.9** |
| GroupContribution | 91.9 |
| dGPredictor (retrained) | 103.3 |

dGPredictor is built on a group decomposition, so it and GroupContribution are
NOT independent -- the two methods claiming ~100 kJ/mol share the basis that is
mis-representing thiol/disulfide chemistry. eQuilibrator and QM are the two that
do not share it, and both land near experiment.

So although QM's error on these reactions is 37-64 kJ/mol and its sign is wrong,
its MAGNITUDE independently excludes the ±100 regime -- and QM is the only method
in the comparison trained on no thermodynamic data whatsoever. That is
corroboration of the repo's finding that cannot be accused of circularity with
TECRDB, which eQuilibrator alone can be.

**This, not a competitive MAE, is the defensible QM contribution.**

## CPCM-X per conformer — scored, and rejected (added 2026-07-23)

`recompute_dgsolv_cpcmx.py` had already been run (all 23 compounds, per
conformer, same geometries) but never scored. It is now, by `score_cpcmx.py`,
on the pKa set — one deprotonation per datum, so the error is attributable to a
single solute instead of cancelling across four.

Per-species free-energy error from experimental pKa (kJ/mol):

| group | kind | n | ALPB mean | ALPB MAE | CPCM-X mean | CPCM-X MAE |
|---|---|--:|--:|--:|--:|--:|
| ammonium | cationic | 3 | −2.0 | 3.4 | −32.9 | 32.9 |
| carboxyl | anionic | 8 | +31.2 | 31.2 | −19.3 | 20.3 |
| phenol | anionic | 1 | +53.5 | 53.5 | +8.1 | 8.1 |
| thiol | anionic | 4 | +35.4 | 35.4 | +9.4 | 17.9 |
| **phosphate** | anionic | 7 | −11.9 | 23.9 | **−150.6** | **150.6** |
| ALL | anionic | 20 | +18.1 | 30.6 | −58.2 | 64.8 |

**CPCM-X does exactly what was claimed for it, and it is still worse.** On the
groups it was argued from, it works: carboxylates 31.2 → 20.3, thiols
35.4 → 17.9, phenol 53.5 → 8.1, and the acetic-acid pKa moves −8.4 units as
`recompute_dgsolv_cpcmx.py` predicted. But phosphates collapse — −150.6 kJ/mol
mean error, up to −198 on GTP — and every metabolite that matters here is a
phosphate. Anionic scatter, which is what propagates to reactions, goes
**28.3 → 71.8**.

Two further disqualifiers:

- The **cationic control fails**. BH+ → B creates no anion, so under a pure
  ion-solvation change it must not move. It moves −32.9 kJ/mol. Whatever CPCM-X
  is doing is not confined to anion solvation, so the "physics not a fitted
  constant" argument does not hold: it trades a +20.0 anion-specific error for a
  −25.3 one plus a −32.9 baseline shift.
- Its own calibration ladder is nonsense as a correction: +3.5 / −52.3 / −144.8
  / −328.6 per cumulative charge, i.e. it wants to *remove* 329 kJ/mol from ATP.

At the reaction level the two are a wash — pH-matched MAE **38.3 (ALPB) vs 37.6
(CPCM-X)** on the ten — which is exactly the cancellation the pKa set exists to
see through. Do not read the 0.7 kJ/mol as CPCM-X being fine.

One real gain, and it is diagnostic rather than predictive: uncorrected
GSH thiolate-vs-thiol sensitivity drops **104.1 → 18.3 kJ/mol**. Consistent with
the thiol row above, CPCM-X genuinely fixes thiolate solvation. If the redox
corroboration argument is written up, CPCM-X is the better solvation model *for
that specific claim* — the microspecies choice stops mattering — while remaining
unusable for anything phosphorylated.

**Bug this uncovered.** `final_model.py` read a hardcoded
`pka_calibration.json`, which on disk was the **CPCM-X** calibration, while the
default `--breakdown` is an **ALPB** composite. Every `+anion cal` and
`+species` number produced that way mixed the two and is void — including the
"33.0" and "122.6" reported earlier today. With matched calibrations the ALPB
ladder gives `+anion cal` **44.3** (worse than the uncorrected 38.3, not
better) and `+species` 69.0. Both `analyze_pka.py` and `final_model.py` now take
`PKA_G_JSON` / `PKA_OUT` / `PKA_CAL` env overrides so the pairing is explicit.

## Why the absolute formulation fails — the error budget

Experimental floor 2.5, predict-zero 11.0, so "useful" means 3–8 kJ/mol.

A solvated polyanion computed absolutely carries continuum-solvation error on
ions (15–20 kJ/mol best case), conformational error, RRHO error and electronic
error. At ~15 kJ/mol per species and four species per reaction, uncorrelated:

    sqrt(4) x 15 ~ 30 kJ/mol

Measured: scatter 42 on 130 reactions, 33.7 on the cleanest subset. The gap to
the target is a factor of 4–10 and is structural, not a tuning problem.

This independently reproduces Jinich et al., Sci Rep 2014 (DFT + continuum):
works on isomerisations, fails on multiply-charged anions, error correlates with
charge. Read that paper before writing anything up.

## Mechanisms eliminated (each with a specific test)

| mechanism | test | verdict |
|---|---|---|
| electronic structure | r2SCAN-3c substituted for MACE-POLAR, same geometry/dGsolv/G_RRHO | MAE 14.9 → 17.3, **worse**. Exonerated. |
| MLIP architecture | UMA vs MACE-POLAR per-reaction | r = 0.981, spread 7.1 vs error 40 |
| solvation | corr(error, Δ dGsolv) on the clean set | r = −0.07. Not the driver. |
| conformer undersampling | sign argument + error vs size | undersampling raises G; observed defect is negative. r(err, size) = −0.07 |
| Alberty transform | tabulated ΔfG° through the transform | ATP hydrolysis −35.3 vs published −32.5; DH scales as Δ(z²) exactly 1:4:9:16; pH slope exact. **Passes.** |
| bond-graph rearrangement | distance-perceived topology vs input, all conformers | 0 changed of 29/53/7/19/34 |
| duplicate conformer counting | RMSD-unique count | −0.9 kJ/mol, negligible |
| protonation state | pKa₂ vs assigned charge | correct |
| species-additive correction | ridge, compound-grouped CV, 49 species | **R² = 0.177** (in-sample 0.769) |

## The decisive negative result

Compound-grouped CV on per-species offsets gives **R² = 0.177**, residual MAE
28.2. To tie predict-zero requires R² ≈ 0.89; to match eQuilibrator, ≈ 0.97.

This rules out *every* per-species anchoring scheme at once: μ_H, μ_water,
cofactor E°′, anion-site corrections, and the reference-compound reformulation.
The error is not in the subspace those schemes operate on.

A pre-registered prediction ("reactions containing flagged species show ≥3× the
error of clean ones") returned **1.19×** at n = 130 and is rejected. An expanded
flag list (aldehyde hydration, open-chain sugars, tautomers) gave 0.80×.

## What did work, partially

Isodesmic referencing on **bond-change similarity** (difference Morgan
fingerprint), not shared species:

| cosine | n pairs | mean isodesmic error |
|---|---|---|
| 0.98–1.00 | 22 | 14.8 |
| 0.90–0.98 | 7 | 13.7 |
| 0.70–0.90 | 336 | 29.9 |
| < 0.40 | 4806 | 47.5 |

At cosine ≥ 0.90 (29% coverage): MAE 13.5 vs 34.7 absolute — a 2.6× gain, but
predict-zero on that subset is 7.6, so it still loses to the trivial baseline.
Shared-*species* referencing made things worse (50.4 vs 38.4, ≈ √2 × absolute:
errors add in quadrature rather than cancelling).

Caveat: isodesmic inherits the reference's error. rxn01675 improved 38.9 → 6.1;
rxn01005 degraded 1.5 → 39.5 because its best reference was itself bad.

## Bugs found and fixed

- **Standard-state term absent.** xtb RRHO is 1 atm (verified against
  Sackur-Tetrode), xtb dGsolv is "1 M gas/solution". Missing RT·ln(24.46) =
  7.93 kJ/mol per species. Cancels only when Δn = 0, so it hid in isomerisations.
  Fixed: `config.gas_1atm_to_1M_kJ`. Fixing it made MAE *worse*, which localised
  a compensating error in the same Δn≠0 reactions.
- **Quality screen too aggressive.** Counted any mode < −1 cm⁻¹ as a saddle
  point, so a methyl rotor rejected acetic acid. Now screens on magnitude
  (`IMAG_CM_TOL`, default 50 cm⁻¹); −184 cm⁻¹ is real, −5 is a rotor.
- **Name collision** — correction-ladder dict shadowed by a same-named list,
  silently returning tuples.
- **Invalid aggregation** — multiplying a mixed-charge-index group mean by site
  count over-corrected polyanions 4× (+161 vs +37 measured at q=−4).

## Unmodelled experimental variables

TECRDB reports `ionic_strength` in 25% of records and `p_mg` in 34%. Ionic
strength is fine — our 0.25 M assumption matches the TECRDB median where known.

**Mg²⁺ speciation — measured, small, NOT applied** (`mg_speciation.py`, added
2026-07-23). The composite is computed at pMg = ∞ (no Mg); many nucleotide/
phosphate measurements are in an Mg buffer at pMg 2–4. Mg enters the Alberty
transform like the proton — each binding species shifts by −RT ln(1 + K_Mg[Mg²⁺])
with tabulated log K (NTP ≈ 4.0, NDP/PPi ≈ 3.0–3.3, PRPP ≈ 4.2, mono-P/Pi ≈
1.6–1.9). On the 14 Mg-buffered reactions the **net** per-reaction correction is
+0.3 kJ/mol mean (range −4.2 … +3.2), and it moves MAE 35.5 → 36.4 — i.e. it
does not help.

This **corrects an earlier claim** here that Mg was "10+ kJ/mol unmodelled": that
was a per-*species* figure. These are phosphoryl-transfer reactions with strong
Mg binders on both sides (e.g. UTP + glucose-1-P → PPi + UDP-glucose: log K
4.0+1.6 vs 3.3+3.0), so Mg largely cancels and the net term is ≤4 kJ/mol —
verified not to be a missing-binder artifact. The "pMg-reported reactions have
higher error" correlation is **confounding, not causal**: TECRDB buffers Mg
precisely for the polyphosphate reactions, which are where the continuum
anion-solvation error is worst. The error rides on the phosphates, not the Mg.
So Mg is ruled out as the culprit, and the correction is kept as a diagnostic
only — never added to the reported composite (caveats: single-microspecies
approximation, ~2–3 kJ/mol per-K uncertainty, median-pMg vs per-measurement).

## Retracted numbers

Three headline figures died on contact with more data. Do not quote them.

- **27.7** — buggy additive extrapolation
- **7.4** — n = 4, 95% CI [0.6, 14.3], and "clean" was partly defined by QM's own errors
- **species-defect hypothesis** — pre-registered test returned 1.19×

## Where QM should be pointed instead

Not as a replacement for eQuilibrator on TECRDB: the physics cannot reach
5 kJ/mol on polyanions and the data-driven methods are already there.

The defensible case is compounds eQuilibrator cannot score (16% of matched
reactions, more of ModelSEED at large), stereochemistry (group contribution is
blind by construction), and novel metabolites. But that regime has no validation
data by definition — which is the unsolved problem and should be designed for
before more compute: held-out compound classes, stereoisomer pairs with known
relative stability, or independently measured hydration/isotope-exchange
equilibria.

Longer term the formulation with a real ceiling is explicit-solvent MD with an
MLIP plus alchemical free energy, which removes the continuum approximation that
the error budget says is fatal.
