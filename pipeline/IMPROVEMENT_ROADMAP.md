# Improving QC ΔrG′° accuracy — chemistry-aware, budget-aware roadmap

Goal: make the *physics* pipeline accurate enough to be the ΔG source at
deployment scale (10⁴–10⁵ reactions) **without** empirical linear-regression
correction (used only where a specific DFT/method limitation is understood and
not economically fixable by better physics) and **without** falling back to
group/component contribution — QC's value is being method-independent (novel
compounds, stereochemistry, no group basis). Derived from the full 367-reaction
TECRDB set (`results/benchmark/tecrdb_full_scored.json`); companion to
`FINDINGS.md` and `FINDINGS_correction_layer.md`.

## Reframe (think from scratch): don't compute absolute μ per species

ΔrG′° = Σ νᵢ μᵢ, and metabolic reactions are group transfers, so the shared
scaffold cancels. The winning move in thermochemistry is to **formulate so the
hard part cancels**, not to compute better absolute numbers. Two consequences:

1. **Solvation → relative reaction alchemy, not absolute hydration.** Absolute
   hydration of ATP vs ADP+Pi is ≈ −2000 kJ/mol each and needs 0.1% accuracy;
   the reaction difference is small. Explicit-cluster data here confirms absolute
   hydration is a losing game *both ways*: continuum under-stabilises poly-anions
   (+bias), explicit clusters over-stabilise them (PPi⁴⁻ −76, PO4³⁻ −66 kJ/mol),
   and the best model is group-dependent. Compute the reaction (or the
   deprotonation) as one alchemical transformation in explicit solvent so the
   scaffold cancels by construction.
2. **Solvation and speciation are the same physics.** pKa = deprotonation free
   energy = anion solvation. Fixing relative free energies in explicit solvent
   fixes solvation *and* the pH-7 protonation ensemble at once — they are not
   separate workstreams.

## It is not only solvation — the full error-channel budget

| channel | size | best method (physics-first) | cost |
|---|---|---|---|
| ion solvation of species that change charge | dominant, 11.6→50.8 with \|z\| | relative reaction alchemy in explicit solvent (MLIP) | expensive, cached/triaged |
| protonation microstate at pH 7 (pKa near 7 for phosphate/thiol) | 12–23 kJ/mol on sensitive rxns; 11/23 metabolites flagged | predict microstate ensemble → Alberty transform (coupled to solvation above) | mixed |
| tautomer / anomer / hydrate of the neutral | part of the ~12 neutral floor (fructose furanose↔pyranose, gem-diol, keto-enol) | cheminformatics: pick dominant species | cheap |
| conformer sampling / RRHO entropy (floppy sugars, nucleotides, CoA) | rest of neutral +6.8 bias | deeper conformer ensembles, quasi-RRHO / MD entropy | cheap–moderate |
| Mg²⁺ / metal binding | net small (+0.3) but real per nucleotide | logK in the transform; keep as diagnostic | cheap |
| electronic structure | **not the bottleneck** (r2SCAN-3c worse than MLIP) | keep MLIP (UMA/MACE) | — |

## Tiered deployment architecture (budget-aware)

Choose the method per reaction by its chemistry; spend expensive compute only
where the cheap tier's own charge signal predicts large error (validated: charge
predicts error).

- **Tier 0 — speciation front-end (always, cheap):** resolve dominant
  tautomer/anomer/hydrate and pH-7 protonation microstates for every metabolite.
  Fixes the neutral-floor speciation channel; currently only *triaged*, not applied.
- **Tier 1 — cheap physics (easy reactions):** neutral / monovalent / no
  charge-change → MLIP gas + continuum + RRHO. Already MAE ~12. Large coverage,
  ~zero cost. Improve via conformer depth.
- **Tier 2 — relative reaction alchemy (hard reactions):** poly-anion charge
  change (max\|z\|≥2) → explicit-solvent alchemical free energy of the reaction /
  deprotonation, MLIP-driven, scaffold cancels. Cached per group-transfer
  primitive (phosphoryl→Pi, hydride→NAD) so cost is one-time, amortised across
  the 10⁴–10⁵ reactions that reuse the same cofactors.

## The error budget (tested)

QC error scales monotonically with formal charge — this is continuum
ion-solvation, and it is the dominant, targetable term:

| max \|z\| in reaction | n | QC MAE | signed bias |
|---|--:|--:|--:|
| 0 (neutral) | 29 | 11.6 | +6.8 |
| 1 | 40 | 16.6 | +9.2 |
| 2 | 160 | 32.8 | +16.0 |
| ≥3 | 138 | 50.8 | +34.7 |

Decomposition of the 36.1 raw MAE:
- **71% of the variance is per-species** (additive over compounds) → fixable by
  better per-species solvation. Oracle per-species floor: **MAE 15.9**.
- **~29% is non-additive per-reaction noise** (conformer sampling, RRHO,
  speciation/tautomers, geometry) → shows up even in neutral reactions as the
  ~12 kJ/mol / +6.8 floor.

No cheap analytic descriptor recovers the per-species part: blind 13-group
LCO 28.7; charge/Born z²/r_eff 29.5 (transfers perfectly but removes only ~7);
plain Δz² 32.4. The per-species solvation error is not a function of charge/size
— it depends on each anion's specific hydration. **That is the empirical case for
explicit solvent**: you must compute each species' solvation better, per species.

## Why per-species is affordable at scale (amortization)

Solvation is a per-species property and metabolic reactions reuse a small pool of
metabolites. Compute expensive solvation **once per unique species, cache it**,
reuse across all reactions → marginal cost per reaction ≈ 0.

Correcting only the top-k recurring species (oracle):

| top-k species treated | residual MAE | % of error removed |
|---|--:|--:|
| 10 | 23.8 | 34% |
| 20 | 20.6 | 43% |
| 40 | 17.5 | 51% |

The **hot list** (highest leverage = |offset|×frequency): NAD, NADH, NADP, NADPH,
ATP, ADP, Pi, PPi, acetyl-CoA, GSSG, GSH, GAP, DHAP, PRPP. A one-time expensive
treatment of ~40 recurring charged cofactors captures ~half the total error
across the database.

## Chemistry-aware method routing (do NOT use one solvation model for all)

Per-species pKa errors from the repo's own data (`FINDINGS.md`) are group-specific.
Route the solvation method by functional group:

| species class | continuum behaviour | method to use | cost |
|---|---|---|---|
| neutral / monovalent organics | ALPB already ~ floor | keep ALPB | cheap |
| carboxylate / phenolate | ALPB 31/53 → CPCM-X 20/8 | **CPCM-X** | cheap (done) |
| thiol / thiolate (GSH, CoA) | ALPB 35 → CPCM-X 18; GSH sens 104→18 | **CPCM-X** | cheap (done) |
| **phosphate mono/di-anion** (ATP, ADP, Pi, PPi, sugar-P) | ALPB ~24 but reaction bias +16…+35; CPCM-X **collapses −150** | **explicit solvent** (microsolv ensemble or FEP), cached | expensive, one-time |
| nicotinamide redox (NAD(P)/NAD(P)H) | ring +1→0 solvation; PP-adenine backbone cancels in redox Δ | explicit on the ring, or external E°′ anchor (53→14, done) | mixed |
| cations (ammonium) | ALPB 3 (fine); CPCM-X 33 (worse) | keep ALPB | cheap |

Key chemical point: for NAD(P)-linked reactions the large per-species offset is
mostly the phosphate backbone, which **cancels** between oxidized/reduced forms;
the non-cancelling part is the nicotinamide ring redox solvation. Treat the ring,
not the whole molecule.

## The non-additive floor (second, cheaper workstream)

The ~12 kJ/mol neutral floor is not solvation. Likely contributors, each testable:
- ModelSEED speciation/tautomers (furanose vs pyranose fructose, open-chain
  sugars, gem-diol hydrates) — `FINDINGS.md` glycosyl analysis already shows this
  moves reactions 45→12.
- conformer sampling depth and RRHO quasi-harmonic treatment.
These are cheap (no new solvation physics) and independent of the solvation work.

## Recommended next build (highest leverage / lowest deployment cost)

1. **Explicit-solvent solvation for the phosphate hot-list**, cached per species.
   Prior finite-cluster attempt failed the <10 gate on phosphate
   (`experiments/explicit_water/RESULTS.md`); the rigorous route is periodic
   explicit-solvent alchemical FEP (`experiments/bulk_fep/`, currently only a
   partial leg). Gate on the two phosphate increments (H2PO4⁻→HPO4²⁻, methyl
   phosphate) reproducing experiment to ~10 kJ/mol before any reaction use.
   Because it is cached per species, the FEP cost is one-time, not per-reaction.
2. **CPCM-X swap for thiol/carboxylate/phenolate** species — already validated,
   cheap, deploy now.
3. **Speciation/tautomer + conformer fixes** for the neutral floor — parallel,
   cheap.

Compute available: 4× V100-32GB (idle). The FEP env (`desd_fep`) is not currently
present and would need rebuilding (OpenMM/openmmtools/pymbar/OpenFF).
