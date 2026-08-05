# QM composite ΔrG′° pipeline

Independent QM adjudication of dGPredictor↔TECRDB disagreements. Nothing in the
pipeline is specific to a particular reaction set: reactions, metabolites and
measurement conditions are all read from data, and every correction is
calibrated on external experimental pKa values, never on the reactions scored.

## Stages

| # | script | env | where | what |
|---|--------|-----|-------|------|
| 1 | `build_inputs.py` | palm | CPU | ModelSEED reactions/metabolites → `bench_{reactions,species,metabolites}.json` |
| 1a | `audit_stereochemistry.py` | palm | CPU | structure integrity: degenerate reactions, diastereomeric ambiguity, collisions |
| 1b | `resolve_stereochemistry.py` | palm | CPU | enumerate explicit isomer states for the ambiguous compounds |
| 2 | `build_ensembles_fast.py` | palm | CPU | ETKDG embed → xtb-GFN2 opt (ALPB) → one Hessian; energy **and** RMSD dedup |
| 3 | `run_uma_ensemble_parallel.py` or `run_macepolar_parallel.py` | GPU env | **multi-GPU** | model-specific gas energy per conformer; shared composite assembly |
| 4 | `final_model.py` | palm | CPU | scores the reactions, writes `results/benchmark/perreaction_dG.csv` |
| 5 | `plot_comparison.py` | palm | CPU | figure under `results/benchmark/` |

## Stereochemical integrity (stages 1a/1b)

Two input defects were found on 2026-08-04 and both silently corrupted scored
reactions.

**Degenerate reactions.** ModelSEED stores the *keto* SMILES for
`cpd02469` "enol-Oxaloacetate" and `cpd01784` "enol-Phenylpyruvate", identical
to their keto partners (`cpd00032`, `cpd00143`; same InChIKey). Both
`rxn00266` and `rxn01004` therefore have one structure on each side, so their
Δ<sub>r</sub>G′° is exactly zero for *any* method. They were being scored and
counted as 5.7 kJ/mol errors. They are predictions of nothing and must be
excluded until the enol structures are curated.

**Diastereomeric ambiguity.** 7 compounds carry an undefined anomeric carbon.
ETKDG resolves it independently per embedding, so the "conformer ensemble"
mixes two substances: D-glucose 6-phosphate embeds 14 conformers split 5/9
across both anomers, with the ratio set by embedding luck and by the relative
energies this project has shown to be unreliable. `qm_thermo/structures.py`
now refuses such inputs (`ALLOW_AMBIGUOUS_STEREO=1` reproduces the old
numbers), and `resolve_stereochemistry.py` enumerates the explicit states —
labelled from ModelSEED's own α/β entries, not guessed here.

Counting raw "undefined stereocentres" would have reported 48 of 165 compounds.
Most of those are **phantom**: 73 phosphate/sulfate centres whose four oxygens
are resonance-equivalent (ATP alone reports three), plus delocalised systems
such as a guanidinium C=N that enumerate to a single InChIKey. After excluding
them the real count is 7, all the same chemistry. Enantiomeric ambiguity is
reported separately and deliberately allowed: mirror images share a free energy
in an achiral solvent.

`speciation.IsomerFamily` combines resolved states, using `G_mix = G_ref +
RT ln f_ref`. Populations must come from measurement and are `null` until
curated; an unresolved family reports the state spread as an uncertainty rather
than inventing a weight.

## Protonation ensemble (`--speciation chemaxon`)

The baseline evaluates one protonation state per compound; the real compound is
an equilibrium mixture. `--speciation chemaxon` adds the missing mixing term
from ModelSEED's atom-level predicted pKa/pKb sites, treating sites as
independent (the model microscopic per-atom values support):

    dG_site = -RT ln(1 + 10^-|pH - pK|)

It is applied to `G` **before** any reaction is scored, so it composes with
`--pH-mode referenced` instead of forming another parallel column.

**It is a correctness term, not an accuracy lever.** The correction is bounded
at −RT ln 2 = −1.72 kJ/mol per site, reached only when pK equals pH, and decays
to −0.24 one pH unit away. Largest compounds here are PPi (−1.82) and phosphate
(−1.58, pKa₂ ≈ 7.0); the net effect on any of the ten reactions is under
0.8 kJ/mol. Measured: baseline MAE 31.7 → 31.2, referenced 16.1 → 15.7, no sign
changes. Expect that size.

Two things it does *not* do. It does not fix a wrongly chosen microspecies —
that would be worth tens of kJ, and a stored-vs-predicted charge audit found no
such case in the benchmark's 24 compounds (the apparent NAD/NADP mismatches are
an artefact of counting only titratable charge and ignoring the permanent
pyridinium N⁺; their reduced partners NADH/NADPH match exactly, which confirms
it). And independence ignores statistical factors between equivalent sites and
electrostatic coupling between nearby ones; the bound above is what makes that
acceptable.

The empirical pKa-based anion-solvation calibration was rejected and archived
outside the active repository at
`../backup/thermodynamic_calc/anion_solvent_calibration/`. It is not part of
the production path.

The active, reproducible inputs are `build_inputs.py`,
`build_ensembles_fast.py`, and `ensemble_deep_xtb.json`. Generated reaction
scores and figures are written under `../results/benchmark/`.

## Composite

    G_aq = E_UMA(gas, per conformer)      # uma-s-1p2, OMol25 head
         + dGsolv(ALPB)                   # xtb GFN2
         + G_RRHO(thermal)                # xtb --ohess, one Hessian per compound
    → Boltzmann average → Alberty transform at pH 7 for the reported baseline

## Archived calibration result

Measured on the 10-reaction benchmark (MAE vs TECRDB, kJ/mol):

| model | MAE | signs correct | status |
|---|---|---|---|
| dGPredictor | 61.2 | 8/10 | reference |
| QM, deep ensemble, pH 7 | 40.1 | 5/10 | — |
| QM, deep ensemble, fixed-species pH midpoint | 38.3 | 5/10 | sensitivity diagnostic only |
| + anion-solvation correction (charge ladder) | 44.3 | 9/10 | REJECTED — diagnostic only |
| + microspecies on top of it | 69.0 | 9/10 | REJECTED — diagnostic only |

The reported model is the **uncorrected pH-7 fixed-microspecies** deep ensemble.
`speciation_sensitivity.py` evaluates the current pH-midpoint approximation and
the two already-computed alternative structures; it is not a general pKa model.
The calibration scripts and their inputs/outputs have been archived; the active
`final_model.py` intentionally has no empirical-correction option.

### pH-specific microspecies ensembles

`final_model.py --pH-mode families` is a separate, provenance-tracked result.
For each compound listed in `microspecies_families.json`, it selects the
explicitly computed reference microspecies and applies
`-RT ln(Z/w_reference)` from curated sequential pKas. This is a true Legendre
transform over the protonation-state partition function, rather than changing
the pH while holding a charged structure fixed. Compounds without a curated pKa
family are explicitly left as fixed microspecies; the output records coverage
and pKa provenance. It is therefore not yet a claim of complete pH treatment
for the whole ten-reaction set.

The current GSH/GSSG redox pilot uses experimentally grounded *macroscopic*
protonation ladders.  It is exact for the macrostate partition functions but
does not claim to resolve GSSG's coupled same-proton-count microstates; that
would require the published microconstants plus QM structures for the relevant
microstates.  On the four redox benchmark entries, adding GSSG to the already
selected GSH-thiol reference changed the midpoint result by only about 0.3
kJ/mol.  The observed improvement is therefore structural GSH selection, not
an unaccounted-for GSSG pH correction.

The first phosphate gate, aqueous PPi, is also now included.  Its measured
25-C pKa ladder leaves the stored HPPi3- state about 92% populated at pH
8–8.2 and changes the two PPi reactions by only −0.19 and −0.21 kJ/mol.
It therefore does not explain their much larger residuals.  Nucleotide and
sugar-phosphate families remain out of the production diagnostic until
compound-specific, metal-aware constants are assembled.

### Reaction-class calibration (experimental, opt-in)

`reaction_class_correction.py` is an intentionally separate calibration layer.
It learns only a shrunk residual by reaction class (currently
phosphate-transfer, thiolate-redox, and glycosyl-transfer labels), never a
per-metabolite offset. Evaluation is leave-*reaction-signature*-out, so forward
and reverse versions of the same reaction cannot leak into each other's fit.
The ten-reaction set is too small for calibration and correctly yields zero
calibrated out-of-fold rows with the default four-independent-signature gate.
Use `final_model.py --write-calibration-input ...` followed by the correction
CLI only after assembling an independent, substantially larger labelled set.

The anion correction is rejected, not merely unvalidated: scored against a
calibration built with the same gas and solvation models, it makes the MAE
**worse** (38.3 → 44.3). It is retained behind a flag for one reason — it
improves the signs 5/10 → 9/10, so the ladder captures the *direction* of the
anion error while failing on magnitude, and that split is evidence about where
the error lives. Never quote its MAE as a result.

Superseded numbers, for anyone holding an older draft: 41.9 / 43.7 / 40.0 / 61.9
predate the standard-state and quality-screen fixes, and the 40.0 additionally
came from an ALPB composite scored with a **CPCM-X** calibration — a silent
file-pairing bug, since made unreachable (`--cal` is required with
`--anion-corr` and has no default; calibrations are now named by model, e.g.
`pka_cal_mp.json`). A "33.0" from the same bug was never published. See
FINDINGS.md.

Two ordering traps worth knowing. The deep ensemble alone is *worse* than the
undersampled one (43.7 vs 35.5) because undersampling was accidentally cancelling
part of the anion-solvation error — sampling and solvation are only correct
together. And an earlier reported 27.7 came from a buggy additive extrapolation;
it should not be quoted.

### Anion-solvation correction — DIAGNOSIS SOLID, CORRECTION REJECTED

**Do not apply this, to this or any other reaction class.** Two candidate
functional forms each fail the test the other passes:

| form | MAE | self-consistency (thiolate vs thiol) |
|---|---|---|
| additive per anionic site, group-specific | 27.7 ✗ buggy | 6.0 kJ/mol ✓ |
| cumulative charge ladder (correct aggregation) | 44.3 | 74.6 kJ/mol ✗ |

(Both MAEs in that table are historical: 27.7 came from the buggy additive
extrapolation, and the ladder's own figure was 40.0 under the mismatched
calibration. 44.3 / 74.6 are the current, correctly-paired values.)

Each fails the test the other passes. The additive form over-corrects polyanions
~4x (+161.5 vs +37.3 measured at q=-4); the cumulative form scales correctly but
is group-blind, charging a generic 3rd-charge increment where the thiol-specific
value is far larger.

Adding the references predicted to fix this (polyanionic thiols at k=2,
tricarballylate at k=3, GTP at k=4 -- 11 -> 18 -> 22 pairs) did **not** converge
it: MAE 32.9 -> 40.0 -> 40.0, self-consistency 76.7 -> 66.4. That is the
signature of a wrong functional form, not of insufficient data.

The structural problem: the error depends on BOTH functional group and charge
state, a 4x4 grid that 22 references populate very unevenly (no thiol at k=3/4,
no phenol past k=1, phosphate spanning k=2-4 with sd ~20 kJ/mol). Within-cell
scatter (12-23 kJ/mol) may already exceed the signal.

Recommended next step is NOT more pKa pairs. Calibrate instead against tabulated
experimental ion hydration free energies -- hundreds of anions, directly
measuring the quantity that is wrong, with no increment/cumulative ambiguity --
or drop the empirical term for explicit microsolvation. And note that with ~8
independent benchmark points no functional form can actually be validated;
expanding the benchmark is a prerequisite, not a follow-up.


Continuum solvent models under-solvate anions; this is well documented in the
literature (proton-exchange/isodesmic pKa schemes and cluster-continuum
microsolvation are the standard remedies). Measured here against experimental
pKa of small acids:

    cationic acids (BH+ → B):   mean error  −2.0 kJ/mol   → proton reference is sound
    anionic  acids (AH → A−):   mean error +18.1 kJ/mol   → anion solvation is not

The cationic family is the control, and it is what makes the diagnosis
attributable: BH+ → B creates no anion, so a shift there would mean a broken
proton reference or baseline instead. It stays put under ALPB. It does *not*
under CPCM-X (−32.9), which is one of the reasons CPCM-X was rejected — see
FINDINGS.md and `score_cpcmx.py`.

Self-consistency check: the microspecies choice should stop mattering once the
anion error is handled. It does not, under the correction (GSH thiolate vs thiol
moves the redox reactions 104.1 kJ/mol uncorrected, 74.6 corrected). Switching
the *solvation model* to CPCM-X does what the correction could not — 104.1 →
18.3 — which is why CPCM-X remains the right choice for the thiol/redox
corroboration specifically, while being unusable for phosphates.

**Calibration as it stands** (`pka_cal_mp.json`, 23 pairs, after the quality
screen). The error per *added* negative charge is not constant, and reverses
sign at high charge:

    k=1: +40.5 (n=6)   k=2: +16.9 (n=8)   k=3: +14.8 (n=4)   k=4: -38.0 (n=2)

The k=4 reversal is reproducible across ATP and GTP (sd 11.9), so for large,
well-separated polyanions ALPB slightly OVER-stabilises. Small highly charged
references (PO4³⁻, P2O7⁴⁻) are excluded by the quality screen — they are
genuinely not bound in a continuum solvent without explicit waters.

### Quality screen

References and metabolites are rejected when the lowest conformer is not a true
minimum or has negative G_RRHO (impossible — the zero-point term alone is
positive). Applied *before* any comparison to experiment, so it cannot be
circular.

Screen on the imaginary mode's MAGNITUDE, not its presence: a hindered methyl/OH
rotor sits at a few negative cm⁻¹ and is harmless, a real saddle point at
hundreds (`phosphate3` is −184 cm⁻¹). An earlier presence-only test rejected
acetic acid, which should have been an immediate red flag; tolerance is now
`IMAG_CM_TOL` (default 50 cm⁻¹). Of the metabolites only methylglyoxal fails —
the same compound the speciation auditor independently flags.

## Cost

Measured on this set (23 metabolites, 605 conformers, 80 cores + 8×V100):

| stage | wall | scaling |
|---|---|---|
| conformer ensembles (CPU, xtb) | ~35 min | **dominates**; ~2 core-hours per metabolite |
| UMA scoring (8 GPUs, 16 workers) | ~3.5 min | ~0.35 GPU-min per metabolite |
| pKa calibration | ~10 min | one-time, reused across all runs |

Extrapolating: ~1000 metabolites ≈ 2000 core-hours ≈ **~25 h on 80 cores**,
with GPU time negligible. Cost is linear in (compounds × conformers) and is
tunable via `FAST_NSTART` (64 here; 24 cuts ~2.5× for modest accuracy loss),
`FAST_EMBED`, and `FAST_WINDOW_KJ`.

## Caveats on the benchmark itself

6 of 10 reactions rest on a single measurement with no reported uncertainty;
rxn00086 has sd = 6.2 kJ/mol over 13 measurements. The 4 redox reactions are 2
chemistries written both directions, so the 10 rows are ~8 independent points.
A realistic target is 10–15 kJ/mol MAE, not ~2.
