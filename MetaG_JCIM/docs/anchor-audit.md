# Anchor Audit

Status: scientific review of the legacy anchor system against frozen
`2026-10-01c`. Anchor corrections are disabled in production
(`ANCHOR_CORRECT=False`), and this audit does not enable them.

## Standard For A Defensible Anchor

An anchor is not justified merely because subtracting a class mean improves a
benchmark. A deployable anchor must have all of the following:

1. **Defined transformation.** The corrected quantity is tied to a specific,
   structurally detectable bond or solvation change, with direction and
   stoichiometric extent.
2. **Causal error model.** Evidence identifies the missing physical term
   (electronic energy, solvation, speciation, or another term), rather than
   only observing a residual.
3. **Transferability.** The missing term is approximately constant, or has a
   physically motivated scaling law, across substrates in the declared domain.
4. **Independent reference.** The magnitude comes from experiment or a
   demonstrably higher-level calculation outside the evaluation set. A
   TECRDB-fitted mean is calibration, not independent physical validation.
5. **Uncertainty and domain.** Reference uncertainty, within-class variation,
   exclusions, and out-of-domain behavior are explicit.
6. **Method specificity.** An anchor is valid only for the exact electronic,
   solvation, microstate, and routing stack for which it was derived.

DFT agreement alone is not enough. PBE0 and UMA can share an error, and a
gas-phase comparison can only localize an aqueous discrepancy; it does not
measure the missing solvation or speciation free energy.

## Current-Pipeline Evidence

The table below evaluates the old numerical corrections on the current ALPB,
experimental-water, no-anchor production results. `Current mean` and `SD` are
signed per-transformation residuals in kJ/mol, scored against the openTECR standardized reference
(benchmark reference since 2026-10-02; against the uncorrected TECRDB only `thioester_pi` (+10.9, effect +9.1)
and `amide_hydrolysis` (+1.7 ± 19.7, effect +1.3) differ). `Old-anchor effect` is the
change in class MAE that would result from applying the stored COSMO-era
offset; positive values are worse.

| Subclass | n | Current mean ± SD | Old offset | Old-anchor effect | Disposition |
|---|---:|---:|---:|---:|---|
| `phosphagen` | 4 | +33.7 ± 10.2 | +56.1 | -11.3 | Keep as a diagnosed limitation; do not apply the old offset |
| `phosphatase_monoester` | 9 | +3.2 ± 4.0 | +23.3 | +16.3 | Retire numerical correction |
| `thioester_ppi` | 2 | +1.9 ± 4.5 | +44.6 | +39.6 | Retire numerical correction |
| `thioester_pi` | 4 | +10.8 ± 10.8 | +30.9 | +9.3 | Reject one-offset model; investigate heterogeneous members |
| `carboxyP` | 2 | -6.1 ± 13.9 | +25.2 | +21.5 | Retire numerical correction |
| `amide_hydrolysis` | 7 | +4.6 ± 16.1 | -6.6 | +1.4 | Reject one-offset model; retain only a broad limitation label |
| `adenylylate_aliphatic` | 2 reference models | not represented in current TECRDB class | +15.6 | not testable here | Provisional research hypothesis only |
| `adenylylate_aminoacid` | 3 reference models | not represented in current TECRDB class | +25.6 | not testable here | Provisional research hypothesis only |

The class counts above are diagnostic, not external validation. They must not
be used to refit and then advertise performance on the same reactions.

## Per-Family Interpretation

### Phosphagen

**Intended purpose:** correct a shared aqueous free-energy error associated
with forming a phosphoguanidinium P-N group after NTP core reduction and pH-0
routing.

**Evidence:** a neutral model reaction gives UMA-PBE0 = +7.8 kJ/mol while the
current four-reaction residual mean is +33.7 kJ/mol. This argues against a
large UMA electronic error and points toward solvation or speciation.

**Gap:** the calculation does not measure the missing aqueous term. The four
TECRDB residuals range from +19.6 to +43.3 kJ/mol, and the old +56.1 kJ/mol
offset belongs to the obsolete COSMO stack.

**Verdict:** the family detector and limitation flag are useful. A numerical
correction is not yet justified. Qualify one only with an independent
phosphoguanidinium hydration/speciation cycle or direct solution thermodynamics.

### Phosphatase Monoester

**Intended purpose:** correct systematic phosphate-monoester hydrolysis
solvation error while excluding adjacent cations and P-O-P chemistry.

**Current evidence:** ALPB reduced the mean residual to +3.2 kJ/mol with a
4.0 kJ/mol SD over nine reactions. Applying the old correction would increase
class MAE by 16.3 kJ/mol.

**Verdict:** the correction has served its historical purpose and should be
retired. The structural detector can remain as a diagnostic or uncertainty
feature, separate from point routing.

### Acyl-CoA Ligase, PPi-Producing

**Intended purpose:** correct the shared error in ATP-dependent thioester
formation through an acyl-adenylate pathway.

**Current evidence:** the two TECRDB members have residuals +5.0 and
-1.3 kJ/mol. The old +44.6 kJ/mol correction would be strongly harmful.

**Verdict:** retire the numerical correction. Two reactions cannot establish
a transferable constant, and the current method no longer exhibits the old
bias.

### Acyl-CoA Ligase, Pi-Producing

**Intended purpose:** separate direct acyl-phosphate/ADP-forming ligases from
the PPi-producing mechanism.

**Current evidence:** four residuals span +3.2 to +26.5 kJ/mol. ATP citrate
lyase is the dominant outlier. A single constant does not describe this
spread, and the old correction worsens class MAE.

**Verdict:** reject a class-wide offset. Investigate reaction representation,
microstates, metal binding, and substrate-specific solvation for the outlier
before proposing a narrower physical correction.

### Acyl-Adenylate

**Intended purpose:** correct the solution-phase error associated with forming
a carboxyl-phosphate mixed anhydride in acyl-AMP.

**Evidence:** a neutral model gives UMA-PBE0 = +1.1 kJ/mol, supporting the
claim that the electronic bond swap is not the main error. The numerical
targets, however, come from an indirect +25 kJ/mol cycle, and the aliphatic
versus amino-acid split was inferred from only two and three model reactions.

**Verdict:** chemically plausible but not deployment-qualified. Preserve it as
a research hypothesis until the complete reference cycle, source data,
standard-state transformations, propagated uncertainty, and transfer tests are
reproducible. Do not treat a gas-phase DFT agreement as validation of the
solution-phase offset.

### Carboxy-Phosphate

**Intended purpose:** correct ATP-dependent carboxylation whose
carboxy-phosphate intermediate is absent from the net equation.

**Problem:** the detector infers an unobserved mechanism from net
stoichiometry. The two current residuals have opposite practical behavior
(-15.9 and +3.7 kJ/mol), and the old offset has the wrong sign for their mean.

**Verdict:** retire the numerical correction. A mechanistic intermediate
cannot justify a net-reaction offset unless a thermodynamic cycle quantitatively
connects that intermediate to the computed error.

### Acyclic Amide Hydrolysis

**Intended purpose:** correct a shared solvation error in amide hydrolysis after
the electronic bond change was found to agree with PBE0.

**Current evidence:** seven residuals span -15.2 to +27.3 kJ/mol against the openTECR reference
(-25.6 to +27.3 against the uncorrected TECRDB; openTECR's corrected anandamide K' values move the low end). The class
mixes ordinary amides, carbamides, beta-lactam-like substrates, and long-chain
lipid measurements. Its mean is near zero and its SD is large.

**Verdict:** reject a single offset. Keep hydrolysis chemistry as a structural
descriptor, and split physical domains only when supported independently.
Lipid-phase measurements should not define an aqueous correction.

## Recommended Architecture

- Done: `routing.anchor` no longer influences full-versus-truncation decisions
  unless `ANCHOR_CORRECT` is on; route-neutral on the frozen 364-reaction
  TECRDB and 300-reaction ModelSEED panels.
- Done: structural detectors moved into `routing/reaction_families.py`, which
  carries no offsets. Detection does not imply correction.
- Archive the COSMO-era offsets with their exact method fingerprint; do not
  ship them as live defaults for ALPB.
- Represent any future correction as a typed reference cycle containing:
  target transformation, source/reference calculation, conditions, method
  fingerprint, coefficient scaling, uncertainty, applicability predicate, and
  validation dataset.
- Require a falsifiable ablation: the proposed physical term should predict
  individual residual changes, not merely reduce a fitted class mean.
