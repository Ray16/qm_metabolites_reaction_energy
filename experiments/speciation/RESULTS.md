# Route 2: QC for speciation (tautomer / hydration / anomer)

Can QC pick the right microspecies from RELATIVE energies? Tested 2026-08-12 on
12 curated aqueous equilibria with literature reference dG. Each pair differs by
a proton shift, a water addition, or a ring configuration -- same or nearly-same
charge, so the polyanion solvation error that sinks absolute dG should cancel.

## The core hypothesis holds: solvation cancels

| quantity | MAE |
|---|--:|
| absolute reaction dG (full TECRDB) | 36.1 |
| **relative microspecies dG (this set)** | **10.3** |

Differencing near-identical structures removes ~2/3 of the error. This is the
regime where QC is usable, and it is exactly what group-contribution methods
are blind to (they cannot represent a tautomer or a hydrate at all).

## By class

| class | n | MAE | dominant species correct |
|---|--:|--:|--:|
| hydration | 7 | 10.2 | 5/7 |
| tautomer  | 4 | 11.5 | **4/4** |
| anomer    | 1 |  6.6 | 0/1 |
| overall   | 12 | 10.3 | 9/12 |

## What is reliable and what is not

- **Decisive equilibria: QC calls the dominant species every time.** Formaldehyde,
  acetone, glyoxylate, methylglyoxal (hydration); all four tautomers; uracil and
  cytosine -- every strongly-favored case is called correctly.
- **Near-degenerate equilibria: unreliable.** All 3 sign misses are equilibria
  within ~7 kJ of 50:50 -- glucose anomer (+/-1.4), pyruvate hydrate (+6.8),
  glyceraldehyde (-3). QC's ~10 kJ noise cannot resolve which side of ~0 these
  fall, which is intrinsic: a 10 kJ error is a factor ~50 in population.
- **Aromatic N-heterocycle tautomers: a real electronic miss.** 2-pyridone
  (QC -1.0 vs exp -18) and 4-pyridone (-2.8 vs -14) get the right dominant form
  but badly underestimate the magnitude. MACE-POLAR under-stabilises the
  amide/pyridone tautomer -- consistent with the "aromatic N" biased group in
  the full-TECRDB analysis. Worth an r2SCAN/DFT single-point check.

## Verdict for the architecture

QC speciation is a usable, generalizable module IF scoped to **decisive**
equilibria and reported as a dominant-species call plus a coarse ratio, not a
precise fraction. It does the one thing the data-driven methods cannot, at an
error (10 kJ) far below the absolute-dG wall (36). The failure modes are named
and bounded: near-degenerate cases, and aromatic-N tautomers pending a DFT check.

Reference values are literature-curated with citations in
`pipeline/speciation_validation.json`; confidence varies (formaldehyde/acetone/
2-pyridone high; glyceraldehyde/4-pyridone/cytosine medium). Expanding the set
is the next step before quoting a firm per-class MAE.
