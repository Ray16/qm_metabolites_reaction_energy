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

---

## Update (2026-08-12): expanded to 22 pairs + DFT check on aromatic-N

Expanded set (11 hydration, 8 tautomer, 3 anomer). Numbers hold:

| class | n | MAE | dominant species correct |
|---|--:|--:|--:|
| hydration | 11 | 10.8 | 9/11 |
| tautomer  |  8 | 11.4 | 7/8 |
| anomer    |  3 |  4.2 | 1/3 |
| overall   | 22 | 10.1 | 17/22 |

New tautomer misses confirm MACE-POLAR has class-specific electronic errors:
acetylacetone enol over-stabilised (QC -20 vs exp +4.3; intramolecular H-bond),
pyridones under-stabilised. Anomers remain unresolvable (all within ~2 kJ of
50:50 -- intrinsic, not fixable).

### DFT check (r2SCAN-3c/CPCM single points + xtb RRHO)

Tested whether the aromatic-N tautomer miss is electronic (MLIP) or solvation:

| tautomer | exp | MACE err | DFT err |
|---|--:|--:|--:|
| 2-pyridone | -18 | +17.0 | **-5.1** |
| 4-pyridone | -14 | +11.2 | **-4.4** |
| cytosine   |  -8 |  +1.8 | -24.4 |
| uracil     | -18 | -16.0 | -24.6 |

**Two-headed, not one fix.** DFT decisively corrects the pyridones (a real ~20
kJ MLIP electronic error -- hydroxypyridine/pyridone), but overshoots the
nucleobases. The latter is the known continuum-solvation failure for nucleobase
tautomers (implicit solvent under-stabilises the lactim; explicit water needed),
not a DFT electronic problem -- MACE-POLAR is closer here, possibly for the
wrong reasons. Aggregate MACE 11.5 vs DFT 14.6 on these four.

### Verdict
QC speciation is usable for **decisive** equilibria (~10 kJ, dominant species
17/22) and is the only method that addresses tautomers/hydrates at all. Its
ceiling is set by the same two errors as everywhere, now small: class-specific
MLIP electronic error (DFT-fixable for O-heterocycles) and continuum solvation
(nucleobases need explicit water). A production speciation module should: use
the MLIP for hydration/anomer/simple tautomers; flag aromatic-N and
1,3-dicarbonyl tautomers for a DFT single point; and treat near-degenerate
(<5 kJ) equilibria as unresolved rather than forcing a call.
