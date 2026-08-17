# Path to a fully general reactive-core localizer

The accuracy lever is **localization**: compute the transformation + its local environment, cap the
conserved distal scaffold uniformly (isodesmic → the scaffold cancels in ΔG), verify. This one
principle subsumes spectator truncation, the NAD ring-cofactor, and the GSH thiol-cap. We want ONE
general mechanism, not a growing list of per-cofactor fixes.

## Current state (hybrid — pragmatic, works now)
- **`localize.py`** — general MCS localizer (grow-without-mirror + balance verify). Handles the
  **substrate tail** well (galactose→lactone, glycerate→oxoglycerate). This is the general path.
- **`cofactor_truncate.py`** — a small **curated table** of canonical cofactor cores (nicotinamide,
  cysteine-thiol; couples compose → glutathione reductase auto-handled). Extensible by one row.

## Why the general path fails *today* (the measured blocker)
`localize.py` keeps the **full NAD** (cut only 1 OH). Root cause: NAD is large and **symmetric**
(two riboses + adenine), so RDKit's `GetSubstructMatch(mcs_smarts)` returns the *wrong* symmetric
embedding → the reactant→product atom map is misaligned → `reaction_center` flags almost every atom
as "changed" → nothing is cut. GSH additionally breaks the 1-to-1 pairing (2 GSH → 1 GSSG dimer).
So the failure is **not** the localization idea — it's the **atom-mapping** underneath it.

## Concrete work to reach full generality (retires the table)
1. **Symmetry-robust atom mapping.** Replace `GetSubstructMatch` (single arbitrary embedding) with a
   map chosen to MAXIMIZE conserved-bond agreement across *all* substructure embeddings
   (`GetSubstructMatches`), or use a maximum-common-*edge*-subgraph / RascalMCES, or seed the MCS
   with the reaction center. Success test: `localize(rxn00810)` must cut NAD to ~the nicotinamide
   ring, and its ΔG must match the curated core.
2. **Many-to-one pairing** for dimer couples (2 GSH → GSSG): expand the dimer into its symmetric
   halves (truncate-v2 already does this: `GSH#0_t/GSH#1_t`) and pair each half.
3. **Radius-sensitivity as the universal verifier.** Escalate the cut radius until ΔG is stable; if
   it won't stabilize, refuse (don't trust the cut). This makes localization self-checking for *any*
   chemistry, cofactor or not.
4. **Retire the table.** Once (1)–(3) reproduce the curated NAD/GSH cores' ΔG within a few kJ,
   `cofactor_truncate` becomes redundant — keep it only as a fast cache seed for the top-10 cofactors
   (they're the most common compounds, so caching their cores is free-at-scale regardless).

## End state
One call — `localize_reaction` — that, for an arbitrary balanced reaction, extracts the reactive
core, caps the conserved scaffold, verifies by radius-sensitivity, and returns a localized reaction
whose ΔG the QM engine computes faithfully. No per-cofactor code; the table is at most a cache.
