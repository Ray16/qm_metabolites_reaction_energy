# Lessons learned — modeling choices, rationale, and dead ends

A running ledger of *why* the pipeline is built the way it is. Each entry: the problem, the choice we
made, the rationale, the evidence, and the status. Negative results are kept deliberately — they are the
expensive knowledge. Companion: `EXPLORATION_LOG.md` (term-by-term ledgers), `ERROR_SOURCE_DECOMP.md`
(where the error comes from), `CBH_PLAN.md` (the current fix).

---

## Part I — The meta-lessons (the ones that transfer)

1. **Score reactions, not molecules. Everything that cancels is free accuracy.** Absolute solvation /
   conformer / reference errors are enormous (10²–10⁵ kJ) but they cancel when the same object appears on
   both sides. Every design choice below is really "arrange the calculation so the big errors cancel."
   Corollary: a giant absolute error in a spectator is harmless; a small error in a *created/destroyed*
   group is fatal.

2. **Diagnose before building.** We measured the error source (undersampling? floppy? imbalance?
   speciation?) before writing a fix — and it repeatedly overturned the "obvious" guess (undersampling
   ruled out at 96% well-sampled; imbalance was a marker not a cause; "pH-0 erases Mg" was refuted by a
   two-minute baseline comparison). A cheap measurement saves an expensive wrong build.

3. **Separate bias from scatter — they need opposite fixes.** A sign-consistent, size-independent error
   (deamination −43 std 4) is a *reference* error → fix with physics/isodesmic referencing, and a single
   correction kills the whole class. A near-zero-mean, high-variance error (NAD-alcohol +0.1 std 14) is
   *scatter* → only method (truncation, sampling) reduces it; no offset helps. Applying the wrong tool
   (calibrating scatter, or averaging-away a bias) wastes effort.

4. **Validate before shipping, on held-out classes.** Fitted species/group corrections looked great
   in-sample and died at grouped-CV (R²=0.177). A correction only counts if it beats "predict the shared
   mean" on a class it was NOT built from. CBH is a fixed physical decomposition precisely to dodge this.

5. **Boltzmann over an ensemble, never the single minimum.** For floppy/solvated species the UMA min-E
   drifts down as you add seeds (non-convergent); the Boltzmann average converges. "More sampling" only
   helps if you average correctly.

6. **Keep clean figures and a living log.** dpi 300, no baked-in gray captions; regenerate on every new
   result. Log negative results the day you get them — they are half the value.

---

## Part II — Modeling choices and their rationale

### Truncation (auto-truncate, cofactor cores)
- **Problem:** full-molecule QM on a big floppy cofactor (NAD, CoA, UDP-sugar) carries conformer noise on
  the ADP-ribose/pantetheine tail that does NOT numerically cancel in ΔG even though the tail is chemically
  identical on both sides.
- **Choice:** replace the identical scaffold with a small covalently-faithful core (nicotinamide ring for
  NAD(P), cysteine-thiol for GSH) — an *isodesmic substitution* — keeping only the redox/reactive center.
- **Rationale:** the tail cancels analytically instead of being sampled noisily; you pay for the core once.
- **Evidence:** NAD dehydrogenases 44.6→10.4; glutathione reductase +34.7→+19.2 (both couples compose
  automatically). Truncated ≈ full-molecule error after pH-0 (~9.6) → the tail WAS the noise.
- **Status:** KEEP (`cofactor_truncate.py`, table-driven). Curated table beats general MCS because
  symmetric cofactors (two riboses+adenine) defeat atom-mapping — MCS picks the wrong copy.

### pH-0 auto-routing (neutral-species QM + analytic pKa transform)
- **Problem:** scoring pH-7 polyanions (phosphate, NTP, carboxylate) directly hits ALPB's diffuse-anion
  under-solvation → a −20..−100 kJ "anion-solvation catastrophe."
- **Choice:** protonate every anionic site to its neutral microspecies (UMA's comfortable regime), compute
  ΔG there, then bridge to pH 7 with the EXACT Alberty pKa transform `−RT ln(1+10^(pH−pKa))`.
- **Rationale:** move the calculation into the regime where continuum solvation is valid; do the
  pH/protonation bookkeeping analytically with textbook pKa's (no DB fitting).
- **Evidence:** phosphate reactions 91.5→18; helps every class EXCEPT isomerase.
- **Gotchas / guards:** (a) **isomerase gate** — pH-0 only injects spectator-anion noise when there's no
  net solvation change; skip it. (b) **H-mass-balance guard** — neutralizing net-proton redox/deamination
  produced ±1150 kJ garbage; refuse and fall back. (c) **n_H+=0** with the pKa ladder — the charged n_H+
  double-counts +1170 kJ (a bug a unit test caught). (d) Do NOT read it as "neutralize always": for
  phosphoryl transfer it is still net-beneficial vs baseline but leaves a +47 P–N/Mg residual (below).
- **Status:** KEEP, default-on with the two guards (`ph0_auto.py`).

### Explicit microsolvation only for a CREATED/DESTROYED compact anion (cluster-continuum)
- **Problem:** continuum solvation under-binds a compact high-charge-density anion (PPi, Pi) that is
  created/destroyed (no spectator partner to cancel).
- **Choice:** add a deterministic first shell of waters (`water_count.py`: 2/hard O, 1/soft S⁻, 1/cation
  N-H), Boltzmann over cluster seeds, and reference the waters to bulk liquid via the Bryantsev monomer
  cycle (`water_ref_G`) so the count need not cancel.
- **Rationale:** the first shell is where continuum fails; referencing to bulk liquid stops the ~−2·10⁵
  kJ/water from leaking into unbalanced-water reactions.
- **Guards / negative results:** explicit water for a created anion with NO partner *leaks* the ~125 kJ
  binding (acetate err −126, rxn01713 +166) → the pipeline REFUSES explicit for created/destroyed anions
  and routes them to pH-0 instead. Thermal on the BARE solute (floppy water librations don't cancel).
- **Status:** KEEP but narrow. The count must be ENOUGH + CONSISTENT, not "exactly right" — waters cancel.

### Fast split for thermal + solvation
- **Choice:** UMA-Hessian thermal (batched finite-diff on GPU) + `xtb --sp --cosmo` single-point solvation,
  replacing the CPU-bound `xtb --ohess --cosmo` bundle.
- **Rationale/evidence:** accuracy-neutral for ΔG (±3 kJ) at ~10× speed; thermal on bare solute unifies
  the implicit and explicit paths. **Status:** KEEP.

### Redox reference — canonical cofactor cores, not the giant molecule
- **Problem:** the −49 NAD bias is the floppy cofactor TAIL, not electronic; pH-0 does nothing for it.
- **Choice:** isodesmic ring cores (nicotinamide, cysteine-thiol). **Status:** KEEP; the NAD⁺/NADH couple
  is then accurate (NAD-alcohol dehydrogenases mean err +0.1). This is the anchor for everything: it proves
  UMA's redox reference is fine, so residual deamination error is on the AMINE side, not the couple.

---

## Part III — Dead ends (kept so we don't repeat them)

- **Truncation+DFT for the glycosyl floor — DEAD.** DFT(wB97M-V) ≈ UMA (+0.7 kJ) on the glycosyl core →
  the residual is the *reference-method ceiling*, not UMA. Only DLPNO-CCSD(T) moves it.
- **Occupancy self-selection (grand-potential water count) — ABANDONED + DELETED.** The self-selected
  water-count peak is a noisy, method-dependent observable AND irrelevant to ΔG (waters cancel). Replaced
  by the deterministic coordination rule. More waters ≠ safer.
- **AIMNet2 gas-phase conformer opt — fragile.** Lower aggregate MAE but breaks signs (7→5) and specific
  reactions; gas geometries under-solvate scaling with charge. Not robust.
- **COSMO-RS in the conformer ensemble — regressed** 35→108 on the Cooper ten; plain ALPB conformer
  ensemble wins. Fancier solvation ≠ better here.
- **Fitted species/group corrections — DEAD at grouped-CV** (R²=0.177); marginal error per charge even
  flips sign (polyanions partially cancel). Data (n≈367) is the bottleneck, not architecture — GNN ties
  linear group-CC held-out (~6.7). This is why the current fix is *isodesmic/CBH* (fixed physical
  decomposition), not regression.
- **Absolute QM ΔG as a metabolic predictor — DEAD** (composite 38 vs predict-zero 11); eQuilibrator
  already solves the curated ten at 3.6. QM's value is *frontier coverage* (the 42% GCM cannot score), not
  beating trained methods on the benchmark.
- **"pH-0 erases the Mg physics" for phosphagen kinases — REFUTED.** Baseline (charged) is WORSE
  (49–118 vs 44–77 after pH-0). pH-0 correctly fixes anion solvation; the +47 is genuine P–N/Mg physics on
  top → the trigger to finally test explicit Mg (a real positive systematic-bias test, unlike before).

---

## Part III½ — CBH library via microsolvation — DEAD (2026-08-17)
The CBH-2 *decomposition* is sound and scalable (one-time cached library, O(functional groups), marginal
per-reaction cost ≈0). But sourcing the library δ from per-fragment explicit **microsolvation FAILED**:
`[NH4+]` clusters wouldn't converge (δ=0 for the key species) and alkylammoniums gave implausible −175
deltas (over-binding + proton transfer R-NH₃⁺·(H₂O)ₙ→R-NH₂·H₃O⁺). Deamination Δcorr +195 vs wanted +43;
guard blew up. **Lesson: never microsolvate a Brønsted-acidic cation with explicit waters — the proton
walks.** The library needs a PT-safe, uniformly-converging reference: DLPNO-CCSD(T)+implicit (SMD/CPCM,
no explicit water to accept the proton) or tabulated single-ion ΔG_solv. Decomposition kept; library
method pivoted.

## Part IV — Deamination/lyase SOLVED by extending pH-0 to bases (2026-08-17)
The −43 deamination reference error was charged-species SOLVATION of the created NH₄⁺ and the destroyed
α-amino-acid zwitterion. **Fix = extend the validated pH-0 route to CATIONS** (`ph0_auto.build_ph0_
reaction_v2`, `PH0_BASES`): deprotonate NH₄⁺→NH₃, −NH₃⁺→−NH₂, guanidinium→guanidine to the neutral
microspecies (UMA-comfortable), bridge back with the textbook base pKa via the Alberty MIRROR form
−RT ln(1+10^(pKa−pH)); `n_H+ = −h_residual` carries the redox proton. Validated (GPU): glutamate DH
−43→−15, alanine −50→−6, leucine −39→+11, aspartate ammonia-lyase −26→+2. Experiment-free (textbook
pKa), scalable (analytic + neutral scoring), reuses proven machinery. The pKa arithmetic is only ~+2.6
kJ — **the win is scoring the NEUTRAL species**, confirming the error was ion solvation not electronics.

**Two lessons banked:**
- *Gate on created/destroyed, not presence.* Neutralizing a SPECTATOR (matched both sides) injects
  neutral-vs-ion sampling noise for zero benefit (malate-DH +23→+40 when it fired on spectator
  carboxylates). Gate: fire only when a chargeable amine C–N bond is created/destroyed
  (`_amine_cn_change`). Mirror of the anion-pH-0 created/destroyed logic.
- *Match the SMARTS to the chemistry.* The first amine-gate SMARTS caught the NAD(P)H dihydropyridine
  ring N (an enamine on sp² C, not basic) and mis-fired on malate-DH. Restrict to basic aliphatic amine
  on sp³ C (`[CX4]-[NX3;H1,H2;!$(NC=O)]`).

Residual outliers (fire but not fixed): diaminopentanoate DH (2 amines) and histidine ammonia-lyase
(imidazole) — a SECOND ionizable N my uniform pKa mishandles → per-N-environment pKa refinement TODO.

### CBH-2 microsolvation library — DEAD (see Part III½); pH-0-bases is the scalable replacement
The generic CBH-2 *decomposition* was sound but its microsolvation *library* failed (proton transfer).
pH-0-bases achieves the same physics (neutral-species scoring) WITHOUT a per-fragment library — analytic
pKa instead of microsolvation, O(1) per reaction, no proton-transfer fragility. This is the scalable win.

### Other frontier (parallel): glycosyl/thioester electronic ceiling
DLPNO-CCSD(T) pilot (`DLPNO_GLYCOSYL_PILOT.md`): confirmed real electronic ceiling — DLPNO moves the
rxn01362 core −12.6 kJ vs DFT (correlation, not solvation), error-correcting. GO for the class overnight
(cc-pVTZ + Boltzmann ensemble). DLPNO reserved for these ELECTRONIC classes only; NOT for the
solvation-driven deamination/phosphagen (where pH-0-bases / neutral-scoring is the right, scalable tool).
