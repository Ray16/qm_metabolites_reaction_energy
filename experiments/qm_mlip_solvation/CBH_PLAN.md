# Generic isodesmic (CBH) correction — build plan

Goal: an **experiment-free, generic** correction that cancels UMA's shared reference errors, so it
generalizes to unseen reactions. NOT fitted to TECRDB. Validated target: deamination cluster
−42.9→~4, phosphagen +47→~8 (shown cancellable via difference reactions, MAE 7.5 / 8.4).

## Method — Connectivity-Based Hierarchy, rung 2 (CBH-2 / "isoatomic")
Per molecule M:  δ(M) = Σ_atoms δ(atom-centered frag) − Σ_bonds δ(bond frag)
where atom-frag = a heavy atom + ALL its heavy neighbours (H-capped), bond-frag = the two atoms of
each heavy–heavy bond (H-capped). δ(frag) = G_high(frag) − G_UMA(frag).
Reaction correction: Δcorr = Σ_species coeff · δ(M_species).
- CBH-2 (not CBH-1) because carboxylate / phosphate / guanidinium / ammonium are charged & delocalised
  — CBH-2 keeps each atom's FULL first shell intact (acetate, methylphosphate, methylguanidinium,
  methylammonium survive as whole reference molecules), so the charge/resonance is never split.
- Spectator scaffolds (adenine, ribose, NAD tail) are UNCHANGED across the reaction → their fragments
  cancel exactly in Σ coeff·δ(M). Only the reactive-core fragments survive. This is why it generalizes
  and why the library is small.

## Correctness gates (each MUST pass before the next)
1. **Decomposition balance** (no GPU): for every reaction, the net CBH-2 reference reaction must
   conserve every element, total H, and total charge. Assert exactly. — Phase 1.
2. **Spectator cancellation** (no GPU): NAD/adenine/ribose fragments must vanish from the net set.
   Inspect the surviving library; it must be small reactive-core molecules only. — Phase 1.
3. **Library energies** (GPU/DLPNO): compute G_UMA and G_high for each surviving library molecule
   ONCE. Small set. — Phase 2.
4. **Deamination validation**: Δcorr must reproduce −43 from the COMPUTED library (not exp), collapsing
   the cluster to ~4. If it does not, the method is wrong — do not ship. — Phase 3.
5. **Held-out generality**: apply the SAME library to phosphagen + ammonia-lyase (not used to build it);
   must move them the right direction without new fitting. — Phase 4.

## Anti-overfitting discipline
- Library δ come from theory only (DLPNO-CCSD(T)+solvation, or independent small-molecule reference
  thermochem) — never from TECRDB ΔG. Memory warns fitted species-corrections die at grouped-CV
  R²=0.177; CBH is a FIXED physical decomposition, not a fit — verify it beats predict-shared-mean on
  held-out classes, else it is not adding physics.
