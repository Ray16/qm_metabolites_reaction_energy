# Exploration Log — Foundation-MLIP QM for biochemical reaction ΔG

A running methodology + reasoning record: which energy terms enter each
prediction, how cancellation is set up, and *why* each correction helped (or
didn't). Newest findings appended to §5. Companion to `README.md` (status
checklist) and `artifacts/*.json` (raw numbers).

---

## 1. Question & hypothesis
Can a modern **foundation MLIP (UMA / OMol25)** + explicit-solvation machinery get
**absolute** biochemical reaction ΔG right, on the reactions where the incumbents
(dGPredictor, eQuilibrator, and the prior xTB composite) fail?

The prior QM composite (xTB/GFN2 + ALPB/CPCM-X, absolute ΔG=S·G_aq) failed for two
*engine-independent* reasons, not bad electronics:
1. **Conformer noise** ±50 kJ on flexible cofactors (NAD, UDP-sugars).
2. **Anion under-solvation** by implicit continuum on pH-7 polyanions.
Hypothesis: UMA fixes coverage (charge-aware, computes polyanions) and speed, but
the two failure modes above must be beaten by *method design* (cancellation),
not by the engine alone.

## 2. Engine
`uma-s-1p2` (OMol25-trained), `fairchem-core 2.21.0`, run in the `uma` conda env.
Key fact: OMol25 task reads **total charge + spin** per structure
(`atoms.info={"charge":int,"spin":int}`) → handles ions natively. Geometry opt /
Hessian via ASE on UMA forces. Solvation add-on from the `xtb` 6.7.1 binary
(`conda run -n xtb xtb … --alpb water --sp`).

## 3. Anatomy of a predicted ΔG  (the master equation)
For a reaction, we assemble the aqueous standard ΔG'° as a sum of separable terms:

```
ΔG_aq  =  ΔE_elec        (UMA gas-phase electronic reaction energy, S·E)
       +  ΔG_thermal     (ZPE + H_thermal − T·S; UMA Hessian → ASE IdealGasThermo)
       +  ΔΔG_solv       (Σ products − Σ reactants, xTB-ALPB(water) single points)
       +  n_H⁺ · G(H⁺,aq,pH7)   (only if the reaction has a NET released/consumed proton)
```

| term | symbol | how computed | typical size | reliability here |
|---|---|---|--:|---|
| electronic | ΔE_elec | UMA min-conformer energies, S·E | 100s–1000 kJ | high (near-DFT) once noise controlled |
| thermal | ΔG_thermal | UMA Hessian → IdealGasThermo G(298) | 10–45 kJ | good **iff Δn_molecules=0** (see §4) |
| solvation | ΔΔG_solv | xTB-ALPB water − gas, on UMA geom | 10s–200s kJ | good for **cations/neutrals**; risky for polyanions unless cancelling |
| proton | G(H⁺,aq,pH7) | fixed literature const = −1130.8 (pH0) − 5.71·pH | −1171 (pH7) | load-bearing constant; ±10 kJ convention uncertainty |

`G(H⁺,aq,pH7)` = G_gas(H⁺) −26.3 + ΔGsolv(H⁺) −1104.5 − RT·ln10·pH.

## 4. Cancellation strategy — the core idea
Absolute ΔG of flexible polyanionic biomolecules is hopeless directly (§5 Step 1).
Three cancellations make it tractable:

- **(a) Spectator truncation.** In every one of the ten reactions a large moiety
  is *identical* on both sides (NAD adenine-phosphate tail; the UDP/uridine of
  UDP-sugars; the γ-Glu/Gly of glutathione). Truncate it to a small cap (methyl,
  etc.). The spectator's energy — and, critically, its **conformer noise** —
  cancels exactly. Validity requires the cap not perturb the reacting center's
  electronics; use a faithful cap when it might (capped cysteine > methanethiol).
- **(b) Molecule-count conservation → entropy cancels.** Gas-phase ideal-gas
  entropy over-counts translational/rotational freedom that is quenched in
  solution. When **Δn_molecules = 0** across the reaction (redox: 3→3; the
  transglycosylation: 2→2), those large trans/rot terms cancel and the ideal-gas
  thermal treatment is reliable. This is why we pick balanced reactions.
- **(c) Anion cancellation.** xTB-ALPB under-solvates anions, but if the same
  number/type of phosphate anion sits on **both** sides (transfers conserve the
  q−2 diphosphate), the error cancels — so we never rely on an accurate absolute
  anion ΔGsolv.

**Rule of thumb learned:** the method works when the *reactive center* is small &
rigid (nicotinamide ring) and the flexible/charged parts are spectators that
cancel. It is at risk when the reactive center itself is flexible (sugars, §5
Step 4) — then truncation removes the tail but not the local floppiness.

## 5. Chronological log (setup → result → lesson)

### Step 0 — feasibility: does UMA run on polyanions?  ✅
`probe_uma_charge.py`. Charge ladder neutral → PO₄³⁻ → pyrophosphate³⁻ (C/H/O/N/P/S).
All 9 species: finite UMA energy+forces, optimized to fmax≤0.05. **UMA covers the
polyanion regime xTB-ALPB mangled** (coverage, not yet accuracy).

### Step 1 — naive gas-phase ΔE on 6 isomerizations.  ❌ (diagnostic)
`step1_isomerization_dE.py`. Independent-conformer sampling of each species, ΔE=S·E.
MAE **23.7** kJ on reactions whose true ΔG ≤ 8 kJ; sign 33%; **median conformer
spread 49 kJ**. Reproduces the prior xTB conformer-noise wall *with UMA*.
**Lesson:** electronics were never the bottleneck. Two independent ±50 kJ absolutes
don't cancel. Isomerizations are a *worst case* — the whole molecule reorganizes,
so nothing cancels. → cancellation is mandatory.

### Step 2 — matched-scaffold (truncated) redox, gas + CHE proton.  ✅ on the key test
`step2_redox_matched.py`. rxn00070/86: 2 GSH + NAD(P)⁺ → NAD(P)H + GSSG + H⁺.
Truncated models: **1-methylnicotinamide⁺/·H** (NAD(P) redox center),
**methanethiol/dimethyldisulfide** (glutathione thiol). NADP≡NAD under truncation
(2′-phosphate is a spectator) → predicts their small 6 kJ experimental gap as ~0.
**Conformer noise collapsed 49 → 7.2 kJ.** Cancellation defeats the wall.
Assembled gas+proton ΔG −213 kJ (no solvation yet — expected).

### Step 3 — add xTB-ALPB solvation.  ◑ mixed
`step3_redox_solvation.py`. **Term ledger (methanethiol model):**
| term | kJ/mol |
|---|--:|
| ΔE_elec (UMA gas) | +957.3 |
| ΔΔG_solv (xTB-ALPB) | +207.5 |
| G(H⁺,aq,pH7) | −1170.8 |
| **ΔG_aq** | **−6.0** |
| exp | +18.0 / +11.9 |
Off by ~20 kJ, sign wrong. It's a **delicate cancellation of ~1000 kJ terms** →
any 1–2 % component error = 10–20 kJ. The missing physics: **thermal corrections**
(used ΔE not ΔG). Note the charged species here are a **cation (MNA⁺) + proton**,
NOT polyanions → ALPB is in its reliable regime.

### Step 3b — add thermal (UMA Hessian) + faithful cysteine model.  ✅ strong
`step3b_redox_thermal.py`. **Term ledger (capped-cysteine model + thermal):**
| term | kJ/mol |
|---|--:|
| ΔE_elec (UMA gas) | +929.7 |
| ΔG_thermal shift | +43.7 |
| ΔΔG_solv (xTB-ALPB) | +213.1 |
| G(H⁺,aq,pH7) | −1170.8 |
| **ΔG_aq (with thermal)** | **+15.8** |
| exp | +18.0 (NAD) / +11.9 (NADP) |
**Error −2.2 / +3.9 kJ → MAE ~3, from first principles (nothing fit).** vs
dGP-retrained off ~90, old QM composite ~38. *Both* corrections mattered: thermal
(+32→44 kJ) and the faithful cap (methanethiol +26.1 → cysteine +15.8).
**Why thermal helped:** the reaction makes/breaks bonds (S–H→S–S, aromatic→dihydro)
and releases a proton; ZPE+thermal+entropy of those changes is ~+40 kJ, and since
Δn=0 the ideal-gas entropy is trustworthy.
**Why the cap helped:** methanethiol misses the peptide inductive/solvation
environment of the cysteine thiol; capping restores it.
**Caveats (do not over-claim):** ±~10 kJ true uncertainty from the 1000-kJ
cancellation (the thiol model alone swung 10 kJ); n=2 sharing GSH/GSSG ≈ one
independent couple; the −1171 proton constant is load-bearing.

### Step 4 — INDEPENDENCE TEST: glycosyl transfer (rxn00579).  ⚠️ good number, UNTRUSTED
`step4_glycosyl_transfer.py`. UDP-glucose + fructose → UDP + sucrose (exp −4.2).
Truncate uridine→methyl. Model **verified atom- & charge-balanced, NO net proton**
→ drops the load-bearing proton constant; q−2 diphosphate on both sides → anion
solvation cancels (cancellation (c)). **Term ledger:**
| term | kJ/mol |
|---|--:|
| ΔG_elec (UMA gas) | +93.3 |
| ΔG_thermal shift | −0.8 |
| ΔΔG_solv (xTB-ALPB) | −95.5 |
| **ΔG_aq (with thermal)** | **−3.0** |
| exp | −4.2 |
Error **+1.2 kJ** (dGP-retrained −46.9, off ~43). **BUT: max conformer spread 85.8
kJ** (MeUDPGlc 85, MeUDP 63, Suc 86, Fru 38) — vs 7 kJ for the rigid nicotinamide.
The result is a cancellation of +93/−95 where each term carries that ~85 kJ noise,
so **the 1.2 kJ agreement is very possibly fortuitous** (right answer, partly wrong
reason). **Confirms the boundary:** truncation removed the *spectator* noise but the
*reactive center is the flexible part* here (sugars) → noise stays. **Key open
question: is −3.0 REPRODUCIBLE across conformer seeds?** (within ~5 kJ ⇒ noise
genuinely cancels because sugar moieties are conserved across the transfer; large
swing ⇒ luck). That reproducibility test is the next step and the diagnostic for
how to fix the sugar-conformer problem.
**Lesson:** within-species conf-spread (85) is NOT the error bar on the reaction —
the real diagnostic is reaction-ΔG reproducibility across independent samplings.

### Step 4b — reproducibility across 5 conformer seeds.  ❌ NOT reproducible (decisive)
`step4b_reproducibility.py` (parallelized 1 seed/GPU). rxn00579 ΔG_aq per seed:
−3.0, +8.6, +10.0, +4.1, +34.0 kJ → **mean +10.7, std ~12.5, range 37 kJ** (exp −4.2).
**The Step-4 −3.0 was LUCK** (seed 1). Real conformer uncertainty ±13 kJ; seed 5 blew
to +34. **Architecture verdict:** per-compound absolute G does NOT converge with
cheap 24-conformer min-sampling for flexible molecules → the clean "sample once,
cache, ΔG=S·G" scalable option is **not viable as-is**. Confirms the boundary
sharply: the method is trustworthy when the reactive center is rigid (redox, 7 kJ),
untrustworthy when flexible (sugars). The min-conformer estimate is the culprit —
24 conformers don't find a stable global min for a floppy sugar-diphosphate.
**Next fork:** (i) does convergent sampling (CREST/metadynamics or Boltzmann over
100s of conformers, cached once per compound) make per-compound G stable? — rescues
option 2 if yes; (ii) matched-transfer conformers (cancel sugar noise, reaction-
level); (iii) fragment references (scales, additivity error). All must be BATCHED.

### Step 5 — BATCHED relaxation infrastructure (for scale).  ✅ built + verified
`batched_relax.py` (`batched_fire`): relax ALL conformers/species in ONE forward
pass per step (custom batched FIRE, per-structure dt/alpha, `maxstep=0.2` cap,
converged structures freeze). Needed because the ~50k-reaction database can't use
sequential per-structure BFGS.
**Bug found + fixed (important):** first batched energies were ~1180 kJ too high —
NOT the optimizer (FIRE converged 8/8 to fmax 0.03) and NOT maxstep. Root cause:
`AtomicData.from_ase` needs **`r_data_keys=["spin","charge"]`** to read charge/spin
from `atoms.info`; without it every structure was computed NEUTRAL (batched q−2 ==
q0). One-line fix → batched energy now matches the single-structure FAIRChemCalculator
to **0.19 kJ**. Lesson: verify batched == reference on a single charged structure
before trusting any batched pipeline. Speedup grows with batch size (batch ALL
species' conformers together, not per-species).

### Step 4e — energy-TARGETED batched Boltzmann (rxn00579).  ✅ reproducible, ❌ biased
`step4e_targeted.py`. ETKDG pool 128 → batched UMA single-point RANK → relax lowest
24 → xTB solvation → Boltzmann. 5 seeds: ΔG = +21,+21,+23,+26,+29 kJ →
**std 3.1 (was 12.5), keep-k FLAT (k=10..24 within <0.2 kJ).**
**RESOLUTIONS:** (1) reproducibility SOLVED by energy-targeting + Boltzmann;
(2) **~10 conformers suffice → per-compound caching is cheap + viable (scalable)**;
(3) the Step-4 −3.0 was LUCK — proper sampling converges the method's true value
**+24, a reproducible +28 kJ BIAS vs exp −4.2.** Reframes the problem from noise
(solved) to a diagnosable systematic error. Suspects for the glycosyl +28 kJ:
uridine→methyl **truncation** not a clean spectator for a phosphoester transfer;
**xTB-ALPB anion-solvation asymmetry** (MeUDPGlc glucosyl-diester vs MeUDP free
phosphate don't cancel perfectly); microspecies/protonation. Next: decompose the
bias term-by-term. (Infra bug fixed: `conda run` xtb = 30 s/call → direct binary
0.04 s; see CLAUDE.md.)

### Step 5 — bias decomposition (rxn00579 +28 kJ).  ✅ it's SOLVATION; COSMO fixes it
`step5_bias_diag.py`. **(A) solvation model:** ALPB ΔG +18.6 (err +22.8), GBSA +17.1
(+21.3), **COSMO +0.4 (err +4.6)** — COSMO solvates the exposed phosphate anion ~18
kJ more → near-experiment. The +28 bias was xTB-ALPB/GBSA anion UNDER-solvation (the
documented wall); reaction is anion-dominated (q−2 diphosphates ~−800 kJ each) so the
anion model dominates. **(B) truncation:** methyl vs ethyl cap shifts ΔE_elec +10.5 kJ
→ uridine cap NOT a perfectly clean spectator (secondary ~10 kJ effect, << 24 kJ solv).
**Caveats:** COSMO under-solvates NEUTRALS (Fructose −10 vs ALPB −66; conductor model)
so its win may be partly fortuitous on anion-dominated reactions; MUST regression-check
COSMO on REDOX (good with ALPB, cation+neutrals) — if COSMO breaks redox, it's a
per-charge-class choice, not a universal upgrade.

### ⚠ CORRECTION — COSMO was over-claimed (single-geometry artifact)
The Step 5 COSMO +0.4 and Step 5b redox "COSMO preserves it" were computed on ONE
geometry each. The **full 5-seed Boltzmann pipeline with `--cosmo`** (per-conformer,
proper ensemble) gives glycosyl **mean +12.0, std 6.2 (was +24/3.1 with ALPB)** →
COSMO HELPS (err 28→16) but does NOT fully fix AND HURTS reproducibility (ddCOSMO is
geometry-sensitive, re-ranks the ensemble). Software used: xtb 6.7.1 `--cosmo water`
= ddCOSMO at GFN2 level (Gsolv, ε=80). Solvation fix NOT settled. **Next: test
`--cpcmx` (CPCM-X, the rigorous variant, adds non-electrostatic terms) in the FULL
pipeline — may be more accurate AND less noisy than bare ddCOSMO.**

### Step 5c — full-pipeline solvation comparison (glycosyl, 5 seeds, keep=10).  ✅ done, set aside
`step5c_solv_compare.py` — ALL four implicit models on the SAME UMA geometries.
Glycosyl rxn00579 (exp −4.2): ALPB +22.6 (std 5.4), GBSA +23.7 (4.5), COSMO +12.0
(6.2), CPCM-X +8.7 (7.1). Conclusions: ALPB/GBSA UNDER-solvate polyanions (err ~27);
COSMO/CPCM-X cut it ~15 kJ (err ~13–16). **CPCM-X TRACKS COSMO (~4 kJ below, ~same
std) — a fast COSMO surrogate, NOT independently more accurate** (user's point). std
~5–7 kJ for ALL models → reproducibility is model-agnostic, limited by keep=10 (NOT
the 3.1 at keep=24 → cross-seed wants ~24 conformers, "10 suffice" was within-seed).
**~13 kJ residual after best solvation → the ~10 kJ TRUNCATION is now the next term.**
Solvation set aside per steer; carry all models through the next reactions and judge
accuracy across the SET, not one reaction.

### Step 5b — COSMO regression on REDOX + cost (single-geometry — see correction above)
`step5b_redox_solv.py` (xtb solvation swapped on the saved step3b geometries, gas
free energy + CHE proton fixed). Redox ΔG: ALPB 15.8, GBSA 5.5, **COSMO 14.9**
(err NAD −3.1 / NADP +3.0). **COSMO KEEPS redox accurate (≈ALPB) AND fixes glycosyl
(+28→+0.4) → universal, not a per-class tradeoff.** GBSA HURTS redox (5.5) → out.
**Cost:** COSMO ~8–9× ALPB per call but absolute tiny (MNA⁺ 0.54s, CysSSCys 1.6s vs
ALPB 0.06–0.19s) → ~2 h threaded for the whole DB. NOTE: xtb `--cosmo` = cheap
implicit-continuum COSMO, NOT the expensive COSMO-RS (COSMOtherm/CPCM-X).
**DECISION: adopt COSMO as the solvation model.** Next: confirm it generalizes on
the other transfers (nucleotidyl rxn01675/01005, glycosyl rxn00605/01713) before
wiring in as default.

### Step 4d — BATCHED Boltzmann ensemble ΔG (rxn00579).  ⚠️ superseded by 4e
`step4d_batched_boltzmann.py`. Batched relax of all species' conformers + threaded
per-conformer xTB solvation + Boltzmann free energy + per-conformer disk cache.
Tests whether Boltzmann (correct statistic) + more conformers (cheap now) makes the
glycosyl ΔG reproducible (Step-4b min-only was std 12.5). [result TBD]

### Step 6 — nucleotidyl transfer (rxn01675/01005).  ❌ hard; SOLVATION RANKING FLIPS
`step6_nucleotidyl_proper.py`. Truncated MeP(−2)+MePPP(−3)→MePPMe(−2)+PPi(−3)
(balanced −5/−5, Δn=0). Proper per-species UMA-Hessian thermal + all solvation
models (fixing an earlier attempt that used the glycosyl thermal −0.8 and only
CPCM-X). ΔG (−3 microspecies, exp ~+2): **ALPB −23.8, GBSA −28.8, COSMO −37.7,
CPCM-X −52.4.** **Solvation ranking FLIPPED vs glycosyl** — ALPB BEST, COSMO/CPCM-X
WORST. Reason: this reaction CREATES **PPi** (compact charge-dense −3); COSMO/CPCM-X
OVER-solvate compact anions → too stable → ΔG too negative. (Glycosyl created an
EXPOSED phosphate → ALPB under-solvates → COSMO fixed it.) **→ NO universal
solvation model; it's anion-charge-density-dependent.** Even ALPB (best) err −25 →
nucleotidyl genuinely hard (PPi solvation + truncation). The −4 protonation (ATP⁴⁻,
CHE proton) gives garbage (−2380): gas-phase trianion deprotonation (~+1230) doesn't
cancel the proton ref without pKa treatment → use the −3 proton-balanced microspecies.
**Truncation RULED OUT** (3-seed methyl vs ethyl caps): Et−Me shift only −2.9 (ALPB)
to +6.6 (CPCM-X) — nowhere near the ~28 kJ residual. **Gas/solv decomposition** (the
anatomy): **ΔE_elec(gas) = +48** (PPi compact −3 → huge intramolecular Coulomb
repulsion destabilizes products in gas) vs **ΔΔG_solv −69 (ALPB) to −86 (CPCM-X)**
(PPi strongly solvated) → net −21 to −38 + thermal. **So it's a delicate cancellation
of a +48 gas term against a ~−75 solvation term, both dominated by the compact PPi
anion**; the ~24 kJ model spread in ΔΔG_solv IS the failure. **Verdict: nucleotidyl
fails because ΔG hinges on PPi's gas-repulsion-vs-solvation balance — a large
cancellation implicit solvation can't nail. NOT truncation. Domain boundary: the
method works when the reaction doesn't create/destroy a compact high-charge-density
anion (PPi). Fix would need explicit/cluster-continuum solvation of PPi.**

## 6. "What helped and why" ledger
| change | Δ on result | why |
|---|---|---|
| spectator truncation | noise 49→7 kJ | removes flexible spectator + its conformer noise |
| pick Δn=0 reactions | makes thermal usable | trans/rot entropy cancels |
| + xTB-ALPB solvation | −213 → toward exp | supplies the (cation) desolvation magnitude |
| + UMA-Hessian thermal | +32–44 kJ, right direction | ZPE/thermal/entropy of bond changes |
| faithful cysteine cap | −10 kJ toward exp | peptide environment on the reactive thiol |
| proton via CHE constant | enables redox | standard aqueous proton free energy |

## 7. Open caveats / risks
- Every ΔG is a small residual of ~1000 kJ terms → ±10 kJ intrinsic sensitivity.
- Proton reference is a fixed constant; reactions without a net proton (transfers)
  are cleaner and a better generality test.
- xTB-ALPB anion solvation only trustworthy when it cancels across sides.
- Truncation validity degrades if the reactive center is flexible/charged.
- Sample sizes tiny; this is proof-of-concept mapping, not a validated method.

## 8. Reproduce
Env: `uma` (UMA + rdkit + ase) for energies/thermal; calls `xtb` env binary for
solvation. Scripts in `scripts/step*`. Raw outputs in `artifacts/*.json`, geoms in
`artifacts/geom_*`. Each script prints its full term ledger.

## 9. Nucleotidyl SOLVED — explicit charge-balanced first-shell waters (Step 7b/c)
The PPi failure (§5) was PPi's compact-anion over-solvation by implicit continuum.
Fixed with explicit first-shell waters + UMA electronics + xtb(RRHO+COSMO) correction.

**Step 7b — charge-balanced fixed count (mu_water-free):** n_water = WPC*|charge| per
species. The reaction conserves charge (-5 both sides) so total waters balance
(5*WPC each side) and the bulk-water reference CANCELS -> no mu_water calibration.
| WPC | waters/side | ΔG | err (exp +1.9) |
|-----|-------------|------|-----|
| implicit | 0 | -28..-52 | -30..-54 |
| 1 | 5   | -32.3 | -34.2 |
| 2 | 10  | +5.9  | +4.0 |
| 3 | 15  | +4.0  | +2.2 |
Converged (WPC 2->3: +5.9->+4.0, both within QM noise of exp). **Nucleotidyl solved.**

**Step 7c — cluster-cycle grand potential (physics-based count, no fitting):** fixes
the step7 pinning by referencing added waters to their OWN same-size water cluster
G_wc(n) (Bryantsev/Ho cluster cycle). Ω = -RT log Σ_n exp(-(G_clus(A,n)-G_wc(n)
-G_clus(A,0))/RT). The same-size water cluster carries the same bound-water
low-freq entropy error -> cancels -> occupancy PEAKS naturally (no mu_water). Also
proves G_wc(n) cancels across same-charge species in the balanced rxn -> validates 7b.

**How many waters (transferable to the DB):** NOT a global WPC. (1) grand-potential
peak self-selects n per species by free-energy convergence; (2) structural seed rule
= ~2-3 waters per anionic O/S H-bond site + ~1 per strong donor (coordination
numbers), which is why WPC~2-3 matched here (phosphates ~2 O- per charge); (3)
converge until incremental ΔΔGsolv(n->n+1)->0; (4) geometric first-shell check.
Ref: Bryantsev-Diallo-Goddard JPCB 2008; Pliego-Riveros.

### Systematic reaction-level spectator truncation (`scripts/truncate.py`)  ✅ built + validated
Automates the hand-built caps *at the reaction level* (we have both sides in hand, so
the spectator is exactly the atom-mapped, bonding-unchanged sub-structure). Pipeline:
MCS atom-map → reaction center = {unmapped ∪ mapped-but-bonding-changed} → grow radius R
→ H-cap the severed bonds (cutting a C–C bond auto-yields a methyl). Guards: GLOBAL
atom/charge/H balance + cap-consistency (removed multiset identical both sides) +
radius-sensitivity (R vs R+1 = the Me/Et convergence knob) + rotatable-bond rigidity gate.
Known limitation: bijective MCS pairing can't represent one reactant splitting across two
products (UTP→UDP-Glc+PPi) → cap-consistency flags False there, but global BALANCE (the
stronger guarantee) still passes. Proper fix = real atom-mapper (RXNMapper) later.

**Validation — reproduces the hand caps:** nucleotidyl UTP+Glc-1-P→UDP-Glc+PPi truncates
at radius 5 to **Me-PPP + Glc-1-P → Me-PP-Glc + PPi** (uridine→methyl on both sides), i.e.
the hand model, automatically, balanced.

**Fixes the hard-ten glycosyl blow-ups (full-molecule scheme unreliable):**
| rxn | full-molecule err | truncated err | cause |
|-----|------|------|-------|
| rxn00605 (→trehalose-6-P) | **−45.2** | **+12.6** (r2) | disaccharide catastrophic cancellation (subtracting two ~1.6 M-kJ glucosyl energies); truncate acceptor to ≤1 ring → noise cancels. Residual +12.6 ≈ the genuine glycosyl-class ~+11 kJ electronic floor. |
| rxn01713 (→sinapoyl-Glc) | +41.2 | testing | carboxylate anion DESTROYED (sinapate⁻→ester) under implicit → over-solvated; truncation shrinks it but does NOT fix solvation → needs explicit first-shell water on the carboxylate (per-species triage now in pipeline). |

Lesson: the full-molecule unified scheme fails on (a) large floppy multi-ring species
(absolute-energy noise ∝ size) and (b) created/destroyed compact anions under implicit.
Truncation addresses (a) directly; (b) needs the explicit-water triage, not truncation.

---

## 2026-08-16 — Full-367 multi-node sweep + pH-0 auto-fix + truncation-v2 + DFT floor verdict

### DFT C–X floor probe → it is the REFERENCE CEILING, not UMA (negative result, closes a branch)
Ran wB97M-V/def2-TZVPD (OMol25's OWN level) on the UMA-relaxed **truncated** rxn00605 glucosyl core.
Gas ΔE_elec: **DFT +61.2 vs UMA +60.5 → DFT−UMA = +0.7 kJ** (per-species diffs +1.7/+10.4/+6.3/+5.2).
UMA reproduces its DFT reference almost exactly on the charged glycosyl core → the +12–16 glycosyl
residual is NOT UMA's electronic error and NOT a locality/charge-embedding failure; it is ωB97M-V's
own residual for anomeric/negative-hyperconjugation energetics + def2-TZVPD's light diffuseness.
**Lesson:** the "C–X electronic floor" is a reference-method ceiling, not a bond-type artifact.
Truncation+DFT-electronics hybrid is DEAD (DFT gives the same wrong gas ΔE). Fix would need
double-hybrid / DLPNO-CCSD(T). The three "floor" motifs (glycosyl C–O, nucleotidyl P–O, thioester
C–S) share not a bond but *thermochemistry dominated by charge delocalisation/resonance on a
polarised heteroatom* — here that error lives in the functional.

### pH-0 auto-routing for the Mg/NTP/PPi/phosphate anion class (`scripts/ph0_auto.py`, PH0_AUTO=1)
The full-367 sweep's dominant error engine = compact polyphosphate anions: implicit continuum
mis-solvates each charge state by ±20–50 kJ and, because phosphoryl transfer CHANGES charge
concentration, the errors DON'T cancel (baseline rxn00695 −96, rxn10427 +61 — OPPOSITE signs =
sign-varying scatter, not a subtractable bias). This is also UMA's softest regime (formal anionic
charge → all three weaknesses at once). FIX = Jinich/Alberty pH-0: protonate every anionic site to
its NEUTRAL microspecies before QM (no formal charge, no diffuse-anion basis need, continuum-solv
valid), then bridge to pH7 with EXACT Alberty transform Σ sign·RT·ln(1+10^(pH−pKa)) over textbook
per-group pKa ladders (free Pi [2.15,7.20,12.35]; terminal [1.5,6.5]; internal [1.5]; carboxyl 4.75).
No atom-mapper: neutralise all sites + per-side terms → matched spectators cancel analytically.

Three refinements, each caught/validated by testing vs TECRDB ground truth:
1. **max-anion canonicalization before neutralising** → source-protonation-independent ladder →
   fixed the 61 FRAGILE charge-state cases (adenylate-kinase transform +57 → +0.0, correct: it is
   genuinely pH-independent). The TECRDB/ChemAxon source draws the SAME terminal phosphate at
   DIFFERENT protonation across ATP/ADP/AMP → spurious transforms until canonicalized.
2. **exact Alberty form** not linear (pH−pKa): the linear form wrongly counts Pi's 12.35 site as −5 kJ.
3. **n_H+ = 0 in the pH-0 route (CRITICAL BUG the ground-truth test caught).** ΔfG'(H+)=0 in the
   transformed framework → all proton exchange is in the pKa transforms; keeping the charged rxn's
   n_H+ double-counts +G_HPLUS (~+1170 kJ). Symptom: adenylate kinase gave ΔG **+1173** while its
   ΔG_QM(neutral) was ~+2.6 (correct!). Matches the hand pH-0 reactions (all n_H+=0).

**Validation (baseline err → pH-0 err), GPU vs TECRDB:** rxn00695 −96.0→+0.2 | rxn10427 +61.3→+4.8 |
rxn00216 +32.4→−11.3 | rxn00151 −117.5→+24.0 | rxn01362 +150.2→+49.7. **MAE 91.5 → 18.0 kJ (n=5).**
Also helps phosphatases (create Pi): rxn00132 25→17.6, rxn00549 23→15.9. Residuals are diagnostic:
rxn01362 = glycosyl electronic (the DFT ceiling above) ON TOP of the anion; rxn00216 = truncation-fail
noise; rxn00151 = PEP enol-phosphate special chemistry. **Scope:** 343/367 trigger, 282 SAFE
(|transform|<8, proton-symmetric), pH-0 HELPS the created/destroyed-anion subset; harmlessness on
spectator-anion redox NOT yet confirmed (scope test inconclusive on contended T4s) → rollout gated.

### Truncation v2 (global-MCS) + the radius-sensitivity validity lesson
v1 (1:1-MCS) REFUSES 20% of TECRDB before trying: 9 multi-coeff + 64 unequal-side (atom splits
A→B+C). `truncate_v2` expands multi-coeff to unit species and takes ONE global MCS over combined
all-reactants vs all-products → isolates the reaction center across splits. Recovers ~5/9 of a
refused sample. BUT multi-case validation (the essential discipline) showed it **helped 2/3, HURT 1/3**:
rxn03643 −10.1→+1.4 (16→7 heavy, big shrink), rxn00336 +37→+26, but **rxn00065 +25→−27** — a marginal
cut (22→21 heavy) that passed balance + n_H+ + removed-fragment-consistency guards yet flipped ΔG 52 kJ.
**Lesson:** structural guards are NECESSARY but NOT SUFFICIENT. The reaction-agnostic validity test is
**radius-sensitivity**: a true spectator removal leaves ΔG invariant to cut radius. Confirmed on the good
case — rxn03643 ΔG(r2)=−24.1, ΔG(r3)=−25.4, |ΔΔG|=1.3 kJ (stable, near exp −25.5). A tuned shrinkage
threshold was REJECTED as overfitting (per review: changes must be empirical/general, not reaction-
specific). Coherent end-state: fold global-MCS + radius-sensitivity into truncate.py, delete the v2 file.

### Infra: multi-node sweep (memory [[lambda-multinode-sweep]]) + genericity audit
Sweep now spans ~37 GPUs (lambda0/1/5/6, claim-based mkdir locks, preflight + circuit-breaker) vs 8.
Node-local envs differ (lambda5 py3.11 lacked rdkit; lambda5+6 lacked the xtb env) — fixed. Genericity
audit: all pKa's are textbook functional-group constants assigned by SMARTS (not fitted to any ΔG); NO
code branches on reaction/species identity; removed the shrinkage + min_conserved_frac magic numbers.

---
## 2026-08-17 — Error-source decomposition + generic CBH corrector (Phase 1)

**Decomposition (tools/error_tail_analysis, mechanism_bias, error_source_decomp; ERROR_SOURCE_DECOMP.md).**
Measured the source of the 14.6 kJ full-pipeline MAE, not guessed:
- Undersampling RULED OUT: 96% of the >20 tail well-sampled (max per-species σ<3, U_samp~2-3);
  corr(|err|,σ)=0.13. rxn00184 σ~1.1 yet 43 off.
- Floppy MINOR: corr(|err|,rotbonds)=0.23 — cofactor-core truncation already removed the floppy tails.
- Charge imbalance real but route-specific (corr 0.10). ModelSEED already balances these
  (mass_charge_balance/landscape.tsv: 138/142 OK); defect is the thermo builder, pipeline re-patches most.
- DOMINANT = bond-type REFERENCE errors (sign-consistent, size-independent → physics not fitting):
  deamination (amino-acid→keto+NH4+) -42.9 std 4.0; NAD-alcohol +0.1 (NAD couple FINE); phosphagen P-N/Mg
  +47; ammonia-lyase -24; hydratase +15; phosphatase +14 (Pi dianion under-solv); glycosyl/thioester ceiling.

**Validation, no GPU:** deamination LOO-isodesmic 42.9→3.8; cofactor-cancelled difference reactions (real
transaminases, from logged per-species G) MAE 7.5; phosphagen difference MAE 8.4 (excl taurocyamine
sulfonate outlier). Phosphagen "pH-0 erases Mg" REFUTED — baseline is WORSE (49-118 vs 44-77); the +47 is
genuine P-N/Mg physics ON TOP of a correct pH-0.

**Generic CBH-2 corrector (scripts/cbh_correct.py, CBH_PLAN.md) — experiment-free, no TECRDB fitting.**
δ(M)=Σ_atoms atom-frag − Σ_bonds bond-frag; ΔG_corr = ΔG_UMA + Σ_frag n·[G_high−G_UMA]. CBH-2 keeps
carboxylate/phosphate/guanidinium/ammonium intact (charge/resonance never split). Phase 1 PROVEN:
per-molecule + reaction balance gate passes (elements+H+charge); spectators (adenine/ribose) cancel;
13/13 cluster rxns → a 26-molecule library of 1-5 heavy-atom fragments dominated by the implicated ions
([NH4+], methyl-phosphate, [NH3+]P) + 6 strained ring fragments (iminium/enamine from Kekulizing
nicotinamide) flagged to verify. Next: Phase 2 library δ via explicit microsolvation of the CHARGED
fragments (tiny isolated ions — sidesteps the water-bookkeeping wall); GUARD that alcohol-DH stays ~0.

### 2026-08-27 — Microspecies input-correctness audit (tools/microspecies_audit.py) + two A/B tests
Premise: a wrong input microspecies (tautomer / protonation / anomer / carbonyl hydration) gives a wrong
ΔG regardless of QM quality. Audited all 453 benchmark species, joined to per-rxn error. Naive flags
over-count — refined out phantom phosphate-P "stereocentres", stereo-strip tautomer FPs, guanidinium
amine FPs. **TRIAGE LENS: a defect causes error only if it neither CANCELS across the reaction nor
SELF-HEALS through conformer sampling.**
- **Sugar anomer — NEGATIVE (inert).** Only the anomeric C is undefined (ring C's defined). Pinning the
  dominant anomer (geometry β-detector, validated vs β-D-glucopyranose) on 5 clean reactions: β never
  beats undefined-orig, U_samp never drops. WHY: ETKDG+multiseed+Boltzmann already averages α/β = the real
  solution mixture (self-heals), and the anomer cancels (sugar both sides). Residual = anion wall. DROP.
- **Aldehyde hydration — POSITIVE (real fix).** Passes both lens tests (carbonyl created/destroyed → no
  cancel; diol needs an added water → no self-heal). gem-diol swap + water-balanced stoichiometry:
  MAE 28.4→21.8 over 6 rxns, every rxn improved (correct sign), magnitude scales with hydration tendency
  (glyoxylate 23.2→**1.8**; glycolaldehyde −29→−21; GAP weakly hydrated → small). Production form =
  QM population-weighted carbonyl⇌gem-diol MIXTURE (self-gating), not a hard swap. See memory
  [[aldehyde-hydration-signal]], [[sugar-anomer-inert]], [[microspecies-audit]].
Fleet: tools/fleet_run.sh (lambda5, 20 GPUs; lambda13 now allowed too), shared-NFS log collection.

### 2026-08-28 — Quasi-chemical theory (QCT) attempt on the anion/Mg wall — NEGATIVE (see memory [[qct-wall-verdict]])
Grounded in literature (Rempe/Pratt/Asthagiri QCT; Rufa ML/MM FEP; Bonnet-Marzari): QCT (inner-shell cluster
+ continuum) is THE established method for ion hydration. Built + validated:
- UMA on Mg: Mg(H2O)6 geom 2.09 Å, binding DFT-quality (+4.9 kJ/water). Prior Mg failures = ill-conditioned
  setups, not UMA. (mg_validation.py)
- QCT abs. Mg²⁺ hydration = −1852 (exp −1830, within convention). ~50 kJ/water n-dependence (inner-shell
  def not converged). (mg_qct.py)
- Anion QCT−COSMO gaps: acetate −21, methylphosphate²⁻ −63.7, HPO4²⁻ −45.8 kJ (QCT captures the under-
  solvation COSMO misses). (anion_qct.py)
- **BUT reaction-level FAILS: phosphatase net = gap(Pi)−gap(ester) = +17.9 kJ — WRONG SIGN (worsens the
  +17 wall) AND ill-conditioned (small diff of large uncertain terms). The reaction ΔG is a difference of
  near-equal large solvation terms → any absolute-solvation method inherits the ill-conditioning.**
Coordination-MD (coordination_md.py): built clash-free T-monitored droplet MD, fixed 2 bugs (seeding
blowup; site detection), validated sound + physically-correct per-case coordination on 8 metabolites — but
MD is NOT needed for the QCT free energy (cluster-opt beats it for rigid ions; n-stationarity moots precise
n). MD's real use = the hydratase flexible-conformer problem, not the wall.
NEXT: isodesmic / reference-reaction cancellation (redox-cofactor trick) — cancel the large phosphate
solvation against a MEASURED reference rather than computing absolute anion solvation.

### 2026-08-28 (cont.) — Wall decomposition + thioester anchor + anchor-physics VERIFIED
Isodesmic LOO per subclass (tools/isodesmic_test.py, coupled_chem_reframe.py, mg_binding_test.py):
- Wall = FIXABLE systematic classes + IRREDUCIBLE per-reaction scatter. Wired into route_anchor.py
  (ANCHOR_CORRECT): phosphagen 59.6->9.4 (+57.3), phosphatase 15->5.1 (+20.3), **thioester acyl-CoA
  25.6->6.8 (+30.0, NEW)** — gated thioester-formed AND phosphoanhydride-consumed (excludes redox
  glyoxylate-DH). Full-367 12.87->11.63 fixed-offset, 11.66 LEAVE-ONE-OUT (circularity ~0.03 kJ -> not
  overfit). Anchors are TECRDB (LOO-within-benchmark; independent Alberty = TODO).
- Mg-binding hypothesis for Mg/NTP scatter TESTED+REFUTED (corr 0.04). Mg/NTP error is the diverse COUPLED
  chemistry (amide/thioester/C-N/sulfate), only thioester forms a clean fixable class; kinase phospho-ester
  is acceptor-specific scatter (irreducible).
- ANCHOR PHYSICS VERIFIED (tools/verify_anchor_physics.py): UMA gas ΔE == DFT(PBE0) to <5 kJ (phosphagen
  +0.8, thioester -4.1) -> offsets are SOLVATION of charged groups, NOT bond-type errors (earlier label
  corrected). Lesson: verify electronic-vs-solvation, don't assert; check mass balance (unbalanced model
  gave garbage 1e5 kJ). See memory [[qct-wall-verdict]].
