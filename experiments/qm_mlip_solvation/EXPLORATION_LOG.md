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

### Step 5b — COSMO regression on REDOX + cost.  ✅ COSMO is a UNIVERSAL upgrade
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
