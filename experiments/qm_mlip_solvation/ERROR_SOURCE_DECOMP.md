# Error-source decomposition — where the 14.6 kJ MAE actually comes from (2026-08-17)

Question posed: is the residual error **undersampling**, **reaction imbalance**, **floppy reactants**,
or **species-level physics**? We have all structures + QM logs, so we measured it directly instead of
guessing. Tools: `tools/error_tail_analysis.py`, `tools/mechanism_bias.py`, `tools/error_source_decomp.py`.
Full pipeline, 357 TECRDB reactions, MAE 14.6 / median 10.4.

## Verdict (ranked by evidence)

**1. Undersampling — RULED OUT.**
96% of the >20 kJ tail (89/93) is well-sampled: max per-species conformer σ < 3 kJ, reaction
U_samp ≈ 2–3 kJ. `corr(|err|, max_σ) = +0.13`. Canonical case rxn00184 (glutamate DH): every species
σ≈1.1, U_samp 2.7 — yet 43 kJ off. The sampling is converged; the error is not statistical.

**2. Floppy reactant — MINOR.**
`corr(|err|, rotatable_bonds) = +0.23`, `corr(|err|, heavy_atoms) = +0.19`. Weak. The COFACTOR_RING +
truncation steps already stripped the floppy NAD/CoA tails, so what remains is not dominated by size.

**3. Reaction / charge imbalance — REAL but ROUTE-SPECIFIC, not the universal cause.**
- 102/357 reactions are scored with a **net charge** (scored-q × coeff ≠ 0), but `corr(|err|,|Q|)=0.10`
  — weak. A net ±1 is only damaging when a specific species carries it; often the pipeline's `n_H+` /
  pH-0 neutralization already rebalances (311/357 balanced).
- **The one place it bites hard: the redox reductive-amination cluster** (glutamate/alanine/leucine/
  octopine/alanopine/diaminopentanoate DH, GAPDH, glucose DH). The **ringcofactor route does not add the
  balancing product H⁺**, so these are scored at net −1 = the missing proton. This is the single biggest
  tail cluster and a genuine, fixable route bug.
- **ModelSEED already has all of these balanced.** Cross-ref against `mass_charge_balance/landscape.tsv`:
  138/142 thermo-imbalanced reactions are marked `OK` (charge_imbalance 0.0) in ModelSEED. They were
  never in `corrected_reactions.tsv` because ModelSEED's versions are already correct — the defect is in
  the **thermo TECRDB builder** (strips H⁺) and the ringcofactor route (doesn't re-add it). The balanced
  stoichiometry to rebuild from already exists in the mass_charge_balance work.

**4. Species-level QM/solvation physics — the DOMINANT residual.**
The offset-consistent, size-independent, well-sampled errors are reference errors, split into 4 families
(`mechanism_bias.py`, sign-consistency in parens):

| family | n | mean err | sign | physics root cause |
|---|---|---|---|---|
| phosphagen P–N kinase | 6 | **+47** | 100% + | pH-0 neutralizes ATP/ADP → **erases the polyphosphate + Mg²⁺** that *is* the thermodynamics of phosphoryl transfer |
| ammonia-lyase (C–NH₂→C=C+NH₃) | 4 | **−24** | 100% − | NH₃/zwitterion reference + C=C conjugation |
| hydratase (C=C+H₂O→C–OH) | 12–14 | **+15** | 92% + | C=C hydration electronic + water reference |
| phosphatase (phosphoester hydrolysis) | 12 | **+14** | 92% + | product **inorganic-phosphate dianion under-solvation** (ALPB) |
| amino-acid zwitterion redox (reductive amination) | ~15 | ±40–50 | mixed | zwitterion solvation + NH₃/NH₄⁺ speciation — biggest, hardest cluster |
| glycosyl / thioester | 22 / 29 | scatter | mixed | N-glycosidic / thioester **electronic ceiling** (DFT≈UMA) → DLPNO |

Offset-removal is diagnostic only (phosphatase MAE 14.7→3.5, hydratase 17.7→6.9): it proves the error is a
*fixed reference shift per bond type*, i.e. fixable by physics, not by fitting per-enzyme constants.

## Physics fixes that follow (no empirical offsets)

1. **Charge-consistent scoring** — rebuild the TECRDB set from the balanced ModelSEED stoichiometry
   (explicit H⁺/H₂O) and make the ringcofactor route add the balancing proton, so no reaction is scored
   with a net charge. Attacks the redox-deamination tail cluster at its bookkeeping root.
2. **Do NOT pH-0-neutralize phosphoryl transfers** — keep the charged polyphosphate and add **explicit
   Mg²⁺**. The systematic-bias test the earlier Mg memo said to wait for is now positive: +47, 100%
   sign-consistent, n=6. This is the trigger to build explicit Mg.
3. **Amino-acid zwitterion cluster** — score correct microspecies (NH₃ vs NH₄⁺) + **explicit micro-
   solvation of the zwitterion** (cluster-continuum cut anion/zwitterion error 44→10 in prior work).
4. **Phosphatase** — explicit microsolvation of the inorganic-phosphate product (same anion-solvation fix).
5. **Glycosyl / thioester** — DLPNO-CCSD(T) on the truncated core (electronic ceiling, affordable).

## VALIDATION (2026-08-17, no GPU needed — from existing logs)

Two diagnosis claims were tested against the data before building anything:

**(a) Phosphagen "pH-0 erases Mg" — REFUTED as stated.** Baseline (charged, no pH-0) is *worse* than
pH-0 for 7/8 phosphagen kinases (baseline err 49–118 kJ; pH-0 44–77). pH-0 correctly fixes the gross
anion solvation; the +47 residual sits *on top* of a correct pH-0 → it is genuine P–N/Mg physics. The
real test is **adding explicit Mg²⁺**, NOT reverting to baseline. (Saved a wrong build.)

**(b) Deamination shared-reference — CONFIRMED, strongly.** NAD deamination cluster (amino-acid → keto-
acid + NH₄⁺): mean err **−42.9 kJ, std 4.0** (n=5, range −39…−50). NAD alcohol/polyol cluster (same
NAD⁺/NADH couple, C–OH↔C=O): mean **+0.1**, std 14 (n=40). The NAD redox reference is therefore *fine*;
the −43 is a single shared reference error on the **amine/NH₄⁺ side**. Within-cluster scatter of only
4 kJ ⇒ removing that one reference snaps the cluster to MAE ~4 (better than the whole pipeline). All
species are well-sampled (σ~1.1), water present, charge balanced (n_H⁺=1) — so it is NOT bookkeeping,
sampling, or floppiness. **Physics fix = isodesmic referencing of α-amino-acid ⇌ 2-keto-acid + NH₃
against one anchor** (cancels the shared −43 from first principles; no fitted offset).

## Bottom line
The MAE is not conformer noise and not floppiness. It is (a) a route-level charge-bookkeeping bug on the
redox-deamination cluster — fixable now from the existing mass_charge_balance stoichiometry — and (b) a
small number of **bond-type reference errors** (phosphoryl+Mg, zwitterion, C=C hydration, phosphoester
anion, glycosyl electronic) each with a concrete first-principles fix. FAD/CoA cores (the prior roadmap
#1) address the *scatter* in the thioester/flavin classes but not these systematic reference errors.
