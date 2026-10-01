# MetaG physics review + improvement campaign (2026-10-01)

Goal: physics-based fixes for the reactions MetaG still gets wrong. TECRDB (std-conditions reference,
`reactions_tecrdb_std.json`) is used to CHECK each change, never to fit it; each change needs an
identified wrong physical term and, where possible, independent evidence (FreeSolv, K_hyd, pKa tables).

## Baseline (physics_20260930b sweep, CPU reassembly, std reference)
COSMO + anchors MAE 12.39 (median 9.45, n=340); COSMO no anchors 13.73. pH-0-routed reactions MAE 13-19 vs
8.5-9 for the rest. 23 reactions returned no estimate.

## Findings (code review + per-reaction decomposition)
1. **xtb `--cosmo` has no H-bond term (Ghb = 0, Gshift = 0)**. FreeSolv (642) group decomposition of the
   solvation error (kJ/mol per group, model - exp): COSMO alcohol OH +17.8, COOH +18.8, amine +11.7;
   ALPB alcohol OH +4.3, COOH -4.9, amine +12.2; COSMO water -3.2 vs exp -26.4 (ALPB -38.4).
   -> COSMO under-solvates every polar group; the class anchors and the hydro-lyase water patch were
   absorbing this. (analysis: FreeSolv lstsq by SMARTS group, freesolv_solv_models.json)
2. **Liquid-water reference**: the chemical potential of liquid water is known exactly; using a continuum
   model of water-in-water is unphysical. Experimental ΔG_hyd(H2O) = -26.4 kJ/mol.
3. **H1 = ALPB + experimental water + NO anchors**: MAE 11.64 (median 8.20, n=331) vs COSMO+anchors 12.39.
   Glycosyl/PRT bias -23.7 -> -3.6, phosphatase +22 -> +5, CoA 19 -> 11.6, ammonia-lyase 11.9 -> 6.7.
   Δn (solute count) trend flat (std-state term confirmed). Residual: kinase/ATP class bias -10.7,
   phosphagen +22, hydratase +13, water-producing reactions +6..+10.
4. On top of H1, the textbook environment-specific carboxyl pKa table (PKA_ENV) now HELPS (11.64->11.45;
   under COSMO it hurt -- the "cancellation" was COSMO's). ZWITTERION_PH0 helps (+5 scored, 11.46).
   FREE_PPI_PKA neutral (correctness). Aldehyde hydration and PH0_BASES both needed.
5. Failures (23): 20 now score with current code (mode following). Fixed: (a) tight re-opt that does not
   converge no longer skips the mode-following test (dimethylmaleic acid); (b) thiolate/phenolate added to
   the pH-0 anion classes (GSH drawn as thiolate by ModelSEED -> unphysical thiolate/ammonium neutral form,
   xtb failures; pKa 8.7/10.0, transform -0.1 kJ, insensitive).
6. ARYLAMINE_NONBASIC (flag): an N bonded to an aromatic atom is not a basic amine (aniline 4.6, adenine N6);
   counting it hid the destroyed Asp ammonium in adenylosuccinate synthase (err -80) and spuriously fired
   the base path on adenylosuccinate lyase and THF one-carbon reactions (7 reactions change).
7. Per-reaction species-representation errors (sign-consistent with missing more-stable forms):
   indolepyruvate scored keto only (enol/hydrate missing; rxn01447 +53); truncation removed the ring-closing
   OH of 2-keto-gluconate (rxn01277 +28); DHFR truncation turned the N10 aniline into an aliphatic amine
   (rxn00686 -38; fixed by C-C-only cuts); lactim tautomers drawn for hydroxypteridines (rxn02902 +42).

## In flight
- species sweep (ALPB primary, COSMO/CPCM-X on same conformers) for current routing + variants:
  `species_worker.py`, `launch_species.sh`, lists `list_bigfirst.json`/`list_small.json` (797 species)
- H3 SOLV_RELAX: relax lowest minima on E_UMA + ΔG_solv(ALPB) (xtb gradient difference) -- test species
  `list_srtest.json` -> cache_sr
- H4 carbonyl hydration vs experimental K_hyd (`khyd_set.json`, independent of TECRDB)

## Update (later 2026-10-01)
8. **Fumarate pKa (PKA_ENV completion)**: alpha,beta-unsaturated carboxyl class added (3.75 when conjugated
   to a 2nd carboxyl -- fumaric/mesaconic/cis-aconitic; 4.35 isolated -- acrylic/crotonic/cinnamic). The
   missing class made fumarate's transform 11.6 kJ too small. H1+PKA_ENV+ZW+FREE_PPI: MAE 11.28 -> 11.04,
   hydratase 17.4 -> 10.4, tail>20 57 -> 53.
9. **H4 carbonyl hydration vs experiment (16 carbonyls, khyd_set.json)**: log K_hyd MAE COSMO 3.58 (bias
   -3.55, ~20 kJ under-hydration) vs ALPB 1.10 (bias +1.0, r 0.96). CARBONYL_HYDRATION_ALL implemented
   (every aldehyde/ketone, <=2 sites, per-form acid transform before mixing). TECRDB A/B pending sweep.
10. **H3 solution-phase relaxation: REJECTED (tested)**. Relaxing the lowest minima on E_UMA+ΔG_ALPB lowers G
    by 2-9 kJ (ATP-H4 -8.7, ADP-H3 -7.0, FBP -6.0, glucose -2.6, malate2- -4.7, citrate3- -14.7), mostly the
    same on both sides of a reaction, at 5-20x cost; does not rescue gas-phase zwitterions. Code kept behind
    SOLV_RELAX (default off).
11. **H6 keto-enol: REJECTED from literature** (no compute): phenylpyruvate is keto in water; oxaloacetate
    74% keto/18% enol/8% hydrate (0.7 kJ); simple enols 1e-7..1e-9.
12. **Lipid-phase references**: the 4 reactions that cleave/transfer a >=C12 acyl chain (retinyl-palmitate
    esterase K'=0.0028, anandamide amidohydrolase x2, carnitine palmitoyltransferase) all err -20..-46;
    measured K' imply condensation-favoured aqueous ester/amide hydrolysis -> micellar partitioning, not an
    aqueous standard-state quantity. Report as a separate, structurally pre-defined category.
13. **Bisphosphate over-stabilisation (H5 in test)**: making a neutral species with two separate phosphoryl
    groups err -20..-58 (5/5 sign-consistent: PFK, PGK, GAPDH, dephospho-CoA kinase; FBP aldolase +48).
    Isodesmic FBP + fructose -> F6P + F1P = +18.6 (ALPB) / +18.7 (COSMO) / +14.6 (CPCM-X) vs ~0 implied by
    FBP's near-additive pKa's. ACID_HB_FILTER drops conformers where an acidic O-H of one acid group H-bonds
    another acid group's O (impossible at pH 7). Test species: list_hbtest.json -> cache_hb.
14. NTP_CORE (nucleoside -> methyl cap) was never validated on kinases (only rejected on phosphagens); the
    nucleotide kinases (ATP+NMP/NDP, near-isodesmic) err -15..-26. A/B pending sweep.
