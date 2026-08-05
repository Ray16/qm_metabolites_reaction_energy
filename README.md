# QM/ML hybrid approach for metabolic ΔrG′° Prediction

Can a quantum-mechanical / machine-learning composite method compute accurate
standard transformed reaction Gibbs energies (Δ<sub>r</sub>G′°) for metabolic
reactions — accurately enough to *adjudicate* disagreements between the retrained
dGPredictor and experiment (TECRDB)?

**Short answer: no, not competitively — and it does not need to.** On the
benchmark reactions the data-driven method eQuilibrator already agrees with
experiment to within its noise (MAE **3.6 kJ/mol**, experimental sd 6.2), while
the pure-QM composite is at **31.7 kJ/mol**. A parameter-free external-reference
layer brings this to **16.1 kJ/mol (10/10 correct signs)** but still does not beat
the trivial predict-zero baseline (10.5) on MAE — QM cannot referee 16–100 kJ/mol
disagreements when its own error is ~20–30. The full, self-critical account —
including everything that was tried and rejected — is in
[`pipeline/FINDINGS.md`](pipeline/FINDINGS.md),
**which is the status of record; read it first.**

What QM *can* do is narrower and defensible: on the glutathione disulfide/thiol
redox reactions its magnitude (~36 kJ/mol) independently excludes dGPredictor's
~103, from a method trained on **zero** thermodynamic data and sharing none of
the group-decomposition basis that dGPredictor and GroupContribution both rest
on. That corroboration — not a competitive MAE — is the usable result.

## The method actually used

    G_aq = E_MLIP(gas, per conformer)   # UMA (uma-s-1p2, OMol25 head) or MACE-POLAR-1
         + dGsolv(ALPB)                 # xtb GFN2, implicit water
         + G_RRHO(thermal)              # xtb --ohess, quasi-RRHO, one Hessian per compound
    → Boltzmann-average conformers
    → Alberty transform to pH 7 for the reported baseline; fixed-species pH-midpoint
      values are retained only as a sensitivity diagnostic

Geometries are optimised in ALPB water; `E_gas` is a single point on that
structure, so the electronic energy and the solvation term share one geometry.
Corrections are calibrated only on **external experimental pKa values**, never on
the reactions being scored.

The structures are fixed ModelSEED-style microspecies. A Legendre term alone
does not create an equilibrium protonation-state ensemble; at pH values other
than 7, the midpoint result is sensitivity analysis only. Run
`pipeline/speciation_sensitivity.py` to quantify the leverage of the
already-computed GSH thiol and methylglyoxal hydrate states.

A note on history: an earlier design (see below) used **ORCA r2SCAN-3c DFT +
SMD**. Substituting real DFT for the MLIP at fixed geometry made the result
*worse* (MAE 14.9 → 17.3 on the clean set), which is one of several results
showing the electronic-structure method is **not** the bottleneck. The dominant
error is continuum solvation of multiply-charged anions (~15 kJ/mol per
polyanion, ~30 kJ/mol per reaction) — structural, not a tuning problem, and it
reproduces Jinich et al., *Sci. Rep.* 2014.

## Results at a glance (10 top-disagreement reactions, MAE vs TECRDB)

| method | MAE (kJ/mol) | signs | note |
|---|---:|---:|---|
| eQuilibrator | 3.6 | — | within the 6.2 experimental sd |
| predict zero | 10.5 | — | the trivial baseline |
| **QM + external references (parameter-free)** | **16.1** | **10/10** | hybrid; external anchors only — see below |
| **QM composite, pH-7 fixed-microspecies baseline** | **31.7** | 7/10 | the honest pure-QM number |
| GroupContribution | 35.4 | — | |
| dGPredictor (retrained) | 61.2 | 8/10 | the method under evaluation |

The pure-QM baseline is **31.7 kJ/mol** (`--pH-mode fixed`, MACE-POLAR-1 + xtb-ALPB,
16-seed ensemble). Adding **parameter-free external references** (`--pH-mode
referenced`) reaches **16.1 kJ/mol, 10/10 correct signs** — still short of predict-zero
(10.5) on MAE, but with the direction correct on every reaction, which is the property
that matters for adjudication.

### The `referenced` column — what it is and is not
Each correction cancels a badly-solvated shared moiety against an **external**
experimental anchor (a reaction *not* in the ten, or a tabulated E°′); it never uses a
scored reaction's own experimental value, and nothing is fitted. Applied corrections:

| class | reaction(s) | correction | err before → after |
|---|---|---|---:|
| redox | rxn00070/34788 | NAD↔NADP equalization via external E°′ | 53 → 14 |
| pyrophosphate transfer | rxn01005/rxn01675 | isodesmic vs UDP-glucose pyrophosphorylase (EC 2.7.7.9) | 23/29 → 5/12 |
| glyoxalase | rxn01834 | methylglyoxal gem-diol hydrate microspecies | 70 → 62 |
| glucosyl transfer | rxn00579 | isodesmic vs sucrose phosphorylase (EC 2.4.1.7), reversed | 45 → 12 |

**The glycosyl correction turns on charge balance, not on sharing a species.** An
earlier attempt referenced the glycosyl reactions to rxn00605 and made them *worse*
(rxn00605 6 → 51): those acceptors differ in charge, so the isodesmic residual is not
charge-balanced and the anion-solvation error does not cancel. Reversed sucrose
phosphorylase works because it cancels fructose and sucrose exactly and leaves
`UDP-glucose + Pi = UDP + G1P`, which is **charge-balanced (−4 on both sides)**. The
two species it removes are the suspect ones: ModelSEED stores fructose as a furanose
though free fructose in water is ~70% β-pyranose, and sucrose carries a double anomeric
linkage. Independent check — the reference's QM value computed here (+63.9) matches the
reversed value from the separately scored 130-reaction set (+62.9) to ~1 kJ/mol.

Caveat: this anchor (rxn00577) is external to the ten but is itself a member of the
130-reaction set, so it is not independent when reporting on that set. See
`pipeline/reference_reactions.json` and `qm_thermo/external_reference.py`.

Database-wide (1550 TECRDB↔ModelSEED matched reactions): eQuilibrator covers 84%
at MAE 5.5; dGPredictor covers 100% at 13.2.

## Repository layout

| path | role |
|---|---|
| **`pipeline/`** | **the current pipeline and all findings** — the main work. Benchmark builders, the composite scorer (`final_model.py`), the pKa-solvation diagnostics, and `FINDINGS.md`. |
| `qm_thermo/` | shared library: conditions/config, the Alberty + Debye–Hückel transform (`reactions.py`), thermochemistry, and the (now-superseded) ORCA DFT backend |
| `mlip/` | multi-GPU UMA/MACE scoring of conformer ensembles; the `G_aq_*.json` energy files the scorer reads |
| `aimnet2_workflow/` | an abandoned AIMNet2 solvation route, kept for reference |
| `data/`, `*.json`, `*.csv` | TECRDB extract, ModelSEED reactions/metabolites, benchmark inputs |
| `backup/` | superseded scripts, figures, and the June presentation (git-ignored) |

Key entry point inside `pipeline/`: `final_model.py` (scores the reactions).
The rejected empirical anion-solvation calibration and its pKa inputs are
archived at `../backup/thermodynamic_calc/anion_solvent_calibration/`; they are
not part of the production workflow. See that directory's own `README.md` for
the stage table.

## Reproducing it

The code assumes a conda env (referred to as `palm`) providing RDKit, ASE,
NumPy, and an xtb binary; UMA/MACE run on GPU. External executables and model
helpers are configured with environment variables (`XTB_BIN`, `ORCA_BIN`, and
`UMA_TOOLS_DIR`), rather than machine-specific paths. Generated results go to
the git-ignored `results/` directory.

Not included in the repo (git-ignored, too large or third-party):

- **MLIP weights** — `MACE-POLAR-1` and the UMA `uma-s-1p2` checkpoint. Obtain
  MACE-POLAR from its upstream release and UMA/OMol25 from the FAIR chemistry
  release; place the `.model` files under `models/`.
- **QM geometry / scratch dumps** (`geometries_*/`, xtb scratch) — regenerable
  from the build scripts.

`env.sh` sets up ORCA + OpenMPI for the legacy DFT path only; the MLIP composite
does not need it.

## Status

This is a completed benchmark study, not an actively developed tool. The
conclusion is stable across several independent lines of evidence (electronic
method, MLIP choice, per-species cross-validation, solvation-model swap, Mg²⁺
bookkeeping): the physics cannot reach the ~5 kJ/mol accuracy the data-driven
methods already deliver on aqueous polyanion chemistry. The place QM is *uniquely*
positioned — compounds with no measured data (novel/stereochemically-resolved
metabolites) — is exactly where there is no validation set, and that is the open
problem, discussed at the end of `FINDINGS.md`.
