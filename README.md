# QM composite for metabolic reaction free energies

Can a quantum-mechanical / machine-learning composite method compute accurate
standard transformed reaction Gibbs energies (Δ<sub>r</sub>G′°) for metabolic
reactions — accurately enough to *adjudicate* disagreements between the retrained
dGPredictor and experiment (TECRDB)?

**Short answer: no, not competitively — and it does not need to.** On the
benchmark reactions the data-driven method eQuilibrator already agrees with
experiment to within its noise (MAE **3.6 kJ/mol**, experimental sd 6.2), while
the QM composite is at **38.3 kJ/mol**. QM cannot referee 16–100 kJ/mol
disagreements when its own error is ~38. The full, self-critical account —
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
    → Alberty Legendre transform to the measured pH + extended Debye–Hückel (ionic strength)

Geometries are optimised in ALPB water; `E_gas` is a single point on that
structure, so the electronic energy and the solvation term share one geometry.
Corrections are calibrated only on **external experimental pKa values**, never on
the reactions being scored.

A note on history: an earlier design (see below) used **ORCA r2SCAN-3c DFT +
SMD**. Substituting real DFT for the MLIP at fixed geometry made the result
*worse* (MAE 14.9 → 17.3 on the clean set), which is one of several results
showing the electronic-structure method is **not** the bottleneck. The dominant
error is continuum solvation of multiply-charged anions (~15 kJ/mol per
polyanion, ~30 kJ/mol per reaction) — structural, not a tuning problem, and it
reproduces Jinich et al., *Sci. Rep.* 2014.

## Results at a glance (10 top-disagreement reactions, MAE vs TECRDB)

| method | MAE (kJ/mol) | note |
|---|---:|---|
| eQuilibrator | 3.6 | within the 6.2 experimental sd |
| predict zero | 10.5 | the trivial baseline |
| GroupContribution | 35.4 | |
| **QM composite (this repo)** | **38.3** | pH-matched deep ensemble |
| dGPredictor (retrained) | 61.2 | the method under evaluation |

Database-wide (1550 TECRDB↔ModelSEED matched reactions): eQuilibrator covers 84%
at MAE 5.5; dGPredictor covers 100% at 13.2.

## Repository layout

| path | role |
|---|---|
| **`pipeline/`** | **the current pipeline and all findings** — the main work. Benchmark builders, the composite scorer (`final_model.py`), the pKa-solvation diagnostics, and `FINDINGS.md`. |
| `qm_thermo/` | shared library: conditions/config, the Alberty + Debye–Hückel transform (`reactions.py`), thermochemistry, and the (now-superseded) ORCA DFT backend |
| `uma_workflow/` | multi-GPU UMA/MACE scoring of conformer ensembles; the `G_aq_*.json` energy files the scorer reads |
| `aimnet2_workflow/` | an abandoned AIMNet2 solvation route, kept for reference |
| `data/`, `*.json`, `*.csv` | TECRDB extract, ModelSEED reactions/metabolites, benchmark inputs |
| `backup/` | superseded scripts, figures, and the June presentation (git-ignored) |

Key entry points inside `pipeline/`: `final_model.py` (scores the
reactions), `analyze_pka.py` (the anion-solvation calibration), and the
diagnostics `score_cpcmx.py` (ALPB vs CPCM-X) and `mg_speciation.py` (Mg²⁺, found
small and **not** applied). See that directory's own `README.md` for the stage
table.

## Reproducing it

The code assumes a conda env (referred to as `palm`) providing RDKit, ASE,
NumPy, and an xtb binary; UMA/MACE run on GPU. **Several scripts hardcode
absolute paths** (`/nfs/lambda_stor_01/...`, the interpreter under
`.../envs/palm/bin/python`) from the machine this was developed on — adjust them
for your environment.

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
