# MetaG

**First-principles reaction free energies for metabolism.** A self-routing QM (UMA machine-learning
interatomic potential) pipeline that scores standard transformed Gibbs energies of reaction (ΔrG′°) from
the reaction's structures, with **calibrated, cross-validated uncertainty** for downstream thermodynamic
flux analysis (TFA/MDF).

MetaG is first-principles at its core and **calibrated (not fitted) on TECRDB**: the physics has no
trainable parameters. The calibrated pieces are (i) a **small set of transparent per-class anchor
offsets** — **8 sub-classes in 6 chemical families**, see *Honesty* — each reported *alongside* the
pure-physics number (`dG_raw`), so a user can always take the unanchored value, and (ii) the
uncertainty layer (class σ, heavy-tail floor, interval multiplier). Every anchor erodes the
first-principles claim, so each must clear a high bar. The method's value is **coverage**: it scores
reactions that group-contribution methods cannot (novel structures, no group decomposition), where a
first-principles method is the only option.

## Layout

```
metag/
  symmetry.py            rotational symmetry number + linearity (RRHO thermal term)
  water_count.py         first-shell water counting / explicit-solvation triage
  routing/               structure-based reaction transforms + physics corrections
    pka_transform.py     pH-0 / Alberty pKa transform (neutral microspecies + analytic pKa)
    cofactor_cores.py    isodesmic NAD(P)/GSH ring cores
    coa_core.py          CoA thioester core reduction
    ntp_core.py          nucleoside-polyphosphate core reduction
    truncate.py          spectator truncation (Δq=0, mass-balance + thioester guarded)
    truncate_global.py   global-MCS truncation (multi-coeff / unequal-side reactions)
    truncation_gate.py   full-vs-truncated routing decision
    aldehyde_hydration.py  carbonyl⇌gem-diol mixture, α-EWG gated
    anchor.py            per-class anchor offsets (8 sub-classes; the only calibrated ΔG pieces)
    applicability.py     out-of-distribution flags (transparency; labels the interval, does not change σ)
    solv_gate.py         structural gates (hydro-lyase water-reference constant, SMD gate)
  energetics/            QM engine — species free energies (needs the `uma` runtime: torch + fairchem + xtb)
    uma.py               batched UMA electronics
    thermal.py           UMA-Hessian RRHO + xtb solvation
    conformers.py        conformer pool + Boltzmann over UNIQUE minima (energy + rotational-constant dedup)
    explicit_solvation.py  explicit-water cluster-continuum
    water_clusters.py    water-cluster seeding
    species_cache.py     content-addressed per-species cache
  uncertainty.py         class-conditional σ + CV-validated prediction interval
  pipeline.py            orchestrator: score_reaction(model, reaction)
  tools/
    calibrate.py         regenerate the uncertainty artifact; fully nested CV (anchor offsets refit per fold)
    cycle_closure.py     ground-truth-free state-function check across reactions sharing compounds
  data/
    sigma_class_calibrated.json   shipped uncertainty calibration
```

## Install

Pure-logic layer (routing, corrections, uncertainty) needs only numpy + rdkit:

```bash
pip install -e .
```

The QM backend additionally needs the heavy `uma` runtime (torch, fairchem-core, ase, and an `xtb`
binary) and a GPU:

```bash
pip install -e ".[qm]"
```

## Use

```python
from metag.energetics.uma import load_uma
from metag.pipeline import score_reaction

pu = load_uma("uma-s-1p2p1")
reaction = {
    "note": "creatine kinase | EC=2.7.3.2",
    "n_Hplus": 0,
    "species": {                        # {name: [coeff (+prod / -react), charge, SMILES]}
        "phosphocreatine": [-1, -2, "..."],
        "ADP":             [-1, -3, "..."],
        "creatine":        [ 1,  0, "..."],
        "ATP":             [ 1, -4, "..."],
    },
}
r = score_reaction(pu, reaction)
print(r["dG"], r["dG_raw"], r["ci95"], r["sigma_pred"])
```

The result carries both the **anchored** `dG` and the **pure-physics** `dG_raw`, the calibrated
`sigma_pred`, a symmetric 95% interval `ci95` built on the same σ (class σ ⊕ conformer-sampling
`U_samp`), `ci_info["externally_calibrated"]` / `calibration_scope` (see *Honesty*), and `routes`
(which structural transforms fired: cofactor ring, truncation, pH-0, and any routing errors).

**Thermodynamic conventions.** Solutes at 1 M, liquid water at 55.34 M, H⁺ at pH 7 (ionic strength 0).
Gas-phase RRHO free energies are at 1 atm, so every species gets the 1 atm → 1 M term RT ln 24.46 =
7.93 kJ/mol (`STD_STATE_KJ`; applied outside the species cache; `STD_STATE_1M=0` disables it for A/B).
The proton free energy uses the Tissandier 1 atm → 1 M value, so it is on the same convention.

**Optional validation gates.** `TRUNC_VALIDATE=1` scores each truncated reaction at cut radius R and R+1
and keeps the truncation only if ΔG is radius-invariant (otherwise full molecules); off by default
(it doubles the cost of truncated reactions). `metag.tools.cycle_closure.closure_report` checks
state-function closure over any set of scored reactions.

The pure-logic layer is usable with no GPU:

```python
from metag import uncertainty
lo, hi, center, info = uncertainty.prediction_interval("fumarate hydratase", ["OC(=O)CC(O)C(=O)O"], 5.0)
```

## Honesty

- **Calibrated on TECRDB, in two places only: anchor offsets and the uncertainty layer.** Both are
  reported transparently (`dG_raw` is always returned). The 8 anchor sub-classes
  (`metag/routing/anchor.py`, `ANCHORS`):

  | family | sub-classes | reference | status |
  |---|---|---|---|
  | phosphagen (P–N) | `phosphagen` | TECRDB members | UMA≈DFT verified; solvation |
  | phosphatase monoester | `phosphatase_monoester` | TECRDB members | electronic gap +7.9 kJ, weaker |
  | acyl-CoA ligase | `thioester_ppi`, `thioester_pi` | TECRDB members | UMA≈DFT verified |
  | acyl-adenylate | `adenylylate_aliphatic`, `adenylylate_aminoacid` | **external** indirect +25 kJ cycle | provisional: reference ±8–10 kJ |
  | carboxy-phosphate | `carboxyP` | TECRDB members (n=2) | UMA≈DFT verified; no LOO possible |
  | acyclic amide hydrolysis | `amide_hydrolysis` | TECRDB members | UMA≈DFT verified |

  Seven sub-classes are referenced to TECRDB members, so the uncertainty calibration
  (`metag/tools/calibrate.py`) **re-fits those offsets inside each CV fold**: held-out reactions are
  scored with offsets fitted without them, together with σ, the heavy-tail floor and the multiplier.
  The reported coverage is therefore for the complete deployed estimator, not only the interval widths.
  The adenylylate offsets use no TECRDB experiment and stay fixed.
  In addition, hydro-lyases get a fixed physical constant (experimental minus xtb-COSMO water solvation,
  −23.2 kJ per net water; `WATER_REF_HYDROLYASE`), reported as `water_ref` in the result.
- **The σ-class is structural where it can be, note-based otherwise.** Reactions matching an anchor
  sub-class get that class from structure (SMARTS), whatever the enzyme note says. All other classes
  come from **enzyme-note keywords** (`uncertainty.mech_class`), so the uncertainty (not the ΔG) depends
  on the annotation; a cryptic or absent note falls to `other/clean`.
- **The interval is calibrated in-distribution only.** There is no computed physics uncertainty yet
  (UMA-vs-MACE ensemble, cycle consistency). A reaction with OOD flags (`routing/applicability.py`:
  divalent-cation coordination with phosphate/carboxylate, rare elements, de-novo aromatic N-heterocycle,
  very large molecules), or in a class absent from the calibration set, gets
  `externally_calibrated = False` and a `calibration_scope` string saying why. Its width is nominal,
  **not a validated coverage**. For TFA/MDF on frontier ModelSEED chemistry, treat those intervals as
  lower bounds on the real uncertainty. OOD flags do not inflate σ (a structural "widen if unusual"
  floor would fire on exactly the frontier reactions the method exists to score). Classes absent from
  the calibration set get at least the overall σ and the global heavy-tail floor, never a narrower interval.
- **Interval form.** Symmetric: half-width = max(m·√(σ_class² + U_samp²), q95abs), with m chosen by
  nested CV so held-out coverage is at least 95%. Only `level=95` is CV-calibrated; other levels are
  Gaussian-scaled and flagged `level_calibrated = False`. The class bias is reported as point-estimate
  metadata (`point_bias`), not baked into a de-biased center (an asymmetric de-bias was CV-rejected).
- **The shipped calibration must match the deployed pipeline.** Any change to physics, routing or anchors
  requires a TECRDB re-sweep and `calibrate()`; `data/sigma_class_calibrated.json` records the
  pipeline configuration it was computed from.

## Tests

```bash
python -m pytest tests/            # or run each tests/test_*.py directly
```

The pure-logic suite (symmetry, pH-0 bookkeeping, anchor detection, aldehyde gate, uncertainty) runs with
no GPU. `tests/test_pipeline_smoke.py` runs an end-to-end scoring and is skipped unless a GPU + the `uma`
runtime are present.
