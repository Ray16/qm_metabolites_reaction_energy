# MetaG

**First-principles reaction free energies for metabolism.** A self-routing QM (UMA machine-learning
interatomic potential) pipeline that scores standard transformed Gibbs energies of reaction (ΔrG′°) from
structure alone, with **calibrated, cross-validated uncertainty** for downstream thermodynamic flux
analysis (TFA/MDF).

MetaG is first-principles at its core. The only departures are a **small set of transparent per-class
empirical anchors** (currently **four** — see *Honesty*), each calibrated to an independent reference ΔG
and reported *alongside* the pure-physics number (`dG_raw`), so a user can always take the unanchored
value. Every anchor added is an erosion of the first-principles claim, so each must clear a high bar; one
of the four (adenylylate) is currently **provisional** and does not yet meet it (see *Honesty*). Its value
is **coverage**: it scores reactions that group-contribution methods cannot — novel structures, no group
decomposition — where a first-principles method is the only option.

## Layout

```
metag/
  symmetry.py          rotational symmetry number + linearity (RRHO thermal term)
  solvation.py         first-shell water counting / explicit-solvation triage
  routing/
    ph0.py             pH-0 / Alberty pKa transform (neutral microspecies + analytic pKa)
    cofactor.py        isodesmic NAD(P)/GSH ring cores
    truncate.py        spectator truncation (Δq=0, mass-balance guarded)
    aldehyde.py        carbonyl⇌gem-diol mixture, α-EWG gated
    anchor.py          per-class empirical anchors (4 sub-classes; the only calibrated pieces)
    route_full.py      full-vs-truncated routing
  backend/             QM engine (needs the `uma` runtime: torch + fairchem + xtb)
    uma.py             batched UMA electronics
    thermal.py         UMA-Hessian RRHO + xtb solvation
    sampling.py        conformer pool + Boltzmann
    waters.py          explicit-water clusters
    clusters.py        cluster seeding
    cache.py           content-addressed species cache
  uncertainty.py       class-conditional σ + CV-validated prediction interval
  pipeline.py          orchestrator: score_reaction(model, reaction)
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
from metag.backend.uma import load_uma
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
`sigma_pred`, and a symmetric CV-validated 95% interval `ci95`.

The pure-logic layer is usable with no GPU:

```python
from metag import uncertainty
lo, hi, center, info = uncertainty.prediction_interval("fumarate hydratase", ["OC(=O)CC(O)C(=O)O"], 5.0)
```

## Honesty

- **Nothing is fit to the ΔG database except the 4 anchor sub-classes**, all reported *alongside* the
  pure-physics number (`dG_raw`) so the unanchored value is always available. They are **not** all equally
  earned:
  - **Three are solid** (phosphagen, phosphatase, thioester): systematic charged-group *solvation* offsets,
    **verified UMA≈DFT** (electronic error ruled out — the offset is solvation, not a model error),
    LOO-validated against Alberty literature ΔrG′°.
  - **One is provisional** (adenylylate, added 2026-08 — ATP + X → X-AMP + PPi): a bond-type reference
    error on the acyl-adenylate mixed anhydride. Weaker on two counts, stated plainly: (1) its reference
    (~+25 kJ) is an **indirect thermodynamic cycle** over six measured parent ligases (no direct
    measurement of the adenylylation step is used), so it carries **~±8–10 kJ beyond** the intra-class
    spread; (2) the electronic-vs-solvation physics is **not yet verified** (no UMA≈DFT check), so we
    cannot yet rule out that UMA is right and the reference is low. The LOO result (MAE 21.6→6.0) shows the
    offset is *consistent across substrates* — that tests **precision, not the accuracy** of the +25
    target. Direction is robust (UMA is too endergonic even at the generous end of the reference range);
    magnitude is soft. TODO to earn it: pin the reference with a direct ATP–PPi-exchange activation Keq,
    run the UMA≈DFT physics check, and widen its σ to ~11 to reflect the reference uncertainty.
- **Uncertainty is honest, not decorative.** `sigma_pred` is the class-level predictive error (~5–25 kJ),
  not the ~1–3 kJ conformer spread. The 95% interval is symmetric ±m·σ (m nested-CV'd so held-out coverage
  ≥ 95%) with a per-class heavy-tail floor; the class *bias* is reported as separate point-estimate
  metadata, deliberately **not** baked into a de-biased center (an asymmetric de-bias was CV-rejected and
  is OOD-fragile). The de-bias metadata is in-distribution-only.

## Tests

```bash
python -m pytest tests/            # or run each tests/test_*.py directly
```

The pure-logic suite (symmetry, pH-0 bookkeeping, anchor detection, aldehyde gate, uncertainty) runs with
no GPU. `tests/test_pipeline_smoke.py` runs an end-to-end scoring and is skipped unless a GPU + the `uma`
runtime are present.
