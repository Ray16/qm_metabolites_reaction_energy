# eQuilibrator (component-contribution) — benchmarking tool

Sibling of `../dGPredictor/` (original) and `../dGPredictor_freiburger/` (Andrew's
retrained). Used here to benchmark eQuilibrator's **ModelSEED compound coverage**
against dGPredictor and the GNN.

## Environment
`equilibrator-api` is installed in the conda env **`eqapi`** (python 3.10,
equilibrator-api 0.6.0). It was NOT previously persistent — an earlier run used an
ephemeral `/tmp/eqenv` that got wiped; this env replaces it.

```bash
conda activate eqapi
python -c "from equilibrator_api import ComponentContribution; ComponentContribution()"  # downloads cache on first use
```

## Compound cache
On first `ComponentContribution()` call, equilibrator-cache downloads a local
`compounds.sqlite` (Zenodo) to `~/.cache/equilibrator/`. Large binary — kept out of
the repo; documented here instead. Coverage lookup uses
`cc.ccache.search_compound_by_inchi_key(...)` + `cc.standard_dg_formation(...)`.

## How eQuilibrator decides it can/can't score a compound
Two gates (either can block it):
1. **Identity** — the compound must be resolvable in eQ's compound cache
   (by InChIKey / KEGG id / other registry). ModelSEED→KEGG mapping
   (`../../pipeline/modelseed_to_kegg.json`) has only **17,555** entries, so the
   *as-deployed* KEGG-gated path is capped there.
2. **Estimability** — even when found, component-contribution must yield a
   FINITE-uncertainty formation energy (compound decomposes into known groups /
   lies in the training reactant set). Infinite sigma ⇒ not coverable.

## Scripts
- `coverage_modelseed.py` — structure-based coverage sweep: for every ModelSEED
  compound with an InChIKey, look it up in the eQ cache and test for a finite Δf G.
  Reports both the structure-based number and the KEGG-gated ceiling.
  Output → `data/eq_coverage_modelseed.json`.

## Prior eQ outputs (already on disk, from the wiped venv run)
`../../results/eq/`: `equilibrator_full.json` (367 TECRDB, in-sample),
`modelseed_all_dG.{csv,json}` (ModelSEED-wide ΔrG′°, 57% reaction coverage).
