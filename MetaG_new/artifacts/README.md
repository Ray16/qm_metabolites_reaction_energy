# MetaG_new pinned artifacts (provenance)

These are frozen copies brought in so MetaG_new is self-contained — no runtime path
reaches into `../MetaG`, `../experiments`, or `../results`. Benchmark reference =
**openTECR re-curation of TECRDB, snapshot dated 2 October 2026**.

Copied-from map (old-tree source → in-package path; sources are read-only, not modified):

| in-package | copied from |
|---|---|
| `../src/metag/data/reactions_opentecr_std.json` | `experiments/qm_mlip_solvation/scripts/reactions_opentecr_std.json` |
| `benchmark/opentecr_standard_report.json` | `MetaG/analysis/opentecr_standard_report.json` |
| `benchmark/tecrdb_full_scored.json` | `results/benchmark/tecrdb_full_scored.json` (native TECRDB-conditions reference) |
| `benchmark/build_opentecr_standard.py` | `MetaG/analysis/build_opentecr_standard.py` |
| `benchmark/opentecr_source/` | `MetaG/analysis/opentecr_source/` (snapshot, provenance) |
| `results/metag_opentecr_calibrated/` | `MetaG/analysis/sweep_20261001/final_20261002_opentecr_calibrated/` (364 scored reactions) |
| `results/generality/` | `MetaG/analysis/sweep_20261001/generality_20261001c/` (300 ModelSEED reactions) |
| `results/generality_inputs.json` | `MetaG/analysis/sweep_20261001/generality_inputs.json` |
| `results/common_reference_comparison.json` | `MetaG/analysis/sweep_20261001/common_reference_comparison.json` |
| `results/generality_report.json` | `MetaG/analysis/sweep_20261001/generality_report.json` |
| `results/baselines/eq_real_tecrdb_std.json` | `MetaG/analysis/sweep_20261001/eq_real_tecrdb_std.json` (eQuilibrator CC) |
| `results/baselines/gc_real_tecrdb_std.json` | `MetaG/analysis/sweep_20261001/gc_real_tecrdb_std.json` (group contribution) |
| `results/baselines/dgp_retrained_heldout.json` | `results/eq/dgp_retrained_heldout.json` (dGPredictor, family-grouped held-out CV) |

## NOT copied
- The per-species QM cache (`MetaG/analysis/sweep_20261001/cache/`, ~36 MB) is a *recompute
  input*, not a paper artifact. Only the UQ/generality **regeneration** (`analysis/uq/features.py`)
  needs it, and that also needs GPUs. It is left in the old tree; `analysis/uq/features.py` is
  guarded so its absence is non-fatal — everything else (numbers, figures, CLI) works without it.
