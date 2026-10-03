# Provenance manifest — MetaG 2026-10-01 revision (post-review)

All paths relative to `MetaG/analysis/sweep_20261001/` unless stated. Inputs are hashed in `input_sha256.txt`.

## Software
UMA `uma-s-1p2p1` (fairchem-core 2.21.0, torch 2.8.0+cu128), xtb 6.7.1 (GFN2; ALPB/COSMO water), RDKit 2026.3.5,
ASE 3.29.0, numpy 2.4.6; eQuilibrator API in the `eqapi` conda env. Final production records share source hash
`ab4c5caa9c25990047e07580bd0706807cec91e8fe757fa2fab7abbc4c2eb2b6` (field `source_hash` in every
production JSON).

## Evidence → script → output

| Evidence | Input | Script (command) | Output |
|---|---|---|---|
| FreeSolv solvation-model errors | `provenance/FreeSolv_v0.52_database.txt` (Mobley FreeSolv v0.52, 642 molecules) | `provenance/freesolv_solv_models.py <database.txt>` (MMFF geometries, xtb single points) | `provenance/freesolv_solv_models.json` |
| FreeSolv per-group regression (COSMO OH +17.8 / COOH +18.8 kJ) | `provenance/freesolv_solv_models.json` | `python provenance/freesolv_group_regression.py` | `provenance/freesolv_group_regression.json` |
| K_hyd validation + hydration calibration (a = 0.639, b = 2.35 kJ) | `khyd_verified.json` (cited: acp-2021-58 review supplement Tables S3/S4) + species cache | `python khyd_validation.py` | `khyd_validation.json` (`calibration_domain_fit`) |
| α-keto-acid keto-form pKa 1.8 | Lopalco et al., J. Pharm. Sci. 2016 (PMC4703567) Table 1 | cited in `metag/routing/pka_transform.py` | — |
| Polyprotic acid macroscopic pKa's | Martell & Smith, Critical Stability Constants (LibreTexts Table E5), I = 0, 25 °C | `metag/routing/pka_transform.py` `POLYACID_PKA` | tests `tests/test_physics_20261001.py` |
| Phosphagen core UMA vs PBE0/def2-TZVP | preset `phosphagen_core` | `analysis/verify_class_physics.py --class phosphagen_core` (GPU) | `provenance/phosphagen_core_dft.log` |
| Thermal ensemble vs single minimum | species cache (`thermal_ensemble` key) | `THERMAL_ENSEMBLE=0/1` reassembly | `thermal_ensemble_test.json` |
| Policy sensitivity + nested selection CV | species cache | `./run_policy_grid.sh` then `python nested_policy_cv.py` | `policy_grid/g_*.json`, `nested_policy_cv.json` |
| Cycle closure by route | `final_20261001c_calibrated/` | `python cycle_closure_by_route.py final_20261001c_calibrated` | `cycle_closure_by_route.json` |
| Common-reference comparison | `final_20261001c_calibrated/`, `eq_real_tecrdb_std.json`, `gc_real_tecrdb_std.json` | `run_equilibrator_std.py`, `run_group_contribution_std.py` (eqapi env), `python common_reference_comparison.py final_20261001c_calibrated` | `common_reference_comparison.json` |
| TECRDB production results | species cache + `reactions_tecrdb_std.json` | `launch_final.sh final_20261001c[_calibrated] ...` (runs `analysis/tecrdb_rescore.py`) | `final_20261001c/*.json`, `final_20261001c_calibrated/*.json` |
| σ calibration | `final_20261001c/` | `python -m metag.tools.calibrate analysis/sweep_20261001/final_20261001c --reactions …/reactions_tecrdb_std.json` | `metag/data/sigma_class_calibrated.json` |
| ModelSEED applicability stress test | `generality_inputs.json` + species cache | `TECRDB_INPUTS=.../generality_inputs.json launch_final.sh generality_20261001c ...`; `python summarize_generality.py` | `generality_20261001c/*.json`, `generality_report.json` |
| Common-reference scatter matrix | calibrated TECRDB results + common-reference inputs | `conda run -n base python common_reference_scatter.py` | `figures/common_reference_scatter.png`, manuscript copy |
| Frozen-results overview | calibrated TECRDB results + comparison and generality summaries | `conda run -n base python make_results_figure.py` | `figures/results_overview.png`, manuscript copy |

## Species computation
Species lists: `species_all.json`, `list_*.json`; workers `species_worker.py` / `launch_species*.sh` (through `gpu_reserve`).
The species cache (`cache/`, ~15 MB, NFS) is not in git; it is reproducible from the lists with the commands above,
and every cached record carries its full settings (model, solvation, sampling, physics version, thermal mode).

## CPU reassembly harness
`reasm_stages.py` (uses `provenance/reassemble.py`): rebuilds reaction ΔG from cached species without QM; agrees
with the production pipeline to ≤ 0.1 kJ per reaction (checked on 363 reactions). Pass `ZWITTERION_PH0=1`
explicitly (the base harness policy forces it off).

## Frozen release outputs

- Species sweep: 1,122 records; 1,115 successful species and seven reviewed failures. No TECRDB species failed.
- TECRDB: 364/364 estimates in each production pass; MAE 9.58, median absolute error 7.24 and RMSE
  13.39 kJ/mol. The calibrated pass differs from the pre-calibration pass only in interval and runtime
  metadata.
- Uncertainty: 347/364 held-out predictions covered (95.3%; exact binomial 95% CI 92.6–97.3%),
  with `m = 2.25` over 245 grouped-CV clusters.
- Cycle closure: 49 independent cycles; RMS residual 1.52 kJ/mol for MetaG and 3.50 kJ/mol for the
  standardized measurements.
- Common set: 319 reactions. Standardized-reference MAEs are 9.42 (MetaG development), 9.97
  (retrained dGPredictor held out), 1.26 (eQuilibrator in sample) and 5.88 kJ/mol (group contribution
  in sample).
- ModelSEED panel: 300 balanced, non-TECRDB reactions; 297 estimates, three explicit failures and
  14 OOD-flagged reactions. This panel has no experimental labels and is not an accuracy validation.

## Artifact hashes

SHA256 values:

- `metag/data/sigma_class_calibrated.json`:
  `32b87c41a348fe5fa5941bed52f963f8a5896a975d5d6b43009ec4c520d04b87`
- `cycle_closure_by_route.json`:
  `6a0bc13b5cd841fb35a072dce0bcea5f1574255b4e585f8267f392f6f3ed94f4`
- `common_reference_comparison.json`:
  `f3ae98548e0fbfb63ada4d67375d03571b3715a404794fdadd7d5de4329893be`
- `generality_report.json`:
  `f5a1f74b44d7277495dd8a6324e54c09efaa291deb85ebbc74caf4163bfee35b`
- `figures/common_reference_scatter.png`:
  `eb63a610927e15e7f0fda2acf53d81861a23991244a40bcea24a9bdc209228ca`
- `figures/results_overview.png`:
  `b1f179c49b3a7e0dd87b2f0a83490842e02e58d75040b8c5beeddd6302ed1026`

The aggregate directory hashes (sorted per-file SHA256 manifests) are
`fb9836c0f80fcdfd5f3ce95492654efec5e26a371f60cc9e64037f5b0e961102` for
`final_20261001c_calibrated/` and
`9d6aef161e1714cb9314d47da5d40d9a3a0cd6316ed3e7797aea2cf969902aa5` for
`generality_20261001c/`.

## Benchmark reference switch to openTECR (2026-10-02)

All reported metrics now use the openTECR standardized reference; see `POST_SWEEP_RUNBOOK.md` section 8.

| Evidence | Input | Script | Output |
|---|---|---|---|
| openTECR snapshot | openTECR Google workbook (CC0), 2026-10-02 | download + `odfpy` sheet export (`../opentecr_source/README.md`) | `../opentecr_source/` |
| openTECR standardized reference | snapshot + `reactions_tecrdb_all.json` + eQuilibrator | `../build_opentecr_standard.py` (eqapi env) | `reactions_opentecr_std.json`, `../opentecr_standard_report.json` |
| openTECR-scored production | species cache + openTECR reference | `launch_final.sh` (default inputs) | `final_20261002_opentecr/`, `final_20261002_opentecr_calibrated/` |
| σ calibration | `final_20261002_opentecr/` | `python -m metag.tools.calibrate … --reactions …/reactions_opentecr_std.json` | `metag/data/sigma_class_calibrated.json` |

SHA256 values:

- `../opentecr_source/openTECR.ods`: `099ce524a95fa4da655175ffc2fd0eaf463b4d8a707aa231a32a3d39bfc2cce8`
- `../opentecr_source/sheet_actual_data.csv`: `bee62b7c71263dd6612457af0b0020376b32fa8f224ba0e8e1a17e4b47c676a5`
- `experiments/qm_mlip_solvation/scripts/reactions_opentecr_std.json`:
  `f9f286579f4da9f0d0558f00416dd43ddb75a250d19e90a40e84288dcba49ecb`
- `metag/data/sigma_class_calibrated.json`: `dc86878f2d2a741a75a44f3668dee87a4080714f6af04ba4e9dd97505c6bfb41`
- `cycle_closure_by_route.json`: `27313d65e362544a4c81ff1d14b13c86a268b371e51842ee1c7fab5822bed512`
- `common_reference_comparison.json`: `5c0ef77f0f2d02f09a298aeffa1eb3f3979d4545bc903ffa2690a2e362019e8a`
- `nested_policy_cv.json`: `5c580df9f82b2e14900af63c09eec7755731c199e682fe5fcaf2771b35bc2b72`
- `final_20261002_opentecr_calibrated/` (sha256 of `sha256sum *.json` run inside the directory):
  `ce6289fd156bdcecd9bb36a48eee0b0fb7ba66b712b7d917389a417852b1d222`

Outputs: MAE 9.54, median 7.15, RMSE 13.32 kJ/mol (364/364); coverage 346/364 (95.1%; CI 92.3–97.0%), m = 2.25;
cycle RMS 1.52 (MetaG) / 3.47 (measurements); common set (319) MetaG 9.38, dGPredictor held-out 9.91,
eQuilibrator 1.34, group contribution 5.82 kJ/mol. The TECRDB-referenced values in the sections above are the
SI comparison.
