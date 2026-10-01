# Provenance manifest — MetaG 2026-10-01 revision (post-review)

All paths relative to `MetaG/analysis/sweep_20261001/` unless stated. Inputs are hashed in `input_sha256.txt`.

## Software
UMA `uma-s-1p2p1` (fairchem-core 2.21.0, torch 2.8.0+cu128), xtb 6.7.1 (GFN2; ALPB/COSMO water), RDKit 2026.3.5,
ASE 3.29.0, numpy 2.4.6; eQuilibrator API in the `eqapi` conda env. Final production records share source hash
`7ca76c65bfeb…` (field `source_hash` in every `final/*.json`).

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
| Cycle closure by route | `final/` | `python cycle_closure_by_route.py final` | `cycle_closure_by_route.json` |
| Common-reference comparison | `final/`, `eq_real_tecrdb_std.json`, `gc_real_tecrdb_std.json` | `run_equilibrator_std.py`, `run_group_contribution_std.py` (eqapi env), `python common_reference_comparison.py final` | `common_reference_comparison.json` |
| Production results | species cache + `reactions_tecrdb_std.json` | `launch_final.sh final final_claims host:idx …` (runs `analysis/tecrdb_rescore.py`) | `final/*.json` |
| σ calibration | `final/` | `python -m metag.tools.calibrate analysis/sweep_20261001/final --reactions …/reactions_tecrdb_std.json` | `metag/data/sigma_class_calibrated.json` |

## Species computation
Species lists: `species_all.json`, `list_*.json`; workers `species_worker.py` / `launch_species*.sh` (through `gpu_reserve`).
The species cache (`cache/`, ~15 MB, NFS) is not in git; it is reproducible from the lists with the commands above,
and every cached record carries its full settings (model, solvation, sampling, physics version, thermal mode).

## CPU reassembly harness
`reasm_stages.py` (uses `provenance/reassemble.py`): rebuilds reaction ΔG from cached species without QM; agrees
with the production pipeline to ≤ 0.1 kJ per reaction (checked on 363 reactions). Pass `ZWITTERION_PH0=1`
explicitly (the base harness policy forces it off).
