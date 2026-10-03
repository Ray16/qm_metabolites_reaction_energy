openTECR snapshot (downloaded 2026-10-02) — the community re-curation of NIST TECRDB (CC0; https://github.com/opentecr).

- `openTECR.ods` — full curation workbook, exported from the openTECR Google Sheet
  (`https://docs.google.com/spreadsheets/d/1jLIxEXVzE2SAzIB0UxBfcFoHrzjzf9euB6ART2VDE8c/export?format=ods`,
  linked from https://w3id.org/opentecr/tecrdb). sha256 099ce524a95fa4da655175ffc2fd0eaf463b4d8a707aa231a32a3d39bfc2cce8.
- `sheet_*.csv` — sheets of that workbook exported with pandas/odfpy: `actual data` (5,816 rows, one per
  measurement, with curation flags; sha256
  bee62b7c71263dd6612457af0b0020376b32fa8f224ba0e8e1a17e4b47c676a5), `table metadata`, `table comments`, `references` (DOI/PMID).
- `data_nightly.csv` — the openTECR nightly CSV (https://opentecr.github.io/TECRDB-data/data.csv); a reduced
  view of `actual data` without ids or flags. Kept for reference only; not used.
- `cc_training_TECRDB.csv` — component-contribution training data (Zenodo 10.5281/zenodo.3978439, v1.1, record
  5495826). Row-for-row identical (same url, same order, 4,544 rows) to `../tecrdb_source/TECRDB.csv`, which is why
  openTECR ids `...TECRDB.csv#entryN` index row N of our file.

Used by `../build_opentecr_standard.py` -> `experiments/qm_mlip_solvation/scripts/reactions_opentecr_std.json`.

This pinned snapshot is the benchmark reference for all reported metrics from 2026-10-02 on (openTECR is a live
sheet; a fresh download can differ). CC0, so the snapshot is deposited with the paper.
