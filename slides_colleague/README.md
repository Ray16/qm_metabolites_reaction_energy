# Colleague discussion slides (2026-08-12)

`qc_thermo_discussion.tex` -> `qc_thermo_discussion.pdf` (11 slides, Beamer).
Build: `pdflatex qc_thermo_discussion.tex` (run twice for the footline page
numbers). Stock Boadilla/seahorse theme -- no metropolis dependency.

Figures in `figures/` regenerate from the committed results:
- `charge_wall.png`, `calibration.png` -- from `pipeline/*` + `results/benchmark/tecrdb_full_scored.json`
- `top10.png` -- copy of `results/benchmark/qm_vs_dgpredictor_top10.png`

Every number is sourced from committed artifacts (`experiments/tecrdb_full/`,
`pipeline/FINDINGS*.md`, `results/eq/HEADTOHEAD.md`).
