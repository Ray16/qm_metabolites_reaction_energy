# Publishability revision (2026-10-27)
CPU-only; no new MetaG inference. Reproduce in order (eqapi env for the first two):
1. `rebuild_mg_bounded.py` -> `reference_mg_bounded.json` (policy fixed in PREREGISTRATION.md before scoring)
2. `cc_novelty.py` -> `cc_novelty.json` (CC retrained from published matrices; 4 hold-out regimes)
3. `score.py`, `within_class.py` -> `scores.json`, `within_class.json`
4. `export_numbers.py` -> `manuscript/bench_numbers.tex`, `bench_ec_table.tex`; `plot_figures.py` (uma env) -> 2 figures
Headline: 274 labels (vs 39 strict); MetaG MAE 8.44 on 239 (zero 11.68). CC 1.8 in-sample -> 8.0 reaction held out
-> 8.9 substrates held out (MetaG 8.4, same 226). dGbyG 3.5 seen -> 6.2 unseen-compound (MetaG 8.0 -> 8.5).
Within-EC r: MetaG > every held-out comparator; MetaG MAE ~ EC-mean LOO baseline (8.75) -> class offsets.
5. `uq_conformal.py` -> `uq_conformal.json` (split-conformal; 90% +/-17.6 kJ, LOO 90.4%, external 16/18)
