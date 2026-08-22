#!/usr/bin/env bash
# Post-sweep refresh: rebuild EVERY downstream artifact from the fresh logs/production/ so all
# predictions, the UQ calibration, and all figures are current. ZERO GPU. Run AFTER the full-367
# production sweep completes (all figure/benchmark tools read logs/production with top precedence).
#   Usage: tools/refresh_all_outputs.sh
set +e
cd "$(dirname "$0")/.."
P=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/modelseed/bin/python
echo "############ 1. UQ calibration (writes artifacts/sigma_class_calibrated.json) ############"
$P tools/calibrate_uncertainty.py
echo; echo "############ 2. UQ figures ############"
$P tools/uq_figures.py
echo; echo "############ 3. error histogram (UMA vs dGP) ############"
$P tools/make_error_histogram.py
echo; echo "############ 4. diagnostic figures (bias-by-class, waterfall, descriptor) ############"
$P tools/diagnostic_figs.py
echo; echo "############ 5. deck figures (per-class MAE, before/after) ############"
$P tools/make_deck_figures.py
echo; echo "############ 6. current error ranking (headline MAE, fresh logs) ############"
$P tools/current_error_ranking.py 2>&1 | head -30
echo; echo "############ 7. six-method benchmark rebuild (per_rxn / per_class) ############"
# needs the gnndgf env (torch + local gnn pkg); fall back to modelseed if absent
GP=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/gnndgf/bin/python
[ -x "$GP" ] || GP=$P
( cd ../gnn_dgf && $GP scripts/build_per_rxn_benchmark.py 2>&1 | tail -25 )
echo; echo "############ 8. pred-vs-exp per-method comparison + shrink-to-mean slopes ############"
$P tools/pred_vs_exp_figs.py
echo; echo "REFRESH DONE. Figures in figures/ ; UQ in artifacts/sigma_class_calibrated.json"
