#!/bin/bash
D=/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/experiments/qm_mlip_solvation
cd "$D"; H=$(hostname); mkdir -p logs/ph0_sweep
for g in "$@"; do setsid nohup bash tools/ph0_worker.sh "$g" > "logs/ph0_sweep/worker_${H}_gpu${g}.log" 2>&1 & sleep 3; done
echo "launched $# pH0 workers on $H (gpus: $*)"
