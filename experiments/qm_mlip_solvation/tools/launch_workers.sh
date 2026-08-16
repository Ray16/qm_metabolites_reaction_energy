#!/bin/bash
# launch_workers.sh <gpu> [<gpu> ...]  -- start one detached claim-based full-367 worker per GPU
# on THIS node. Staggered to avoid a thundering-herd on the NFS UMA-checkpoint load.
D=/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/experiments/qm_mlip_solvation
cd "$D" || exit 1
H=$(hostname)
for g in "$@"; do
  setsid nohup bash tools/full367_worker.sh "$g" > "logs/full367/worker_${H}_gpu${g}.log" 2>&1 &
  sleep 3
done
echo "launched $# workers on $H (gpus: $*)"
