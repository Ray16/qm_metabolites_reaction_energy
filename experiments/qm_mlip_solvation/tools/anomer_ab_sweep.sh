#!/usr/bin/env bash
# Anomer A/B sweep: 15 variant reactions across GPUs 6 and 7 (one job per GPU at a time, resumable).
set -u
cd "$(dirname "$0")/.."
PY=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/uma/bin/python
export RXN_FILE=artifacts/anomer_ab_reactions.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# clean isolation of the anomer effect: FULL molecule (anomeric C always present) + fresh (no cache reuse)
# so all three variants are sampled identically and only the pinned anomer differs.
export AUTO_TRUNCATE=0 COFACTOR_RING=0 ROUTE_FULL=0 ANCHOR_CORRECT=0 PH0_AUTO=1 PH0_BASES=1 SPECIES_CACHE=0
LOGDIR=logs/anomer_ab; mkdir -p "$LOGDIR"

mapfile -t KEYS < <($PY -c "import json;[print(k) for k in json.load(open('artifacts/anomer_ab_reactions.json'))]")
GPUS=(6 7)
run_shard () {
  local g=$1 idx
  for ((idx=g; idx<${#KEYS[@]}; idx+=${#GPUS[@]})); do
    local key=${KEYS[$idx]} log="$LOGDIR/${KEYS[$idx]}.log"
    if [ -f "$log" ] && grep -q "err=" "$log"; then echo "skip $key (done)"; continue; fi
    echo "[GPU${GPUS[$g]}] $key"
    CUDA_VISIBLE_DEVICES=${GPUS[$g]} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      $PY scripts/unified_pipeline.py --only "$key" --seeds 1,2,3,4 --keep 10 --pool 48 > "$log" 2>&1
  done
}
for g in "${!GPUS[@]}"; do run_shard "$g" & done
wait
echo "ALL DONE"
