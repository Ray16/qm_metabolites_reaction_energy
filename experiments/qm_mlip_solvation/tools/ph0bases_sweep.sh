#!/usr/bin/env bash
# Shard the 41 amine-change reactions across GPUs 0-7, each with PH0_AUTO + PH0_BASES (per-N pKa).
# Per-reaction logs -> logs/ph0bases_sweep/<rid>.log ; per-shard driver log -> logs/ph0bases_sweep/gN.log
set +u
cd "$(dirname "$0")/.."
ROOT=/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc
source "$ROOT/env.sh" >/dev/null 2>&1
UP=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/uma/bin/python
P=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/modelseed/bin/python
for g in 0 1 2 3 4 5 6 7; do
  shard="scripts/shards/ph0bases_g${g}.json"
  ids=$($P -c "import json;print(' '.join(json.load(open('$shard'))))")
  (
    for rid in $ids; do
      CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 RXN_FILE="$shard" PH0_AUTO=1 PH0_BASES=1 \
        $UP scripts/unified_pipeline.py --only "$rid" --seeds 1,2 \
        > "logs/ph0bases_sweep/${rid}.log" 2>&1
    done
  ) > "logs/ph0bases_sweep/g${g}.log" 2>&1 &
  echo "shard $g -> GPU $g : $ids"
done
wait
echo "SWEEP DONE"
