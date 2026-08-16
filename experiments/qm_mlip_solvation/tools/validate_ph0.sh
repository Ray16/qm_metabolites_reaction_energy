#!/bin/bash
# GPU before/after validation of the pH-0 AUTO fix against TECRDB ground truth.
# For each rid (args): capture BASELINE err from logs/full367/<rid>.log, run the pH-0 pipeline
# on a free GPU, save the pH-0 result to artifacts/ph0_val/ph0_<rid>.json, then RESTORE the
# baseline artifact so the rolling full-367 table stays the untouched implicit-anion baseline.
# Usage: validate_ph0.sh <gpu0,gpu1,...> <host> <rid> [<rid> ...]
set -u
D=/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/experiments/qm_mlip_solvation
cd "$D" || exit 1
PY=/homes/rzhu/miniforge3/envs/uma/bin/python
IFS=',' read -r -a GPUS <<< "$1"; shift
HOST="$1"; shift
mkdir -p artifacts/ph0_val logs/ph0_val
i=0
for rid in "$@"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  i=$((i+1))
  [ -f "artifacts/unified_pipeline_$rid.json" ] && cp "artifacts/unified_pipeline_$rid.json" "artifacts/ph0_val/baseline_$rid.json"
  (
    AUTO_TRUNCATE=1 TRUNC_RADIUS=2 PH0_AUTO=1 CONV_MAX=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      RXN_FILE=scripts/reactions_tecrdb_all.json CUDA_VISIBLE_DEVICES="$gpu" \
      "$PY" scripts/unified_pipeline.py --only "$rid" > "logs/ph0_val/${rid}.log" 2>&1
    cp "artifacts/unified_pipeline_$rid.json" "artifacts/ph0_val/ph0_$rid.json" 2>/dev/null
    [ -f "artifacts/ph0_val/baseline_$rid.json" ] && cp "artifacts/ph0_val/baseline_$rid.json" "artifacts/unified_pipeline_$rid.json"
  ) &
  # keep at most ${#GPUS[@]} concurrent
  while [ "$(jobs -r | wc -l)" -ge "${#GPUS[@]}" ]; do sleep 5; done
done
wait
echo "VALIDATE_PH0 DONE on $HOST"
