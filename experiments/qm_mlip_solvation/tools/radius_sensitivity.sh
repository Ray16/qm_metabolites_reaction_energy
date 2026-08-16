#!/bin/bash
# Radius-sensitivity truncation-validity test (reaction-agnostic, no tuned thresholds).
# A TRUE spectator removal leaves ΔG invariant to the cut radius. Score each reaction truncated at
# radius 2 AND radius 3; a stable ΔG (|ΔΔG| small) means the cut is a clean spectator and can be
# trusted; a large swing means the cut touches the reactive context -> distrust, use full molecules.
# Writes logs/radius_val/<rid>_r{2,3}.log. Usage: radius_sensitivity.sh <gpus_csv> <rid>...
set -u
D=/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/experiments/qm_mlip_solvation
cd "$D"; PY=/homes/rzhu/miniforge3/envs/uma/bin/python
IFS=',' read -r -a G <<< "$1"; shift
mkdir -p logs/radius_val artifacts/radius_val; i=0
for rid in "$@"; do
  [ -f artifacts/unified_pipeline_$rid.json ] && cp artifacts/unified_pipeline_$rid.json artifacts/radius_val/baseline_$rid.json
  for rad in 2 3; do
    gpu="${G[$((i%${#G[@]}))]}"; i=$((i+1))
    ( AUTO_TRUNCATE=1 TRUNC_V2=1 TRUNC_RADIUS=$rad CONV_MAX=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RXN_FILE=scripts/reactions_tecrdb_all.json \
        CUDA_VISIBLE_DEVICES=$gpu $PY scripts/unified_pipeline.py --only "$rid" > logs/radius_val/${rid}_r${rad}.log 2>&1 ) &
    while [ "$(jobs -r|wc -l)" -ge "${#G[@]}" ]; do sleep 5; done
  done
done
wait
# restore baselines
for rid in "$@"; do [ -f artifacts/radius_val/baseline_$rid.json ] && cp artifacts/radius_val/baseline_$rid.json artifacts/unified_pipeline_$rid.json; done
echo "RADIUS_SENS DONE"
