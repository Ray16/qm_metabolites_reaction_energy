#!/bin/bash
# Validate v2 global-map truncation (AUTO_TRUNCATE + TRUNC_V2, NO pH-0) vs baseline full-molecule.
# Backs up + restores baseline artifacts. Usage: validate_truncv2.sh <gpus_csv> <rid>...
set -u
D=/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/experiments/qm_mlip_solvation
cd "$D"; PY=/homes/rzhu/miniforge3/envs/uma/bin/python
IFS=',' read -r -a G <<< "$1"; shift
mkdir -p artifacts/truncv2_val logs/truncv2_val; i=0
for rid in "$@"; do
  gpu="${G[$((i%${#G[@]}))]}"; i=$((i+1))
  [ -f artifacts/unified_pipeline_$rid.json ] && cp artifacts/unified_pipeline_$rid.json artifacts/truncv2_val/baseline_$rid.json
  ( AUTO_TRUNCATE=1 TRUNC_V2=1 TRUNC_RADIUS=2 CONV_MAX=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RXN_FILE=scripts/reactions_tecrdb_all.json \
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/unified_pipeline.py --only "$rid" > logs/truncv2_val/$rid.log 2>&1
    cp artifacts/unified_pipeline_$rid.json artifacts/truncv2_val/v2_$rid.json 2>/dev/null
    [ -f artifacts/truncv2_val/baseline_$rid.json ] && cp artifacts/truncv2_val/baseline_$rid.json artifacts/unified_pipeline_$rid.json ) &
  while [ "$(jobs -r|wc -l)" -ge "${#G[@]}" ]; do sleep 5; done
done
wait; echo "TRUNCV2_VAL DONE"
