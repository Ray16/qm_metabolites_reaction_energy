#!/bin/bash
# Validate the COHERENT pipeline: AUTO_TRUNCATE (v2 global-MCS) + PH0_AUTO together (truncate the
# floppy backbone AND neutralize the anion) vs baseline. Backs up + restores baseline artifacts.
# Usage: validate_coherent.sh <gpus_csv> <rid>...
set -u
D=/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/experiments/qm_mlip_solvation
cd "$D"; PY=/homes/rzhu/miniforge3/envs/uma/bin/python
IFS=',' read -r -a G <<< "$1"; shift
mkdir -p artifacts/coherent_val logs/coherent_val; i=0
for rid in "$@"; do
  gpu="${G[$((i%${#G[@]}))]}"; i=$((i+1))
  [ -f artifacts/unified_pipeline_$rid.json ] && cp artifacts/unified_pipeline_$rid.json artifacts/coherent_val/baseline_$rid.json
  ( AUTO_TRUNCATE=1 TRUNC_V2=1 PH0_AUTO=1 TRUNC_RADIUS=2 CONV_MAX=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RXN_FILE=scripts/reactions_tecrdb_all.json \
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/unified_pipeline.py --only "$rid" > logs/coherent_val/$rid.log 2>&1
    cp artifacts/unified_pipeline_$rid.json artifacts/coherent_val/coh_$rid.json 2>/dev/null
    [ -f artifacts/coherent_val/baseline_$rid.json ] && cp artifacts/coherent_val/baseline_$rid.json artifacts/unified_pipeline_$rid.json ) &
  while [ "$(jobs -r|wc -l)" -ge "${#G[@]}" ]; do sleep 5; done
done
wait; echo "COHERENT_VAL DONE"
