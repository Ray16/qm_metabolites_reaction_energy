#!/bin/bash
# BEST pipeline (AUTO_TRUNCATE) on ALL 367 TECRDB reactions. Strict 1-job-per-GPU (8 streams).
# Resumable: skips reactions whose log already has a ΔG line.
cd /nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/experiments/qm_mlip_solvation
PY=/homes/rzhu/miniforge3/envs/uma/bin/python
mapfile -t RIDS < <($PY -c "import json;print('\n'.join(json.load(open('scripts/reactions_tecrdb_all.json'))))")
mkdir -p logs/full367
NGPU=8
for gpu in $(seq 0 $((NGPU-1))); do
  ( for idx in "${!RIDS[@]}"; do
      if (( idx % NGPU == gpu )); then
        rid="${RIDS[$idx]}"
        if grep -q "ΔG =" "logs/full367/$rid.log" 2>/dev/null; then continue; fi
        AUTO_TRUNCATE=1 TRUNC_RADIUS=2 CONV_MAX=5 \
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          RXN_FILE=scripts/reactions_tecrdb_all.json CUDA_VISIBLE_DEVICES=$gpu \
          $PY scripts/unified_pipeline.py --only "$rid" > "logs/full367/$rid.log" 2>&1
      fi
    done ) &
  done
wait
echo "FULL367 DONE"
