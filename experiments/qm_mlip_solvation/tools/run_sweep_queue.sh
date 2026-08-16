#!/bin/bash
# Strict 1-job-per-GPU sweep: NGPU sequential streams. No oversubscription.
cd /nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/experiments/qm_mlip_solvation
PY=/homes/rzhu/miniforge3/envs/uma/bin/python
mapfile -t RIDS < <($PY -c "import json;print('\n'.join(json.load(open('scripts/reactions_tecrdb_sample.json'))))")
NGPU=4
for gpu in $(seq 0 $((NGPU-1))); do
  ( for idx in "${!RIDS[@]}"; do
      if (( idx % NGPU == gpu )); then
        rid="${RIDS[$idx]}"
        grep -q "ΔG =" "logs/sweep_$rid.log" 2>/dev/null && continue   # skip done
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RXN_FILE=scripts/reactions_tecrdb_sample.json \
          CONV_MAX=4 CUDA_VISIBLE_DEVICES=$gpu $PY scripts/unified_pipeline.py --only "$rid" \
          > "logs/sweep_$rid.log" 2>&1
      fi
    done ) &
done
wait
echo "SWEEP QUEUE DONE"
