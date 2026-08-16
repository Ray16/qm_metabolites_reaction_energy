#!/bin/bash
# pH-0 AUTO validation on lambda1's IDLE GPUs (no contention with the lambda0 sweep).
# 3 reactions in parallel on GPUs 0,1,2. Saves pH0 artifacts to artifacts/ph0_val/ph0_<rid>.json
# then RESTORES baseline artifacts so the rolling full-367 table stays the implicit-anion baseline.
cd /nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/experiments/qm_mlip_solvation
PY=/homes/rzhu/miniforge3/envs/uma/bin/python
run() {  # rid gpu
  local rid=$1 gpu=$2
  AUTO_TRUNCATE=1 TRUNC_RADIUS=2 PH0_AUTO=1 CONV_MAX=5 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    RXN_FILE=scripts/reactions_tecrdb_all.json CUDA_VISIBLE_DEVICES=$gpu \
    $PY scripts/unified_pipeline.py --only "$rid" > logs/ph0_val_$rid.log 2>&1
  cp artifacts/unified_pipeline_$rid.json artifacts/ph0_val/ph0_$rid.json 2>/dev/null
  cp artifacts/ph0_val/baseline_$rid.json artifacts/unified_pipeline_$rid.json 2>/dev/null
}
run rxn00695 0 &
run rxn01358 1 &
run rxn00216 2 &
wait
echo "PH0_VAL_L1 DONE"
