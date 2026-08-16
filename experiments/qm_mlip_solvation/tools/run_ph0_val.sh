#!/bin/bash
# pH-0 AUTO validation on the phosphate failures (before/after). Saves pH0 artifacts to
# artifacts/ph0_val/ph0_<rid>.json and RESTORES the baseline artifacts so the rolling
# full-367 table stays the untouched implicit-anion baseline.
cd /nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/experiments/qm_mlip_solvation
PY=/homes/rzhu/miniforge3/envs/uma/bin/python
for rid in rxn00695 rxn01358 rxn00216; do
  AUTO_TRUNCATE=1 TRUNC_RADIUS=2 PH0_AUTO=1 CONV_MAX=5 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    RXN_FILE=scripts/reactions_tecrdb_all.json CUDA_VISIBLE_DEVICES=5 \
    $PY scripts/unified_pipeline.py --only "$rid" 2>&1
  cp artifacts/unified_pipeline_$rid.json artifacts/ph0_val/ph0_$rid.json 2>/dev/null
  # restore baseline artifact so rolling table stays baseline
  [ -f artifacts/ph0_val/baseline_$rid.json ] && cp artifacts/ph0_val/baseline_$rid.json artifacts/unified_pipeline_$rid.json
done
echo "PH0_VAL DONE"
