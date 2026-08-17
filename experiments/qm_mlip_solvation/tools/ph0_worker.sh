#!/bin/bash
# pH-0 coherent-pass worker: AUTO_TRUNCATE (v1) + PH0_AUTO on all 367, claim-based/resumable/self-
# healing (same infra as full367_worker). Writes logs/ph0_sweep/. Usage: ph0_worker.sh <GPU> [PY]
set -u
D=/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/experiments/qm_mlip_solvation
cd "$D" || exit 1
GPU="${1:?need GPU index}"; PY="${2:-/homes/rzhu/miniforge3/envs/uma/bin/python}"
H=$(hostname); OUT=logs/ph0_sweep; CLAIMS="$OUT/claims"; mkdir -p "$CLAIMS"
if ! CUDA_VISIBLE_DEVICES="$GPU" "$PY" -c "import rdkit, torch" >/dev/null 2>&1 || [ ! -x /homes/rzhu/miniforge3/envs/xtb/bin/xtb ]; then
  echo "$H gpu$GPU PREFLIGHT FAIL"; exit 3; fi
mapfile -t RIDS < <($PY -c "import json;print('\n'.join(json.load(open('scripts/reactions_tecrdb_all.json'))))")
ran=0; fails=0
for rid in "${RIDS[@]}"; do
  grep -q "ΔG =" "$OUT/$rid.log" 2>/dev/null && continue
  mkdir "$CLAIMS/$rid" 2>/dev/null || continue
  grep -q "ΔG =" "$OUT/$rid.log" 2>/dev/null && continue
  AUTO_TRUNCATE=1 PH0_AUTO=1 TRUNC_RADIUS=2 CONV_MAX=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RXN_FILE=scripts/reactions_tecrdb_all.json \
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/unified_pipeline.py --only "$rid" > "$OUT/$rid.log" 2>&1
  if grep -q "ΔG =" "$OUT/$rid.log" 2>/dev/null; then ran=$((ran+1)); fails=0
  else rmdir "$CLAIMS/$rid" 2>/dev/null; fails=$((fails+1)); [ "$fails" -ge 3 ] && { echo "$H gpu$GPU CIRCUIT-BREAKER"; exit 4; }; fi
done
echo "$H gpu$GPU pH0 done, ran $ran"
