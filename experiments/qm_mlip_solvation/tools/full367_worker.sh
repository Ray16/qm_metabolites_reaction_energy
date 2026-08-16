#!/bin/bash
# One-GPU worker for the multi-node full-367 sweep. Claim-based + resumable + multi-node safe.
#   - PREFLIGHT: verify the interpreter can import rdkit+torch BEFORE touching the queue, so a
#     misconfigured node (e.g. wrong python / missing rdkit) becomes a no-op instead of a
#     claim-eating runaway.
#   - skip any reaction whose log already has a "ΔG =" line (done)
#   - atomically CLAIM via `mkdir` (NFS-atomic); skip if already claimed
#   - RELEASE the claim (rmdir) if the run produced no "ΔG =" line, so another node can retry
#   - CIRCUIT BREAKER: exit after 3 consecutive failures (a broken node stops itself)
# Usage: full367_worker.sh <GPU_INDEX> [PYTHON_BIN]
set -u
D=/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/experiments/qm_mlip_solvation
cd "$D" || exit 1
GPU="${1:?need GPU index}"
PY="${2:-/homes/rzhu/miniforge3/envs/uma/bin/python}"
H=$(hostname)
CLAIMS="logs/full367/claims"
mkdir -p "$CLAIMS" logs/full367

# --- preflight: a node that can't import the deps must NOT claim anything ---
if ! CUDA_VISIBLE_DEVICES="$GPU" "$PY" -c "import rdkit, torch" >/dev/null 2>&1; then
  echo "$H gpu$GPU PREFLIGHT FAIL ($PY lacks rdkit/torch) -> no-op"; exit 3
fi

mapfile -t RIDS < <($PY -c "import json;print('\n'.join(json.load(open('scripts/reactions_tecrdb_all.json'))))")
ran=0; fails=0
for rid in "${RIDS[@]}"; do
  grep -q "ΔG =" "logs/full367/$rid.log" 2>/dev/null && continue      # already done
  mkdir "$CLAIMS/$rid" 2>/dev/null || continue                        # atomic claim (skip if taken)
  grep -q "ΔG =" "logs/full367/$rid.log" 2>/dev/null && continue      # recheck after claim
  AUTO_TRUNCATE=1 TRUNC_RADIUS=2 CONV_MAX=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    RXN_FILE=scripts/reactions_tecrdb_all.json CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" scripts/unified_pipeline.py --only "$rid" > "logs/full367/$rid.log" 2>&1
  if grep -q "ΔG =" "logs/full367/$rid.log" 2>/dev/null; then
    ran=$((ran+1)); fails=0
  else
    rmdir "$CLAIMS/$rid" 2>/dev/null                                  # release so another node retries
    fails=$((fails+1))
    if [ "$fails" -ge 3 ]; then
      echo "$H gpu$GPU CIRCUIT-BREAKER: 3 consecutive failures -> abort (ran $ran)"; exit 4
    fi
  fi
done
echo "$H gpu$GPU finished, ran $ran reactions"
