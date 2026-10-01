#!/bin/bash
# launch_species.sh LIST.json host:idx [host:idx ...]  -- claim-based species workers through the GPU gate
set -u
M=/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/MetaG; D=$M/analysis/sweep_20261001
GATE=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve; PY=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/uma/bin/python
L=$1; shift; mkdir -p $D/logs
for hi in "$@"; do h=${hi%%:*}; i=${hi##*:}
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$h" "cd $M && setsid nohup $GATE run $i -- env \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$M \
    METAG_CACHE=$D/${CACHE_DIR:-cache} CLAIMS=$D/claims_${TAG:-x} DONE=$D/done_${TAG:-x} ${EXTRA_ENV:-} SOLV_MODEL=${SOLV_MODEL:-alpb} SOLV_ALSO=${SOLV_ALSO:-cosmo,cpcmx} \
    XTB_BIN=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/xtb/bin/xtb XTBCPX_BIN=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/xtbcpx/bin/xtb \
    FAIRCHEM_CACHE_DIR=/nfs/lambda_stor_01/homes/rzhu/.cache/fairchem \
    $PY -u $D/species_worker.py $L > $D/logs/sp_${TAG:-x}_${h}_gpu${i}.log 2>&1 < /dev/null &" > /dev/null 2>&1 < /dev/null &
  echo "launched $h gpu$i"; sleep 8
done
