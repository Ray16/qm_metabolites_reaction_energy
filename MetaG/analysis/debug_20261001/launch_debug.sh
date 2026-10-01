#!/bin/bash
# launch_debug.sh host idx rxnlist outdir logname [extra env...]
M=/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/MetaG
h=$1; i=$2; rx=$3; out=$4; log=$5; shift 5
ssh -n -o BatchMode=yes "$h" "cd $M && setsid nohup /nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve run $i -- env \
 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$M \
 METAG_CACHE=${METAG_CACHE:-$M/analysis/debug_20261001/cache} \
 XTB_BIN=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/xtb/bin/xtb XTBCPX_BIN=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/xtbcpx/bin/xtb \
 FAIRCHEM_CACHE_DIR=/nfs/lambda_stor_01/homes/rzhu/.cache/fairchem SOLV_MODEL=cosmo $* \
 /nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/uma/bin/python -u $M/analysis/debug_20261001/debug_rxn.py $rx $out > $M/analysis/debug_20261001/$log 2>&1 < /dev/null &" > /dev/null 2>&1 < /dev/null &
echo launched $h $i $log
