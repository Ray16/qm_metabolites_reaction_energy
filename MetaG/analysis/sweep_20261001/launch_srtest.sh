#!/bin/bash
M=/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/MetaG; D=$M/analysis/sweep_20261001
h=$1; i=$2; tag=$3; shift 3
ssh -n -o BatchMode=yes "$h" "cd $M && setsid nohup /nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve run $i -- env \
 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$M \
 METAG_CACHE=$D/cache_$tag CLAIMS=$D/claims_$tag DONE=$D/done_$tag SOLV_MODEL=alpb SOLV_ALSO=cosmo $* \
 XTB_BIN=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/xtb/bin/xtb XTBCPX_BIN=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/xtbcpx/bin/xtb \
 FAIRCHEM_CACHE_DIR=/nfs/lambda_stor_01/homes/rzhu/.cache/fairchem \
 /nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/uma/bin/python -u $D/species_worker.py $D/list_srtest.json > $D/logs/srtest_${tag}_${h}_${i}.log 2>&1 < /dev/null &" > /dev/null 2>&1 < /dev/null &
echo launched $h $i $tag
