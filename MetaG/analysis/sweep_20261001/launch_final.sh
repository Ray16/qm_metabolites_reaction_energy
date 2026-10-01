#!/bin/bash
# launch_final.sh OUTDIR CLAIMSDIR host:idx ...  -- real pipeline (analysis/tecrdb_rescore.py) on the warm cache
M=/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/MetaG; D=$M/analysis/sweep_20261001; OUT=$1; CL=$2; shift 2
for hi in "$@"; do h=${hi%%:*}; i=${hi##*:}
  ssh -n -o BatchMode=yes $h "cd $M && setsid nohup /nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve run $i -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$M METAG_CACHE=$D/cache ADJ_OUT=$OUT CLAIMS=$CL UMA_FIRE_CHUNK=8 UMA_HESS_CHUNK=32 XTB_BIN=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/xtb/bin/xtb XTBCPX_BIN=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/xtbcpx/bin/xtb FAIRCHEM_CACHE_DIR=/nfs/lambda_stor_01/homes/rzhu/.cache/fairchem /nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/uma/bin/python -u analysis/tecrdb_rescore.py > $D/logs/final_$(basename $OUT)_${h}_${i}.log 2>&1 < /dev/null &" > /dev/null 2>&1 < /dev/null &
  sleep 8; done; wait
