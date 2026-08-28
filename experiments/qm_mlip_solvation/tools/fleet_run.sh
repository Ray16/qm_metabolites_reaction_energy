#!/usr/bin/env bash
# Reusable fleet dispatcher: fan a list of unified_pipeline --only KEYS across the GPUs of one node
# (one job per GPU, backgrounded remotely). All paths are on shared NFS so remote jobs write logs back
# here. Usage: NODE=lambda5 NGPU=15 RXN_FILE=... FLAGS="A=0 B=1" LOGDIR=logs/xxx SEEDS=1,2,3,4 \
#             bash tools/fleet_run.sh key1 key2 ...
set -u
NODE=${NODE:?set NODE}; NGPU=${NGPU:-16}
RXN_FILE=${RXN_FILE:?set RXN_FILE}; LOGDIR=${LOGDIR:?set LOGDIR}
SEEDS=${SEEDS:-1,2,3,4}; KEEP=${KEEP:-10}; POOL=${POOL:-48}; FLAGS=${FLAGS:-}
DIR=/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/experiments/qm_mlip_solvation
PY=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/uma/bin/python
mkdir -p "$DIR/$LOGDIR"
KEYS=("$@")
echo "dispatching ${#KEYS[@]} jobs to $NODE (<=$NGPU gpus)"
i=0
for key in "${KEYS[@]}"; do
  log="$DIR/$LOGDIR/${key}.log"
  if [ -f "$log" ] && grep -q "err=" "$log"; then echo "skip $key (done)"; continue; fi
  g=$(( i % NGPU )); i=$(( i + 1 ))
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$NODE" \
    "cd $DIR && CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RXN_FILE=$RXN_FILE $FLAGS \
     nohup $PY scripts/unified_pipeline.py --only $key --seeds $SEEDS --keep $KEEP --pool $POOL \
     > $LOGDIR/${key}.log 2>&1 &" &
  sleep 0.3
done
wait
echo "all $NODE launches issued"
