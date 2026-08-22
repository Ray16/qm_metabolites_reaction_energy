#!/usr/bin/env bash
# Back up the stale logs/production and launch a fresh clean full-367 sweep under the CURRENT
# unified_pipeline defaults (CoA-NAC truncation now default-on via COFACTOR_RING) so all
# predictions are up to date and free of stale-log artifacts.
set +u
cd "$(dirname "$0")/.."
echo "=== model line ==="
grep -nE "_MODEL\s*=|uma-s-1p2" scripts/unified_pipeline.py scripts/batched_relax.py | head
echo "=== reaction count ==="
/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/modelseed/bin/python -c "import json;d=json.load(open('scripts/reactions_tecrdb_all.json'));print('n_rxn=',len(d))"
echo "=== back up stale logs/production ==="
if [ -d logs/production ] && [ ! -d logs/production.bak_stale_20260821 ]; then
  mv logs/production logs/production.bak_stale_20260821
  echo "moved logs/production -> logs/production.bak_stale_20260821"
fi
mkdir -p logs/production
echo "=== launching fresh sweep on 4 GPUs ==="
nohup bash tools/production_sweep.sh scripts/reactions_tecrdb_all.json logs/production 4 \
  > logs/production_sweep_fresh.driver.log 2>&1 &
echo "sweep PID=$!"
sleep 8
echo "=== first driver output ==="
head -5 logs/production_sweep_fresh.driver.log 2>/dev/null
