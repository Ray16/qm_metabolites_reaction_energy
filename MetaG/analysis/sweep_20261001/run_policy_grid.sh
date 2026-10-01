#!/bin/bash
# All 2^7 combinations of the benchmark-selected switches, CPU reassembly from the species cache.
D="$(cd "$(dirname "$0")" && pwd)"; M="$(cd "$D/../.." && pwd)"; export PY=${PY:-python}
FLAGS=(PH0_ISOMERASE NTP_CORE CARBONYL_HYDRATION_ALL PKA_ENV ZWITTERION_PH0 ARYLAMINE_NONBASIC ACID_HB_FILTER)
cd $M
for c in $(seq 0 127); do
  env=""; tag=""
  for b in $(seq 0 6); do v=$(( (c >> b) & 1 )); env="$env ${FLAGS[$b]}=$v"; tag="$tag$v"; done
  [ -f $D/policy_grid/g_$tag.json ] && continue
  echo "$tag $env"
done | xargs -P 40 -L 1 bash -c 'tag=$0; RS_CACHE='$D'/cache RS_SOLV=alpb RS_VIA=primary RS_WATER='$D'/water_ref_G_expt.json RS_ANCH=off THERMAL_ENSEMBLE=0 OMP_NUM_THREADS=1 $PY '$D'/reasm_stages.py '$D'/policy_grid/g_$tag.json "$@" THERMAL_ENSEMBLE=0 > /dev/null 2>&1'
ls $D/policy_grid | wc -l
