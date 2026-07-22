#!/bin/sh
# Investigation helper: sweep the §46 matmul-hybrid volume gate
# (PJRT_OCL_MM_HYBRID_GPU_MINVOL) for one workload.
# Usage: sh tools/minvolsweep.sh <workload>
set -e
export POCL_CACHE_DIR=/home/ubuntu/project/third_party/pocl-cache
export TMPDIR=/home/ubuntu/project/third_party/tmp
W="$1"
PY=/home/ubuntu/project/.venv/bin/python
for v in 1073741824 134217728 16777216 1048576 1; do
  printf 'minvol=%-12s ' "$v"
  PJRT_OCL_MM_HYBRID_GPU_MINVOL="$v" $PY tools/prof_workload.py --name "$W" \
    --no-floor 2>&1 | grep -v experimental | sed 's/(iters.*//' | tr '\n' ' '
  echo
done
