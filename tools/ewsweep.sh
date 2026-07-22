#!/bin/sh
# Investigation helper: sweep PJRT_OCL_EW_TS (the elementwise/strided-reduce
# tile size = outputs per work-group) for one workload.
# Usage: sh tools/ewsweep.sh <workload> [sizes...]
set -e
export POCL_CACHE_DIR=/home/ubuntu/project/third_party/pocl-cache
export TMPDIR=/home/ubuntu/project/third_party/tmp
W="$1"; shift
TS="$*"
[ -n "$TS" ] || TS="4096 2048 1024 512 256"
PY=/home/ubuntu/project/.venv/bin/python
for t in $TS; do
  printf 'ew_ts=%-6s ' "$t"
  PJRT_OCL_EW_TS="$t" $PY tools/prof_workload.py --name "$W" --no-floor 2>&1 \
    | grep -v experimental | sed 's/(iters.*//' | tr '\n' ' '
  echo
done
