#!/bin/sh
# Investigation helper: sweep PJRT_OCL_VM_LANES (the REAL lane/work-group count
# for the host-dispatch launch grid; PJRT_OCL_NLANES is overwritten by the
# plugin when it spawns the lowering subprocess, so it has no effect here).
# Usage: sh tools/lanesweep.sh <workload> [lanes...]
set -e
export POCL_CACHE_DIR=/home/ubuntu/project/third_party/pocl-cache
export TMPDIR=/home/ubuntu/project/third_party/tmp
W="$1"; shift
LANES="$*"
[ -n "$LANES" ] || LANES="16 32 64 128 256 512"
PY=/home/ubuntu/project/.venv/bin/python
for L in $LANES; do
  printf 'lanes=%-5s ' "$L"
  PJRT_OCL_VM_LANES="$L" $PY tools/prof_workload.py --name "$W" --no-floor 2>&1 \
    | grep -v experimental | tr '\n' ' '
  echo
done
