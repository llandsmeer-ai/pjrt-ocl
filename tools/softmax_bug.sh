#!/bin/sh
# §50 investigation: run tools/softmax_bug.py N times, one process each.
# Usage: [env...] sh tools/softmax_bug.sh <reps> [args to softmax_bug.py]
set -e
export POCL_CACHE_DIR=/home/ubuntu/project/third_party/pocl-cache
export TMPDIR=/home/ubuntu/project/third_party/tmp
N="$1"; shift
i=0
while [ "$i" -lt "$N" ]; do
  /home/ubuntu/project/.venv/bin/python tools/softmax_bug.py "$@" 2>&1 \
    | grep -v experimental | head -1
  i=$((i + 1))
done
