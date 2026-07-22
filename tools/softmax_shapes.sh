#!/bin/sh
# §50 investigation: shape sweep of the fused-softmax wrong-answer repro.
set -e
export POCL_CACHE_DIR=/home/ubuntu/project/third_party/pocl-cache
export TMPDIR=/home/ubuntu/project/third_party/tmp
PY=/home/ubuntu/project/.venv/bin/python
for rc in 64:10 96:10 128:10 64:20 64:5 200:10 64:11 63:10; do
  R=${rc%%:*}; C=${rc##*:}
  echo "--- rows=$R cols=$C  (elems=$((R * C)))"
  i=0
  while [ $i -lt 3 ]; do
    $PY tools/softmax_bug.py --rows "$R" --cols "$C" 2>&1 \
      | grep -v experimental | head -1
    i=$((i + 1))
  done
done
