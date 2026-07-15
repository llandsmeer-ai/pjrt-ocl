#!/usr/bin/env bash
# Per-op perf sweep (docs/roadmap.md Phase 3 dir. 2): lane-scaling + vs JAX CPU.
# Run on NVIDIA (the PoCL barrier is unreliable under iteration — decisions #1).
# Usage: . ./env.sh && tools/bench_ops.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo "op,size,backend,lanes,ms,metric"
for spec in "add 2048" "broadcast 2048" "reduce 2048" "matmul 1024"; do
  set -- $spec; OP=$1; SZ=$2
  JAX_PLATFORMS=cpu timeout 90 $PY tools/bench_ops.py $OP $SZ --ref 2>/dev/null
  for L in 1 8 47 188 376; do
    JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=NVIDIA PJRT_OCL_VM_LANES=$L \
      timeout 90 $PY tools/bench_ops.py $OP $SZ 2>/dev/null
  done
done
