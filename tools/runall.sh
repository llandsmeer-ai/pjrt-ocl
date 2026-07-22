#!/bin/sh
# Investigation helper: run tools/prof_workload.py over the slow non-loop
# workloads on whichever backend the caller selected via JAX_PLATFORMS /
# PJRT_OCL_DEVICE. Usage: JAX_PLATFORMS=cpu sh tools/runall.sh [--phases]
set -e
export POCL_CACHE_DIR=/home/ubuntu/project/third_party/pocl-cache
export CUDA_CACHE_PATH=/home/ubuntu/project/third_party/nv-cache
export TMPDIR=/home/ubuntu/project/third_party/tmp
PY=/home/ubuntu/project/.venv/bin/python
for w in batchnorm nbody embedding_softmax fft monte_carlo layernorm cnn mlp; do
  $PY tools/prof_workload.py --name "$w" "$@" 2>&1 | grep -v experimental || true
done
