#!/usr/bin/env bash
# Run (a subset of) JAX's own unit tests against the OpenCL backend and tally
# which StableHLO ops we're still missing — the test-driven coverage driver
# (CLAUDE.md M3). Needs the jax source checked out at the matching tag:
#   git clone --depth 1 --branch jax-v$(jax.__version__) https://github.com/jax-ml/jax third_party/jax-src
#   .venv/bin/pip install --cache-dir third_party/pip-cache absl-py hypothesis
#
# Single-device only — multi-device / sharding / distributed tests are out of
# scope (we expose one device). Run on NVIDIA (PoCL barrier is flaky under
# iteration; see docs/decisions.md #1).
#
# Usage: tools/jax_coverage.sh [test_file.py ...]   (default: lax_test.py)
set -u
cd "$(dirname "$0")/.."
source ./env.sh
SRC=third_party/jax-src
PY=$(pwd)/.venv/bin/python
FILES=("${@:-lax_test.py}")

# tests whose names match these are skipped (multi-device, and features we
# deliberately don't target yet).
SKIP='pmap or sharding or shard_map or multi_device or distributed or
      Collective or all_gather or all_reduce or ppermute or xmap or
      pjit_sharding or MultiDevice or custom_partition'
SKIP=$(echo "$SKIP" | tr -d '\n')

cd "$SRC"
for f in "${FILES[@]}"; do
  echo "### $f"
  JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=NVIDIA \
    "$PY" -m pytest "tests/$f" -q --no-header --tb=line -p no:cacheprovider \
      -o addopts="" -k "not ($SKIP)" 2>&1 |
    tee "/tmp/jaxcov_$f.out" |
    tail -1
  echo "--- missing ops (ranked) in $f ---"
  grep -oE 'unsupported op: stablehlo\.[a-z_0-9]+' "/tmp/jaxcov_$f.out" |
    sed 's/unsupported op: //' | sort | uniq -c | sort -rn
  echo
done
