"""Per-op performance harness (docs/roadmap.md Phase 3 directive 2).

Two questions per op:
  1) Does it parallelize? -> run at several lane counts, look for throughput
     scaling as execution units increase.
  2) How does it compare to JAX's own CPU backend? -> same op, jax cpu.

Backend + lane count are fixed at client-creation from the environment, so this
script benchmarks ONE config per process; bench_ops.sh drives the sweep.

Usage:
  JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=NVIDIA PJRT_OCL_VM_LANES=188 \
      python tools/bench_ops.py <op> <size>
  JAX_PLATFORMS=cpu python tools/bench_ops.py <op> <size> --ref
op in: add, broadcast, reduce, matmul
Prints a single CSV row: op,size,backend,lanes,ms,gflops_or_gbps
"""
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp


def bench(fn, *args, iters=20, warmup=5):
    r = fn(*args)
    jax.block_until_ready(r)
    for _ in range(warmup):
        r = fn(*args); jax.block_until_ready(r)
    best = float("inf")
    for _ in range(6):
        t = time.perf_counter()
        for _ in range(iters):
            r = fn(*args); jax.block_until_ready(r)
        best = min(best, (time.perf_counter() - t) / iters)
    return best


def main():
    op, size = sys.argv[1], int(sys.argv[2])
    ref = "--ref" in sys.argv
    backend = "cpu" if ref else os.environ.get("PJRT_OCL_DEVICE", "opencl")
    lanes = int(os.environ.get("PJRT_OCL_VM_LANES", "0"))
    rng = np.random.default_rng(0)

    if op == "add":
        a = jnp.asarray(rng.standard_normal((size, size)).astype(np.float32))
        b = jnp.asarray(rng.standard_normal((size, size)).astype(np.float32))
        f = jax.jit(lambda x, y: x + y)
        ms = bench(f, a, b)
        gbps = 3 * size * size * 4 / ms / 1e9  # 2 read + 1 write
        metric = f"{gbps:.1f}GB/s"
    elif op == "broadcast":
        a = jnp.asarray(rng.standard_normal((size,)).astype(np.float32))
        f = jax.jit(lambda x: jnp.broadcast_to(x, (size, size)))
        ms = bench(f, a)
        gbps = size * size * 4 / ms / 1e9  # write-bound
        metric = f"{gbps:.1f}GB/s"
    elif op == "reduce":
        a = jnp.asarray(rng.standard_normal((size * size,)).astype(np.float32))
        f = jax.jit(lambda x: jnp.sum(x))
        ms = bench(f, a)
        gbps = size * size * 4 / ms / 1e9  # read-bound
        metric = f"{gbps:.1f}GB/s"
    elif op == "matmul":
        a = jnp.asarray((rng.integers(0, 3, (size, size))).astype(np.float32))
        b = jnp.asarray((rng.integers(0, 3, (size, size))).astype(np.float32))
        f = jax.jit(lambda x, y: x @ y)
        ms = bench(f, a, b)
        gflops = 2 * size ** 3 / ms / 1e9
        metric = f"{gflops:.0f}GF/s"
    else:
        print(f"unknown op {op}", file=sys.stderr); sys.exit(2)

    print(f"{op},{size},{backend},{lanes},{ms*1e3:.3f},{metric}")


if __name__ == "__main__":
    main()
