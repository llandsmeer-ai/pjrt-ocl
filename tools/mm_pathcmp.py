"""Compare the standalone mm2 fast path against the in-VM MMA tile.

`x @ w` alone takes runtime.cc's LaunchMatmul fast path (mm2). `(x @ w) * c`
adds one elementwise consumer, which (below the §46 hybrid volume gate) keeps
the matmul on the VM's `vmo_mma_tile`. The difference is the cost every
EMBEDDED small matmul pays.

Usage:
    . ./env.sh
    JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=Intel \
      .venv/bin/python tools/mm_pathcmp.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "tools")

SHAPES = [(64, 256, 512), (64, 512, 256), (256, 256, 256), (512, 512, 512),
          (1024, 1024, 1024)]


def main():
    import jax
    import jax.numpy as jnp
    from micro_xe2 import timeit
    rng = np.random.default_rng(0)
    floor = jax.jit(lambda a, b: a.reshape(-1)[0] + b.reshape(-1)[0])
    fast = jax.jit(lambda x, w: x @ w)
    invm = jax.jit(lambda x, w: (x @ w) * 1.0000001)
    for M, K, N in SHAPES:
        x = jnp.asarray(rng.standard_normal((M, K)).astype(np.float32))
        w = jnp.asarray((rng.standard_normal((K, N)) * .05).astype(np.float32))
        b = timeit(floor, (x, w))
        tf = timeit(fast, (x, w)) - b
        tv = timeit(invm, (x, w)) - b
        fl = 2.0 * M * N * K
        print(f"{M}x{K}x{N}: mm2 {tf:.4f} ms ({fl/tf/1e9:6.0f} GF/s)   "
              f"in-VM {tv:.4f} ms ({fl/tv/1e9:6.0f} GF/s)   ratio {tv/tf:.2f}x")


if __name__ == "__main__":
    main()
