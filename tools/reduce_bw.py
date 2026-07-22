"""Measure reduction throughput on the current backend, two ways.

--seg     last-axis (segmented) reduce with the SEGMENT LENGTH swept at a
          constant total element count. Exposes TILE_RED_SEG's cost being
          per-SEGMENT (one work-group per segment) rather than per element.
--strided axis-0 (strided/partial) reduce vs a plain copy at several shapes.
          Exposes TILE_RED_STRIDED's tile count = ceil(n_out / EW_TS), i.e. a
          reduction whose OUTPUT is <= EW_TS elements runs on ONE work-group.

Both report GB/s over the INPUT bytes (copy rows report read+write).

Usage:
    . ./env.sh
    JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=Intel \
      .venv/bin/python tools/reduce_bw.py --seg
    PJRT_OCL_EW_TS=256 JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=Intel \
      .venv/bin/python tools/reduce_bw.py --strided
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "tools")

SEGS = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 1024]
SHAPES = [(128, 256), (1024, 256), (8192, 256), (128, 4096), (1024, 4096),
          (4096, 4096)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", action="store_true")
    ap.add_argument("--strided", action="store_true")
    ap.add_argument("--total", type=int, default=786432)
    a = ap.parse_args()
    import jax
    import jax.numpy as jnp
    from micro_xe2 import timeit
    rng = np.random.default_rng(0)
    floor = jax.jit(lambda x: x.reshape(-1)[0] * 1.0)

    if a.seg or not a.strided:
        print(f"{'shape':>18} {'elems':>9} {'ms':>9} {'above':>9} {'GB/s':>8} "
              f"{'ns/segment':>11}")
        for seg in SEGS:
            rows = max(a.total // seg, 1)
            x = jnp.asarray(rng.standard_normal((rows, seg)).astype(np.float32))
            t = timeit(jax.jit(lambda v: v.sum(-1)), (x,))
            b = timeit(floor, (x,))
            n = rows * seg
            print(f"{str((rows, seg)):>18} {n:9d} {t:9.4f} {t-b:9.4f} "
                  f"{n*4/(t-b)/1e6:8.1f} {(t-b)*1e6/rows:11.1f}")

    if a.strided:
        print(f"{'shape':>16} {'sum0':>9} {'GB/s':>7} {'sum-1':>9} {'GB/s':>7} "
              f"{'copy':>9} {'GB/s':>7}")
        for rows, cols in SHAPES:
            x = jnp.asarray(rng.standard_normal((rows, cols)).astype(np.float32))
            b = timeit(floor, (x,))
            t0 = timeit(jax.jit(lambda v: v.sum(0)), (x,)) - b
            t1 = timeit(jax.jit(lambda v: v.sum(-1)), (x,)) - b
            tc = timeit(jax.jit(lambda v: v * 2.0), (x,)) - b
            n = rows * cols * 4
            print(f"{str((rows, cols)):>16} {t0:9.4f} {n/t0/1e6:7.1f} "
                  f"{t1:9.4f} {n/t1/1e6:7.1f} {tc:9.4f} {2*n/tc/1e6:7.1f}")


if __name__ == "__main__":
    main()
