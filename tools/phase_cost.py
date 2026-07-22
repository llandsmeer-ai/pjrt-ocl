"""Measure the marginal cost of ONE extra compute phase (dataflow level) on the
current backend, and the marginal cost of extra elements.

Builds programs of k chained roll(+1) steps (a roll is a TILE_GATHER, which does
not fuse with its neighbours, so each step is its own phase) over an array of n
f32 elements, times them, and least-squares fits ms = a + b*k. `b` is the
per-phase marginal cost including both host enqueue and device execution.

Usage:
    . ./env.sh
    JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=Intel \
      .venv/bin/python tools/phase_cost.py --n 1024
"""
from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32])
    a = ap.parse_args()
    import jax
    import jax.numpy as jnp
    import sys
    sys.path.insert(0, "tools")
    from micro_xe2 import timeit

    x = jnp.asarray(np.random.default_rng(0).standard_normal(a.n).astype(np.float32))

    def mk(k):
        def f(v):
            for i in range(k):
                v = jnp.roll(v, 1) + np.float32(1.0 + i * 1e-6)
            return v
        return jax.jit(f)

    print(f"n={a.n}")
    rows = []
    for k in a.ks:
        ms = timeit(mk(k), (x,))
        rows.append((k, ms))
        print(f"  k={k:4d}  {ms:.4f} ms")
    ks = np.array([r[0] for r in rows], float)
    ts = np.array([r[1] for r in rows], float)
    A = np.stack([np.ones_like(ks), ks], 1)
    coef, *_ = np.linalg.lstsq(A, ts, rcond=None)
    print(f"  fit: ms = {coef[0]:.4f} + {coef[1]*1e3:.2f} us * k")


if __name__ == "__main__":
    main()
