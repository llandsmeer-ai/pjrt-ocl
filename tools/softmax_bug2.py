"""§50 follow-up probes: is the fused-softmax tail corruption on the INPUT side?

Variants, all on the same 64x10 f32 data:
  A  softmax(x)                         -- softmax reads the program INPUT
  B  softmax(x*1.0+0.0)                 -- softmax reads an ARENA buffer
  C  (softmax(x), x*1.0)                -- also echo the input back
  D  x*2.0+1.0                          -- plain EW read of the same input
  E  softmax(x) with x used twice       -- input also feeds another consumer

Usage:
    . ./env.sh
    JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=Intel \
      .venv/bin/python tools/softmax_bug2.py
"""
from __future__ import annotations

import numpy as np


def main():
    import jax
    import jax.numpy as jnp
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((64, 10)) * 0.05).astype(np.float32)
    z = x - x.max(-1, keepdims=True)
    e = np.exp(z)
    ref = e / e.sum(-1, keepdims=True)
    xj = jnp.asarray(x)

    def report(tag, got, want):
        d = np.abs(np.asarray(got) - want)
        bad = sorted(set(np.argwhere(d > 1e-6)[:, 0].tolist()))
        print(f"  {tag:<34} maxerr={d.max():.3e} nbad_rows={len(bad)} {bad[:12]}")

    report("A softmax(x)  [ported input]",
           jax.jit(lambda v: jax.nn.softmax(v, -1))(xj), ref)
    report("B softmax(x*1+0) [arena input]",
           jax.jit(lambda v: jax.nn.softmax(v * 1.0 + 0.0, -1))(xj), ref)
    o = jax.jit(lambda v: (jax.nn.softmax(v, -1), v * 1.0))(xj)
    report("C softmax(x) of (sm,x) tuple", o[0], ref)
    report("C echo of x", o[1], x)
    report("D x*2+1 [ported input]",
           jax.jit(lambda v: v * 2.0 + 1.0)(xj), x * 2.0 + 1.0)
    report("E softmax(x)+0.001*x",
           jax.jit(lambda v: jax.nn.softmax(v, -1) + 0.001 * v)(xj),
           ref + 0.001 * x)


if __name__ == "__main__":
    main()
