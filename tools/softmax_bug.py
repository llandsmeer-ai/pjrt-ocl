"""Minimal reproducer for the §50 Xe2 fused-softmax wrong-answer bug.

Prints, for the given shape, the max |error| vs a numpy reference and the row
indices that are wrong, plus whether each wrong row came back UNIFORM (all
classes equal, i.e. the staged row read as a constant / zeros).

Usage:
    . ./env.sh
    JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=Intel \
      .venv/bin/python tools/softmax_bug.py --rows 64 --cols 10
"""
from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pre", action="store_true",
                    help="feed softmax from a preceding elementwise op")
    ap.add_argument("--post", action="store_true",
                    help="consume the softmax result with a following EW op")
    a = ap.parse_args()
    import jax
    import jax.numpy as jnp
    x = (np.random.default_rng(a.seed).standard_normal((a.rows, a.cols)) * 0.05
         ).astype(np.float32)
    z = x - x.max(-1, keepdims=True)
    e = np.exp(z)
    ref = e / e.sum(-1, keepdims=True)
    def body(v):
        if a.pre:
            v = v * 1.0 + 0.0
        r = jax.nn.softmax(v, axis=-1)
        if a.post:
            r = r * 1.0 + 0.0
        return r
    f = jax.jit(body)
    got = np.asarray(f(jnp.asarray(x)))
    d = np.abs(got - ref)
    bad = sorted(set(np.argwhere(d > 1e-6)[:, 0].tolist()))
    uni = [int(r) for r in bad if float(got[r].std()) < 1e-9]
    print(f"rows={a.rows} cols={a.cols} pre={a.pre} maxerr={d.max():.3e} "
          f"nbad={len(bad)} bad_rows={bad[:40]}")
    print(f"  uniform-row (all classes equal) among bad: {len(uni)} {uni[:20]}")
    if bad:
        r = bad[0]
        print(f"  row {r}: got={got[r][:6]} ref={ref[r][:6]}")


if __name__ == "__main__":
    main()
