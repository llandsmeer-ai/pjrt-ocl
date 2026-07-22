"""§50 follow-up: is the fused LAYERNORM_SEG kernel affected like softmax_seg?

Runs the bench-suite layernorm shape (16,64,256) through jax.jit on the current
backend and compares against a float64 numpy reference.

Usage:
    . ./env.sh
    JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=Intel .venv/bin/python tools/norm_bug.py
"""
from __future__ import annotations

import numpy as np


def main():
    import jax
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    B, T, D = 16, 64, 256
    x = rng.standard_normal((B, T, D)).astype(np.float32)
    g = np.ones(D, np.float32)
    b = np.zeros(D, np.float32)
    xd = x.astype(np.float64)
    mu = xd.mean(-1, keepdims=True)
    var = ((xd - mu) ** 2).mean(-1, keepdims=True)
    ref = ((xd - mu) * (var + 1e-5) ** -0.5 * g + b)

    def fn(x, g, b):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) * (var + 1e-5) ** -0.5 * g + b

    got = np.asarray(jax.jit(fn)(jnp.asarray(x), jnp.asarray(g), jnp.asarray(b)))
    d = np.abs(got - ref)
    rows = d.reshape(-1, D).max(1)
    bad = np.argwhere(rows > 1e-3).ravel().tolist()
    print(f"layernorm {B}x{T}x{D}: maxabs={d.max():.3e} "
          f"nbad_rows={len(bad)}/{B*T} first={bad[:10]} last={bad[-10:]}")


if __name__ == "__main__":
    main()
