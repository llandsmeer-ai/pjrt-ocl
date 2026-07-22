"""Ad-hoc micro-benchmark harness for isolating slow ops on a backend.

Times a list of named (fn, args) snippets with the same methodology as
tools/bench_suite/run_suite.py (warmup, adaptive iters, median of 5) and
prints ms plus ms-above-a-matching-dispatch-floor.

Snippet sets are selected with --set; see SETS at the bottom.

Usage:
    . ./env.sh
    JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=Intel \
      .venv/bin/python tools/micro_xe2.py --set nbody
    JAX_PLATFORMS=cpu .venv/bin/python tools/micro_xe2.py --set nbody
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def timeit(f, args, budget=0.05):
    import jax
    for _ in range(3):
        jax.block_until_ready(f(*args))
    t0 = time.perf_counter()
    jax.block_until_ready(f(*args))
    one = time.perf_counter() - t0
    iters = max(3, min(500, int(budget / max(one, 1e-6))))
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(iters):
            r = f(*args)
        jax.block_until_ready(r)
        ts.append((time.perf_counter() - t) / iters)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def run(snippets):
    import jax
    import jax.numpy as jnp
    # floor control: N inputs -> 1 scalar, per distinct arg count
    floors = {}

    def floor_for(args):
        n = len(args)
        if n not in floors:
            def ident(*a):
                acc = None
                for l in a:
                    v = l.reshape(-1)[0].astype(jnp.float32)
                    acc = v if acc is None else acc + v
                return acc
            floors[n] = timeit(jax.jit(ident), args)
        return floors[n]

    print(f"{'snippet':<34} {'ms':>9} {'floor':>8} {'above':>9}")
    for name, fn, args in snippets:
        args = tuple(jnp.asarray(a) for a in args)
        f = jax.jit(fn)
        try:
            ms = timeit(f, args)
        except Exception as e:
            print(f"{name:<34} ERROR {type(e).__name__}: {str(e)[:90]}")
            continue
        fl = floor_for(args)
        print(f"{name:<34} {ms:9.4f} {fl:8.4f} {ms-fl:9.4f}")


# --------------------------------------------------------------- snippet sets
def set_nbody():
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    Np = 64
    pos = rng.standard_normal((Np, 3)).astype(np.float32)
    vel = (rng.standard_normal((Np, 3)) * 0.1).astype(np.float32)
    mass = (np.abs(rng.standard_normal((Np,))) + 0.5).astype(np.float32)
    d_np = (pos[:, None, :] - pos[None, :, :]).astype(np.float32)
    r2_np = (d_np * d_np).sum(-1).astype(np.float32) + np.float32(1e-2)

    s = []
    s.append(("d = pos[:,None]-pos[None,:]",
              lambda p: p[:, None, :] - p[None, :, :], (pos,)))
    s.append(("r2 = (d*d).sum(-1)",
              lambda d: (d * d).sum(-1), (d_np,)))
    s.append(("inv = r2**-1.5",
              lambda r: r ** -1.5, (r2_np,)))
    s.append(("inv = 1/(r*sqrt(r))",
              lambda r: 1.0 / (r * jnp.sqrt(r)), (r2_np,)))
    s.append(("rsqrt3 = rsqrt(r)**3",
              lambda r: jax.lax.rsqrt(r) ** 3, (r2_np,)))
    s.append(("sum(axis=1) of (64,64,3)",
              lambda d: d.sum(axis=1), (d_np,)))
    s.append(("sum(axis=-1) of (64,64,3)",
              lambda d: d.sum(axis=-1), (d_np,)))
    s.append(("mul3 (64,64,3)",
              lambda d, i: d * i[..., None], (d_np, r2_np)))
    s.append(("full nbody step",
              lambda pos, vel, mass: vel + 0.01 * (
                  ((pos[:, None, :] - pos[None, :, :])
                   * (((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1) + 1e-2)
                   ** -1.5)[..., None] * mass[None, :, None]).sum(axis=1),
              (pos, vel, mass)))
    return s


def set_batchnorm():
    rng = np.random.default_rng(0)
    B, D = 128, 256
    x = rng.standard_normal((B, D)).astype(np.float32)
    g = np.ones(D, np.float32)
    b = np.zeros(D, np.float32)
    s = []
    s.append(("mean(axis=0) [128,256]", lambda x: x.mean(0, keepdims=True), (x,)))
    s.append(("mean(axis=-1) [128,256]", lambda x: x.mean(-1, keepdims=True), (x,)))
    s.append(("x - mean0", lambda x: x - x.mean(0, keepdims=True), (x,)))
    s.append(("batchnorm full", lambda x, g, b: (
        (x - x.mean(0, keepdims=True))
        * (((x - x.mean(0, keepdims=True)) ** 2).mean(0, keepdims=True) + 1e-5)
        ** -0.5 * g + b), (x, g, b)))
    return s


def set_layernorm():
    rng = np.random.default_rng(0)
    B, T, D = 16, 64, 256
    x = rng.standard_normal((B, T, D)).astype(np.float32)
    g = np.ones(D, np.float32)
    b = np.zeros(D, np.float32)
    s = []
    s.append(("copy x [16,64,256]", lambda x: x * 2.0, (x,)))
    s.append(("mean(-1)", lambda x: x.mean(-1, keepdims=True), (x,)))
    s.append(("x-mu", lambda x: x - x.mean(-1, keepdims=True), (x,)))
    s.append(("var", lambda x: ((x - x.mean(-1, keepdims=True)) ** 2)
              .mean(-1, keepdims=True), (x,)))
    s.append(("layernorm full", lambda x, g, b: (
        (x - x.mean(-1, keepdims=True))
        * (((x - x.mean(-1, keepdims=True)) ** 2).mean(-1, keepdims=True) + 1e-5)
        ** -0.5 * g + b), (x, g, b)))
    return s


def set_fft():
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((512,)).astype(np.float32)
    s = []
    s.append(("abs(fft(512))", lambda x: jnp.abs(jnp.fft.fft(x)), (sig,)))
    s.append(("fft(512).real", lambda x: jnp.fft.fft(x).real, (sig,)))
    return s


def set_mc():
    import jax
    import jax.numpy as jnp
    N = 1 << 16
    dummy = np.zeros(1, np.float32)
    s = []
    s.append(("uniform(2^16,2) only", lambda d: jax.random.uniform(
        jax.random.PRNGKey(0), (N, 2)).sum() + 0.0 * d[0], (dummy,)))
    s.append(("bits(2^16*2) only", lambda d: (jax.random.bits(
        jax.random.PRNGKey(0), (N, 2), jnp.uint32).sum().astype(jnp.float32)
        + 0.0 * d[0]), (dummy,)))
    s.append(("monte_carlo full", lambda d: 4.0 * (
        (jax.random.uniform(jax.random.PRNGKey(0), (N, 2))[:, 0] ** 2
         + jax.random.uniform(jax.random.PRNGKey(0), (N, 2))[:, 1] ** 2)
        <= 1.0).mean(keepdims=True) + 0.0 * d, (dummy,)))
    return s


def set_cnn():
    from jax import lax
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    B, H, W, Cin, Cout = 8, 32, 32, 3, 16
    x = rng.standard_normal((B, H, W, Cin)).astype(np.float32)
    w = (rng.standard_normal((3, 3, Cin, Cout)) * 0.1).astype(np.float32)
    y = rng.standard_normal((B, H, W, Cout)).astype(np.float32)
    s = []
    s.append(("conv only", lambda x, w: lax.conv_general_dilated(
        x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")),
        (x, w)))
    s.append(("relu+mean(1,2) of conv-out", lambda y: jnp.maximum(y, 0.).mean(
        axis=(1, 2)), (y,)))
    s.append(("cnn full", lambda x, w: jnp.maximum(lax.conv_general_dilated(
        x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")),
        0.).mean(axis=(1, 2)), (x, w)))
    return s


def set_mlp():
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    B, D0, D1, D2, D3 = 64, 256, 512, 256, 64
    x = rng.standard_normal((B, D0)).astype(np.float32)
    w1 = (rng.standard_normal((D0, D1)) * .05).astype(np.float32)
    w2 = (rng.standard_normal((D1, D2)) * .05).astype(np.float32)
    w3 = (rng.standard_normal((D2, D3)) * .05).astype(np.float32)
    b1 = np.zeros(D1, np.float32)
    s = []
    s.append(("x@w1 [64,256]x[256,512]", lambda x, w: x @ w, (x, w1)))
    s.append(("x@w1+b relu", lambda x, w, b: jnp.maximum(x @ w + b, 0.),
              (x, w1, b1)))
    s.append(("mlp full", lambda x, w1, b1, w2, w3: jnp.maximum(
        jnp.maximum(x @ w1 + b1, 0.) @ w2, 0.) @ w3,
        (x, w1, b1, w2, w3)))
    return s


def set_embsm():
    import jax
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    V, D, C, B = 1000, 64, 10, 32
    emb = (rng.standard_normal((V, D)) * .05).astype(np.float32)
    ids = rng.integers(0, V, size=(B,)).astype(np.int32)
    clsw = (rng.standard_normal((D, C)) * .05).astype(np.float32)
    logits = (rng.standard_normal((B, C)) * .05).astype(np.float32)
    s = []
    s.append(("gather emb[ids]", lambda e, i: e[i], (emb, ids)))
    s.append(("gather@clsw", lambda e, i, w: e[i] @ w, (emb, ids, clsw)))
    s.append(("softmax(32,10)", lambda z: jax.nn.softmax(z, -1), (logits,)))
    s.append(("embsm full", lambda e, i, w: jax.nn.softmax(e[i] @ w, -1),
              (emb, ids, clsw)))
    return s


SETS = {
    "nbody": set_nbody, "batchnorm": set_batchnorm, "layernorm": set_layernorm,
    "fft": set_fft, "monte_carlo": set_mc, "cnn": set_cnn, "mlp": set_mlp,
    "embedding_softmax": set_embsm,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(SETS))
    a = ap.parse_args()
    import jax  # noqa: F401  (backend selected by JAX_PLATFORMS)
    run(SETS[a.set]())
