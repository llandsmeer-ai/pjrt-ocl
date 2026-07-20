"""Autoregressive DECODE benchmark — the regime megakernels are built for.

One token through a GPT-style model with a KV cache of length C. At batch-1
decode every matmul is a matVEC (M=1): memory-bound (read the weights once),
NOT compute-bound — so cuBLAS's tensor-core tuning is irrelevant and the cost is
weight-HBM-bandwidth + the ~100 kernel launches/token that a megakernel folds
into one. This is the opposite of the prefill benchmark (tools/bench_transformer.py),
where big compute-bound matmuls are cuBLAS's home turf.

Backend is process-global, so ours vs the reference run in separate processes.

Usage:
    . ./env.sh && JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=NVIDIA python tools/bench_decode.py --config base
    . ./env.sh && JAX_PLATFORMS=cuda python tools/bench_decode.py --config base   # reference
"""
from __future__ import annotations
import argparse, time
import numpy as np

# (d_model, n_heads, d_ff, n_layers, context_len)
CONFIGS = {
    "small": (256, 4, 1024, 4, 128),
    "base":  (512, 8, 2048, 6, 256),
    "large": (1024, 16, 4096, 6, 512),
}


def make_decode(jnp):
    def ln(x, g, b, eps=1e-5):
        m = x.mean(-1, keepdims=True)
        v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) * (v + eps) ** -0.5 * g + b

    def gelu(x):
        return 0.5 * x * (1.0 + jnp.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

    def step(x, kc, vc, params, H):
        # x: (1, D)   kc/vc: per-layer (H, C, hd)
        D = x.shape[-1]; hd = D // H
        for li, p in enumerate(params):
            h = ln(x, p["ln1_g"], p["ln1_b"])
            q = (h @ p["wq"]).reshape(H, 1, hd)          # (H,1,hd)
            k = (h @ p["wk"]).reshape(H, 1, hd)
            v = (h @ p["wv"]).reshape(H, 1, hd)
            kk = jnp.concatenate([kc[li], k], axis=1)    # (H, C+1, hd)
            vv = jnp.concatenate([vc[li], v], axis=1)
            scores = (q @ jnp.swapaxes(kk, -1, -2)) * (hd ** -0.5)  # (H,1,C+1)
            a = jax_softmax(scores)
            o = (a @ vv).reshape(1, D)                   # (1,D)
            x = x + o @ p["wo"]
            h = ln(x, p["ln2_g"], p["ln2_b"])
            x = x + gelu(h @ p["w1"]) @ p["w2"]
        return x

    def jax_softmax(s):
        s = s - s.max(-1, keepdims=True)
        e = jnp.exp(s)
        return e / e.sum(-1, keepdims=True)
    return step


def make_params(cfg, seed=0):
    D, H, F, L, C = cfg; hd = D // H
    rng = np.random.default_rng(seed); f32 = np.float32
    def r(*s, sc=0.02): return (rng.standard_normal(s) * sc).astype(f32)
    params = [{
        "ln1_g": np.ones(D, f32), "ln1_b": np.zeros(D, f32),
        "ln2_g": np.ones(D, f32), "ln2_b": np.zeros(D, f32),
        "wq": r(D, D), "wk": r(D, D), "wv": r(D, D), "wo": r(D, D),
        "w1": r(D, F), "w2": r(F, D),
    } for _ in range(L)]
    x = r(1, D, sc=1.0)
    kc = r(L, H, C, hd); vc = r(L, H, C, hd)
    return x, kc, vc, params


def bytes_moved(cfg):
    D, H, F, L, C = cfg
    # weights read once per token (the decode floor) + KV cache read
    w = L * (4 * D * D + 2 * D * F) * 4
    kv = L * 2 * H * C * (D // H) * 4
    return w + kv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base", choices=list(CONFIGS))
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    import jax, jax.numpy as jnp
    cfg = CONFIGS[a.config]; D, H, F, L, C = cfg
    step = make_decode(jnp)
    x, kc, vc, params = make_params(cfg)
    jx, jkc, jvc = jnp.asarray(x), jnp.asarray(kc), jnp.asarray(vc)
    jp = [{k: jnp.asarray(v) for k, v in p.items()} for p in params]
    fn = jax.jit(lambda x, kc, vc, p: step(x, kc, vc, p, H))
    if a.check:
        got = np.asarray(fn(jx, jkc, jvc, jp))
        print(f"decode {a.config} out={got.shape} finite={np.isfinite(got).all()} "
              f"mean={got.mean():.4f} std={got.std():.4f}")
        return
    for _ in range(5): jax.block_until_ready(fn(jx, jkc, jvc, jp))
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(a.iters): r = fn(jx, jkc, jvc, jp)
        jax.block_until_ready(r); ts.append((time.perf_counter() - t) / a.iters)
    ms = float(np.median(ts)) * 1e3
    gbps = bytes_moved(cfg) / (ms * 1e-3) / 1e9
    print(f"decode {a.config}  {ms*1000:.1f} us/token  {gbps:.0f} GB/s (weight+KV BW)")


if __name__ == "__main__":
    main()
