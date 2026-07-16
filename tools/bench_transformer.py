"""A small GPT-style transformer forward pass (random weights) as a realistic
end-to-end workload for the OpenCL backend, benchmarked against native CUDA.

It's not a trained model — random matrices — but it stresses the library the way
a real transformer does: matmuls (QKV/out/FFN projections), batched attention
(QKᵀ, softmax, AV per head), layernorm (reduce mean/var, rsqrt), GELU, residual
adds, and the reshapes/transposes that multi-head attention needs.

Usage:
    . ./env.sh && python tools/bench_transformer.py [--check] [--config small]
"""
from __future__ import annotations

import argparse
import time

import numpy as np


# --- model (pure jnp; params are plain dicts of arrays) ----------------------

def make_model(jnp):
    def layernorm(x, g, b, eps=1e-5):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) * (var + eps) ** -0.5 * g + b

    def gelu(x):                       # tanh approximation (what GPT-2 uses)
        return 0.5 * x * (1.0 + jnp.tanh(
            0.7978845608 * (x + 0.044715 * x ** 3)))

    def softmax(x):
        x = x - x.max(-1, keepdims=True)
        e = jnp.exp(x)
        return e / e.sum(-1, keepdims=True)

    def attention(x, p, n_heads):
        B, T, D = x.shape
        H, hd = n_heads, D // n_heads
        q = x @ p["wq"]
        k = x @ p["wk"]
        v = x @ p["wv"]
        # (B,T,D) -> (B,H,T,hd)
        def split(t):
            return t.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        q, k, v = split(q), split(k), split(v)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5)   # (B,H,T,T)
        out = softmax(scores) @ v                               # (B,H,T,hd)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return out @ p["wo"]

    def block(x, p, n_heads):
        x = x + attention(layernorm(x, p["ln1_g"], p["ln1_b"]), p, n_heads)
        h = gelu(layernorm(x, p["ln2_g"], p["ln2_b"]) @ p["w1"]) @ p["w2"]
        return x + h

    def model(x, params, n_heads):
        for p in params:
            x = block(x, p, n_heads)
        return x

    return model


CONFIGS = {
    # (batch, seq, d_model, n_heads, d_ff, n_layers)
    "tiny":  (1, 64, 128, 4, 512, 2),
    "small": (1, 128, 256, 4, 1024, 4),
    "base":  (4, 128, 512, 8, 2048, 6),
    # compute-bound: large D/F make the projections + FFN dominate, where TF32
    # tensor cores should shine and the small-op/barrier overhead amortizes.
    "large": (8, 256, 1024, 16, 4096, 6),
    # single large layer: fits the arena today; the full 6-layer `large` needs
    # arena liveness-reuse (bump allocator overflows the u32 offset cap — §15).
    "large_l1": (8, 256, 1024, 16, 4096, 1),
}


def make_params(cfg, seed=0):
    B, T, D, H, F, L = cfg
    rng = np.random.default_rng(seed)
    f32 = np.float32
    def r(*shape, s=0.02):
        return (rng.standard_normal(shape) * s).astype(f32)
    params = []
    for _ in range(L):
        params.append({
            "ln1_g": np.ones(D, f32), "ln1_b": np.zeros(D, f32),
            "ln2_g": np.ones(D, f32), "ln2_b": np.zeros(D, f32),
            "wq": r(D, D), "wk": r(D, D), "wv": r(D, D), "wo": r(D, D),
            "w1": r(D, F), "w2": r(F, D),
        })
    x = r(B, T, D, s=1.0)
    return x, params


def flops(cfg):
    B, T, D, H, F, L = cfg
    # 4 DxD projections + attn (2*T*T*D) + FFN (2*D*F), x2 for mul-add, per layer
    per_layer = (4 * B * T * D * D + 2 * B * H * T * T * (D // H) * 2
                 + 2 * B * T * D * F) * 2
    return per_layer * L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="small", choices=list(CONFIGS))
    ap.add_argument("--check", action="store_true",
                    help="compare against a JAX-CPU reference and exit nonzero on mismatch")
    ap.add_argument("--dump-ref", metavar="PATH",
                    help="(internal) run the model on the current JAX backend and "
                         ".npy the output to PATH; used by --check's reference subprocess")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    cfg = CONFIGS[args.config]
    model = make_model(jnp)
    x, params = make_params(cfg)
    jx = jnp.asarray(x)
    jp = [{k: jnp.asarray(v) for k, v in p.items()} for p in params]
    fn = jax.jit(lambda x, p: model(x, p, cfg[3]))

    if args.dump_ref:
        np.save(args.dump_ref, np.asarray(fn(jx, jp)))
        return

    if args.check:
        got = np.asarray(fn(jx, jp))
        # Reference on JAX CPU (deterministic params, seed=0) in a subprocess —
        # backends can't be switched in-process. Tolerances match the TF32
        # tensor-core path (docs/decisions.md §10c): abs ~5e-3 on a ~1-std signal.
        import subprocess, sys, tempfile, os as _os
        with tempfile.TemporaryDirectory() as td:
            ref_path = _os.path.join(td, "ref.npy")
            env = dict(_os.environ, JAX_PLATFORMS="cpu")
            subprocess.run([sys.executable, __file__, "--config", args.config,
                            "--dump-ref", ref_path], check=True, env=env)
            ref = np.load(ref_path)
        max_abs = float(np.max(np.abs(got - ref)))
        max_rel = float(np.max(np.abs(got - ref) / (np.abs(ref) + 1e-6)))
        ok = np.allclose(got, ref, atol=5e-2, rtol=2e-2)
        print(f"config={args.config} out shape={got.shape} "
              f"finite={np.isfinite(got).all()} mean={got.mean():.4f} "
              f"std={got.std():.4f} | vs JAX-CPU: max_abs={max_abs:.2e} "
              f"max_rel={max_rel:.2e} allclose(5e-2,2e-2)={'PASS' if ok else 'FAIL'}")
        sys.exit(0 if (ok and np.isfinite(got).all()) else 1)

    for _ in range(4):
        jax.block_until_ready(fn(jx, jp))
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(args.iters):
            r = fn(jx, jp)
        jax.block_until_ready(r)
        ts.append((time.perf_counter() - t) / args.iters)
    ms = float(np.median(ts)) * 1e3
    gf = flops(cfg) / (ms * 1e-3) / 1e9
    print(f"config={args.config} {ms:.4f} ms/iter  {gf:.1f} GFLOP/s")


if __name__ == "__main__":
    main()
