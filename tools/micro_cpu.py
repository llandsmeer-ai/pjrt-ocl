"""Microbenchmarks isolating the CPU tile path: large EW / reduce / norm.
Run one backend per process (JAX_PLATFORMS chosen by caller).
    .venv/bin/python tools/micro_cpu.py <case> <backend-label>
prints ##MJSON## {json}
"""
import sys, time, json, numpy as np


def build(case):
    import jax, jax.numpy as jnp
    rng = np.random.default_rng(0)
    if case == "ew_add_16m":
        n = 16 << 20
        a = jnp.asarray(rng.standard_normal(n, dtype=np.float64).astype(np.float32))
        b = jnp.asarray(rng.standard_normal(n, dtype=np.float64).astype(np.float32))
        return jax.jit(lambda a, b: a + b), (a, b)
    if case == "ew_chain_16m":  # 4-op chain, memory-bound
        n = 16 << 20
        a = jnp.asarray(rng.standard_normal(n, dtype=np.float64).astype(np.float32))
        b = jnp.asarray(rng.standard_normal(n, dtype=np.float64).astype(np.float32))
        return jax.jit(lambda a, b: (a * b + a) * 2.0 - b), (a, b)
    if case == "reduce_sum_16m":
        n = 16 << 20
        a = jnp.asarray(rng.standard_normal(n, dtype=np.float64).astype(np.float32))
        return jax.jit(lambda a: a.sum()), (a,)
    if case == "reduce_max_16m":
        n = 16 << 20
        a = jnp.asarray(rng.standard_normal(n, dtype=np.float64).astype(np.float32))
        return jax.jit(lambda a: a.max()), (a,)
    if case == "layernorm_big":       # 4096 rows x 1024 (suffix reduce)
        x = jnp.asarray(rng.standard_normal((4096, 1024), dtype=np.float64).astype(np.float32))
        g = jnp.ones(1024, np.float32); b = jnp.zeros(1024, np.float32)
        def fn(x, g, b):
            mu = x.mean(-1, keepdims=True)
            var = ((x - mu) ** 2).mean(-1, keepdims=True)
            return (x - mu) * (var + 1e-5) ** -0.5 * g + b
        return jax.jit(fn), (x, g, b)
    if case == "softmax_big":         # 4096 rows x 1024 (suffix reduce)
        x = jnp.asarray(rng.standard_normal((4096, 1024), dtype=np.float64).astype(np.float32))
        import jax.nn
        return jax.jit(lambda x: jax.nn.softmax(x, axis=-1)), (x,)
    if case == "reduce_rows_big":     # 8192 x 512 sum over last axis (segmented)
        x = jnp.asarray(rng.standard_normal((8192, 512), dtype=np.float64).astype(np.float32))
        return jax.jit(lambda x: x.sum(-1)), (x,)
    if case == "reduce_cols_big":     # 512 x 8192 sum over axis 0 (strided)
        x = jnp.asarray(rng.standard_normal((512, 8192), dtype=np.float64).astype(np.float32))
        return jax.jit(lambda x: x.sum(0)), (x,)
    raise SystemExit("unknown case " + case)


def main():
    case, label = sys.argv[1], sys.argv[2]
    import jax
    fn, args = build(case)
    out = fn(*args); jax.block_until_ready(out)
    for _ in range(3):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter(); jax.block_until_ready(fn(*args))
    one = time.perf_counter() - t0
    iters = max(3, min(300, int(0.15 / max(one, 1e-6))))
    ts = []
    for _ in range(7):
        t = time.perf_counter()
        for _ in range(iters):
            r = fn(*args)
        jax.block_until_ready(r)
        ts.append((time.perf_counter() - t) / iters)
    ts.sort()
    o = np.asarray(out)
    print("##MJSON## " + json.dumps({
        "case": case, "label": label, "ms": ts[len(ts)//2]*1e3,
        "ms_min": ts[0]*1e3, "iters": iters,
        "sum": float(o.sum()), "shape": list(o.shape)}))


if __name__ == "__main__":
    main()
