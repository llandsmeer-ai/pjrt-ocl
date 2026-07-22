"""Workload registry for the bench suite.

Each workload is built by ``build(name)`` which returns ``(fn, tree, meta)``:

- ``fn(tree) -> jnp.ndarray``  — a single jittable function returning ONE array
  (so both backends can be compared with a plain ``allclose``).
- ``tree``                     — a pytree (dict/list) of **numpy** f32/i32 arrays,
  seeded and deterministic. The worker maps ``jnp.asarray`` over the leaves in
  the target process, so inputs land on the target device / backend.
- ``meta``                     — ``{"cat": ..., "note": ..., "expect": ...}``.

IMPORTANT: this module must NOT import jax at top level. The driver imports it
just to read ``WORKLOAD_NAMES``; jax is imported lazily inside ``build`` so that
``JAX_PLATFORMS`` (set by the worker's parent before the subprocess starts)
selects the backend correctly.
"""
from __future__ import annotations

import numpy as np

# Ordered: AI workloads first, then scientific, then physics/neuro analogues.
WORKLOAD_NAMES = [
    # --- AI ---
    "mlp",
    "cnn",
    "lstm",
    "gru",
    "transformer",
    "attention",
    "layernorm",
    "batchnorm",
    "embedding_softmax",
    # --- scientific ---
    "heat2d",
    "nbody",
    "rk4_ode",
    "logistic_map",
    "monte_carlo",
    "fft",
    # --- physics / neuro (brax real env; jaxley analogue) ---
    "spring_mass",
    "hh_neuron",
    "brax_step",
]


def _rng(seed=0):
    return np.random.default_rng(seed)


def _f32(a):
    return np.asarray(a, dtype=np.float32)


def build(name):
    """Return (fn, tree, meta) for the named workload. Imports jax lazily."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    f32 = np.float32

    # ------------------------------------------------------------------ AI ---
    if name == "mlp":
        rng = _rng(0)
        B, D0, D1, D2, D3 = 64, 256, 512, 256, 64
        tree = {
            "x": _f32(rng.standard_normal((B, D0))),
            "w1": _f32(rng.standard_normal((D0, D1)) * 0.05),
            "b1": np.zeros(D1, f32),
            "w2": _f32(rng.standard_normal((D1, D2)) * 0.05),
            "b2": np.zeros(D2, f32),
            "w3": _f32(rng.standard_normal((D2, D3)) * 0.05),
            "b3": np.zeros(D3, f32),
        }

        def fn(p):
            h = jnp.maximum(p["x"] @ p["w1"] + p["b1"], 0.0)
            h = jnp.maximum(h @ p["w2"] + p["b2"], 0.0)
            return h @ p["w3"] + p["b3"]

        return fn, tree, {"cat": "AI", "note": "3-layer MLP + relu", "expect": "PASS"}

    if name == "cnn":
        rng = _rng(0)
        B, H, W, Cin, Cout = 8, 32, 32, 3, 16
        tree = {
            "x": _f32(rng.standard_normal((B, H, W, Cin))),
            "w": _f32(rng.standard_normal((3, 3, Cin, Cout)) * 0.1),
        }

        def fn(p):
            y = lax.conv_general_dilated(
                p["x"], p["w"], window_strides=(1, 1), padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"))
            y = jnp.maximum(y, 0.0)
            # global average pool -> (B, Cout)
            return y.mean(axis=(1, 2))

        return fn, tree, {"cat": "AI", "note": "conv2d+relu+pool (flax nn.Conv equiv)",
                          "expect": "FAIL(convolution)"}

    if name in ("lstm", "gru"):
        rng = _rng(0)
        B, T, In, Hid = 16, 24, 32, 64
        gates = 4 if name == "lstm" else 3
        tree = {
            "x": _f32(rng.standard_normal((B, T, In))),
            "wx": _f32(rng.standard_normal((In, gates * Hid)) * 0.05),
            "wh": _f32(rng.standard_normal((Hid, gates * Hid)) * 0.05),
            "b": np.zeros(gates * Hid, f32),
        }

        def sigmoid(z):
            return jax.nn.sigmoid(z)

        if name == "lstm":
            def cell(carry, x_t, p):
                h, c = carry
                z = x_t @ p["wx"] + h @ p["wh"] + p["b"]
                i, f_, g, o = jnp.split(z, 4, axis=-1)
                i, f_, o = sigmoid(i), sigmoid(f_), sigmoid(o)
                g = jnp.tanh(g)
                c = f_ * c + i * g
                h = o * jnp.tanh(c)
                return (h, c), h

            def fn(p):
                x = p["x"].transpose(1, 0, 2)  # (T,B,In)
                B_, Hid_ = x.shape[1], p["wh"].shape[0]
                h0 = jnp.zeros((B_, Hid_))
                (h, _), _ = lax.scan(lambda ca, xt: cell(ca, xt, p), (h0, h0), x)
                return h

            note = "hand-rolled LSTM over lax.scan (sigmoid+tanh)"
        else:
            def cell(h, x_t, p):
                z = x_t @ p["wx"] + h @ p["wh"] + p["b"]
                r, u, n = jnp.split(z, 3, axis=-1)
                r, u = sigmoid(r), sigmoid(u)
                n = jnp.tanh(n * r)
                h = (1.0 - u) * n + u * h
                return h, h

            def fn(p):
                x = p["x"].transpose(1, 0, 2)
                B_, Hid_ = x.shape[1], p["wh"].shape[0]
                h0 = jnp.zeros((B_, Hid_))
                h, _ = lax.scan(lambda hh, xt: cell(hh, xt, p), h0, x)
                return h

            note = "hand-rolled GRU over lax.scan (sigmoid+tanh)"

        return fn, tree, {"cat": "AI", "note": note, "expect": "PASS?"}

    if name == "transformer":
        # Reuse the tuned transformer model definition.
        import importlib.util
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        bt_path = os.path.join(os.path.dirname(here), "bench_transformer.py")
        spec = importlib.util.spec_from_file_location("bench_transformer", bt_path)
        bt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bt)
        cfg = bt.CONFIGS["small"]
        x, params = bt.make_params(cfg, seed=0)
        model = bt.make_model(jnp)
        n_heads = cfg[3]
        tree = {"x": x, "params": params}

        def fn(p):
            return model(p["x"], p["params"], n_heads)

        return fn, tree, {"cat": "AI", "note": "GPT-style transformer block (small)",
                          "expect": "PASS"}

    if name == "attention":
        rng = _rng(0)
        B, T, D, Hh = 8, 64, 128, 4
        hd = D // Hh
        r = lambda *s: _f32(rng.standard_normal(s) * 0.05)
        tree = {"x": r(B, T, D), "wq": r(D, D), "wk": r(D, D),
                "wv": r(D, D), "wo": r(D, D)}

        def fn(p):
            x = p["x"]
            B_, T_, D_ = x.shape
            def split(t):
                return t.reshape(B_, T_, Hh, hd).transpose(0, 2, 1, 3)
            q, k, v = split(x @ p["wq"]), split(x @ p["wk"]), split(x @ p["wv"])
            scores = (q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5)
            attn = jax.nn.softmax(scores, axis=-1)
            out = (attn @ v).transpose(0, 2, 1, 3).reshape(B_, T_, D_)
            return out @ p["wo"]

        return fn, tree, {"cat": "AI", "note": "multi-head self-attention layer",
                          "expect": "PASS"}

    if name == "layernorm":
        rng = _rng(0)
        B, T, D = 16, 64, 256
        tree = {"x": _f32(rng.standard_normal((B, T, D))),
                "g": np.ones(D, f32), "b": np.zeros(D, f32)}

        def fn(p):
            x = p["x"]
            mu = x.mean(-1, keepdims=True)
            var = ((x - mu) ** 2).mean(-1, keepdims=True)
            return (x - mu) * (var + 1e-5) ** -0.5 * p["g"] + p["b"]

        return fn, tree, {"cat": "AI", "note": "layernorm (reduce over last axis)",
                          "expect": "PASS"}

    if name == "batchnorm":
        rng = _rng(0)
        B, D = 128, 256
        tree = {"x": _f32(rng.standard_normal((B, D))),
                "g": np.ones(D, f32), "b": np.zeros(D, f32)}

        def fn(p):
            x = p["x"]
            mu = x.mean(0, keepdims=True)         # reduce over BATCH (axis 0)
            var = ((x - mu) ** 2).mean(0, keepdims=True)
            return (x - mu) * (var + 1e-5) ** -0.5 * p["g"] + p["b"]

        return fn, tree, {"cat": "AI", "note": "batchnorm (reduce over axis 0)",
                          "expect": "FAIL(partial-axis reduce)"}

    if name == "embedding_softmax":
        rng = _rng(0)
        V, D, C, B = 1000, 64, 10, 32
        ids = rng.integers(0, V, size=(B,)).astype(np.int32)
        tree = {
            "emb": _f32(rng.standard_normal((V, D)) * 0.05),
            "ids": ids,
            "clsw": _f32(rng.standard_normal((D, C)) * 0.05),
        }

        def fn(p):
            e = p["emb"][p["ids"]]           # embedding lookup -> gather
            return jax.nn.softmax(e @ p["clsw"], axis=-1)

        return fn, tree, {"cat": "AI", "note": "embedding lookup + softmax classifier",
                          "expect": "FAIL(gather)"}

    # ------------------------------------------------------- scientific ---
    if name == "heat2d":
        rng = _rng(0)
        N, STEPS = 96, 30
        u0 = _f32(rng.standard_normal((N, N)))
        tree = {"u": u0}

        def step(u, _):
            lap = (jnp.roll(u, 1, 0) + jnp.roll(u, -1, 0)
                   + jnp.roll(u, 1, 1) + jnp.roll(u, -1, 1) - 4.0 * u)
            return u + 0.2 * lap, None

        def fn(p):
            u, _ = lax.scan(step, p["u"], None, length=STEPS)
            return u

        return fn, tree, {"cat": "SCI", "note": "2D heat-equation stencil (roll+EW, scan)",
                          "expect": "PASS?"}

    if name == "nbody":
        rng = _rng(0)
        Np = 64
        tree = {
            "pos": _f32(rng.standard_normal((Np, 3))),
            "vel": _f32(rng.standard_normal((Np, 3)) * 0.1),
            "mass": _f32(np.abs(rng.standard_normal((Np,))) + 0.5),
        }

        def fn(p):
            pos = p["pos"]
            # pairwise displacement (Np,Np,3), softened gravity, sum over sources
            d = pos[:, None, :] - pos[None, :, :]
            r2 = (d * d).sum(-1) + 1e-2
            inv = r2 ** -1.5
            f = (d * inv[..., None] * p["mass"][None, :, None]).sum(axis=1)
            return p["vel"] + 0.01 * f

        return fn, tree, {"cat": "SCI", "note": "N-body gravity step (pairwise + reduce)",
                          "expect": "PASS?"}

    if name == "rk4_ode":
        rng = _rng(0)
        B, STEPS, dt = 32, 100, 0.01
        y0 = _f32(rng.standard_normal((B, 3)) * 0.5 + np.array([1., 1., 1.]))
        tree = {"y": y0}

        def deriv(y):  # Lorenz system, batched
            x, yy, z = y[..., 0], y[..., 1], y[..., 2]
            s, rr, bb = 10.0, 28.0, 8.0 / 3.0
            return jnp.stack([s * (yy - x), x * (rr - z) - yy, x * yy - bb * z], -1)

        def step(y, _):
            k1 = deriv(y)
            k2 = deriv(y + 0.5 * dt * k1)
            k3 = deriv(y + 0.5 * dt * k2)
            k4 = deriv(y + dt * k3)
            return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None

        def fn(p):
            y, _ = lax.scan(step, p["y"], None, length=STEPS)
            return y

        return fn, tree, {"cat": "SCI", "note": "RK4 Lorenz integrator (scan)",
                          "expect": "PASS?"}

    if name == "logistic_map":
        rng = _rng(0)
        B, ITERS = 256, 200
        x0 = _f32(rng.uniform(0.1, 0.9, size=(B,)))
        r = _f32(np.linspace(3.5, 4.0, B))
        tree = {"x": x0, "r": r}

        def fn(p):
            def body(_, x):
                return p["r"] * x * (1.0 - x)
            return lax.fori_loop(0, ITERS, body, p["x"])

        return fn, tree, {"cat": "SCI", "note": "logistic-map iteration (fori_loop/while)",
                          "expect": "PASS?"}

    if name == "monte_carlo":
        N = 1 << 16
        tree = {"dummy": np.zeros(1, f32)}

        def fn(p):
            key = jax.random.PRNGKey(0)
            xy = jax.random.uniform(key, (N, 2))
            inside = (xy[:, 0] ** 2 + xy[:, 1] ** 2) <= 1.0
            return 4.0 * inside.mean(keepdims=True) + 0.0 * p["dummy"]

        return fn, tree, {"cat": "SCI", "note": "Monte-Carlo pi (threefry RNG)",
                          "expect": "FAIL(threefry/shift)"}

    if name == "fft":
        rng = _rng(0)
        N = 512
        tree = {"sig": _f32(rng.standard_normal((N,)))}

        def fn(p):
            return jnp.abs(jnp.fft.fft(p["sig"]))

        return fn, tree, {"cat": "SCI", "note": "1D FFT magnitude (needs complex dtype + fft)",
                          "expect": "FAIL(complex/fft)"}

    # --------------------------------------------- physics / neuro ---
    if name == "spring_mass":
        # brax analogue: 1D chain of masses joined by springs, semi-implicit Euler.
        rng = _rng(0)
        Nm, STEPS, dt, k, damp = 64, 120, 0.05, 5.0, 0.02
        tree = {
            "x": _f32(np.linspace(0, 1, Nm) + rng.standard_normal(Nm) * 0.01),
            "v": _f32(rng.standard_normal(Nm) * 0.0),
            "x0": _f32(np.linspace(0, 1, Nm)),
        }

        def step(carry, _):
            x, v = carry
            # spring force from left/right neighbours (fixed ends via roll+mask)
            left = jnp.concatenate([x[:1], x[:-1]])
            right = jnp.concatenate([x[1:], x[-1:]])
            f = k * (left + right - 2.0 * x) - damp * v
            v = v + dt * f
            x = x + dt * v
            return (x, v), None

        def fn(p):
            (x, v), _ = lax.scan(step, (p["x"], p["v"]), None, length=STEPS)
            return x

        return fn, tree, {"cat": "PHYS", "note": "spring-mass chain (brax analogue, scan)",
                          "expect": "PASS?"}

    if name == "hh_neuron":
        # jaxley analogue: Hodgkin-Huxley single compartment, forward Euler.
        rng = _rng(0)
        B, STEPS, dt = 64, 200, 0.01
        tree = {
            "V": _f32(-65.0 + rng.standard_normal(B)),
            "m": _f32(np.full(B, 0.05)),
            "h": _f32(np.full(B, 0.6)),
            "n": _f32(np.full(B, 0.32)),
            "I": _f32(np.full(B, 10.0)),
        }
        gNa, gK, gL, ENa, EK, EL, Cm = 120.0, 36.0, 0.3, 50.0, -77.0, -54.4, 1.0

        def rates(V):
            am = 0.1 * (V + 40.0) / (1.0 - jnp.exp(-(V + 40.0) / 10.0))
            bm = 4.0 * jnp.exp(-(V + 65.0) / 18.0)
            ah = 0.07 * jnp.exp(-(V + 65.0) / 20.0)
            bh = 1.0 / (1.0 + jnp.exp(-(V + 35.0) / 10.0))
            an = 0.01 * (V + 55.0) / (1.0 - jnp.exp(-(V + 55.0) / 10.0))
            bn = 0.125 * jnp.exp(-(V + 65.0) / 80.0)
            return am, bm, ah, bh, an, bn

        def step(carry, _):
            V, m, h, n = carry
            am, bm, ah, bh, an, bn = rates(V)
            m = m + dt * (am * (1 - m) - bm * m)
            h = h + dt * (ah * (1 - h) - bh * h)
            n = n + dt * (an * (1 - n) - bn * n)
            INa = gNa * m ** 3 * h * (V - ENa)
            IK = gK * n ** 4 * (V - EK)
            IL = gL * (V - EL)
            V = V + dt / Cm * (tree["I"] - INa - IK - IL)
            return (V, m, h, n), None

        def fn(p):
            (V, m, h, n), _ = lax.scan(
                step, (p["V"], p["m"], p["h"], p["n"]), None, length=STEPS)
            return V

        return fn, tree, {"cat": "PHYS", "note": "Hodgkin-Huxley neuron (jaxley analogue, scan)",
                          "expect": "PASS?"}

    if name == "brax_step":
        # Real brax env: reset + one physics step, jitted as one program.
        #
        # LAYER 1 (host device allowlist): brax 0.14.2 routes every backend
        # through MuJoCo-MJX's io._resolve_impl, which raises "Unsupported
        # device" for any platform other than gpu/tpu/cpu (our 'opencl' PJRT
        # device is device-agnostic pure jax — the physics doesn't care). Bypass
        # by pinning the JAX impl (§41). This is a host-side monkeypatch, not a
        # backend change; it is a no-op on cuda (which resolves normally).
        try:
            import mujoco.mjx._src.io as _mjxio
            from mujoco.mjx._src import types as _mjxtypes
            _orig = _mjxio._resolve_impl
            _mjxio._resolve_impl = (lambda d: _mjxtypes.Impl.JAX
                                    if d.platform not in ("gpu", "tpu", "cpu")
                                    else _orig(d))
        except Exception:  # noqa: BLE001 — no mjx: brax's own backends still build
            pass
        from brax import envs
        env = envs.create("inverted_pendulum", backend="positional")
        key = jax.random.PRNGKey(0)
        A = env.action_size
        tree = {"act": np.zeros(A, f32)}

        def fn(p):
            state = env.reset(key)
            nxt = env.step(state, p["act"])
            return nxt.obs

        # LAYER 2 (sdy dialect) is now FIXED in lowering.py: the VHLO artifact's
        # Shardy sharding hints deserialize + collapse to identity (§41). What
        # still blocks a full end-to-end brax run (catalogued, next M3 gaps):
        #   * combined jit(reset+step) trips a JAX-internal 'opencl'-platform
        #     lowering bug: @_threefry_split is outlined with a ui32 signature
        #     but called with i32 (MLIR verifier: 'operand type mismatch
        #     tensor<2xui32> vs tensor<2xi32>'). Not our code — JAX's random
        #     lowering for the experimental platform. jit(reset) alone lowers.
        #   * mjx physics (ant/humanoid): reduce with an `and`/`or` reducer body
        #     (from jp.allclose/jp.all — needs an integer min/max reduce),
        #     general data-dependent stablehlo.scatter, and chlo.erf_inv.
        return fn, tree, {"cat": "PHYS",
                          "note": "brax inverted_pendulum reset+step (real env). "
                                  "Layer-1 device allowlist bypassed; sdy dialect fixed; "
                                  "ui32 dtype reporting fixed (§42); OP_CONV/scatter/"
                                  "reduce-and (§39/§42) — runs end-to-end vs CUDA",
                          "expect": "PASS"}

    raise KeyError(f"unknown workload: {name}")
