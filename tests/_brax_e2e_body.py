"""E2E: a real brax env (inverted_pendulum, positional backend) reset + one
physics step, jitted as ONE program, run through whatever JAX backend the
process is configured for. Prints the observation + reward so a caller can
compare our OpenCL backend against native CUDA.

Exercises the full brax-step op set end-to-end: stablehlo.case (N-way switch,
lowered to flat sibling OP_IFs), non-canonical dot_general (transpose-
canonicalized), gather/scatter, reduce-and/or, threefry RNG, sdy hints.

Runs in a fresh process (jax backend is process-global). Emits one JSON line
prefixed BRAX_RESULT so the harness can parse obs/reward regardless of backend.
"""
import json
import sys

import numpy as np

import jax

# LAYER 1 (host device allowlist): brax routes every backend through MJX's
# io._resolve_impl, which rejects any platform other than gpu/tpu/cpu. Our
# 'opencl' PJRT device is device-agnostic pure jax — pin the JAX impl (a host
# shim, no-op on cuda/cpu). See docs/decisions.md §41.
try:
    import mujoco.mjx._src.io as _mjxio
    from mujoco.mjx._src import types as _mjxtypes
    _orig = _mjxio._resolve_impl
    _mjxio._resolve_impl = (lambda d: _mjxtypes.Impl.JAX
                            if d.platform not in ("gpu", "tpu", "cpu")
                            else _orig(d))
except Exception:  # noqa: BLE001
    pass

from brax import envs

env = envs.create("inverted_pendulum", backend="positional")
key = jax.random.PRNGKey(0)
act = np.zeros(env.action_size, np.float32)


def fn(p):
    state = env.reset(key)
    nxt = env.step(state, p["act"])
    return nxt.obs, nxt.reward


obs, reward = jax.jit(fn)({"act": act})
obs = np.asarray(obs, np.float64)
reward = float(np.asarray(reward))
platform = jax.devices()[0].platform

assert np.all(np.isfinite(obs)), f"obs not finite: {obs}"
assert np.isfinite(reward), f"reward not finite: {reward}"

print("BRAX_RESULT " + json.dumps({
    "platform": platform,
    "obs": obs.tolist(),
    "reward": reward,
}))
print("BRAX E2E PASS", file=sys.stderr)
