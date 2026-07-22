"""E2E: jax.random (threefry2x32) through the real plugin on OpenCL.

Runs in a fresh opencl-only process (jax's backend is process-global). Since we
cannot also spin up the CPU backend here for a reference, the threefry output is
compared against a GOLDEN captured from JAX's CPU backend
(``jax.random.bits(PRNGKey(0), (8,), uint32)``) — a bit-exact regression that
proves the ui64-counter shift/xor/iota path matches XLA. Also sanity-checks
uniform range, normal finiteness, and a Monte-Carlo pi estimate.
"""
import numpy as np

import jax
import jax.numpy as jnp

(dev,) = jax.devices()
assert dev.platform == "opencl", dev.platform

# Golden: JAX CPU backend, jax.random.bits(PRNGKey(0), (8,), dtype=uint32).
GOLDEN = np.array([0xF29A4FA7, 0xFA843692, 0x55110E28, 0x77FAA835,
                   0x91E43BB7, 0x2A5E6943, 0x4F68EA7D, 0xB081CCFE],
                  dtype=np.uint32)

bits = np.asarray(jax.jit(
    lambda: jax.random.bits(jax.random.PRNGKey(0), (8,), dtype=jnp.uint32))())
np.testing.assert_array_equal(bits.astype(np.uint32), GOLDEN)
print("threefry bits bit-exact vs CPU golden: ok")

# ui32 host round-trip: a device uint32 buffer must report U32 (not S32) at the
# PJRT boundary, so np.asarray keeps uint32. Regression guard for the dtype bug
# that made np.asarray(key) yield int32 → materialised the closed-over threefry
# key as tensor<2xi32>, tripping JAX's own ui32 @_threefry_split verifier
# (blocked brax reset+step). Assert the *reported* dtype directly (no astype).
assert bits.dtype == np.uint32, ("device uint32 buffer reported as "
                                 f"{bits.dtype}, expected uint32")
key_host = np.asarray(jax.random.PRNGKey(0))
assert key_host.dtype == np.uint32, (
    f"PRNGKey round-tripped as {key_host.dtype}, expected uint32")
print("ui32 host round-trip dtype ok")

# uniform in [0, 1)
uni = np.asarray(jax.jit(
    lambda: jax.random.uniform(jax.random.PRNGKey(0), (1 << 14, 2)))())
assert np.isfinite(uni).all(), "uniform not finite"
assert uni.min() >= 0.0 and uni.max() < 1.0, (uni.min(), uni.max())
inside = (uni[:, 0] ** 2 + uni[:, 1] ** 2) <= 1.0
pi = 4.0 * inside.mean()
assert abs(pi - np.pi) < 0.1, f"pi estimate off: {pi}"
print(f"monte-carlo pi={pi:.4f}: ok")

# normal: finite, roughly standard
nrm = np.asarray(jax.jit(
    lambda: jax.random.normal(jax.random.key(0), (20000,)))())
assert np.isfinite(nrm).all(), "normal not finite"
assert abs(nrm.mean()) < 0.05 and abs(nrm.std() - 1.0) < 0.05, (
    nrm.mean(), nrm.std())
print("normal ~N(0,1): ok")

print("RANDOM E2E PASS")
