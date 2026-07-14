"""E2E assertions, run in a fresh process by test_e2e.py (jax's backend is
process-global, so this must not share a process with CPU-backend tests)."""
import numpy as np

import jax
import jax.numpy as jnp

(dev,) = jax.devices()
assert dev.platform == "opencl", dev.platform

a = np.arange(8, dtype=np.float32)
b = a * 2

f = jax.jit(lambda x, y: (x + y) * x)
np.testing.assert_array_equal(np.asarray(f(jnp.asarray(a), jnp.asarray(b))),
                              (a + b) * a)
print("basic: ok")

g = jax.jit(lambda x, y: (x + y, x * y, x - y))
for r, want in zip(g(jnp.asarray(a), jnp.asarray(b)), (a + b, a * b, a - b)):
    np.testing.assert_array_equal(np.asarray(r), want)
print("multi-output: ok")

x, xe = jnp.asarray(a), a
for _ in range(5):
    x = f(x, jnp.asarray(b))
    xe = (xe + b) * xe
np.testing.assert_array_equal(np.asarray(x), xe)
print("chained: ok")

np.testing.assert_array_equal(np.asarray(jax.jit(lambda t: t)(jnp.asarray(a))), a)
print("identity/aliasing: ok")

m = np.arange(12, dtype=np.float32).reshape(3, 4)
h = jax.jit(lambda t, u: t * u - t)
np.testing.assert_array_equal(np.asarray(h(jnp.asarray(m), jnp.asarray(m))),
                              m * m - m)
print("2d: ok")

print("E2E PASS")
