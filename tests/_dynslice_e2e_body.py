"""dynamic_slice-with-runtime-offset E2E, run in a fresh process by
test_e2e.py. Regression for docs/decisions.md §20: the start scalars' aux
byte offsets must be patched at LOAD time — an offset passed as a program
INPUT lives in an I/O port, and arena offsets recorded at lowering time are
invalidated by the arena-reuse pass. The numpy validators address the scalars
by buffer id, so ONLY a real-plugin run catches this (it used to silently
slice at offset 0, and segfault PoCL when chained)."""
import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

(dev,) = jax.devices()
assert dev.platform == "opencl", dev.platform

n, half = 4096, 2048
a = np.arange(n, dtype=np.float32)

# offset arrives as a program input (an I/O port in the loaded program)
f = jax.jit(lambda x, k: lax.dynamic_slice(x, (k,), (half,)))
for start in (0, 7, n // 4, n):  # n clamps to n - half
    got = np.asarray(f(jnp.asarray(a), jnp.asarray(np.int32(start))))
    lo = min(max(start, 0), n - half)
    np.testing.assert_array_equal(got, a[lo:lo + half])
print("input offset: ok")

# offset chained through optimization_barrier aliases (same buffer id each
# link); data-dependent so the repeated slices cannot be DCE'd
def chain(x, k):
    z = lax.optimization_barrier(jnp.float32(0))
    y = lax.dynamic_slice(x, (k,), (half,))
    for _ in range(7):
        k = k + (y[0] * z).astype(jnp.int32)
        y = lax.dynamic_slice(x, (k,), (half,))
    return y

got = np.asarray(jax.jit(chain)(jnp.asarray(a), jnp.asarray(np.int32(n // 4))))
np.testing.assert_array_equal(got, a[n // 4:n // 4 + half])
print("barrier-chained: ok")

# dynamic_update_slice at a runtime input offset (same aux patching path)
g = jax.jit(lambda x, u, k: lax.dynamic_update_slice(x, u, (k,)))
u = -np.arange(half, dtype=np.float32) - 1
got = np.asarray(g(jnp.asarray(a), jnp.asarray(u), jnp.asarray(np.int32(13))))
want = a.copy()
want[13:13 + half] = u
np.testing.assert_array_equal(got, want)
print("update input offset: ok")

print("DYNSLICE E2E PASS")
