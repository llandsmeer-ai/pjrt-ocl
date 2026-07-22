"""E2E: stablehlo.case / stablehlo.if through the real plugin (both the
megakernel and host-dispatch engines). N-way switch lowers to flat sibling
OP_IFs; the device reads each branch flag between barrier phases and runs only
the selected arm. Integer-valued f32 so results are bit-exact.
"""
import numpy as np

import jax
import jax.numpy as jnp

(dev,) = jax.devices()
assert dev.platform == "opencl", dev.platform

branches = [lambda x: x * 2.0,
            lambda x: x + 10.0,
            lambda x: -x,
            lambda x: x * x]
a = np.arange(6, dtype=np.float32)

for idx in (0, 1, 2, 3, 7, -1):   # incl. out-of-range (clamps to last branch)
    f = jax.jit(lambda x, i=idx: jax.lax.switch(i, branches, x))
    got = np.asarray(f(a))
    exp = np.asarray(branches[min(max(idx, 0), len(branches) - 1)](a))
    np.testing.assert_array_equal(got, exp), (idx, got, exp)

# runtime index computed on device (not a python constant)
def rt(x):
    i = (x[0] > 0).astype(jnp.int32) + (x[1] > 0).astype(jnp.int32)
    return jax.lax.switch(i, branches, x)

for arr in ([-1.0, -1.0, 2, 3, 4, 5], [1.0, -1.0, 2, 3, 4, 5],
            [1.0, 1.0, 2, 3, 4, 5]):
    v = np.asarray(arr, np.float32)
    got = np.asarray(jax.jit(rt)(v))
    ii = int(v[0] > 0) + int(v[1] > 0)
    np.testing.assert_array_equal(got, np.asarray(branches[ii](v)))

# stablehlo.if via lax.cond
def cnd(p, x):
    return jax.lax.cond(p[0] > 0, lambda z: z * 3.0, lambda z: z - 1.0, x)

for pv in ([1.0], [-1.0]):
    got = np.asarray(jax.jit(cnd)(np.asarray(pv, np.float32), a))
    exp = a * 3.0 if pv[0] > 0 else a - 1.0
    np.testing.assert_array_equal(got, exp)

print("CASE E2E PASS")
