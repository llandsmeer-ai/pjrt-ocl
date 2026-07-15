"""E2E while-loop assertions through the real plugin, run in a fresh process
(jax backend is process-global). Integer-valued f32 so results are bit-exact.
Compares the opencl backend against numpy references."""
import numpy as np
import jax
import jax.numpy as jnp

(dev,) = jax.devices()
assert dev.platform == "opencl", dev.platform


def run(f, *args):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), jax.jit(f)(*args))


# 1. scalar mixed carry: counter (i32) + value (f32)
def while_double(x):
    return jax.lax.while_loop(lambda c: c[0] < 10,
                              lambda c: (c[0] + 1, c[1] * 2.0),
                              (jnp.int32(0), x))
i, v = run(while_double, jnp.float32(1.0))
assert int(i) == 10 and float(v) == 1024.0, (i, v)
print("while scalar mixed carry: ok", int(i), float(v))

# 2. fori_loop scalar accumulate
def fori_add(x):
    return jax.lax.fori_loop(0, 5, lambda k, a: a + 3.0, x)
r = run(fori_add, jnp.float32(2.0))
assert float(r) == 17.0, r
print("fori_loop scalar: ok", float(r))

# 3. vector carry (multi-tile) elementwise
vin = np.arange(40000, dtype=np.float32)
def fori_vec(v):
    return jax.lax.fori_loop(0, 3, lambda k, a: a + 1.0, v)
rv = run(fori_vec, jnp.asarray(vin))
np.testing.assert_array_equal(rv, vin + 3.0)
print("vector multi-tile carry: ok")

# 4. body with two dataflow levels
def while_multilevel(x):
    def body(c):
        k, a = c
        b = a * 2.0
        d = b + a
        return (k + 1, d)
    return jax.lax.while_loop(lambda c: c[0] < 3, body, (jnp.int32(0), x))
_, r4 = run(while_multilevel, jnp.float32(1.0))
assert float(r4) == 27.0, r4   # a *= 3 each iter: 1->3->9->27
print("body multi-level: ok", float(r4))

# 5. zero iterations
def while_zero(x):
    return jax.lax.while_loop(lambda c: c[0] < 0,
                              lambda c: (c[0] + 1, c[1] * 2.0),
                              (jnp.int32(0), x))
zi, zv = run(while_zero, jnp.float32(7.0))
assert int(zi) == 0 and float(zv) == 7.0, (zi, zv)
print("zero iterations: ok")

# 6. nested while
def nested(x):
    def body(c):
        k, s = c
        _, s2 = jax.lax.while_loop(lambda d: d[0] < 3,
                                   lambda d: (d[0] + 1, d[1] + 1.0),
                                   (jnp.int32(0), s))
        return (k + 1, s2)
    return jax.lax.while_loop(lambda c: c[0] < 4, body, (jnp.int32(0), x))
_, rn = run(nested, jnp.float32(0.0))
assert float(rn) == 12.0, rn
print("nested while: ok", float(rn))

# 7. while feeding a downstream op
def while_then(x):
    _, y = jax.lax.while_loop(lambda c: c[0] < 5,
                              lambda c: (c[0] + 1, c[1] + 2.0),
                              (jnp.int32(0), x))
    return y * y
rt = run(while_then, jnp.float32(1.0))
assert float(rt) == 121.0, rt
print("while then elementwise: ok", float(rt))

print("WHILE E2E PASS")
