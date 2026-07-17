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

# 8. scan with stacked outputs — the in-place dynamic_update_slice-into-carry
# path (ys is a program OUTPUT bound as an I/O port; only a real-plugin run
# exercises that binding). Multi-tile rows so the scatter spans lanes.
def scan_stack(c, xs):
    def step(c, xt):
        c = c + xt
        return c, c * 2.0
    return jax.lax.scan(step, c, xs)
n, T = 5000, 6
c0 = np.zeros(n, np.float32)
xs8 = np.arange(n * T, dtype=np.float32).reshape(T, n)
cf, ys = run(scan_stack, jnp.asarray(c0), jnp.asarray(xs8))
c_ref = c0.copy()
ys_ref = np.empty_like(xs8)
for t in range(T):
    c_ref = c_ref + xs8[t]
    ys_ref[t] = c_ref * 2.0
np.testing.assert_array_equal(cf, c_ref)
np.testing.assert_array_equal(ys, ys_ref)
print("scan stacked outputs (in-place DUS): ok")

# 9. DUS carry also read in the body — the in-place fold must bail and the
# two-phase copy path must still be correct on device.
def dus_read(x):
    def body(st):
        i, ys, acc = st
        upd = jnp.full((1, 8), 2.0, jnp.float32)
        ys2 = jax.lax.dynamic_update_slice(ys, upd, (i, jnp.int32(0)))
        return i + 1, ys2, acc + jnp.sum(ys)
    st = (jnp.int32(0), jnp.zeros((5, 8), jnp.float32), x)
    return jax.lax.while_loop(lambda st: st[0] < 5, body, st)[1:]
ys9, acc9 = run(dus_read, jnp.float32(0.0))
np.testing.assert_array_equal(ys9, np.full((5, 8), 2.0, np.float32))
assert float(acc9) == 2.0 * 8 * (0 + 1 + 2 + 3 + 4), acc9
print("DUS carry-also-read (copy path): ok")

print("WHILE E2E PASS")
