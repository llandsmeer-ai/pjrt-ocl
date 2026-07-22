"""E2E assertions for INTEGER and BOOL reductions through the real plugin.

Run in a fresh process by test_e2e.py on the opencl platform. §41 flagged that
our reduce kernels assumed f32 slots, so reduce-and / reduce-or over bool and
unsigned min/max over u32 produced WRONG results. This body drives every reduce
PATH (full two-phase, suffix RED_SEG, strided/prefix RED_STRIDED, windowed
RED_WINDOW) for i32 / u32 / bool over sum/max/min/and/or and checks each against
numpy. Values are chosen to DISCRIMINATE the bug: the u32 cases use elements
above INT_MAX so a signed max()/min() would pick the wrong element.
"""
import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

(dev,) = jax.devices()
assert dev.platform == "opencl", dev.platform

RNG = np.random.default_rng(41)


def chk(name, got, want):
    got = np.asarray(got)
    want = np.asarray(want)
    # dtype CATEGORY must match (a float result here would be the §41 bug);
    # numpy promotes int sums to int64, so compare values in got's dtype.
    assert got.dtype.kind == want.dtype.kind, \
        f"{name}: dtype kind {got.dtype} vs {want.dtype}"
    np.testing.assert_array_equal(got.reshape(want.shape),
                                  want.astype(got.dtype),
                                  err_msg=f"{name} mismatch")
    print(f"{name}: ok")


def band(x, axis):
    return lax.reduce(x, np.array(-1, x.dtype), lax.bitwise_and, tuple(axis))


def bor(x, axis):
    return lax.reduce(x, np.array(0, x.dtype), lax.bitwise_or, tuple(axis))


# ---- i32 full / suffix / strided : sum, max, min --------------------------
i = RNG.integers(-50, 50, (4, 6)).astype(np.int32)
chk("i32 sum full", jax.jit(lambda x: jnp.sum(x))(jnp.asarray(i)), i.sum())
chk("i32 max full", jax.jit(lambda x: jnp.max(x))(jnp.asarray(i)), i.max())
chk("i32 min full", jax.jit(lambda x: jnp.min(x))(jnp.asarray(i)), i.min())
chk("i32 sum axis-1", jax.jit(lambda x: jnp.sum(x, -1))(jnp.asarray(i)), i.sum(-1))
chk("i32 max axis-1", jax.jit(lambda x: jnp.max(x, -1))(jnp.asarray(i)), i.max(-1))
chk("i32 min axis0", jax.jit(lambda x: jnp.min(x, 0))(jnp.asarray(i)), i.min(0))
chk("i32 sum axis0", jax.jit(lambda x: jnp.sum(x, 0))(jnp.asarray(i)), i.sum(0))

# ---- u32 : UNSIGNED max/min must not sign-flip on values > INT_MAX ---------
BIG = np.uint32(3_000_000_000)          # > 2^31, negative if read signed
u = np.array([[7, BIG, 5, 1],
              [BIG - 1, 2, BIG, 9]], dtype=np.uint32)
chk("u32 max full", jax.jit(lambda x: jnp.max(x))(jnp.asarray(u)), u.max())
chk("u32 min full", jax.jit(lambda x: jnp.min(x))(jnp.asarray(u)), u.min())
chk("u32 max axis-1", jax.jit(lambda x: jnp.max(x, -1))(jnp.asarray(u)), u.max(-1))
chk("u32 min axis-1", jax.jit(lambda x: jnp.min(x, -1))(jnp.asarray(u)), u.min(-1))
chk("u32 max axis0", jax.jit(lambda x: jnp.max(x, 0))(jnp.asarray(u)), u.max(0))
chk("u32 min axis0", jax.jit(lambda x: jnp.min(x, 0))(jnp.asarray(u)), u.min(0))

# ---- integer bitwise and/or reduce (i32) ----------------------------------
m = RNG.integers(0, 1 << 20, (3, 5)).astype(np.int32)
chk("i32 and full", jax.jit(lambda x: band(x, (0, 1)))(jnp.asarray(m)),
    np.bitwise_and.reduce(m, axis=None))
chk("i32 or full", jax.jit(lambda x: bor(x, (0, 1)))(jnp.asarray(m)),
    np.bitwise_or.reduce(m, axis=None))
chk("i32 and axis-1", jax.jit(lambda x: band(x, (1,)))(jnp.asarray(m)),
    np.bitwise_and.reduce(m, axis=1))
chk("i32 or axis0", jax.jit(lambda x: bor(x, (0,)))(jnp.asarray(m)),
    np.bitwise_or.reduce(m, axis=0))

# ---- bool all/any : full / suffix / strided -------------------------------
b = RNG.integers(0, 2, (4, 5)).astype(bool)
b[1, :] = True                          # guarantee a mixed pattern
chk("bool all full", jax.jit(lambda x: jnp.all(x))(jnp.asarray(b)), np.all(b))
chk("bool any full", jax.jit(lambda x: jnp.any(x))(jnp.asarray(b)), np.any(b))
chk("bool all axis-1", jax.jit(lambda x: jnp.all(x, -1))(jnp.asarray(b)), np.all(b, -1))
chk("bool any axis-1", jax.jit(lambda x: jnp.any(x, -1))(jnp.asarray(b)), np.any(b, -1))
chk("bool all axis0", jax.jit(lambda x: jnp.all(x, 0))(jnp.asarray(b)), np.all(b, 0))
chk("bool any axis0", jax.jit(lambda x: jnp.any(x, 0))(jnp.asarray(b)), np.any(b, 0))

# allclose-style: bool all over a comparison mask (the MJX jp.allclose idiom)
p = RNG.standard_normal((6,)).astype(np.float32)
q = p.copy()
q[3] += 1.0
chk("bool allclose-idiom",
    jax.jit(lambda a, c: jnp.all(jnp.abs(a - c) < 1e-3))(jnp.asarray(p), jnp.asarray(q)),
    np.all(np.abs(p - q) < 1e-3))

# ---- windowed reduce (RED_WINDOW): i32 sum-pool + u32 max-pool -------------
wi = RNG.integers(0, 20, (1, 8, 1)).astype(np.int32)
sp = jax.jit(lambda x: lax.reduce_window(
    x, np.int32(0), lax.add, (1, 2, 1), (1, 2, 1), "VALID"))(jnp.asarray(wi))
chk("i32 sum-pool window", sp,
    wi.reshape(1, 4, 2, 1).sum(2))

wu = np.array([[[7], [BIG], [5], [BIG - 1], [2], [BIG], [1], [9]]], dtype=np.uint32)
mp = jax.jit(lambda x: lax.reduce_window(
    x, np.uint32(0), lax.max, (1, 2, 1), (1, 2, 1), "VALID"))(jnp.asarray(wu))
chk("u32 max-pool window", mp,
    wu.reshape(1, 4, 2, 1).max(2))

print("REDUCE INT/BOOL E2E PASS")
