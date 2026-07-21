"""Coverage tests: integer shifts + integer iota — the ops jax.random's
threefry2x32 RNG needs (shift_left, shift_right_logical, xor on the counter).

check_typed compares both python validators (tensor interp + schedule sim)
against jax bit-exactly. Our lowering canonicalises signless 32-bit ints to
DT_I32, so these use int32 inputs (jax x64 is off, so uint64 is unavailable at
the jnp level anyway — the real ui64 counter path lives *inside* threefry and is
validated end-to-end through the plugin in test_e2e.py::test_e2e_random_*).
"""
import jax
import jax.numpy as jnp
import numpy as np

from oputil import check_typed
from pjrt_ocl import lowering as L


# --- integer shifts (int32) -------------------------------------------------

def test_shift_left_i32():
    x = jnp.asarray(np.array([1, 2, 255, 0x123456, 7, -3], dtype=np.int32))
    s = jnp.asarray(np.array([1, 3, 4, 8, 0, 2], dtype=np.int32))
    check_typed(lambda a, b: jax.lax.shift_left(a, b), x, s)


def test_shift_right_logical_zero_fill_i32():
    # a negative (high-bit-set) int shifted right *logically* must zero-fill,
    # not sign-extend — the property threefry's word-mixing relies on.
    x = jnp.asarray(np.array([-1, -8, 0x7FFFFFFF, -2147483648, 16],
                             dtype=np.int32))
    s = jnp.asarray(np.array([1, 2, 4, 1, 3], dtype=np.int32))
    check_typed(lambda a, b: jax.lax.shift_right_logical(a, b), x, s)


def test_shift_right_arithmetic_i32():
    # arithmetic shift sign-extends (fills with the sign bit).
    x = jnp.asarray(np.array([-8, -1, 16, 0x7FFFFFFF, -2147483648],
                             dtype=np.int32))
    s = jnp.asarray(np.array([1, 2, 2, 4, 1], dtype=np.int32))
    check_typed(lambda a, b: jax.lax.shift_right_arithmetic(a, b), x, s)


def test_shift_by_zero_is_identity():
    x = jnp.asarray(np.array([1, -1, 12345, -2147483648], dtype=np.int32))
    z = jnp.asarray(np.zeros(4, dtype=np.int32))
    check_typed(lambda a, b: jax.lax.shift_left(a, b), x, z)
    check_typed(lambda a, b: jax.lax.shift_right_logical(a, b), x, z)


# --- a threefry-shaped rotate (shl | shr_l) ---------------------------------

def test_rotate_left_idiom_i32():
    # (x << r) | (x >>> (32 - r)) — the core threefry mixing step.
    def rotl(x, r):
        rr = jnp.asarray(np.full(x.shape, 32, np.int32)) - r
        return jax.lax.shift_left(x, r) | jax.lax.shift_right_logical(x, rr)
    x = jnp.asarray(np.array([0x12345678, 1, -1, 0x0BCDEF01, 42, 7],
                             dtype=np.int32))
    r = jnp.asarray(np.array([13, 15, 26, 6, 17, 29], dtype=np.int32))
    check_typed(rotl, x, r)


# --- integer iota (the counter generator) -----------------------------------

def test_iota_int32_values_not_float_bits():
    # arange lowers to stablehlo.iota; the result must be integer coordinates
    # [0,1,2,...], NOT their f32 bit-pattern (the bug that broke threefry).
    check_typed(lambda z: jnp.arange(16, dtype=jnp.int32) + z,
                jnp.asarray(np.zeros(16, dtype=np.int32)))


def test_iota_2d_int32():
    check_typed(lambda z: jax.lax.broadcasted_iota(jnp.int32, (3, 4), 1) + z,
                jnp.asarray(np.zeros((3, 4), dtype=np.int32)))


# --- the ops register under the expected stablehlo names --------------------

def test_shift_ops_registered():
    for name in ("stablehlo.shift_left", "stablehlo.shift_right_logical",
                 "stablehlo.shift_right_arithmetic"):
        assert name in L.OP_HANDLERS, f"{name} not registered"
