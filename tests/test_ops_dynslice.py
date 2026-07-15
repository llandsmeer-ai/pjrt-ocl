"""Coverage tests: dynamic_slice / dynamic_update_slice via runtime-offset
gather/scatter (start indices are traced i32 arguments, so jax emits real
stablehlo.dynamic_slice / dynamic_update_slice rather than folding to a static
slice). check_typed keeps the f32 operand and i32 index dtypes distinct."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from oputil import check_typed
from pjrt_ocl import scheduler
from pjrt_ocl.lowering import LoweringError

RNG = np.random.default_rng(23)


def arr(*shape, hi=20):
    return RNG.integers(0, hi, shape).astype(np.float32)


def i32(v):
    return np.int32(v)


# --- dynamic_slice ----------------------------------------------------------

def test_dynslice_1d():
    check_typed(lambda x, i: jax.lax.dynamic_slice(x, (i,), (3,)),
                arr(8), i32(2))


def test_dynslice_1d_clamp_high():
    # start 10 clamps to 8 - 3 = 5
    check_typed(lambda x, i: jax.lax.dynamic_slice(x, (i,), (3,)),
                arr(8), i32(10))


def test_dynslice_1d_clamp_negative():
    check_typed(lambda x, i: jax.lax.dynamic_slice(x, (i,), (3,)),
                arr(8), i32(-4))


def test_dynslice_2d():
    check_typed(lambda x, i, j: jax.lax.dynamic_slice(x, (i, j), (2, 3)),
                arr(4, 6), i32(1), i32(2))


def test_dynslice_2d_clamp():
    check_typed(lambda x, i, j: jax.lax.dynamic_slice(x, (i, j), (2, 3)),
                arr(4, 6), i32(9), i32(9))


def test_dynslice_3d():
    check_typed(
        lambda x, i, j, k: jax.lax.dynamic_slice(x, (i, j, k), (2, 2, 2)),
        arr(3, 4, 5), i32(1), i32(1), i32(2))


def test_dynslice_then_add():
    check_typed(
        lambda x, y, i: jax.lax.dynamic_slice(x, (i,), (4,)) + y,
        arr(8), arr(4), i32(3))


def test_dynslice_multitile():
    prog = check_typed(
        lambda x, i: jax.lax.dynamic_slice(x, (i,), (20000,)),
        arr(30000), i32(5000))
    dg = [t for t in prog.schedule.tasks
          if t.tile_op == scheduler.TILE_DYN_GATHER]
    assert dg and dg[0].n_tiles() > 1


# --- dynamic_update_slice ---------------------------------------------------

def test_dynupdate_1d():
    check_typed(
        lambda x, u, i: jax.lax.dynamic_update_slice(x, u, (i,)),
        arr(8), arr(3), i32(2))


def test_dynupdate_1d_clamp():
    check_typed(
        lambda x, u, i: jax.lax.dynamic_update_slice(x, u, (i,)),
        arr(8), arr(3), i32(20))


def test_dynupdate_2d():
    check_typed(
        lambda x, u, i, j: jax.lax.dynamic_update_slice(x, u, (i, j)),
        arr(4, 6), arr(2, 3), i32(1), i32(2))


def test_dynupdate_2d_clamp_negative():
    check_typed(
        lambda x, u, i, j: jax.lax.dynamic_update_slice(x, u, (i, j)),
        arr(4, 6), arr(2, 3), i32(-2), i32(-5))


def test_dynupdate_then_mul():
    check_typed(
        lambda x, u, i: jax.lax.dynamic_update_slice(x, u, (i,))
        * np.float32(2.0),
        arr(6), arr(2), i32(3))


# --- i64 indices ------------------------------------------------------------

def test_dynslice_i64_index():
    jax.config.update("jax_enable_x64", True)
    try:
        check_typed(lambda x, i: jax.lax.dynamic_slice(x, (i,), (3,)),
                    arr(8), np.int64(2))
    finally:
        jax.config.update("jax_enable_x64", False)
