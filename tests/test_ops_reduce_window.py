"""Coverage tests: reduce_window (pooling) via the windowed-reduction tile op.

Supported: sum/max/min pooling, no dilation, VALID / SAME / explicit
non-negative padding, f32 and i32. Values are small integers (exact under the
f32 accumulation). Rejections (dilation) asserted at the bottom."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from oputil import check, check_typed
from pjrt_ocl import scheduler
from pjrt_ocl.lowering import LoweringError

RNG = np.random.default_rng(29)


def arr(*shape, hi=9):
    return jnp.asarray(RNG.integers(0, hi, shape).astype(np.float32))


def iarr(*shape, lo=-9, hi=9):
    return jnp.asarray(RNG.integers(lo, hi, shape).astype(np.int32))


def _sum(x, win, stride, pad="VALID"):
    return jax.lax.reduce_window(x, 0.0, jax.lax.add, win, stride, pad)


def _max(x, win, stride, pad="VALID"):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, win, stride, pad)


def _min(x, win, stride, pad="VALID"):
    return jax.lax.reduce_window(x, jnp.inf, jax.lax.min, win, stride, pad)


# --- sum pooling ------------------------------------------------------------

def test_sumpool_2x2_stride2():
    check(lambda x: _sum(x, (2, 2), (2, 2)), arr(4, 4))


def test_sumpool_overlap_stride1():
    check(lambda x: _sum(x, (3, 3), (1, 1)), arr(5, 5))


def test_sumpool_1d():
    check(lambda x: _sum(x, (3,), (1,)), arr(8))


def test_sumpool_1d_stride2():
    check(lambda x: _sum(x, (2,), (2,)), arr(10))


def test_sumpool_3d():
    check(lambda x: _sum(x, (2, 2, 2), (1, 1, 1)), arr(3, 4, 3))


# --- max / min pooling ------------------------------------------------------

def test_maxpool_2x2_stride2():
    check(lambda x: _max(x, (2, 2), (2, 2)), arr(4, 4, hi=50))


def test_maxpool_overlap():
    check(lambda x: _max(x, (3, 3), (1, 1)), arr(6, 6, hi=50))


def test_minpool_2x2():
    check(lambda x: _min(x, (2, 2), (2, 2)), arr(4, 4, hi=50))


# --- padding ----------------------------------------------------------------

def test_sumpool_explicit_padding():
    check(lambda x: jax.lax.reduce_window(
        x, 0.0, jax.lax.add, (3, 3), (1, 1), [(1, 1), (1, 1)]), arr(4, 4))


def test_maxpool_same_padding():
    check(lambda x: _max(x, (2, 2), (1, 1), "SAME"), arr(5, 5, hi=50))


def test_sumpool_same_padding():
    check(lambda x: _sum(x, (2, 2), (2, 2), "SAME"), arr(5, 5))


# --- integer pooling --------------------------------------------------------

def test_sumpool_i32():
    check_typed(lambda x: jax.lax.reduce_window(
        x, np.int32(0), jax.lax.add, (2, 2), (2, 2), "VALID"), iarr(4, 4))


def test_maxpool_i32():
    check_typed(lambda x: jax.lax.reduce_window(
        x, np.iinfo(np.int32).min, jax.lax.max, (2, 2), (2, 2), "VALID"),
        iarr(4, 4))


# --- multi-tile (output > TILE_SIZE) ----------------------------------------

def test_sumpool_multitile():
    prog = check(lambda x: _sum(x, (1,), (1,)), arr(20000))
    rw = [t for t in prog.schedule.tasks
          if t.tile_op == scheduler.TILE_RED_WINDOW]
    assert rw and rw[0].n_tiles() > 1


# --- rejections -------------------------------------------------------------

def _lower(f, *args):
    from oputil import to_artifact
    scheduler.lower_and_schedule(to_artifact(f, *args))


def test_window_dilation_rejected():
    with pytest.raises(LoweringError):
        _lower(lambda x: jax.lax.reduce_window(
            x, 0.0, jax.lax.add, (2, 2), (1, 1), "VALID",
            window_dilation=(2, 2)), arr(6, 6))
