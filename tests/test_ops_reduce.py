"""Coverage tests: full reductions (sum/max/min/prod) via REDUCE_PART/COMB.

Only FULL reductions (all axes -> scalar) are supported by the flat two-phase
model; partial-axis reductions are rejected at lowering time (asserted below).
All values are integer-valued f32 so sum/prod are bit-exact under jit (no FMA /
reassociation error); prod uses small arrays to stay well under 2^24.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from oputil import check, check_typed
from pjrt_ocl import scheduler, vmreader
from pjrt_ocl.lowering import LoweringError

RNG = np.random.default_rng(11)


def arr(*shape, lo=0, hi=8):
    return jnp.asarray(RNG.integers(lo, hi, shape).astype(np.float32))


def iarr(*shape, lo=0, hi=8):
    return jnp.asarray(RNG.integers(lo, hi, shape).astype(np.int32))


# --- full reductions, 1D/2D/3D ---------------------------------------------

@pytest.mark.parametrize("shape", [(6,), (3, 4), (2, 3, 4)])
def test_sum_full(shape):
    check(lambda x: jnp.sum(x), arr(*shape))


@pytest.mark.parametrize("shape", [(6,), (3, 4), (2, 3, 4)])
def test_max_full(shape):
    check(lambda x: jnp.max(x), arr(*shape, hi=50))


@pytest.mark.parametrize("shape", [(6,), (3, 4), (2, 3, 4)])
def test_min_full(shape):
    check(lambda x: jnp.min(x), arr(*shape, lo=-50, hi=50))


@pytest.mark.parametrize("shape", [(5,), (3, 3)])
def test_prod_full(shape):
    # small arrays + small values so the integer product stays exact in f32
    check(lambda x: jnp.prod(x), arr(*shape, lo=1, hi=4))


# --- integer (i32) full reductions ------------------------------------------

@pytest.mark.parametrize("shape", [(6,), (3, 4), (2, 3, 4)])
def test_sum_full_i32(shape):
    check_typed(lambda x: jnp.sum(x), iarr(*shape, lo=-20, hi=20))


@pytest.mark.parametrize("shape", [(6,), (3, 4), (2, 3, 4)])
def test_max_full_i32(shape):
    check_typed(lambda x: jnp.max(x), iarr(*shape, lo=-50, hi=50))


@pytest.mark.parametrize("shape", [(6,), (3, 4), (2, 3, 4)])
def test_min_full_i32(shape):
    check_typed(lambda x: jnp.min(x), iarr(*shape, lo=-50, hi=50))


@pytest.mark.parametrize("shape", [(5,), (3, 3)])
def test_prod_full_i32(shape):
    check_typed(lambda x: jnp.prod(x), iarr(*shape, lo=1, hi=4))


def test_sum_large_multichunk_i32():
    # > TILE_SIZE (16384) elements => several REDUCE_PART tiles across lanes
    prog = check_typed(lambda x: jnp.sum(x), iarr(100000, lo=-2, hi=3))
    parts = [t for t in prog.schedule.tasks
             if t.tile_op == scheduler.TILE_REDUCE_PART]
    assert parts and parts[0].n_tiles() > 1
    # reduce tasks + output buffer must carry i32 (DT_I32 == 1), not f32 default
    assert parts[0].dtype == 1
    assert prog.buffers[prog.outputs[0]].dtype == 1


# --- reduction feeding arithmetic (exercises the barrier/level join) --------

def test_sum_then_add():
    check(lambda x, y: jnp.sum(x) + y, arr(3, 4), arr())


def test_max_then_mul():
    check(lambda x: jnp.max(x) * jnp.asarray(2.0, np.float32), arr(4, 5, hi=30))


def test_two_reductions_combined():
    check(lambda x: jnp.sum(x) - jnp.max(x), arr(10, hi=20))


# --- large array => multi-chunk partial reduce (n_parts > 1, split lanes) ---

def test_sum_large_multichunk():
    # > TILE_SIZE (16384) elements => several REDUCE_PART tiles across lanes
    prog = check(lambda x: jnp.sum(x), arr(100000, hi=2))
    # confirm the partial phase really produced more than one tile
    parts = [t for t in prog.schedule.tasks
             if t.tile_op == scheduler.TILE_REDUCE_PART]
    assert parts and parts[0].n_tiles() > 1


def test_max_large_multichunk():
    check(lambda x: jnp.max(x), arr(60000, hi=1000))


# --- rejection: partial-axis reductions are out of coverage -----------------

def _lower(f, *args):
    from oputil import to_artifact
    scheduler.lower_and_schedule(to_artifact(f, *args))


@pytest.mark.parametrize("f", [
    lambda x: jnp.sum(x, axis=0),
    lambda x: jnp.sum(x, axis=1),
    lambda x: jnp.max(x, axis=0),
])
def test_partial_axis_rejected(f):
    with pytest.raises(LoweringError):
        _lower(f, arr(3, 4))
