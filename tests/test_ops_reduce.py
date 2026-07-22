"""Coverage tests: full reductions (sum/max/min/prod) via REDUCE_PART/COMB.

Only FULL reductions (all axes -> scalar) are supported by the flat two-phase
model; partial-axis reductions are rejected at lowering time (asserted below).
All values are integer-valued f32 so sum/prod are bit-exact under jit (no FMA /
reassociation error); prod uses small arrays to stay well under 2^24.
"""
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from oputil import check, check_typed
from pjrt_ocl import scheduler, vmreader
from pjrt_ocl.lowering import LoweringError

RNG = np.random.default_rng(11)


def arr(*shape, lo=0, hi=8):
    return jnp.asarray(RNG.integers(lo, hi, shape).astype(np.float32))


def iarr(*shape, lo=0, hi=8):
    return jnp.asarray(RNG.integers(lo, hi, shape).astype(np.int32))


def uarr(*shape, lo=0, hi=8):
    return jnp.asarray(RNG.integers(lo, hi, shape).astype(np.uint32))


def barr(*shape):
    return jnp.asarray(RNG.integers(0, 2, shape).astype(bool))


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


# --- unsigned (u32) full / suffix / strided reductions ----------------------
# §41: unsigned max/min must NOT sign-flip; the operand's ui32 type is recovered
# in lowering (signless i32 collapses otherwise) and routed to DT_U32 so the
# kernel uses UNSIGNED compare + 0 / UINT_MAX identities. These validator tests
# stay in the int32 range so the signed-view interpreter agrees; full-range u32
# (values > INT_MAX) is proven on-device by tests/_reduce_int_e2e_body.py.
@pytest.mark.parametrize("shape", [(6,), (3, 4)])
def test_sum_full_u32(shape):
    check_typed(lambda x: jnp.sum(x), uarr(*shape, lo=0, hi=20))


@pytest.mark.parametrize("shape", [(6,), (3, 4)])
def test_max_full_u32(shape):
    check_typed(lambda x: jnp.max(x), uarr(*shape, lo=0, hi=1000))


@pytest.mark.parametrize("shape,axis", [((3, 4), 1), ((3, 4), 0)])
def test_u32_axis_max_min(shape, axis):
    check_typed(lambda x: jnp.max(x, axis), uarr(*shape, lo=0, hi=1000))
    check_typed(lambda x: jnp.min(x, axis), uarr(*shape, lo=0, hi=1000))


# --- integer bitwise reducers (and / or / xor) over i32 ---------------------
# From stablehlo.and/or/xor reducer bodies (jp.all / jp.any lower to these over
# bool; explicit lax.reduce for ints). §41: these had NO kernel dispatch.
def _band(x, axis):
    return lax.reduce(x, np.array(-1, x.dtype), lax.bitwise_and, tuple(axis))


def _bor(x, axis):
    return lax.reduce(x, np.array(0, x.dtype), lax.bitwise_or, tuple(axis))


def _bxor(x, axis):
    return lax.reduce(x, np.array(0, x.dtype), lax.bitwise_xor, tuple(axis))


@pytest.mark.parametrize("red", [_band, _bor, _bxor])
def test_int_bitwise_reduce_full(red):
    check_typed(lambda x: red(x, (0, 1)), iarr(3, 5, lo=0, hi=1 << 20))


@pytest.mark.parametrize("red", [_band, _bor, _bxor])
@pytest.mark.parametrize("axis", [1, 0])
def test_int_bitwise_reduce_axis(red, axis):
    # axis=1 -> suffix (RED_SEG); axis=0 -> strided (RED_STRIDED)
    check_typed(lambda x: red(x, (axis,)), iarr(4, 6, lo=0, hi=1 << 20))


# --- bool reductions: jp.all (and) / jp.any (or) ----------------------------
# §41 core blocker: MJX's jp.allclose needs reduce-and over bool. Bool is stored
# 1-byte (uchar 0/1); the reduce kernels used to read it as f32 -> garbage.
@pytest.mark.parametrize("shape", [(6,), (3, 4), (2, 3, 4)])
def test_bool_all_full(shape):
    check_typed(lambda x: jnp.all(x), barr(*shape))


@pytest.mark.parametrize("shape", [(6,), (3, 4), (2, 3, 4)])
def test_bool_any_full(shape):
    check_typed(lambda x: jnp.any(x), barr(*shape))


@pytest.mark.parametrize("axis", [1, 0])
def test_bool_all_any_axis(axis):
    # axis=1 -> suffix RED_SEG; axis=0 -> strided RED_STRIDED
    check_typed(lambda x: jnp.all(x, axis), barr(4, 5))
    check_typed(lambda x: jnp.any(x, axis), barr(4, 5))


def test_bool_allclose_idiom():
    # the exact jp.allclose pattern MJX leans on: all(|a-b| < atol)
    a = arr(8, hi=5)
    b = a
    check_typed(lambda x, y: jnp.all(jnp.abs(x - y) < 1e-3), a, b)


def test_bool_reduce_uses_bool_buffer():
    # the reduce output must carry DT_BOOL (3), not f32 — the kernel dispatch
    # keys on the result dtype byte.
    prog = check_typed(lambda x: jnp.all(x), barr(20))
    assert prog.buffers[prog.outputs[0]].dtype == 3


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


# --- rejection: only NON-CONTIGUOUS axis sets remain out of coverage --------

def _lower(f, *args):
    from oputil import to_artifact
    scheduler.lower_and_schedule(to_artifact(f, *args))


@pytest.mark.parametrize("f", [
    lambda x: jnp.sum(x, axis=(0, 2)),  # non-contiguous {0,2}: needs transpose
    lambda x: jnp.max(x, axis=(0, 2)),
])
def test_noncontiguous_axis_rejected(f):
    # a non-contiguous axis set still needs a permuting transpose first
    with pytest.raises(LoweringError):
        _lower(f, arr(3, 4, 5))


# --- partial-axis (prefix / interior) reduce via OP_REDUCE_STRIDED ----------

@pytest.mark.parametrize("shape,axis", [
    ((3, 4), 0),                         # prefix axis of rank-2 (batchnorm idiom)
    ((5, 6), 0),
    ((2, 3, 4), 0),                      # prefix axis of rank-3
    ((2, 3, 4), 1),                      # interior axis of rank-3 (nbody idiom)
    ((2, 3, 4), (0, 1)),                 # contiguous prefix block, inner=4
    ((4, 5, 2), 1),
])
@pytest.mark.parametrize("red", ["sum", "max", "min"])
def test_partial_axis_strided_matches_numpy(shape, axis, red):
    """Interior / prefix partial-axis reductions now lower to the strided
    partial-axis reduce tile op; both validators must match jax."""
    fn = {"sum": jnp.sum, "max": jnp.max, "min": jnp.min}[red]
    prog = check(lambda x: fn(x, axis=axis), arr(*shape))
    # confirm it actually used the strided tile op (not seg / two-phase)
    assert any(t.tile_op == scheduler.TILE_RED_STRIDED
               for t in prog.schedule.tasks)


def test_batchnorm_mean_axis0():
    # batchnorm's reduce: mean over the BATCH axis (axis 0), keepdims
    check(lambda x: jnp.mean(x, axis=0, keepdims=True), arr(8, 16, hi=4))


@pytest.mark.parametrize("shape,axis", [
    ((3, 4), 1),                         # last axis of rank-2
    ((2, 3, 4), 2),                      # last axis of rank-3
    ((2, 3, 4), (1, 2)),                 # innermost two axes (suffix)
])
@pytest.mark.parametrize("red", ["sum", "max", "min"])
def test_suffix_reduction_matches_numpy(shape, axis, red):
    """Innermost-suffix partial reductions (softmax/layernorm) now lower to the
    segmented-reduce tile op; both validators must match jax."""
    fn = {"sum": jnp.sum, "max": jnp.max, "min": jnp.min}[red]
    check(lambda x: fn(x, axis=axis), arr(*shape))
