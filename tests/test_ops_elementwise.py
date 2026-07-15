"""Coverage tests: expanded elementwise ops (div/max/min/pow, unary
transcendentals, compare, select) — all route through the TILE_EW tile op
with a subop (pjrt_ocl/ops/elementwise.py)."""
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from oputil import check

RNG = np.random.default_rng(11)


def arr(*shape, lo=0, hi=20):
    """Integer-valued f32 array in [lo, hi) — exact under f32 arithmetic."""
    return jnp.asarray(RNG.integers(lo, hi, shape).astype(np.float32))


def farr(*shape, lo=-5.0, hi=5.0):
    """Fractional f32 array in [lo, hi) — for floor/ceil/sign/tanh domains."""
    return jnp.asarray(RNG.uniform(lo, hi, size=shape).astype(np.float32))


# --- binary ------------------------------------------------------------

def test_divide():
    check(lambda x, y: x / y, arr(20, hi=50), arr(20, lo=1, hi=10))


def test_divide_multi_tile():
    check(lambda x, y: x / y, arr(100, 100, hi=50), arr(100, 100, lo=1, hi=10))


def test_maximum():
    check(lambda x, y: jnp.maximum(x, y), arr(20), arr(20))


def test_minimum():
    check(lambda x, y: jnp.minimum(x, y), arr(20), arr(20))


def test_power():
    # positive base, small exponent: avoids base<0/fractional-exponent domain
    # mismatches between numpy's and XLA's pow() implementations.
    check(lambda x, y: x ** y, arr(20, lo=1, hi=5), arr(20, lo=0, hi=4),
          rtol=1e-4, atol=1e-4)


# --- unary ---------------------------------------------------------------

def test_negate():
    check(lambda x: -x, arr(20))


def test_exponential():
    check(lambda x: jnp.exp(x), arr(20, lo=0, hi=5), rtol=1e-4, atol=1e-4)


def test_log():
    check(lambda x: jnp.log(x), arr(20, lo=1, hi=50), rtol=1e-4, atol=1e-4)


def test_sqrt():
    check(lambda x: jnp.sqrt(x), arr(20, lo=0, hi=50), rtol=1e-4, atol=1e-4)


def test_rsqrt():
    check(lambda x: lax.rsqrt(x), arr(20, lo=1, hi=50), rtol=1e-4, atol=1e-4)


def test_tanh():
    check(lambda x: jnp.tanh(x), farr(20), rtol=1e-4, atol=1e-4)


def test_abs():
    check(lambda x: jnp.abs(x), farr(20, lo=-20.0, hi=20.0))


def test_floor():
    check(lambda x: jnp.floor(x), farr(20))


def test_ceil():
    check(lambda x: jnp.ceil(x), farr(20))


def test_sign():
    check(lambda x: jnp.sign(x), farr(20))


def test_unary_multi_tile():
    check(lambda x: jnp.exp(-jnp.abs(x)), farr(100, 100),
          rtol=1e-4, atol=1e-4)


# --- compare --------------------------------------------------------------

@pytest.mark.parametrize("op", [
    lambda x, y: x == y, lambda x, y: x != y, lambda x, y: x < y,
    lambda x, y: x <= y, lambda x, y: x > y, lambda x, y: x >= y,
], ids=["eq", "ne", "lt", "le", "gt", "ge"])
def test_compare_each_direction(op):
    def f(x, y):
        pred = op(x, y)
        return lax.select(pred, jnp.ones_like(x), jnp.zeros_like(x))
    # small integer range -> guarantees ties, so EQ/NE are exercised too.
    check(f, arr(30, hi=5), arr(30, hi=5))


def test_compare_multi_tile():
    def f(x, y):
        return lax.select(x < y, jnp.ones_like(x), jnp.zeros_like(x))
    check(f, arr(100, 100, hi=5), arr(100, 100, hi=5))


# --- select -----------------------------------------------------------

def test_select_basic():
    check(lambda x, y: lax.select(x < y, x, y), arr(20), arr(20))


def test_select_multi_tile():
    check(lambda x, y: lax.select(x < y, x, y), arr(100, 100), arr(100, 100))


# --- fusion combos ----------------------------------------------------------

def test_fusion_div_then_add():
    check(lambda x, y, z: x / y + z,
          arr(20, hi=50), arr(20, lo=1, hi=10), arr(20))


def test_fusion_exp_of_neg():
    check(lambda x: jnp.exp(-x), arr(20, lo=0, hi=5), rtol=1e-4, atol=1e-4)


def test_fusion_compare_select_chain():
    # abs via compare + select instead of stablehlo.abs
    check(lambda x, y: lax.select(x > y, x - y, y - x), arr(20), arr(20))


def test_fusion_max_then_sqrt():
    check(lambda x, y: jnp.sqrt(jnp.maximum(x, y)), arr(20), arr(20),
          rtol=1e-4, atol=1e-4)


def test_fusion_select_of_transcendentals():
    check(lambda x, y: lax.select(x < y, jnp.exp(x), jnp.log(y)),
          arr(20, lo=0, hi=5), arr(20, lo=1, hi=50), rtol=1e-4, atol=1e-4)


# --- coverage batch: log1p/expm1/cbrt/sin/cos/tan/round/atan2/remainder/
# clamp/is_finite/bitwise (docs/coverage-baseline.md "easy EW ops") ---------
#
# NOTE on scope: oputil.check()/vmreader.execute() force every input arg
# through a float32 round-trip (`np.ascontiguousarray(arg, dtype=np.float32)`
# before writing it into the arena) — a testing-harness limitation, not a
# lowering/kernel one. That's transparent for is_finite (float in, bool out)
# and for int32 and/or/xor/not (4-byte int, same slot width as f32, small
# values survive the f32 round-trip exactly). It breaks for *bool*-typed
# and/or/xor/not: bool is a 1-byte arena slot, so a f32-cast bool input is
# 4x the buffer's byte size and vmreader.execute raises before the kernel
# logic is even exercised. Those are verified against the real plugin on
# hardware instead (see the session report), not via this harness.
#
# NOTE on chlo/composite: lax.asin/acos/sinh/cosh/atanh/asinh/acosh lower to
# chlo.* ops that XLA legalizes to plain stablehlo ops (atan -> atan2) or a
# stablehlo.composite wrapping a private-function decomposition depending on
# the op, decided by JAX/XLA internals outside this lowering pipeline. atan
# happens to fully decompose to atan2 (which this batch adds, so `lax.atan`
# now works end-to-end through the real plugin) — verified on hardware, not
# here, because chlo ops can't round-trip through oputil.check()'s
# `.lower().compiler_ir('stablehlo')` + manual portable-artifact serialize
# path (that path skips whatever chlo-legalization pass full `jax.jit`
# execution runs; `stablehlo.serialize_portable_artifact` then rejects the
# leftover chlo op). asin/acos/sinh/cosh/atanh/asinh/acosh hit
# stablehlo.composite either way, which is unimplemented (a separate,
# already-tracked gap — docs/coverage-baseline.md "composite 96/118" — not
# an elementwise op, out of scope here).

def test_log1p():
    check(lambda x: jnp.log1p(x), farr(20, lo=0.1, hi=5.0),
          rtol=1e-4, atol=1e-4)


def test_expm1():
    check(lambda x: jnp.expm1(x), farr(20, lo=-3.0, hi=3.0),
          rtol=1e-4, atol=1e-4)


def test_cbrt():
    # lax.cbrt -> stablehlo.cbrt directly; jnp `x ** (1/3)` decomposes to
    # power+broadcast instead and doesn't exercise the new op.
    check(lambda x: lax.cbrt(x), farr(20, lo=-20.0, hi=20.0),
          rtol=1e-4, atol=1e-4)


def test_sine():
    check(lambda x: jnp.sin(x), farr(20, lo=-3.0, hi=3.0),
          rtol=1e-4, atol=1e-4)


def test_cosine():
    check(lambda x: jnp.cos(x), farr(20, lo=-3.0, hi=3.0),
          rtol=1e-4, atol=1e-4)


def test_tan():
    check(lambda x: jnp.tan(x), farr(20, lo=-1.0, hi=1.0),
          rtol=1e-4, atol=1e-4)


def test_round_nearest_even():
    # lax.round(..., TO_NEAREST_EVEN) -> stablehlo.round_nearest_even
    # directly; plain jnp.round wraps in a func.call (out of scope — no
    # func.call inlining in this lowering pipeline).
    check(lambda x: lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN),
          farr(20))


def test_round_nearest_afz():
    check(lambda x: lax.round(x, lax.RoundingMethod.AWAY_FROM_ZERO),
          farr(20))


def test_round_multi_tile():
    check(lambda x: lax.round(x, lax.RoundingMethod.AWAY_FROM_ZERO),
          farr(100, 100))


def test_atan2():
    check(lambda x, y: jnp.arctan2(x, y), farr(20), farr(20, lo=1.0, hi=5.0),
          rtol=1e-4, atol=1e-4)


def test_remainder():
    # lax.rem -> stablehlo.remainder (C fmod semantics, sign of the
    # dividend); jnp.remainder wraps in a func.call for python-mod semantics
    # instead (out of scope, same func.call limitation as jnp.round).
    check(lambda x, y: lax.rem(x, y), farr(20, lo=-10.0, hi=10.0),
          farr(20, lo=1.0, hi=5.0), rtol=1e-4, atol=1e-4)


def test_clamp():
    # lax.clamp(min, x, max) -> stablehlo.clamp directly (ternary, no new
    # opcode: lowered as OP_MIN_F32 then OP_MAX_F32).
    check(lambda lo, x, hi: lax.clamp(lo, x, hi),
          jnp.full((20,), -1.0, dtype=jnp.float32), farr(20),
          jnp.full((20,), 1.0, dtype=jnp.float32))


def test_is_finite():
    def f(x):
        pred = jnp.isfinite(x)
        return lax.select(pred, jnp.ones_like(x), jnp.zeros_like(x))
    check(f, farr(20))


# --- bitwise (int32; bool covered on hardware — see NOTE above) -----------

def _iarr(*shape, lo=0, hi=16):
    return jnp.asarray(RNG.integers(lo, hi, shape).astype(np.int32))


def test_and_i32():
    check(lambda x, y: x & y, _iarr(20), _iarr(20))


def test_or_i32():
    check(lambda x, y: x | y, _iarr(20), _iarr(20))


def test_xor_i32():
    check(lambda x, y: x ^ y, _iarr(20), _iarr(20))


def test_not_i32():
    check(lambda x: ~x, _iarr(20))
