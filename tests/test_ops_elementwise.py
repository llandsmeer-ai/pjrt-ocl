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
