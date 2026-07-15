"""Coverage tests: bitcast_convert (bit reinterpret, same element size).

bitcast_convert relabels the raw bytes of each element as another same-size
dtype WITHOUT numeric conversion (f32<->i32<->u32). These go through
oputil.check_typed (dtype-preserving) since the value dtype changes.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from oputil import check_typed

RNG = np.random.default_rng(7)


def _f32(*shape):
    return jnp.asarray(RNG.uniform(-10.0, 10.0, shape).astype(np.float32))


# NOTE: our type system is signless — i32 and u32 both map to DT_I32 (same
# 4-byte storage, same bits), so a bitcast whose *result* is uint32 comes back
# labelled int32 (values bit-identical). The u32 path is the i32 path; we test
# it via the i32 label to avoid a spurious dtype-label mismatch.

def test_bitcast_f32_to_i32():
    check_typed(lambda a: jax.lax.bitcast_convert_type(a, jnp.int32), _f32(16))


def test_bitcast_i32_to_f32():
    xi = _f32(16).view(jnp.int32)   # valid finite-float bit patterns
    check_typed(lambda a: jax.lax.bitcast_convert_type(a, jnp.float32), xi)


def test_bitcast_roundtrip_f32():
    # bitcast to i32 then back to f32 is the identity on the bits
    check_typed(
        lambda a: jax.lax.bitcast_convert_type(
            jax.lax.bitcast_convert_type(a, jnp.int32), jnp.float32),
        _f32(16))


def test_bitcast_multi_tile():
    # > TILE_SIZE (16384) elements => more than one EW tile across lanes
    check_typed(lambda a: jax.lax.bitcast_convert_type(a, jnp.int32),
                _f32(20000))


def test_bitcast_2d():
    check_typed(lambda a: jax.lax.bitcast_convert_type(a, jnp.int32),
                _f32(8, 9))
