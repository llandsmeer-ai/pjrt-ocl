"""Coverage tests: shape ops (broadcast_in_dim, transpose) via GATHER_STRIDED."""
import jax.numpy as jnp
import numpy as np
import pytest

from oputil import check

RNG = np.random.default_rng(7)


def arr(*shape, hi=20):
    return jnp.asarray(RNG.integers(0, hi, shape).astype(np.float32))


def test_broadcast_vector_to_matrix():
    check(lambda x: jnp.broadcast_to(x, (4, 5)), arr(5))


def test_broadcast_scalar_axis():
    # (1, 5) stretched on axis 0
    check(lambda x: jnp.broadcast_to(x, (3, 5)), arr(1, 5))


def test_broadcast_multi_tile():
    check(lambda x: jnp.broadcast_to(x, (100, 100)), arr(100))


def test_broadcast_then_add():
    check(lambda x, y: jnp.broadcast_to(x, (3, 4)) + y, arr(4), arr(3, 4))


def test_transpose_2d():
    check(lambda x: x.T, arr(3, 4))


def test_transpose_3d():
    check(lambda x: jnp.transpose(x, (2, 0, 1)), arr(2, 3, 4))


def test_transpose_involution():
    check(lambda x: x.T + x.T, arr(5, 6))


@pytest.mark.parametrize("perm", [(0, 1, 2), (1, 0, 2), (2, 1, 0), (0, 2, 1)])
def test_transpose_all_perms(perm):
    check(lambda x: jnp.transpose(x, perm), arr(2, 3, 4))


# --- reshape (buffer alias) ---

def test_reshape_2d_to_2d():
    check(lambda x: x.reshape(6, 2), arr(3, 4))


def test_reshape_flatten():
    check(lambda x: x.reshape(-1), arr(2, 3, 4))


def test_reshape_expand():
    check(lambda x: x.reshape(4, 6), arr(24))


def test_reshape_fused():
    check(lambda x, y: (x.reshape(2, 6) + y).reshape(3, 4), arr(3, 4), arr(2, 6))


# --- slice (strided gather) ---

def test_slice_1d_strided():
    check(lambda v: v[1:7:2], arr(8))


def test_slice_2d():
    check(lambda v: v[1:3, 2:5], arr(4, 6))


def test_slice_2d_strided():
    check(lambda v: v[::2, ::3], arr(8, 9))


def test_slice_then_add():
    check(lambda a, b: a[:2, :3] + b[1:3, 2:5], arr(4, 6), arr(4, 6))


# --- reverse (negative-stride gather) ---

def test_reverse_1d():
    check(lambda v: v[::-1], arr(8))


def test_reverse_axis0():
    check(lambda v: v[::-1, :], arr(4, 5))


def test_reverse_then_slice():
    check(lambda v: v[1:5][::-1], arr(8))
