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
