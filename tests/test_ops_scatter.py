"""General data-dependent scatter (stablehlo.scatter -> OP_SCATTER_INDEX, §42).

The mirror of the general gather (test_ops_gather.py): each update element's
operand target is read at runtime from a scatter_indices tensor, and combined
into the operand via set / add / max / min. Both validators (numpy tensor
interpreter + schedule simulator) are checked against jax/numpy.

Duplicate-index cases deliberately exercise the atomic accumulate path for add
(order-independent); set uses unique indices (stablehlo leaves duplicate
overwrite order unspecified). Args are jnp arrays so the `.at[]` reference runs
under jax; check_typed executes our VM on np.asarray() of the same buffers.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from oputil import check_typed
from pjrt_ocl import lowering as L


def _rng(s=0):
    return np.random.default_rng(s)


def test_index_set_1d():
    x = jnp.arange(8, dtype=jnp.float32)
    idx = jnp.array([0, 3, 5], dtype=jnp.int32)
    v = jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32)
    prog = check_typed(lambda x, i, v: x.at[i].set(v), x, idx, v)
    assert L.OP_SCATTER_INDEX in [i.op for i in prog.instrs]


def test_index_add_1d_unique():
    x = jnp.arange(8, dtype=jnp.float32)
    idx = jnp.array([1, 4, 7], dtype=jnp.int32)
    v = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    check_typed(lambda x, i, v: x.at[i].add(v), x, idx, v)


def test_index_add_1d_duplicate():
    # duplicate targets: scatter-add MUST accumulate (2 -> 1.0+3.0+7.0).
    x = jnp.zeros(5, dtype=jnp.float32)
    idx = jnp.array([2, 2, 0, 2, 4], dtype=jnp.int32)
    v = jnp.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=jnp.float32)
    check_typed(lambda x, i, v: x.at[i].add(v), x, idx, v)


def test_zeros_index_add_histogram():
    # the canonical jnp.zeros().at[idx].add(x) (histogram / segment-sum idiom).
    rng = _rng(7)
    idx = jnp.asarray(rng.integers(0, 16, size=(200,)).astype(np.int32))
    v = jnp.asarray(rng.standard_normal((200,)).astype(np.float32))
    check_typed(lambda i, v: jnp.zeros((16,), jnp.float32).at[i].add(v),
                idx, v, rtol=1e-4, atol=1e-4)


def test_index_max_1d_duplicate():
    x = jnp.full((4,), -1.0, dtype=jnp.float32)
    idx = jnp.array([0, 0, 1, 3, 1], dtype=jnp.int32)
    v = jnp.array([2.0, 5.0, 1.0, 9.0, 3.0], dtype=jnp.float32)
    check_typed(lambda x, i, v: x.at[i].max(v), x, idx, v)


def test_index_min_1d_duplicate():
    x = jnp.full((4,), 100.0, dtype=jnp.float32)
    idx = jnp.array([0, 0, 1, 3, 1], dtype=jnp.int32)
    v = jnp.array([2.0, 5.0, 1.0, 9.0, 3.0], dtype=jnp.float32)
    check_typed(lambda x, i, v: x.at[i].min(v), x, idx, v)


def test_row_set_2d():
    # N-D window: set whole ROWS of a matrix (window over the trailing axis).
    rng = _rng(2)
    x = jnp.asarray(rng.standard_normal((6, 4)).astype(np.float32))
    idx = jnp.array([0, 5, 2], dtype=jnp.int32)
    v = jnp.asarray(rng.standard_normal((3, 4)).astype(np.float32))
    check_typed(lambda x, i, v: x.at[i].set(v), x, idx, v)


def test_row_add_2d_duplicate():
    # scatter-add of whole rows with a duplicate target row.
    rng = _rng(3)
    x = jnp.zeros((5, 3), dtype=jnp.float32)
    idx = jnp.array([1, 1, 4], dtype=jnp.int32)
    v = jnp.asarray(rng.standard_normal((3, 3)).astype(np.float32))
    check_typed(lambda x, i, v: x.at[i].add(v), x, idx, v, rtol=1e-4, atol=1e-4)


def test_scatter_2d_index_batch_add():
    # 2-D batch of scalar scatter-adds into a matrix at (row, col) pairs.
    rng = _rng(4)
    x = jnp.zeros((10, 10), dtype=jnp.float32)
    r = jnp.asarray(rng.integers(0, 10, size=(20,)).astype(np.int32))
    c = jnp.asarray(rng.integers(0, 10, size=(20,)).astype(np.int32))
    v = jnp.asarray(rng.standard_normal((20,)).astype(np.float32))
    check_typed(lambda x, r, c, v: x.at[r, c].add(v), x, r, c, v, rtol=1e-4,
                atol=1e-4)


def test_scatter_i64_indices_add():
    jax.config.update("jax_enable_x64", True)
    try:
        x = jnp.zeros(12, dtype=jnp.float32)
        idx = jnp.array([0, 3, 3, 11, 3], dtype=jnp.int64)
        v = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
        check_typed(lambda x, i, v: x.at[i].add(v), x, idx, v)
    finally:
        jax.config.update("jax_enable_x64", False)


def test_scatter_int32_add():
    x = jnp.zeros(6, dtype=jnp.int32)
    idx = jnp.array([0, 2, 2, 5], dtype=jnp.int32)
    v = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
    check_typed(lambda x, i, v: x.at[i].add(v), x, idx, v)
