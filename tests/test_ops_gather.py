"""General data-dependent gather (stablehlo.gather -> OP_GATHER_INDEX, §38).

Distinct from the strided-view gather (OP_GATHER_STRIDED) covered elsewhere:
here the operand base offset of each output element is read at runtime from a
start_indices tensor (embedding lookup and friends). Both validators (numpy
tensor interpreter + schedule simulator) are checked against jax/numpy.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from oputil import check_typed
from pjrt_ocl import lowering as L


def _lowered_ops(f, *args):
    from oputil import to_artifact
    prog = L.lower_artifact(to_artifact(f, *args))
    return [i.op for i in prog.instrs]


def _rng(s=0):
    return np.random.default_rng(s)


def test_embedding_lookup_1d():
    rng = _rng(0)
    emb = rng.standard_normal((1000, 64)).astype(np.float32)
    ids = rng.integers(0, 1000, size=(32,)).astype(np.int32)
    prog = check_typed(lambda e, i: e[i], emb, ids)
    assert L.OP_GATHER_INDEX in [i.op for i in prog.instrs]


def test_embedding_softmax_classifier():
    rng = _rng(1)
    emb = (rng.standard_normal((1000, 64)) * 0.05).astype(np.float32)
    ids = rng.integers(0, 1000, size=(32,)).astype(np.int32)
    clsw = (rng.standard_normal((64, 10)) * 0.05).astype(np.float32)

    def fn(e, i, w):
        return jax.nn.softmax(e[i] @ w, axis=-1)

    check_typed(fn, emb, ids, clsw, rtol=2e-3, atol=2e-3)


def test_gather_2d_index_batch():
    rng = _rng(2)
    emb = rng.standard_normal((100, 8)).astype(np.float32)
    ids = rng.integers(0, 100, size=(4, 5)).astype(np.int32)
    check_typed(lambda e, i: e[i], emb, ids)


def test_gather_i64_indices():
    rng = _rng(3)
    emb = rng.standard_normal((100, 8)).astype(np.float32)
    ids = rng.integers(0, 100, size=(7,)).astype(np.int64)
    check_typed(lambda e, i: e[i], emb, ids)


def test_gather_two_component_scalar():
    # (row, col) -> scalar gather: index_vector has two components.
    rng = _rng(4)
    a = rng.standard_normal((20, 20)).astype(np.float32)
    r = rng.integers(0, 20, size=(9,)).astype(np.int32)
    c = rng.integers(0, 20, size=(9,)).astype(np.int32)
    check_typed(lambda m, rr, cc: m[rr, cc], a, r, c)


def test_gather_clip_out_of_bounds():
    # stablehlo.gather clamps starts to [0, dim - slice_size]; jnp 'clip' mode.
    rng = _rng(5)
    emb = rng.standard_normal((100, 8)).astype(np.float32)
    ids = np.array([-1, 0, 3, 150, -50, 99], dtype=np.int32)
    check_typed(lambda e, i: jnp.take(e, i, axis=0, mode="clip"), emb, ids)
