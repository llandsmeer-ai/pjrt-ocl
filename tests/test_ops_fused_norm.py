"""Coverage tests: fused segmented softmax / layernorm-core ops (§19).

`_fuse_norm` recognizes the softmax and layernorm reduce->broadcast idioms in
our lowered VM-instr stream and collapses each into ONE fused local-memory op
(OP_SOFTMAX / OP_LAYERNORM). These tests assert (a) the fusion actually fires,
(b) both validators (tensor interp + schedule sim) match jax/numpy on the
re-parsed bytecode, and (c) the gates: PJRT_OCL_FUSE_NORM=0 and seg > 1024 fall
back to the correct decomposed lowering.
"""
import os

import jax.numpy as jnp
import numpy as np

from oputil import check, to_artifact
from pjrt_ocl import lowering as L

RNG = np.random.default_rng(19)


def farr(*shape, scale=1.0):
    return jnp.asarray((RNG.standard_normal(shape) * scale).astype(np.float32))


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = jnp.exp(x)
    return e / e.sum(-1, keepdims=True)


def _layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) * (var + eps) ** -0.5 * g + b


def _lowered_ops(f, *args):
    prog = L.lower_artifact(to_artifact(f, *args))
    return [i.op for i in prog.instrs[:prog.main_len] if i.op != L.OP_NOP]


# --- fusion fires -----------------------------------------------------------

def test_softmax_fuses():
    ops = _lowered_ops(_softmax, farr(4, 8, 16, 16))
    assert L.OP_SOFTMAX in ops
    assert L.OP_REDUCE_SEG not in ops       # both reduces collapsed away


def test_layernorm_fuses():
    x, g, b = farr(4, 16, 32), farr(32), farr(32)
    ops = _lowered_ops(_layernorm, x, g, b)
    assert L.OP_LAYERNORM in ops
    assert L.OP_REDUCE_SEG not in ops       # mean + var reduces collapsed away
    # the trailing per-channel affine (*g + b) stays as separate EW ops
    assert ops.count(L.OP_MUL_F32) >= 1 and L.OP_ADD_F32 in ops


# --- correctness on both validators (fused path) ----------------------------

def test_softmax_2d():
    check(_softmax, farr(6, 20), atol=1e-5)


def test_softmax_4d_attention_shape():
    check(_softmax, farr(2, 4, 16, 16), atol=1e-5)


def test_softmax_large_scale():
    # large magnitudes exercise the max-subtraction for numerical stability
    check(_softmax, farr(8, 64, scale=40.0), atol=1e-5)


def test_layernorm_3d():
    x, g, b = farr(4, 16, 32), farr(32), farr(32)
    check(lambda z, gg, bb: _layernorm(z, gg, bb), x, g, b, atol=2e-4)


def test_layernorm_2d():
    x, g, b = farr(8, 64), farr(64), farr(64)
    check(lambda z, gg, bb: _layernorm(z, gg, bb), x, g, b, atol=2e-4)


def test_layernorm_wide_segment():
    x, g, b = farr(4, 512), farr(512), farr(512)
    check(lambda z, gg, bb: _layernorm(z, gg, bb), x, g, b, atol=2e-4)


# --- gates: disabled + oversized segment fall back to decomposed lowering ----

def test_fuse_norm_disabled_env():
    os.environ["PJRT_OCL_FUSE_NORM"] = "0"
    try:
        ops = _lowered_ops(_softmax, farr(4, 8, 16, 16))
        assert L.OP_SOFTMAX not in ops          # not fused
        assert L.OP_REDUCE_SEG in ops           # decomposed path intact
        check(_softmax, farr(6, 20), atol=1e-5)  # and still correct
    finally:
        del os.environ["PJRT_OCL_FUSE_NORM"]


def test_softmax_oversized_segment_not_fused():
    # seg = 2048 > _NORM_SEG_MAX (1024): must fall back to the decomposed reduce
    ops = _lowered_ops(_softmax, farr(2, 2048))
    assert L.OP_SOFTMAX not in ops
    assert L.OP_REDUCE_SEG in ops
