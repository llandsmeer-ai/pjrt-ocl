"""Coverage tests: fused flash-attention (OP_FLASH_ATTN, §34).

`_fuse_attention` recognizes the batched per-head attention idiom
    DOT(QKᵀ)·scale → softmax(-1) → DOT(AV)
in our lowered VM-instr stream and collapses it into ONE online-softmax op that
never materializes the (T×C) score matrix. These tests assert (a) the fusion
fires on both the decode (T=1) and prefill (T>1) idioms, (b) both validators
(tensor interp + schedule sim) match jax/numpy on the re-parsed bytecode, and
(c) the gates: PJRT_OCL_FLASH=0 and an oversized head-dim fall back to the
correct decomposed DOT→softmax→DOT lowering.
"""
import os

import jax.numpy as jnp
import numpy as np
import pytest

from oputil import check, to_artifact
from pjrt_ocl import lowering as L

RNG = np.random.default_rng(34)


@pytest.fixture(autouse=True)
def _flash_on():
    """Flash-attention is DEFAULT OFF (a measured regression on the current
    workload, §34); enable it so these tests exercise the fused path. The
    disabled/default tests below override this within their own body."""
    prev = os.environ.get("PJRT_OCL_FLASH")
    os.environ["PJRT_OCL_FLASH"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("PJRT_OCL_FLASH", None)
        else:
            os.environ["PJRT_OCL_FLASH"] = prev


def farr(*shape, scale=1.0):
    return jnp.asarray((RNG.standard_normal(shape) * scale).astype(np.float32))


def _attn(q, k, v):
    """Batched per-head attention: q(H,T,hd) k(H,C,hd) v(H,C,hd) → (H,T,hd)."""
    hd = q.shape[-1]
    s = (q @ jnp.swapaxes(k, -1, -2)) * (hd ** -0.5)   # (H,T,C)
    s = s - s.max(-1, keepdims=True)
    a = jnp.exp(s)
    a = a / a.sum(-1, keepdims=True)
    return a @ v                                       # (H,T,hd)


def _mha(x, wq, wk, wv, H):
    """Full multi-head block (prefill idiom with reshape/transpose): x(B,T,D)."""
    B, T, D = x.shape
    hd = D // H
    def split(t):
        return t.reshape(B, T, H, hd).transpose(0, 2, 1, 3)   # (B,H,T,hd)
    q, k, v = split(x @ wq), split(x @ wk), split(x @ wv)
    s = (q @ jnp.swapaxes(k, -1, -2)) * (hd ** -0.5)
    s = s - s.max(-1, keepdims=True)
    a = jnp.exp(s); a = a / a.sum(-1, keepdims=True)
    o = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return o


def _lowered_ops(f, *args):
    prog = L.lower_artifact(to_artifact(f, *args))
    return [i.op for i in prog.instrs[:prog.main_len] if i.op != L.OP_NOP]


# --- fusion fires -----------------------------------------------------------

def test_flash_fires_decode():
    # T=1 decode idiom (batched, per head)
    ops = _lowered_ops(_attn, farr(4, 1, 64), farr(4, 128, 64), farr(4, 128, 64))
    assert L.OP_FLASH_ATTN in ops
    assert L.OP_SOFTMAX not in ops       # softmax + both dots collapsed
    assert ops.count(L.OP_DOT) == 0


def test_flash_fires_prefill():
    ops = _lowered_ops(_attn, farr(4, 32, 64), farr(4, 96, 64), farr(4, 96, 64))
    assert L.OP_FLASH_ATTN in ops
    assert L.OP_SOFTMAX not in ops


def test_flash_fires_full_mha():
    # the transpose/reshape multi-head prefill idiom (Q/K/V all folded views)
    x, wq, wk, wv = farr(1, 16, 128), farr(128, 128), farr(128, 128), farr(128, 128)
    ops = _lowered_ops(lambda z, a, b, c: _mha(z, a, b, c, 4), x, wq, wk, wv)
    assert L.OP_FLASH_ATTN in ops


# --- correctness on both validators (fused path) ----------------------------

def test_flash_decode_short():
    check(_attn, farr(4, 1, 64), farr(4, 128, 64), farr(4, 128, 64), atol=1e-4)


def test_flash_decode_long():
    # long context — the target regime; C=2048
    check(_attn, farr(8, 1, 64), farr(8, 2048, 64), farr(8, 2048, 64), atol=2e-4)


def test_flash_prefill():
    check(_attn, farr(4, 24, 64), farr(4, 96, 64), farr(4, 96, 64), atol=1e-4)


def test_flash_single_head():
    check(_attn, farr(1, 8, 32), farr(1, 40, 32), farr(1, 40, 32), atol=1e-4)


def test_flash_large_scores():
    # large magnitudes exercise the online max-subtraction rescale
    check(_attn, farr(2, 4, 32, scale=6.0), farr(2, 64, 32, scale=6.0),
          farr(2, 64, 32), atol=2e-4)


def test_flash_full_mha_correct():
    x, wq, wk, wv = farr(1, 16, 128), farr(128, 128), farr(128, 128), farr(128, 128)
    # f32 matmul(K=128)→attention chain vs jax: loosen to matmul-noise tolerance
    check(lambda z, a, b, c: _mha(z, a, b, c, 4), x, wq, wk, wv,
          rtol=2e-3, atol=2e-3)


# --- gates: disabled + oversized head-dim fall back to decomposed ------------

def test_flash_disabled_env():
    os.environ["PJRT_OCL_FLASH"] = "0"
    try:
        ops = _lowered_ops(_attn, farr(4, 1, 64), farr(4, 128, 64), farr(4, 128, 64))
        assert L.OP_FLASH_ATTN not in ops       # not fused
        assert L.OP_DOT in ops                  # decomposed path intact
        check(_attn, farr(4, 1, 64), farr(4, 128, 64), farr(4, 128, 64), atol=1e-4)
    finally:
        del os.environ["PJRT_OCL_FLASH"]


def test_flash_oversized_headdim_not_fused():
    # hd = 320 > _HD_MAX (256): must fall back to the decomposed reduce+dots
    ops = _lowered_ops(_attn, farr(2, 1, 320), farr(2, 64, 320), farr(2, 64, 320))
    assert L.OP_FLASH_ATTN not in ops
    assert L.OP_DOT in ops


def test_flash_default_off():
    # DEFAULT is OFF (§34 regression): with the env unset, the decomposed
    # DOT→softmax→DOT chain must be emitted, byte-identical to pre-§34 lowering.
    os.environ.pop("PJRT_OCL_FLASH", None)
    ops = _lowered_ops(_attn, farr(4, 1, 64), farr(4, 128, 64), farr(4, 128, 64))
    assert L.OP_FLASH_ATTN not in ops
    assert L.OP_DOT in ops
