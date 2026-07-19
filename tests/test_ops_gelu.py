"""Coverage tests: fused GELU tanh-approx op (§19b/§24).

`_fuse_gelu` recognizes the GPT-2 tanh-approx GELU idiom
`0.5*x*(1+tanh(0.7978845608*(x + 0.044715*x^3)))` in our lowered VM-instr
stream and collapses the ~8-op elementwise chain into ONE dedicated unary op
(OP_GELU) that computes the whole thing per element (one global read + write).
These tests assert (a) the fusion fires on jax.nn.gelu(approximate=True),
(b) it does NOT fire on lookalikes (wrong constants, non-reused arg, the exact
erf variant), (c) both validators (tensor interp + schedule sim) match jax on
the re-parsed bytecode, and (d) the PJRT_OCL_FUSE_GELU=0 gate falls back to the
correct decomposed lowering.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from oputil import check, to_artifact
from pjrt_ocl import lowering as L

RNG = np.random.default_rng(56)


def farr(*shape, scale=1.0):
    return jnp.asarray((RNG.standard_normal(shape) * scale).astype(np.float32))


def _gelu(x):
    return jax.nn.gelu(x, approximate=True)


def _lowered_ops(f, *args):
    prog = L.lower_artifact(to_artifact(f, *args))
    return [i.op for i in prog.instrs[:prog.main_len] if i.op != L.OP_NOP]


# --- fusion fires -----------------------------------------------------------

def test_gelu_fuses():
    ops = _lowered_ops(_gelu, farr(4, 128))
    assert L.OP_GELU in ops
    # the whole tanh-approx chain collapses away
    assert L.OP_TANH_F32 not in ops
    assert L.OP_AFFINE_F32 not in ops
    assert ops.count(L.OP_GELU) == 1


def test_gelu_fuses_various_shapes():
    for shape in [(16,), (4, 8), (2, 3, 5), (4, 128, 256)]:
        ops = _lowered_ops(_gelu, farr(*shape))
        assert L.OP_GELU in ops, f"did not fire on {shape}"


# --- fusion does NOT fire on lookalikes -------------------------------------

def test_no_fire_wrong_cubic_const():
    # 0.05 instead of 0.044715 -> gate rejects; decomposed chain kept.
    def f(x):
        inner = 0.7978845608 * (x + 0.05 * x ** 3)
        return 0.5 * x * (1.0 + jnp.tanh(inner))
    ops = _lowered_ops(f, farr(4, 32))
    assert L.OP_GELU not in ops
    assert L.OP_TANH_F32 in ops


def test_no_fire_wrong_sqrt_const():
    # wrong sqrt(2/pi) scale -> gate rejects.
    def f(x):
        inner = 0.8 * (x + 0.044715 * x ** 3)
        return 0.5 * x * (1.0 + jnp.tanh(inner))
    ops = _lowered_ops(f, farr(4, 32))
    assert L.OP_GELU not in ops


def test_no_fire_wrong_half_const():
    # final scale 0.6 instead of 0.5 -> gate rejects.
    def f(x):
        inner = 0.7978845608 * (x + 0.044715 * x ** 3)
        return 0.6 * x * (1.0 + jnp.tanh(inner))
    ops = _lowered_ops(f, farr(4, 32))
    assert L.OP_GELU not in ops


def test_no_fire_non_reused_arg():
    # the final multiply and the cube use DIFFERENT tensors: not a real gelu of
    # a single arg, so the reused-X gate must reject it.
    def f(x, y):
        inner = 0.7978845608 * (x + 0.044715 * x ** 3)
        return 0.5 * y * (1.0 + jnp.tanh(inner))
    ops = _lowered_ops(f, farr(4, 32), farr(4, 32))
    assert L.OP_GELU not in ops


# NOTE: the exact (erf) variant `jax.nn.gelu(approximate=False)` lowers to a
# chlo.erf chain that this jaxlib's VHLO serializer cannot even emit (fails in
# to_artifact, before our lowering) — so there is nothing for the recognizer to
# match. The recognizer is tanh-idiom-only by construction; the erf variant is a
# documented follow-up (§19b). No test here because the artifact won't serialize.


# --- correctness on both validators (fused path) ----------------------------

def test_gelu_1d():
    check(_gelu, farr(64), atol=1e-5)


def test_gelu_2d():
    check(_gelu, farr(6, 40), atol=1e-5)


def test_gelu_ffn_shape():
    # (4, 128, 2048)-style FFN activation, smaller for test speed.
    check(_gelu, farr(4, 32, 128), atol=1e-5)


def test_gelu_large_magnitude():
    # saturating tanh region (|x| large): tanh -> +-1, gelu -> x or 0.
    check(_gelu, farr(8, 64, scale=8.0), atol=1e-5)


def test_gelu_negative():
    check(_gelu, -jnp.abs(farr(4, 50)), atol=1e-5)


# --- the FUSE_GELU=0 revert lever -------------------------------------------

def test_fuse_gelu_off_falls_back():
    os.environ["PJRT_OCL_FUSE_GELU"] = "0"
    try:
        ops = _lowered_ops(_gelu, farr(4, 32))
        assert L.OP_GELU not in ops
        assert L.OP_TANH_F32 in ops       # decomposed chain intact
        check(_gelu, farr(4, 32), atol=1e-5)
    finally:
        del os.environ["PJRT_OCL_FUSE_GELU"]


def test_fused_matches_decomposed():
    # fused OP_GELU and the decomposed chain must agree (both vs jax) — the
    # recognizer must not change results.
    x = farr(4, 128)
    os.environ["PJRT_OCL_FUSE_GELU"] = "0"
    try:
        check(_gelu, x, atol=1e-5)        # decomposed
    finally:
        del os.environ["PJRT_OCL_FUSE_GELU"]
    check(_gelu, x, atol=1e-5)            # fused (default)
