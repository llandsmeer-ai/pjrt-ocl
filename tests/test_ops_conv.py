"""stablehlo.convolution -> OP_CONV / TILE_CONV (§39).

Direct N-D convolution, canonical NHWC input / HWIO kernel / NHWC output. Both
validators (numpy tensor interpreter + schedule simulator) plus the jax
reference are cross-checked over the common conv variants: unit/strided,
SAME/VALID padding, kernel (rhs) dilation, 1x1, and 1-D conv.
"""
from __future__ import annotations

import jax
import numpy as np
from jax import lax

from oputil import check
from pjrt_ocl import lowering as L


def _rng(s=0):
    return np.random.default_rng(s)


def _conv2d(x, w, strides, pad, rhs_dil=(1, 1)):
    return lax.conv_general_dilated(
        x, w, window_strides=strides, padding=pad, rhs_dilation=rhs_dil,
        dimension_numbers=("NHWC", "HWIO", "NHWC"))


def test_conv2d_same_unit():
    rng = _rng(0)
    x = rng.standard_normal((2, 8, 8, 3)).astype(np.float32)
    w = (rng.standard_normal((3, 3, 3, 5)) * 0.3).astype(np.float32)
    prog = check(lambda x, w: _conv2d(x, w, (1, 1), "SAME"), x, w,
                 rtol=2e-4, atol=2e-4)
    assert L.OP_CONV in [i.op for i in prog.instrs]


def test_conv2d_valid_unit():
    rng = _rng(1)
    x = rng.standard_normal((2, 8, 8, 3)).astype(np.float32)
    w = (rng.standard_normal((3, 3, 3, 5)) * 0.3).astype(np.float32)
    check(lambda x, w: _conv2d(x, w, (1, 1), "VALID"), x, w, rtol=2e-4, atol=2e-4)


def test_conv2d_strided_same():
    rng = _rng(2)
    x = rng.standard_normal((2, 9, 9, 4)).astype(np.float32)
    w = (rng.standard_normal((3, 3, 4, 6)) * 0.3).astype(np.float32)
    check(lambda x, w: _conv2d(x, w, (2, 2), "SAME"), x, w, rtol=2e-4, atol=2e-4)


def test_conv2d_strided_valid():
    rng = _rng(3)
    x = rng.standard_normal((2, 9, 9, 4)).astype(np.float32)
    w = (rng.standard_normal((2, 2, 4, 6)) * 0.3).astype(np.float32)
    check(lambda x, w: _conv2d(x, w, (2, 2), "VALID"), x, w, rtol=2e-4, atol=2e-4)


def test_conv2d_rhs_dilation():
    rng = _rng(4)
    x = rng.standard_normal((2, 10, 10, 3)).astype(np.float32)
    w = (rng.standard_normal((3, 3, 3, 4)) * 0.3).astype(np.float32)
    check(lambda x, w: _conv2d(x, w, (1, 1), "VALID", (2, 2)), x, w,
          rtol=2e-4, atol=2e-4)


def test_conv2d_1x1():
    rng = _rng(5)
    x = rng.standard_normal((2, 7, 7, 5)).astype(np.float32)
    w = (rng.standard_normal((1, 1, 5, 8)) * 0.3).astype(np.float32)
    check(lambda x, w: _conv2d(x, w, (1, 1), "SAME"), x, w, rtol=2e-4, atol=2e-4)


def test_conv1d_same():
    rng = _rng(6)
    x = rng.standard_normal((2, 16, 3)).astype(np.float32)
    w = (rng.standard_normal((3, 3, 7)) * 0.3).astype(np.float32)

    def fn(x, w):
        return lax.conv_general_dilated(
            x, w, window_strides=(1,), padding="SAME",
            dimension_numbers=("NWC", "WIO", "NWC"))

    check(fn, x, w, rtol=2e-4, atol=2e-4)


def test_cnn_conv_relu_pool():
    # The bench_suite `cnn` workload shape: conv2d + relu + global-average pool.
    rng = _rng(7)
    x = rng.standard_normal((8, 32, 32, 3)).astype(np.float32)
    w = (rng.standard_normal((3, 3, 3, 16)) * 0.1).astype(np.float32)

    def fn(x, w):
        y = _conv2d(x, w, (1, 1), "SAME")
        return jax.numpy.maximum(y, 0.0).mean(axis=(1, 2))

    check(fn, x, w, rtol=2e-3, atol=2e-3)
