"""Coverage tests: complex64 (split-pair) + stablehlo.fft (§43).

Complex-typed SSA values are lowered as a PAIR of f32 buffers (real, imag) in
ops/complex_fft.py — no complex arena dtype. These tests drive real jax programs
that produce complex intermediates and reduce back to real outputs (abs / real /
imag), then check BOTH validators (tensor interp + schedule sim) against numpy.

FFT is a direct DFT via a constant twiddle matmul (1-D forward FFT only); its
end-to-end correctness on the real NVIDIA + PoCL devices is exercised by the
bench suite (docs/workload-coverage.md, fft PASS).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from oputil import check

RNG = np.random.default_rng(43)


def farr(*shape, scale=1.0):
    return jnp.asarray((RNG.standard_normal(shape) * scale).astype(np.float32))


# --- complex construction / projection --------------------------------------

def test_complex_real_imag():
    def f(re, im):
        c = jax.lax.complex(re, im)
        return jnp.real(c) + 2.0 * jnp.imag(c)
    check(f, farr(16), farr(16))


def test_complex_abs():
    def f(re, im):
        return jnp.abs(jax.lax.complex(re, im))
    check(f, farr(3, 8), farr(3, 8), atol=1e-4)


# --- complex elementwise algebra --------------------------------------------

def test_complex_add_sub():
    def f(a, b):
        ca = jax.lax.complex(a[0], a[1])
        cb = jax.lax.complex(b[0], b[1])
        c = (ca + cb) - cb
        return jnp.stack([jnp.real(c), jnp.imag(c)])
    check(f, farr(2, 12), farr(2, 12), atol=1e-4)


def test_complex_mul():
    def f(a, b):
        ca = jax.lax.complex(a[0], a[1])
        cb = jax.lax.complex(b[0], b[1])
        c = ca * cb
        return jnp.stack([jnp.real(c), jnp.imag(c)])
    check(f, farr(2, 10), farr(2, 10), atol=1e-4)


def test_complex_negate():
    def f(a):
        c = -jax.lax.complex(a[0], a[1])
        return jnp.stack([jnp.real(c), jnp.imag(c)])
    check(f, farr(2, 7), atol=1e-4)


# --- fft (magnitude spectrum) -----------------------------------------------

@pytest.mark.parametrize("n", [8, 16, 64, 512])
def test_fft_magnitude(n):
    def f(sig):
        return jnp.abs(jnp.fft.fft(sig))
    check(f, farr(n), atol=2e-3, rtol=2e-3)


def test_fft_real_imag():
    def f(sig):
        y = jnp.fft.fft(sig)
        return jnp.stack([jnp.real(y), jnp.imag(y)])
    check(f, farr(32), atol=2e-3, rtol=2e-3)


def test_fft_batched():
    # 1-D FFT over the last axis of a batched input (leading dims flatten to B).
    def f(sig):
        return jnp.abs(jnp.fft.fft(sig, axis=-1))
    check(f, farr(4, 16), atol=2e-3, rtol=2e-3)
