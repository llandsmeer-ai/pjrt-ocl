"""E2E: matmul through the real plugin under the HOST-DISPATCH engine.

Regression for the matmul-geometry bug (docs/decisions.md §17): LaunchMatmul
picked its launch geometry from host_dispatch() rather than the device type, so a
GPU forced onto the host engine (or any fence-less GPU, which uses host-dispatch
by default) launched the CPU register/packed geometry — silently WRONG on a GPU.
Keyed on is_gpu() now. This body forces PJRT_OCL_ENGINE=host and checks a matmul
with N % 16 == 0 (the size class that steers the packed-path decision). Small
integer values keep it bit-exact regardless of TF32 contraction.
"""
import numpy as np

import jax
import jax.numpy as jnp

(dev,) = jax.devices()
assert dev.platform == "opencl", dev.platform

rng = np.random.default_rng(0)
# N % 16 == 0 to hit the packed-path gate; small ints => exact under f32 & TF32.
M, K, N = 96, 64, 128
a = rng.integers(0, 3, size=(M, K)).astype(np.float32)
b = rng.integers(0, 3, size=(K, N)).astype(np.float32)
got = np.asarray(jax.jit(lambda x, y: x @ y)(jnp.asarray(a), jnp.asarray(b)))
ref = a @ b
# The geometry bug produced errors ~O(result magnitude) (hundreds); exact here.
np.testing.assert_allclose(got, ref, atol=1e-1, rtol=0)
print(f"matmul-host {M}x{K}x{N}: max_abs={np.abs(got - ref).max():.4g} ok")

print("MATMUL HOST E2E PASS")
