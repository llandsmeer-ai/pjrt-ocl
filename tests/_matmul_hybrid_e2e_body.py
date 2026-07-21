"""E2E: in-program (hybrid-routed) CPU matmul, plain AND epilogue-fused.

Regression for the CPU in-program matmul hybrid (docs/decisions.md §12a):
LaunchHostDispatch peels a pure-matmul phase out of the megakernel to the
packed mm2p / mm2p_epi kernels. Unlike the pure-matmul FAST PATH (single-task
program, tested by _matmul_host_e2e_body), here the matmul is EMBEDDED in a
larger program (a following relu / bias / second matmul), so it must go through
the phase-routing path and its shared B-panel scratch.

Two shapes exercise both routed kernels:
  * a plain chained matmul  a @ b @ c   -> mm2p   (no epilogue)
  * bias + relu             relu(a @ b + bias)    -> mm2p_epi (p6 epilogue)

M >= 6 and N % 16 == 0 hit the packed geometry; small integers keep it exact.
"""
import numpy as np

import jax
import jax.numpy as jnp

(dev,) = jax.devices()
assert dev.platform == "opencl", dev.platform

rng = np.random.default_rng(0)
M, K, N = 96, 64, 128  # M>=6, N%16==0

# 1) chained matmul (embedded -> routes to mm2p, no epilogue)
a = rng.integers(0, 3, size=(M, K)).astype(np.float32)
b = rng.integers(0, 3, size=(K, N)).astype(np.float32)
c = rng.integers(0, 3, size=(N, N)).astype(np.float32)
got = np.asarray(jax.jit(lambda x, y, z: (x @ y) @ z)(
    jnp.asarray(a), jnp.asarray(b), jnp.asarray(c)))
ref = (a @ b) @ c
np.testing.assert_allclose(got, ref, atol=1e-1, rtol=0)
print(f"chained {M}x{K}x{N}: max_abs={np.abs(got - ref).max():.4g} ok")

# 2) bias + relu (epilogue-fused matmul -> routes to mm2p_epi)
af = rng.standard_normal((M, K)).astype(np.float32)
bf = rng.standard_normal((K, N)).astype(np.float32) * 0.1
bias = rng.standard_normal((N,)).astype(np.float32)
got2 = np.asarray(jax.jit(lambda x, y, bb: jnp.maximum(x @ y + bb, 0.0))(
    jnp.asarray(af), jnp.asarray(bf), jnp.asarray(bias)))
ref2 = np.maximum(af @ bf + bias, 0.0)
# f32-exact accumulation on CPU (no TF32); tight tolerance.
np.testing.assert_allclose(got2, ref2, atol=1e-4, rtol=1e-4)
print(f"bias+relu {M}x{K}x{N}: max_abs={np.abs(got2 - ref2).max():.4g} ok")

print("MATMUL HYBRID E2E PASS")
