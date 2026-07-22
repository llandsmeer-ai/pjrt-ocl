"""In-megakernel matmul TFLOP/s: a pure C=A@B under our OpenCL plugin.
PJRT_OCL_MM_KERNEL=0 forces the matmul to stay in the megakernel (vs the
standalone mm2 fast path), so this measures vmo_mma_tile's in-VM ceiling for
§35. Compare 64-tile (default) vs PJRT_OCL_MEGA_BIGTILE=1 (128x128, 188 lanes).
Usage: JAX_PLATFORMS=opencl PJRT_OCL_MM_KERNEL=0 python mmbench.py [N]"""
import sys, time
import numpy as np
import jax, jax.numpy as jnp

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
a = jnp.asarray(np.random.rand(N, N).astype(np.float32))
b = jnp.asarray(np.random.rand(N, N).astype(np.float32))
f = jax.jit(lambda x, y: x @ y)
for _ in range(3):
    jax.block_until_ready(f(a, b))
ts = []
for _ in range(10):
    t = time.perf_counter()
    r = f(a, b)
    jax.block_until_ready(r)
    ts.append(time.perf_counter() - t)
ms = float(np.median(ts)) * 1e3
tf = 2.0 * N * N * N / (ms * 1e-3) / 1e12
print(f"N={N}  {ms:.3f} ms  {tf:.1f} TFLOP/s")
