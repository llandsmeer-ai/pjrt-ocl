"""E2E: a SHARDED jax program (Shardy 'sdy' sharding hints in the VHLO artifact)
lowers and runs through the real plugin. Runs in a fresh process (jax backend is
process-global). Proves the sdy-dialect infra (deserialize registers the dialect;
value-carrying sdy ops collapse to identity on our single device) — the fix that
unblocks any sharded program, brax/MJX included. See docs/decisions.md.
"""
import numpy as np

import jax
import jax.numpy as jnp  # noqa: F401  (import parity with the other e2e bodies)
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

(dev,) = jax.devices()
assert dev.platform == "opencl", dev.platform

mesh = Mesh(np.array([dev]).reshape(1), ("x",))
a = np.arange(8, dtype=np.float32)


def f(x):
    # with_sharding_constraint emits sdy.mesh + sdy.sharding_constraint into the
    # portable artifact; both are identity on one device.
    x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("x")))
    return x * 2.0 + 1.0


r = np.asarray(jax.jit(f)(a))
np.testing.assert_array_equal(r, a * 2.0 + 1.0)
print("sharding_constraint: ok")

print("SDY E2E PASS")
