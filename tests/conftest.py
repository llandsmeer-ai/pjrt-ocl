"""Force JAX onto the CPU backend for the whole test session BEFORE jax is
imported anywhere. oputil.check() uses jax as the *oracle* — it must not run on
our own (in-development) OpenCL plugin, or the comparison would be circular and,
worse, fail on ops the plugin can't lower yet. Previously this relied on
test_lowering.py setting the env at import time, which broke when running an
op-family test file on its own. Setting it here (conftest is imported first)
makes every invocation robust.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
