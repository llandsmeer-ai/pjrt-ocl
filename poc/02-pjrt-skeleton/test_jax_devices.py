"""Explicit-registration smoke test (no discovery involved).

Usage: JAX_PLATFORMS=opencl /home/ubuntu/project/.venv/bin/python test_jax_devices.py
"""

import os
import sys

_SO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "build", "libpjrt_ocl_skeleton.so")
)

from jax._src import xla_bridge as xb

xb.register_plugin("opencl", priority=500, library_path=_SO, options=None)

import jax

devices = jax.devices()
print("jax.devices() =", devices)
d = devices[0]
print("platform:", d.platform)
print("device_kind:", d.device_kind)
print("repr:", repr(d))
sys.exit(0 if d.platform == "opencl" else 1)
