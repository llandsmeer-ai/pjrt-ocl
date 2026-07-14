"""End-to-end: jax.jit through the real plugin (libpjrt_ocl.so) on OpenCL.

Runs the assertions (tests/_e2e_body.py) in a fresh subprocess because jax's
backend choice is process-global and must not leak into CPU-backend tests.
Uses integer-valued f32 so results are bit-exact regardless of FMA contraction
(docs/decisions.md 5b). Select the backend with PJRT_OCL_DEVICE.
"""
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
PLUGIN = REPO / "pjrt_plugin" / "build" / "libpjrt_ocl.so"
BODY = pathlib.Path(__file__).parent / "_e2e_body.py"


@pytest.mark.skipif(not PLUGIN.exists(), reason="libpjrt_ocl.so not built")
def test_e2e_subprocess():
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "opencl"
    # Keep OpenCL compiler caches off the (full) root overlay on the dev box.
    env.setdefault("POCL_CACHE_DIR", str(REPO / "third_party" / "pocl-cache"))
    proc = subprocess.run([sys.executable, str(BODY)], capture_output=True,
                          text=True, env=env, timeout=120)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "E2E PASS" in proc.stdout
