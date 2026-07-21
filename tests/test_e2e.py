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
WHILE_BODY = pathlib.Path(__file__).parent / "_while_e2e_body.py"
MATMUL_HOST_BODY = pathlib.Path(__file__).parent / "_matmul_host_e2e_body.py"
DYNSLICE_BODY = pathlib.Path(__file__).parent / "_dynslice_e2e_body.py"
RANDOM_BODY = pathlib.Path(__file__).parent / "_random_e2e_body.py"


def _run_body(body: pathlib.Path, marker: str, extra_env: dict | None = None
              ) -> None:
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "opencl"
    # Keep OpenCL compiler caches off the (full) root overlay on the dev box.
    env.setdefault("POCL_CACHE_DIR", str(REPO / "third_party" / "pocl-cache"))
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run([sys.executable, str(body)], capture_output=True,
                          text=True, env=env, timeout=120)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert marker in proc.stdout


@pytest.mark.skipif(not PLUGIN.exists(), reason="libpjrt_ocl.so not built")
def test_e2e_subprocess():
    _run_body(BODY, "E2E PASS")


@pytest.mark.skipif(not PLUGIN.exists(), reason="libpjrt_ocl.so not built")
def test_e2e_random_subprocess():
    """jax.random (threefry2x32) end-to-end: the ui64-counter shift/xor/iota
    path must reproduce XLA's RNG bit-exactly (golden captured from the CPU
    backend). Unlocks the whole jax.random / Monte-Carlo workload class."""
    _run_body(RANDOM_BODY, "RANDOM E2E PASS")


@pytest.mark.skipif(not PLUGIN.exists(), reason="libpjrt_ocl.so not built")
def test_e2e_random_host_dispatch():
    """Same, under the host-dispatch engine (portable clFinish-per-phase)."""
    _run_body(RANDOM_BODY, "RANDOM E2E PASS", {"PJRT_OCL_ENGINE": "host"})


@pytest.mark.skipif(not PLUGIN.exists(), reason="libpjrt_ocl.so not built")
def test_e2e_while_subprocess():
    """stablehlo.while end-to-end through the real plugin (multi-lane now that
    the device-scope barrier fixed the cross-lane race)."""
    _run_body(WHILE_BODY, "WHILE E2E PASS")


@pytest.mark.skipif(not PLUGIN.exists(), reason="libpjrt_ocl.so not built")
def test_e2e_host_dispatch():
    """Force the host-dispatch engine (clFinish-per-phase barrier, no in-kernel
    spin-barrier — the CPU/non-GPU default; docs/decisions.md #1). Works on any
    device, so this exercises the host-driven frame walk + segment kernel in CI
    regardless of which OpenCL device is present."""
    _run_body(WHILE_BODY, "WHILE E2E PASS", {"PJRT_OCL_ENGINE": "host"})


@pytest.mark.skipif(not PLUGIN.exists(), reason="libpjrt_ocl.so not built")
def test_e2e_matmul_host_dispatch():
    """Matmul under the host-dispatch engine — regression for the matmul launch
    geometry keying on host_dispatch() instead of is_gpu() (§17): a GPU on
    the host engine used to launch CPU geometry and silently miscompute. On a
    GPU this exercises the fix; on CPU it just confirms the packed path."""
    _run_body(MATMUL_HOST_BODY, "MATMUL HOST E2E PASS", {"PJRT_OCL_ENGINE": "host"})


@pytest.mark.skipif(not PLUGIN.exists(), reason="libpjrt_ocl.so not built")
def test_e2e_dynslice_runtime_offset():
    """dynamic_slice/update with a runtime INPUT offset (§20): the loader must
    patch the aux idx_byteoff words from buffer ids — offsets recorded at
    lowering time are stale (arena reuse) or wrong (I/O ports). Validators
    address by buffer id and cannot catch this; only a real-plugin run can."""
    _run_body(DYNSLICE_BODY, "DYNSLICE E2E PASS")


@pytest.mark.skipif(not PLUGIN.exists(), reason="libpjrt_ocl.so not built")
def test_e2e_dynslice_runtime_offset_host_dispatch():
    _run_body(DYNSLICE_BODY, "DYNSLICE E2E PASS", {"PJRT_OCL_ENGINE": "host"})
