"""Shared helper for op-coverage tests: run a jax fn through our lowering +
scheduler and check BOTH validators (tensor interpreter + schedule simulator)
against the CPU-backend (numpy) result.

Each op-family test file imports `check` and asserts on real jax programs. This
keeps per-family tests in their own files so coverage grows in parallel.
"""
from __future__ import annotations

import jax
import numpy as np
from jaxlib.mlir.dialects import stablehlo

from pjrt_ocl import scheduler, vmreader


def to_artifact(f, *args) -> bytes:
    art = jax.jit(f).lower(*args).compiler_ir("stablehlo")
    with art.context:
        return stablehlo.serialize_portable_artifact(
            art, stablehlo.get_current_version())


def check(f, *args, rtol=1e-5, atol=1e-5):
    """Assert both validators match jax/numpy. Returns the parsed Program."""
    artifact = to_artifact(f, *args)
    prog = vmreader.parse(scheduler.lower_and_schedule(artifact))
    np_args = [np.asarray(a, np.float32) for a in args]
    got_tensor = vmreader.execute(prog, np_args)
    got_sched = vmreader.execute_schedule(prog, np_args)
    exp = f(*args)
    exp = exp if isinstance(exp, (list, tuple)) else (exp,)
    assert len(got_tensor) == len(exp), "output count mismatch"
    for i, (gt, gs, e) in enumerate(zip(got_tensor, got_sched, exp)):
        e = np.asarray(e, np.float32)
        np.testing.assert_allclose(
            gt.reshape(e.shape), e, rtol=rtol, atol=atol,
            err_msg=f"tensor validator mismatch on output {i}")
        np.testing.assert_allclose(
            gs.reshape(e.shape), e, rtol=rtol, atol=atol,
            err_msg=f"schedule simulator mismatch on output {i}")
    return prog
