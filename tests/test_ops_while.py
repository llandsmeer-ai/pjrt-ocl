"""stablehlo.while coverage: real jax control-flow programs checked against the
CPU backend through BOTH validators (tensor interpreter + schedule simulator).

The schedule simulator (`vmreader._run_control`) mirrors vm2.cl's frame-stack
interpreter, so a green run here means the per-lane WHILE control entries the
scheduler emits (cond/body sub-streams after root_len) execute correctly. The
device (NVIDIA) path is exercised separately (tests/test_e2e is the e2e seam).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxlib.mlir.dialects import stablehlo

from pjrt_ocl import lowering as L, scheduler as S, vmreader as R
from oputil import check, to_artifact


def test_while_scalar_mixed_carry():
    """Counter (i32) + value (f32) carried together; body updates both."""
    def f(x):
        return jax.lax.while_loop(
            lambda c: c[0] < 10, lambda c: (c[0] + 1, c[1] * 2.0),
            (jnp.int32(0), x))
    check(f, np.float32(1.0))          # -> (10, 1024.0)


def test_fori_loop_scalar():
    def f(x):
        return jax.lax.fori_loop(0, 5, lambda i, a: a + 1.0, x)
    check(f, np.float32(3.0))          # -> 8.0


def test_fori_loop_multiply():
    def f(x):
        return jax.lax.fori_loop(0, 6, lambda i, a: a * 2.0, x)
    check(f, np.float32(1.5))          # -> 96.0


def test_while_vector_carry():
    """Loop-carried buffer is an array; body is elementwise over the whole tile."""
    def f(v):
        return jax.lax.fori_loop(0, 4, lambda i, a: a + 1.0, v)
    check(f, np.arange(8, dtype=np.float32))   # each elem += 4


def test_while_multi_tile_vector_carry():
    """Carry spans multiple EW tiles (> TILE_SIZE) so the body task fans out
    across lanes inside the loop."""
    def f(v):
        return jax.lax.fori_loop(0, 3, lambda i, a: a * 1.5, v)
    check(f, np.linspace(-1.0, 1.0, 40000, dtype=np.float32))


def test_while_body_multi_level():
    """Body has a dependent chain (two dataflow levels) => an internal barrier
    inside the body sub-list; loop-carry commit is a further level."""
    def f(x):
        def body(c):
            i, a = c
            b = a * 2.0          # level 0
            d = b + a            # level 1 (depends on b)
            return (i + 1, d)
        return jax.lax.while_loop(lambda c: c[0] < 3, body,
                                  (jnp.int32(0), x))
    check(f, np.float32(1.0))


def test_while_zero_iterations():
    """Cond false at entry => body never runs; results == inits."""
    def f(x):
        return jax.lax.while_loop(lambda c: c[0] < 0, lambda c: (c[0] + 1,
                                  c[1] * 2.0), (jnp.int32(0), x))
    check(f, np.float32(7.0))          # -> (0, 7.0)


def test_nested_while():
    """while inside a while: the inner loop's cond/body sub-streams live beyond
    the outer body (nested-further rule), driven by the frame stack."""
    def f(x):
        def body(c):
            i, s = c
            _, s2 = jax.lax.while_loop(
                lambda d: d[0] < 3, lambda d: (d[0] + 1, d[1] + 1.0),
                (jnp.int32(0), s))
            return (i + 1, s2)
        return jax.lax.while_loop(lambda c: c[0] < 4, body,
                                  (jnp.int32(0), x))
    check(f, np.float32(0.0))          # 4 * 3 = 12.0


def test_while_multilane_schedule_simulator():
    """Exercise the MULTI-LANE while scheduler path (per-lane cond/body sub-
    ranges after root_len) via the exact python lane simulator. Device schedules
    force a single lane (cross-lane data + iteration races on the real barrier),
    but the multi-lane control-entry logic must stay correct and covered."""
    def f(v):
        return jax.lax.fori_loop(0, 4, lambda i, a: a * 1.5 + 1.0, v)
    x = np.linspace(-2.0, 2.0, 50000, dtype=np.float32)   # multi-tile carry
    prog = L.lower_artifact(to_artifact(f, x))
    sched = S.schedule_program(prog, S.DeviceConfig(nlanes=4, costs={}),
                               allow_multilane_while=True)
    parsed = R.parse(prog.serialize(sched))
    assert parsed.schedule.n_lanes == 4
    got_tensor = R.execute(parsed, [x])
    got_sched = R.execute_schedule(parsed, [x])       # 4-lane sim, cross-lane ok
    exp = np.asarray(f(x))
    np.testing.assert_allclose(got_tensor[0].reshape(exp.shape), exp,
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got_sched[0].reshape(exp.shape), exp,
                               rtol=1e-5, atol=1e-5)


def test_while_schedules_multilane_on_device_config():
    """A while program now schedules across ALL lanes (the device-scope-fence
    barrier fixed the cross-lane data race that used to force a single lane;
    poc/07 / docs/decisions.md #1)."""
    def f(x):
        return jax.lax.fori_loop(0, 3, lambda i, a: a + 1.0, x)
    prog = L.lower_artifact(to_artifact(f, np.linspace(0, 1, 50000, np.float32)))
    sched = S.schedule_program(prog, S.DeviceConfig(nlanes=32, costs={}))
    assert sched.n_lanes == 32


def test_while_then_elementwise():
    """A loop feeding a downstream op: exercises root scheduling around the
    WHILE (init copies before, consumers after, barriers between)."""
    def f(x):
        _, y = jax.lax.while_loop(
            lambda c: c[0] < 5, lambda c: (c[0] + 1, c[1] + 2.0),
            (jnp.int32(0), x))
        return y * y
    check(f, np.float32(1.0))          # (1 + 10)^2 = 121.0
