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


# --- fixed-trip detection: OP_FOR + unroll (poc/12) --------------------------
# Counted loops (lax.scan / fori_loop) are detected at lowering time and take
# one of two cond-free paths: bytecode UNROLL (small trips) or OP_FOR (a body
# sub-list run a compile-time number of times). PJRT_OCL_WHILE pins the path.

import os
import pytest


@pytest.fixture(params=["while", "for", "unroll"])
def while_mode(request, monkeypatch):
    monkeypatch.setenv("PJRT_OCL_WHILE", request.param)
    return request.param


def _op_counts(f, *args):
    prog = L.lower_artifact(to_artifact(f, *args))
    ops = [i.op for i in prog.instrs]
    return ops.count(L.OP_WHILE), ops.count(L.OP_FOR)


def test_fixed_trip_all_modes(while_mode):
    """The same counted loop is correct through while, OP_FOR, and unroll."""
    def f(x):
        return jax.lax.fori_loop(0, 7, lambda i, a: a * 1.5 + 0.25, x)
    check(f, np.linspace(-1.0, 1.0, 1000, dtype=np.float32))


def test_scan_stacked_outputs(while_mode):
    """lax.scan carrying state AND stacking per-step outputs (dynamic_update_
    slice into the ys carry). Regression test: _fuse_views used to fold the
    DUS identity gather and orphan the scatter (ys came back all zeros)."""
    def f(c, xs):
        def step(c, xt):
            c = c * 0.9 + xt
            return c, c * 2.0
        return jax.lax.scan(step, c, xs)
    c0 = np.zeros(4, np.float32)
    xs = np.arange(20, dtype=np.float32).reshape(5, 4)
    check(f, c0, xs)


def test_scan_counter_is_any_carry_index(while_mode):
    """scan's counter is carry index 1 (xs is 0); detection must find it via
    the cond compare, not assume position 0."""
    def f(c, xs):
        return jax.lax.scan(lambda c, xt: (c + xt, c), c, xs)
    check(f, np.float32(0.0), np.arange(6, dtype=np.float32))


def test_fixed_trip_detection_lowers_op_for():
    os.environ["PJRT_OCL_WHILE"] = "for"
    try:
        def f(x):
            return jax.lax.fori_loop(0, 100, lambda i, a: a + 1.0, x)
        n_while, n_for = _op_counts(f, np.float32(0.0))
        assert (n_while, n_for) == (0, 1)
    finally:
        os.environ.pop("PJRT_OCL_WHILE", None)


def test_auto_unrolls_small_and_fors_large():
    os.environ["PJRT_OCL_WHILE"] = "auto"
    try:
        def small(x):
            return jax.lax.fori_loop(0, 4, lambda i, a: a + 1.0, x)
        def large(x):
            return jax.lax.fori_loop(0, 1000, lambda i, a: a + 1.0, x)
        assert _op_counts(small, np.float32(0.0)) == (0, 0)   # unrolled
        assert _op_counts(large, np.float32(0.0)) == (0, 1)   # OP_FOR
    finally:
        os.environ.pop("PJRT_OCL_WHILE", None)


def test_data_dependent_while_keeps_op_while():
    """A genuinely data-dependent cond must never be converted."""
    def f(x):
        return jax.lax.while_loop(lambda v: v < 100.0, lambda v: v * 2.0, x)
    n_while, n_for = _op_counts(f, np.float32(1.0))
    assert (n_while, n_for) == (1, 0)
    check(f, np.float32(1.0))


def test_fixed_trip_zero_iterations(while_mode):
    def f(x):
        return jax.lax.fori_loop(3, 3, lambda i, a: a * 100.0, x)
    check(f, np.float32(7.0))


def test_fixed_trip_counter_used_in_body(while_mode):
    """Body consumes the counter value itself (i as data): the unroll path
    const-folds it per iteration, FOR keeps the counter add live."""
    def f(x):
        return jax.lax.fori_loop(
            0, 6, lambda i, a: a + jnp.float32(1.0) * i.astype(jnp.float32), x)
    check(f, np.float32(0.0))          # 0+1+2+3+4+5 = 15


def test_fixed_trip_nonunit_step_via_while():
    """Hand-written counted while with step 3: trip = ceil((10-0)/3) = 4."""
    def f(x):
        return jax.lax.while_loop(
            lambda c: c[0] < 10, lambda c: (c[0] + 3, c[1] + 1.0),
            (jnp.int32(0), x))[1]
    check(f, np.float32(0.0))          # 4 iterations


# --- in-place dynamic_update_slice into the loop carry (scan ys stacking) ---

def test_scan_dus_inplace_elides_identity_gather():
    """In FOR mode the DUS identity gather (full ys copy per iteration) must
    be folded away: the scatter writes the ys carry buffer directly."""
    os.environ["PJRT_OCL_WHILE"] = "for"
    try:
        def f(c, xs):
            return jax.lax.scan(lambda c, xt: (c * 0.9 + xt, c), c, xs)
        prog = L.lower_artifact(to_artifact(
            f, np.zeros(8, np.float32), np.ones((5, 8), np.float32)))
        ops = [i.op for i in prog.instrs]
        assert ops.count(L.OP_DYNAMIC_UPDATE_SLICE) == 1
        # the full-length (5*8 elem) identity copy of ys must be gone; the
        # only gathers left are small views (e.g. the (8,)->(1,8) update row).
        assert not any(i.op == L.OP_GATHER_STRIDED and i.n == 40
                       for i in prog.instrs)
    finally:
        os.environ.pop("PJRT_OCL_WHILE", None)


def test_scan_dus_inplace_values(while_mode):
    """Value-correct scan stacking through the in-place DUS path, with a
    nontrivial carry recurrence and T > n so row indexing bugs surface."""
    def f(c, xs):
        def step(c, xt):
            c = c * 0.5 + xt
            return c, c + 1.0
        return jax.lax.scan(step, c, xs)
    c0 = np.linspace(0.0, 1.0, 3).astype(np.float32)
    xs = np.arange(21, dtype=np.float32).reshape(7, 3)
    check(f, c0, xs)


def test_dus_carry_also_read_keeps_copy(while_mode):
    """The ys carry is ALSO read in the body (sum into a second carry): the
    in-place fold must bail (reads-of-carry != 1) and stay correct."""
    def f(x):
        def body(st):
            i, ys, acc = st
            upd = jnp.full((1, 4), 2.0, jnp.float32)
            ys2 = jax.lax.dynamic_update_slice(ys, upd, (i, jnp.int32(0)))
            return i + 1, ys2, acc + jnp.sum(ys)
        st = (jnp.int32(0), jnp.zeros((5, 4), jnp.float32), x)
        return jax.lax.while_loop(lambda st: st[0] < 5, body, st)[1:]
    check(f, np.float32(0.0))


def test_dus_operand_not_carry_keeps_copy(while_mode):
    """DUS whose operand is not the carry (fresh zeros each iteration): the
    fold must bail (gather source != carry buffer) and stay correct."""
    def f(x):
        def body(st):
            i, out = st
            base = jnp.zeros((4, 2), jnp.float32)
            upd = jnp.full((1, 2), 1.0, jnp.float32) * (i.astype(jnp.float32) + 1.0)
            fresh = jax.lax.dynamic_update_slice(base, upd, (i, jnp.int32(0)))
            return i + 1, out + fresh
        st = (jnp.int32(0), x)
        return jax.lax.while_loop(lambda st: st[0] < 4, body, st)[1]
    check(f, np.zeros((4, 2), np.float32))
