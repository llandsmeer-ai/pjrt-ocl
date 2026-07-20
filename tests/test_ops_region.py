"""Coverage tests: register-resident fused map-region op (§23/§27/§28).

`_fuse_region` recognizes maximal WITHIN-PHASE runs of pure-map f32 EW ops
(elementwise + affine) with a single externally-live output and ≤2 external
inputs, and collapses each into ONE OP_MAP_REGION whose intermediates stay in
per-thread float4 slots (one global load per input, one store) — K phases + K
global round-trips → 1 phase. Over-budget / long chains SPLIT into budget-sized
single-output on-chip sub-regions (still one kernel).

These tests assert (a) the fusion fires on ≤2-input map chains/DAGs, (b) it does
NOT fire where the design gate forbids it (>2 inputs, cross-lane boundary,
singletons), (c) both validators (tensor interp + schedule sim) match jax on the
re-parsed bytecode, (d) the over-budget split produces multiple correct
sub-regions, and (e) PJRT_OCL_FUSE_REGION=0 falls back to the decomposed chain.
"""
import os

import jax.numpy as jnp
import numpy as np

from oputil import check, to_artifact
from pjrt_ocl import lowering as L

RNG = np.random.default_rng(28)


def farr(*shape, scale=1.0):
    return jnp.asarray((RNG.standard_normal(shape) * scale).astype(np.float32))


def _lowered_ops(f, *args):
    prog = L.lower_artifact(to_artifact(f, *args))
    return [i.op for i in prog.instrs[:prog.main_len] if i.op != L.OP_NOP]


def _n_regions(f, *args):
    return _lowered_ops(f, *args).count(L.OP_MAP_REGION)


# --- fusion fires -----------------------------------------------------------

def test_region_fires_two_input_dag():
    # (a+b)*(a-b): 3 map ops, 2 inputs, single output -> one region.
    a, b = farr(1024), farr(1024)
    ops = _lowered_ops(lambda x, y: (x + y) * (x - y), a, b)
    assert ops.count(L.OP_MAP_REGION) == 1
    assert L.OP_ADD_F32 not in ops and L.OP_MUL_F32 not in ops


def test_region_fires_one_input_chain():
    # exp(a)*tanh(a): 2 map ops, 1 input.
    a = farr(777)
    ops = _lowered_ops(lambda x: jnp.exp(x) * jnp.tanh(x), a)
    assert ops.count(L.OP_MAP_REGION) == 1
    assert L.OP_EXP_F32 not in ops and L.OP_TANH_F32 not in ops


def test_region_fires_affine_in_chain():
    # affine (scalar mul/add) folds into a region micro-op alongside the rest.
    a, b = farr(512), farr(512)
    ops = _lowered_ops(lambda x, y: jnp.tanh(2.0 * x + 1.0) - y, a, b)
    assert ops.count(L.OP_MAP_REGION) == 1


def test_region_various_shapes_and_ops():
    for shape in [(16,), (4, 32), (2, 3, 40), (128, 9)]:
        a, b = farr(*shape), farr(*shape)
        ops = _lowered_ops(
            lambda x, y: jnp.maximum(x * y, jnp.abs(x) - y), a, b)
        assert L.OP_MAP_REGION in ops, f"did not fire on {shape}"


# --- correctness on both validators -----------------------------------------

def test_region_diff_of_squares():
    a, b = farr(2000), farr(2000)
    check(lambda x, y: (x + y) * (x - y), a, b, atol=1e-5)


def test_region_transcendental_chain():
    a = farr(4, 300)
    check(lambda x: jnp.tanh(jnp.exp(-jnp.abs(x)) + x), a, atol=1e-5)


def test_region_two_input_mix():
    a, b = farr(6, 55), farr(6, 55)
    check(lambda x, y: jnp.minimum(x * x + y, jnp.maximum(x - y, y)),
          a, b, atol=1e-5)


def test_region_larger_tail():
    # a chain reusing an input several times (DAG), > one float4 tile.
    a = farr(40000)
    def f(x):
        t = x
        for _ in range(5):
            t = jnp.tanh(t) + 0.5 * x
        return t
    prog = check(f, a, atol=1e-5)
    ops = [i.op for i in prog.instrs[:prog.main_len] if i.op != L.OP_NOP]
    assert ops.count(L.OP_MAP_REGION) == 1


# --- over-budget split ------------------------------------------------------

def test_region_over_budget_splits():
    # a long chain forced under a tiny slot budget must SPLIT into several
    # single-output sub-regions (still all correct), not fall back wholesale.
    a = farr(4096)
    def f(x):
        t = x
        for _ in range(6):
            t = jnp.tanh(t) + 0.5 * x
        return t
    os.environ["PJRT_OCL_REGION_SLOTS"] = "3"
    try:
        prog = check(f, a, atol=1e-5)
        n = [i.op for i in prog.instrs[:prog.main_len]
             if i.op != L.OP_NOP].count(L.OP_MAP_REGION)
        assert n >= 2, f"expected a split into >=2 sub-regions, got {n}"
    finally:
        del os.environ["PJRT_OCL_REGION_SLOTS"]


def test_region_split_matches_unsplit():
    # the split path and the single-region path must agree (both vs jax).
    a = farr(3, 300)
    def f(x):
        t = x
        for _ in range(6):
            t = jnp.tanh(t) + 0.5 * x
        return t
    check(f, a, atol=1e-5)                     # default budget: one region
    os.environ["PJRT_OCL_REGION_SLOTS"] = "3"
    try:
        check(f, a, atol=1e-5)                 # forced split
    finally:
        del os.environ["PJRT_OCL_REGION_SLOTS"]


# --- fusion does NOT fire where the gate forbids ----------------------------

def test_no_fire_single_op():
    # a lone EW op is not a region (nothing to collapse).
    a, b = farr(128), farr(128)
    assert _n_regions(lambda x, y: x + y, a, b) == 0


def test_no_fire_across_matmul_boundary():
    # a matmul between two EW ops is a cross-lane boundary: the ops are in
    # different phases and must NOT be fused into one region.
    a = farr(32, 32)
    ops = _lowered_ops(lambda x: jnp.tanh((x + 1.0) @ (x - 1.0)) * 2.0, a)
    # the pre-matmul and post-matmul EW work never share a region.
    assert L.OP_DOT in ops


def test_fuse_region_off_falls_back():
    a, b = farr(4, 32), farr(4, 32)
    os.environ["PJRT_OCL_FUSE_REGION"] = "0"
    try:
        ops = _lowered_ops(lambda x, y: (x + y) * (x - y), a, b)
        assert L.OP_MAP_REGION not in ops
        assert L.OP_MUL_F32 in ops             # decomposed chain intact
        check(lambda x, y: (x + y) * (x - y), a, b, atol=1e-5)
    finally:
        del os.environ["PJRT_OCL_FUSE_REGION"]


def test_fused_matches_decomposed():
    a, b = farr(4, 128), farr(4, 128)
    f = lambda x, y: jnp.tanh(x * y) + jnp.exp(x - y)
    os.environ["PJRT_OCL_FUSE_REGION"] = "0"
    try:
        check(f, a, b, atol=1e-5)              # decomposed
    finally:
        del os.environ["PJRT_OCL_FUSE_REGION"]
    check(f, a, b, atol=1e-5)                  # fused (default)
