"""Arena liveness-reuse pass (lowering._reuse_arena, docs/decisions.md §16).

The old bump allocator gave every buffer a fresh arena offset, so the arena
grew with the SUM of every intermediate ever emitted and multi-layer programs
overflowed the 2^31 offset cap. The reuse pass reassigns offsets by live
interval (in scheduler PHASE time) so the arena is bounded by PEAK concurrent
liveness. These tests prove:

  * a later buffer REUSES a dead earlier buffer's offset (arena < naive sum),
  * a program that would overflow the old bump allocator now fits, and
  * every case stays numerically correct under BOTH vmreader validators (the
    semantic tensor interpreter and the schedule lane simulator, which re-parse
    the serialized bytecode and must agree) — an early free would corrupt one.
"""
from __future__ import annotations

import io
import os

os.environ["JAX_PLATFORMS"] = "cpu"      # lower against CPU; never route via .so

import numpy as np
import pytest

import pjrt_ocl.lowering as L
import pjrt_ocl.scheduler as S
import pjrt_ocl.vmreader as R

CFG = S.DeviceConfig(nlanes=8, costs={})   # fuse=True (matches _reuse_arena's env)


def _artifact(fn, *args):
    import jax
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import stablehlo
    m = jax.jit(fn).lower(*args).compiler_ir("stablehlo")
    buf = io.BytesIO()
    m.operation.write_bytecode(file=buf)
    clone = ir.Module.parse(buf.getvalue(), context=m.context)
    return stablehlo.serialize_portable_artifact(
        clone, stablehlo.get_current_version())


def _lower(fn, *args):
    """Lower + schedule + re-parse (what the executor/validators see)."""
    prog = L.lower_artifact(_artifact(fn, *args))
    sched = S.schedule_program(prog, CFG)
    parsed = R.parse(prog.serialize(sched))
    return prog, parsed


def _both(parsed, args):
    """Run both reference validators, assert they agree, return outputs."""
    sem = R.execute(parsed, list(args))
    sch = R.execute_schedule(parsed, list(args))
    assert len(sem) == len(sch)
    for a, b in zip(sem, sch):
        np.testing.assert_array_equal(a, b)
    return sem


def _aligned(n_bytes: int) -> int:
    return -(-n_bytes // L.ARENA_ALIGN) * L.ARENA_ALIGN


def _naive_sum(prog) -> int:
    """What the old bump allocator would have used = sum of aligned sizes."""
    return sum(_aligned(b.size_bytes) for b in prog.buffers)


# ---------------------------------------------------------------------------
# 1. reuse actually happens: a later buffer takes a dead one's offset
# ---------------------------------------------------------------------------

def test_offset_reuse_in_matmul_chain():
    """A chain of matmuls; each intermediate dies after the next matmul, so a
    later intermediate reuses an earlier one's slot. Offsets are no longer a
    per-buffer bijection, and the arena is far below the naive sum — yet both
    validators reproduce numpy exactly."""
    N = 16
    a = (np.arange(N * N, dtype=np.float32) % 5).reshape(N, N)
    w = (np.arange(N * N, dtype=np.float32) % 3 - 1).reshape(N, N)

    def f(a, w):
        x = a @ w
        y = x @ w
        z = y @ w
        return z @ w

    prog, parsed = _lower(f, a, w)

    offsets = [b.arena_byte_offset for b in prog.buffers]
    # some offset is shared by >1 buffer id  => genuine slot reuse
    assert len(set(offsets)) < len(offsets)
    # peak-bound, not sum-bound
    assert prog.arena_bytes < _naive_sum(prog)

    (got,) = _both(parsed, (a, w))
    np.testing.assert_allclose(got.reshape(N, N), a @ w @ w @ w @ w,
                               rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 2. a program whose SUM overflows but whose PEAK fits
# ---------------------------------------------------------------------------

def test_arena_bounded_by_peak_not_sum():
    """Many sequential reduce/broadcast phases each emit a fresh full-length
    intermediate. The bump allocator would sum them all (here ~40 MiB); the
    reuse pass bounds the arena to a couple of live buffers. Assert a large
    reduction AND numerical correctness."""
    M = 4096
    x = (np.arange(M, dtype=np.float32) % 7).reshape(1, M)

    def step(acc):
        m = acc.mean(axis=1, keepdims=True)        # reduce -> phase boundary
        return (acc - m) * 0.9 + 0.01              # broadcast (fresh buffer)

    def f(x):
        acc = x
        for _ in range(24):
            acc = step(acc)
        return acc

    prog, parsed = _lower(f, x)

    # bump allocator would need the full sum; reuse fits in a small multiple of
    # the working set (a handful of M-length f32 buffers, not 24+).
    assert prog.arena_bytes < _naive_sum(prog) // 4
    assert prog.arena_bytes < 8 * _aligned(M * 4)

    ref = x
    for _ in range(24):
        m = ref.mean(axis=1, keepdims=True)
        ref = (ref - m) * 0.9 + 0.01
    (got,) = _both(parsed, (x,))
    np.testing.assert_allclose(got.reshape(1, M), ref, rtol=1e-5, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. region safety: a counted loop (OP_FOR/WHILE) stays correct
#    (its whole sub-list collapses to one phase — no early free of a carry)
# ---------------------------------------------------------------------------

def test_reuse_correct_across_while_region():
    """A fori_loop with a vector carry, surrounded by matmul phases that create
    reusable dead slots. The loop's carry/body buffers must stay live across
    the entire region; both validators reproduce the reference exactly."""
    import jax
    N = 32
    a = (np.arange(N * N, dtype=np.float32) % 4).reshape(N, N)
    w = (np.arange(N * N, dtype=np.float32) % 3).reshape(N, N)

    def f(a, w):
        x = a @ w                                   # dead-slot producer
        c = jax.lax.fori_loop(0, 5, lambda i, c: c * 1.5 + 1.0, x)
        return c @ w                                # reuse x's slot region

    prog, parsed = _lower(f, a, w)
    (got,) = _both(parsed, (a, w))

    ref = a @ w
    for _ in range(5):
        ref = ref * 1.5 + 1.0
    ref = ref @ w
    np.testing.assert_allclose(got.reshape(N, N), ref, rtol=1e-4, atol=1e-3)


# ---------------------------------------------------------------------------
# 4. viewfold source liveness: a folded gather source must stay live for its
#    viewer (transpose folded into a matmul operand read, §14a)
# ---------------------------------------------------------------------------

def test_reuse_correct_with_viewfold_transpose():
    """`x @ x.T` folds the transpose gather into the matmul operand; the shared
    source buffer must remain live until the (viewing) matmul runs. Reuse must
    not reclaim it early."""
    M, K = 24, 16
    x = (np.arange(M * K, dtype=np.float32) % 5).reshape(M, K)

    def f(x):
        y = x @ x.T          # (M,M): B operand is x viewed as transpose
        return y @ x         # (M,K)

    prog, parsed = _lower(f, x)
    (got,) = _both(parsed, (x,))
    np.testing.assert_allclose(got.reshape(M, K), (x @ x.T) @ x,
                               rtol=1e-4, atol=1e-3)


# ---------------------------------------------------------------------------
# 5. every buffer offset stays in-range and 64B-aligned (runtime contract)
# ---------------------------------------------------------------------------

def test_offsets_valid_after_reuse():
    N = 20
    a = (np.arange(N, dtype=np.float32) % 3)

    def f(a):
        x = a
        for _ in range(8):
            x = x * a + 1.0
        return x, a.sum(keepdims=True)

    prog, _ = _lower(f, a)
    for b in prog.buffers:
        assert b.arena_byte_offset % L.ARENA_ALIGN == 0
        assert b.arena_byte_offset + b.size_bytes <= prog.arena_bytes
