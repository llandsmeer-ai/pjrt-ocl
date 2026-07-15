"""Matmul via OP_DOT / TILE_MMA — the dot_general op family.

Scope (monotone increment; correct over ambitious). We lower ONLY the plain
2D dense matmul that vm2.cl's `mma_tile` already implements:

    C[M,N] = A[M,K] @ B[K,N]      (all row-major, dense, f32, no batching)

stablehlo.dot_general carries a `dot_dimension_numbers` attribute with
lhs/rhs contracting and batching dimensions. We accept ONLY the canonical
plain-matmul layout and raise LoweringError (naming the offending numbers) for
everything else:

  SUPPORTED : lhs rank 2, rhs rank 2,
              lhs_contracting_dimensions == [1], rhs_contracting_dimensions == [0],
              lhs_batching_dimensions == [], rhs_batching_dimensions == [].
  REJECTED  : any batching, non-canonical contracting axes (e.g. [0]x[1], which
              would need an A/B transpose — a separate GATHER family), or rank != 2.

`jax.numpy` 2D matmul (`a @ b`) lowers to exactly the canonical numbers
(verified: `contracting_dims = [1] x [0]`, no batching).

--- M/N/K field encoding (design decision, see NOTES.md) --------------------

The device kernel `mma_tile` reads the tile dims as LITERALS from the task
(`M=t.p0, N=t.p1, K=t.p2`), so the scheduler's `to_task(ins)` must produce
literal M, N, K. But `to_task` is handed only the `Instr` — it cannot
dereference the aux pool (it has no `rt`), and the reader bounds-checks
`Instr.aux <= n_aux`, so a raw integer cannot be smuggled through `aux` either.

An `Instr` has exactly two free scalar fields once dst/a/b hold the buffers:
`n` and `imm`. Three dims must fit in two fields, so we pack:

    Instr.n   = M
    Instr.imm = (N << 16) | K           (N, K each < 2^16)
    Instr.aux = 0                        (DOT uses no aux pool)

Decoders (single source of truth = these two fields):
    M = ins.n ;  N = ins.imm >> 16 ;  K = ins.imm & 0xFFFF

This deviates from docs/vmprogram.md's DOT row (aux=[M,N,K], n=M*N): that form
is unusable here because to_task has no pool access. The C++ engine never reads
the DOT tensor Instr — it consumes the scheduled Task (p0/p1/p2) — so the
Instr encoding only has to satisfy our own interp + scheduler, which it does.
N, K > 65535 raise LoweringError rather than silently corrupting the packing.
"""
from __future__ import annotations

import math

from .. import lowering as L
from .. import opsem
from ..scheduler import Task, TILE_MMA, MMA_T


def _decode(ins):
    """(M, N, K) from the packed Instr fields (see module docstring)."""
    return ins.n, ins.imm >> 16, ins.imm & 0xFFFF


@L.handles("stablehlo.dot_general")
def _dot_general(ctx, op):
    from jaxlib.mlir.dialects import stablehlo
    lhs_shape, _, _ = L.tensor_info(op.operands[0].type)
    rhs_shape, _, _ = L.tensor_info(op.operands[1].type)
    out_shape, _, _ = L.tensor_info(op.results[0].type)

    dn = stablehlo.DotDimensionNumbers(op.attributes["dot_dimension_numbers"])
    lc = list(dn.lhs_contracting_dimensions)
    rc = list(dn.rhs_contracting_dimensions)
    lb = list(dn.lhs_batching_dimensions)
    rb = list(dn.rhs_batching_dimensions)

    if lb or rb:
        raise L.LoweringError(
            f"dot_general: batching unsupported (lhs_batching={lb}, "
            f"rhs_batching={rb}); only plain 2D matmul is covered")
    if len(lhs_shape) != 2 or len(rhs_shape) != 2:
        raise L.LoweringError(
            f"dot_general: only rank-2 @ rank-2 supported "
            f"(got lhs {lhs_shape}, rhs {rhs_shape})")
    if lc != [1] or rc != [0]:
        raise L.LoweringError(
            f"dot_general: only canonical contracting dims [1] x [0] supported "
            f"(got lhs_contracting={lc}, rhs_contracting={rc}); non-canonical "
            f"layouts would need an operand transpose (GATHER family)")

    M, K = lhs_shape
    K2, N = rhs_shape
    if K != K2:
        raise L.LoweringError(
            f"dot_general: contracting dim mismatch (lhs K={K}, rhs K={K2})")
    if out_shape != (M, N):
        raise L.LoweringError(
            f"dot_general: result shape {out_shape} != expected ({M}, {N})")
    if N > 0xFFFF or K > 0xFFFF:
        raise L.LoweringError(
            f"dot_general: N={N}, K={K} exceed the 16-bit packing limit "
            f"(imm = (N<<16)|K); larger matmuls need a wider encoding")

    dst = ctx.new_buffer(M * N)
    ctx.emit(L.Instr(L.OP_DOT, dst=dst,
                     a=ctx.buf_for(op.operands[0]),
                     b=ctx.buf_for(op.operands[1]),
                     n=M, imm=(N << 16) | K))
    ctx.value_to_buf[op.results[0]] = dst


# --- tensor-opcode semantics for OP_DOT -------------------------------------

def _dot_to_task(ins) -> Task:
    M, N, K = _decode(ins)
    return Task(TILE_MMA, dst=ins.dst, a=ins.a, b=ins.b,
                p0=M, p1=N, p2=K, p3=0)


def _dot_interp(ins, rt):
    """Reference semantics (validator a): dense C[MxN] = A[MxK] @ B[KxN]."""
    M, N, K = _decode(ins)
    a = rt.view(ins.a, M * K).reshape(M, K)
    b = rt.view(ins.b, K * N).reshape(K, N)
    rt.view(ins.dst, M * N)[:] = (a @ b).reshape(-1)


def _dot_tile_sim(task, entry, rt):
    """Schedule simulator (validator b): fill the 16x16 output tiles in
    [entry.tile_lo, entry.tile_hi). Mirrors vm2.cl mma_tile tile indexing:
    tiles_n = ceil(N/16); tr = tile // tiles_n; tc = tile % tiles_n; each tile
    is C[tr*16:.., tc*16:..] (clipped to M,N) = A_rows @ B_cols."""
    M, N, K = task.p0, task.p1, task.p2
    a = rt.view(task.a).reshape(M, K)
    b = rt.view(task.b).reshape(K, N)
    c = rt.view(task.dst).reshape(M, N)
    tiles_n = math.ceil(N / MMA_T) if N else 1
    for tile in range(entry.tile_lo, entry.tile_hi):
        tr, tc = tile // tiles_n, tile % tiles_n
        r0, r1 = tr * MMA_T, min(tr * MMA_T + MMA_T, M)
        c0, c1 = tc * MMA_T, min(tc * MMA_T + MMA_T, N)
        if r0 >= r1 or c0 >= c1:
            continue
        c[r0:r1, c0:c1] = a[r0:r1, :] @ b[:, c0:c1]


opsem.register(L.OP_DOT, to_task=_dot_to_task, interp=_dot_interp)
opsem.register_tile_sim(TILE_MMA, _dot_tile_sim)
