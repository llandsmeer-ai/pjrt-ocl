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
from ..scheduler import Task, TILE_MMA, TILE_RED_SEG, MMA_T


def _decode(ins):
    """(M, N, K, G) from the packed Instr fields. G = batch count (imm2)."""
    return ins.n, ins.imm >> 16, ins.imm & 0xFFFF, max(1, ins.imm2)


@L.handles("stablehlo.dot_general")
def _dot_general(ctx, op):
    """General dot_general reduced to a (batched) row-major matmul
    C[G,M,N] = A[G,M,K] @ B[G,K,N]:

    - **Broadcast matmul** (`x @ W`, no batch dims): the lhs's leading free dims
      flatten into M — e.g. (B,T,D)@(D,N) becomes (B*T, D)@(D, N).
    - **Batched matmul** (attention QKᵀ / AV): equal leading batch dims on both
      sides give G contiguous per-batch 2D matmuls.

    Canonical layout only: lhs is [batch…, free…, K] (contract = last axis), rhs
    is [batch…, K, free…] (contract = first non-batch axis) — exactly what
    jax.numpy `@` / attention einsums emit. A non-canonical contract axis would
    need an operand transpose first (raised, not silently wrong)."""
    from jaxlib.mlir.dialects import stablehlo
    lhs_shape, _, _ = L.tensor_info(op.operands[0].type)
    rhs_shape, _, _ = L.tensor_info(op.operands[1].type)

    dn = stablehlo.DotDimensionNumbers(op.attributes["dot_dimension_numbers"])
    lc = list(dn.lhs_contracting_dimensions)
    rc = list(dn.rhs_contracting_dimensions)
    lb = list(dn.lhs_batching_dimensions)
    rb = list(dn.rhs_batching_dimensions)
    lr, rr = len(lhs_shape), len(rhs_shape)

    if len(lc) != 1 or len(rc) != 1:
        raise L.LoweringError(
            f"dot_general: only a single contracting dim (got lhs {lc}, rhs {rc})")
    nb = len(lb)
    if lb != list(range(nb)) or rb != list(range(nb)):
        raise L.LoweringError(
            f"dot_general: batch dims must be the leading axes on both sides "
            f"(got lhs_batching={lb}, rhs_batching={rb})")
    G = 1
    for d in range(nb):
        if lhs_shape[d] != rhs_shape[d]:
            raise L.LoweringError("dot_general: batch dim size mismatch")
        G *= lhs_shape[d]
    if lc[0] != lr - 1:
        raise L.LoweringError(
            f"dot_general: lhs contract dim must be last (got {lc[0]} of rank "
            f"{lr}); a non-canonical layout needs a transpose first")
    if rc[0] != nb:
        raise L.LoweringError(
            f"dot_general: rhs contract dim must be first after the batch dims "
            f"(got {rc[0]}); a non-canonical layout needs a transpose first")

    K = lhs_shape[lc[0]]
    if rhs_shape[rc[0]] != K:
        raise L.LoweringError("dot_general: contracting dim size mismatch")
    M = 1
    for d in range(nb, lr - 1):
        M *= lhs_shape[d]
    N = 1
    for d in range(nb + 1, rr):
        N *= rhs_shape[d]
    if N > 0xFFFF or K > 0xFFFF:
        raise L.LoweringError(
            f"dot_general: N={N}, K={K} exceed the 16-bit packing limit")

    dst = ctx.new_buffer(G * M * N)
    ctx.emit(L.Instr(L.OP_DOT, dst=dst,
                     a=ctx.buf_for(op.operands[0]),
                     b=ctx.buf_for(op.operands[1]),
                     n=M, imm=(N << 16) | K, imm2=G))
    ctx.value_to_buf[op.results[0]] = dst


# --- tensor-opcode semantics for OP_DOT -------------------------------------

def _dot_to_task(ins) -> Task:
    M, N, K, G = _decode(ins)
    # GEMV routing: A[M,K] @ x[K,1] through the 64x64 MMA tile wastes 63/64 of
    # every tile (N=1) and runs the K-loop serially per tile — measured 27-46x
    # off cuBLAS on the chained bench. Route it to the segmented-reduce tile in
    # dot mode instead (p3=1: out[o] = sum_j A[o*K+j] * x[j]): one row per
    # tile, whole-workgroup coalesced dot + local tree — M-way parallel.
    # Folded views (p4/p5) stay on the MMA path (the row read must be
    # contiguous).
    if N == 1 and G == 1 and not ins.aview and not ins.bview:
        return Task(TILE_RED_SEG, dst=ins.dst, a=ins.a, b=ins.b,
                    p0=M, p1=K, p2=0, p3=1)
    # aview/bview (+1; 0 = contiguous): a folded transpose/reshape/broadcast on
    # this operand (see lowering._fuse_matmul_views). The device reads the
    # pre-transpose SOURCE via the gather descriptor at p4/p5-1.
    return Task(TILE_MMA, dst=ins.dst, a=ins.a, b=ins.b,
                p0=M, p1=N, p2=K, p3=G, p4=ins.aview, p5=ins.bview)


def _dot_views(ins, rt):
    """Recover (aview, bview) for a folded dot from the 2-word aux header at
    ins.aux (0 = no fold). Written by lowering._finalize_matmul_views so the
    tensor validator, which runs on re-parsed bytecode, can see the fold."""
    if not ins.aux:
        return 0, 0
    return rt.aux[ins.aux], rt.aux[ins.aux + 1]


def _dot_interp(ins, rt):
    """Reference semantics (validator a): C[G,M,N] = A[G,M,K] @ B[G,K,N].
    A folded operand reads its pre-transpose source through the strided view
    (rt.viewed decodes the contiguous [G,M,K]/[G,K,N] flat index)."""
    M, N, K, G = _decode(ins)
    av, bv = _dot_views(ins, rt)
    a = rt.viewed(ins.a, G * M * K, av).reshape(G, M, K)
    b = rt.viewed(ins.b, G * K * N, bv).reshape(G, K, N)
    rt.view(ins.dst, G * M * N)[:] = (a @ b).reshape(-1)


def _dot_tile_sim(task, entry, rt):
    """Schedule simulator (validator b): fill the MMA_T×MMA_T output tiles in
    [entry.tile_lo, entry.tile_hi). Tile index runs over batch × tiles_m ×
    tiles_n; g = tile // (tiles_m*tiles_n) selects the batch slice, then tr/tc
    within it (mirrors mma.cl's batched tile indexing)."""
    M, N, K, G = task.p0, task.p1, task.p2, max(1, task.p3)
    # task.p4/p5 (+1; 0 = contiguous): folded operand read via the strided view
    # (rt.viewed decodes the contiguous [G,M,K]/[G,K,N] flat index). rt.view()
    # takes no element count here, so use the plain read on the contiguous path.
    a = (rt.viewed(task.a, G * M * K, task.p4) if task.p4
         else rt.view(task.a)).reshape(G, M, K)
    b = (rt.viewed(task.b, G * K * N, task.p5) if task.p5
         else rt.view(task.b)).reshape(G, K, N)
    c = rt.view(task.dst).reshape(G, M, N)
    tiles_n = math.ceil(N / MMA_T) if N else 1
    tiles_m = math.ceil(M / MMA_T) if M else 1
    per = tiles_m * tiles_n
    for tile in range(entry.tile_lo, entry.tile_hi):
        g, loc = tile // per, tile % per
        tr, tc = loc // tiles_n, loc % tiles_n
        r0, r1 = tr * MMA_T, min(tr * MMA_T + MMA_T, M)
        c0, c1 = tc * MMA_T, min(tc * MMA_T + MMA_T, N)
        if r0 >= r1 or c0 >= c1:
            continue
        c[g, r0:r1, c0:c1] = a[g, r0:r1, :] @ b[g, :, c0:c1]


opsem.register(L.OP_DOT, to_task=_dot_to_task, interp=_dot_interp)
opsem.register_tile_sim(TILE_MMA, _dot_tile_sim)
