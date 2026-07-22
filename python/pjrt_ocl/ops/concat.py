"""concatenate + pad via the strided-scatter tile op (OP_SCATTER / TILE_SCATTER).

concatenate(inputs, dim): allocate the output, then scatter each input into its
disjoint slice along `dim`. pad: fill the output with the pad value, then scatter
the operand into the interior. Each source scatters `dst[out_off + affine(i)] =
a[i]`; multiple scatters write the same output buffer but disjoint regions —
they are tagged with a shared `disjoint` group id so the scheduler packs them
into ONE barrier phase instead of WAW-serializing them (§52).

FAST PATH (§52, `_try_index_gather`): a concatenate whose operands are ALL
slices of the SAME source value is a pure permutation/duplication of that
source — `jnp.roll` and the neighbour-shift idiom of every stencil / spring
chain lower to exactly that. Instead of N gathers + N scatters (2N barrier
phases inside a scan body, the dominant per-iteration cost on a small GPU) it
becomes ONE OP_GATHER_INDEX from a compile-time constant index vector: one
phase, and 3n memory traffic instead of 4n. Gated on output size (the index
vector is a const-pool i32 array of that many elements).
"""
from __future__ import annotations

import math
import os

import numpy as np

from .. import lowering as L
from .. import opsem
from ..scheduler import Task, TILE_SCATTER


def _row_major_strides(shape):
    s = [0] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        s[i] = acc
        acc *= shape[i]
    return s


def _emit_scatter(ctx, out_buf, in_buf, in_shape, out_strides, out_off,
                  disjoint=0):
    n = math.prod(in_shape) if in_shape else 1
    rank = len(in_shape)
    aux = [rank] + list(in_shape) + list(out_strides) + [out_off]
    aux_off = ctx.add_aux(aux)
    ctx.emit(L.Instr(L.OP_SCATTER, dst=out_buf, a=in_buf, n=n, aux=aux_off,
                     disjoint=disjoint))


# Max output elements for which a concatenate-of-slices is turned into a single
# index gather. The index vector costs 4 bytes/element in the const pool (and
# one extra 4-byte read per output element at run time), so cap it; above the
# cap the scatter lowering (no index vector) is used. PJRT_OCL_CONCAT_GATHER=0
# disables the fast path entirely (A/B lever).
def _concat_idx_max() -> int:
    try:
        return int(os.environ.get("PJRT_OCL_CONCAT_IDX_MAX", str(1 << 20)))
    except ValueError:
        return 1 << 20


def _slice_of(v):
    """If SSA value `v` is produced by a `stablehlo.slice`, return
    (source_value, start_indices, strides); else None."""
    from jaxlib.mlir import ir
    d = L.defining_op(v)
    if d is None or d.name != "stablehlo.slice":
        return None
    starts = [int(x) for x in
              ir.DenseI64ArrayAttr(d.attributes["start_indices"])]
    strides = [int(x) for x in ir.DenseI64ArrayAttr(d.attributes["strides"])]
    return d.operands[0], starts, strides


def _try_index_gather(ctx, op) -> bool:
    """concatenate(slice(S,…), slice(S,…), …) -> one OP_GATHER_INDEX over a
    constant index vector. Returns True iff it fired.

    Every operand must be a slice of the SAME source value S (an operand that
    IS S counts as the full slice). The result is then `out[i] = S_flat[idx[i]]`
    with idx computed here at compile time — a single barrier phase instead of
    one gather per slice plus one scatter per operand.

    Reading S at the concatenate's position rather than at each slice's position
    is safe for the same reason the existing lowering is: within a block the
    buffer of an SSA value is only rewritten by a loop-carry commit, and the
    in-place commit folds in `_lower_while_regions` are disabled whenever any
    other body instruction (this gather) reads the carry."""
    from jaxlib.mlir import ir
    if os.environ.get("PJRT_OCL_CONCAT_GATHER", "1") == "0":
        return False
    out_shape, out_n, dtype = L.tensor_info(op.results[0].type)
    if out_n <= 0 or out_n > _concat_idx_max() or not out_shape:
        return False
    src = None
    specs = []                        # (starts, strides, operand_shape)
    for v in op.operands:
        if v in ctx.cbuf:             # split-complex pair: not a single buffer
            return False
        sl = _slice_of(v)
        if sl is None:
            s, sh = v, L.tensor_info(v.type)[0]
            starts, strides = [0] * len(sh), [1] * len(sh)
        else:
            s, starts, strides = sl
            sh = L.tensor_info(v.type)[0]
        if src is None:
            src = s
        elif s != src:
            return False
        specs.append((starts, strides, sh))
    if src is None or src in ctx.cbuf:
        return False
    src_shape, src_n, src_dt = L.tensor_info(src.type)
    if src_dt != dtype or src_n <= 0 or len(src_shape) != len(out_shape):
        return False

    dim = int(ir.IntegerAttr(op.attributes["dimension"]).value)
    out_strides = _row_major_strides(out_shape)
    src_strides = _row_major_strides(src_shape)
    idx = np.zeros(out_n, dtype=np.int32)
    offset = 0                                   # running position along `dim`
    for starts, strides, sh in specs:
        if math.prod(sh) == 0:
            continue
        grids = np.indices(sh)
        src_lin = np.zeros(sh, dtype=np.int64)
        out_lin = np.zeros(sh, dtype=np.int64)
        for d in range(len(sh)):
            src_lin += (starts[d] + grids[d] * strides[d]) * src_strides[d]
            out_lin += (grids[d] + (offset if d == dim else 0)) * out_strides[d]
        idx[out_lin.ravel()] = src_lin.ravel()
        offset += sh[dim]
    if int(idx.min()) < 0 or int(idx.max()) >= src_n:
        return False                             # defensive: never clamp-fold

    # Dedup identical index vectors: an unrolled scan emits the SAME roll every
    # iteration, and each copy would otherwise cost out_n*4 const-pool bytes
    # (heat2d: 4.4 MB of duplicates before this).
    raw = idx.tobytes()
    cache = getattr(ctx, "_concat_idx_cache", None)
    if cache is None:
        cache = ctx._concat_idx_cache = {}
    idx_buf = cache.get(raw)
    if idx_buf is None:
        idx_buf = ctx.new_buffer(out_n, L.DT_I32)
        ctx.consts.append((idx_buf, raw))
        cache[raw] = idx_buf
    # gather_index aux (ops/gather_index.py): flat 1-D operand view —
    # out_rank=1, nidx=1, si_vec_stride=0, is64=0, idx_byteoff placeholder,
    # idx_bufid, out_dims=[out_n], op_stride=[0], si_stride=[1],
    # idx_op_stride=[1], clamp_max=[src_n-1].
    aux_off = ctx.add_aux([1, 1, 0, 0, 0, idx_buf, out_n, 0, 1, 1, src_n - 1])
    dst = ctx.new_buffer(out_n, dtype)
    ctx.emit(L.Instr(L.OP_GATHER_INDEX, dst=dst, a=ctx.buf_for(src), n=out_n,
                     aux=aux_off, reads_hint=(idx_buf,)))
    ctx.value_to_buf[op.results[0]] = dst
    return True


@L.handles("stablehlo.concatenate")
def _concatenate(ctx, op):
    from jaxlib.mlir import ir
    if _try_index_gather(ctx, op):
        return
    out_shape, out_n, dtype = L.tensor_info(op.results[0].type)
    dim = int(ir.IntegerAttr(op.attributes["dimension"]).value)
    out_strides = _row_major_strides(out_shape)
    out_buf = ctx.new_buffer(out_n, dtype)
    # group id unique within this program: the scheduler treats scatters sharing
    # it as writing disjoint regions of out_buf (they do, by construction) and
    # packs them into ONE phase instead of a WAW barrier chain.
    group = len(ctx.instrs) + 1
    offset = 0                                   # running position along `dim`
    for operand in op.operands:
        in_shape, _, _ = L.tensor_info(operand.type)
        _emit_scatter(ctx, out_buf, ctx.buf_for(operand), in_shape,
                      out_strides, offset * out_strides[dim], disjoint=group)
        offset += in_shape[dim]
    ctx.value_to_buf[op.results[0]] = out_buf


@L.handles("stablehlo.pad")
def _pad(ctx, op):
    from jaxlib.mlir import ir
    in_shape, in_n, dtype = L.tensor_info(op.operands[0].type)
    out_shape, out_n, _ = L.tensor_info(op.results[0].type)
    low = [int(x) for x in ir.DenseI64ArrayAttr(op.attributes["edge_padding_low"])]
    high = [int(x) for x in ir.DenseI64ArrayAttr(op.attributes["edge_padding_high"])]
    interior = [int(x) for x in
                ir.DenseI64ArrayAttr(op.attributes["interior_padding"])]
    if any(v < 0 for v in low + high):
        raise L.LoweringError("pad: negative edge padding (cropping) unsupported")
    out_strides = _row_major_strides(out_shape)
    out_buf = ctx.new_buffer(out_n, dtype)
    # 1) fill the whole output with the (scalar) pad value via a stride-0 gather
    #    (reads operand[1], the scalar pad value, into every output element).
    fill_aux = ctx.add_aux([len(out_shape)] + list(out_shape) +
                           [0] * len(out_shape) + [0])
    ctx.emit(L.Instr(L.OP_GATHER_STRIDED, dst=out_buf,
                     a=ctx.buf_for(op.operands[1]), n=out_n, aux=fill_aux))
    # 2) scatter the operand into the interior: output coord for input coord c is
    #    low[d] + c*(interior[d]+1)  ->  out_stride[d]*(interior[d]+1), off=sum low.
    scat_strides = [out_strides[d] * (interior[d] + 1)
                    for d in range(len(in_shape))]
    out_off = sum(low[d] * out_strides[d] for d in range(len(in_shape)))
    _emit_scatter(ctx, out_buf, ctx.buf_for(op.operands[0]), in_shape,
                  scat_strides, out_off)
    ctx.value_to_buf[op.results[0]] = out_buf


# --- OP_SCATTER tensor-opcode semantics -------------------------------------

def _scatter_to_task(ins) -> Task:
    return Task(TILE_SCATTER, dst=ins.dst, a=ins.a, b=ins.a, p0=ins.aux,
                p1=ins.n)


def _scatter_coords(rt, base, n):
    rank = rt.aux[base]
    in_dims = [rt.aux_i32(base + 1 + d) for d in range(rank)]
    out_strides = [rt.aux_i32(base + 1 + rank + d) for d in range(rank)]
    out_off = rt.aux_i32(base + 1 + 2 * rank)
    if rank == 0:
        return np.full(n, out_off, dtype=np.int64)
    idx = np.arange(n)
    offs = np.full(n, out_off, dtype=np.int64)
    for d in range(rank - 1, -1, -1):
        offs += (idx % in_dims[d]) * out_strides[d]
        idx //= in_dims[d]
    return offs


def _scatter_interp(ins, rt) -> None:
    offs = _scatter_coords(rt, ins.aux, ins.n)
    rt.view(ins.dst)[offs] = rt.view(ins.a, ins.n)


def _scatter_tile_sim(task, entry, rt):
    n = task.p1
    lo = entry.tile_lo * rt.tile_size
    hi = min(entry.tile_hi * rt.tile_size, n)
    if lo >= hi:
        return
    offs = _scatter_coords(rt, task.p0, n)[lo:hi]
    rt.view(task.dst)[offs] = rt.view(task.a)[lo:hi]


opsem.register(L.OP_SCATTER, to_task=_scatter_to_task, interp=_scatter_interp,
               reads=lambda ins: {ins.a})
opsem.register_tile_sim(TILE_SCATTER, _scatter_tile_sim)
