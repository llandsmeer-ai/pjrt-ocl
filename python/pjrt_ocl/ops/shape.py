"""Shape ops via GATHER_STRIDED — the reference op-family module.

GATHER_STRIDED computes, for each output element i (row-major over out_dims):
    dst[i] = a[src_off + sum_d idx_d(i) * in_stride_d]
where idx_d(i) is the d-th coordinate of i in out_dims. This one tile op covers
a whole class of "pick elements of the input by an affine index map":

- broadcast_in_dim: out_stride 0 on broadcast axes, input's own stride on mapped
  axes (stride 0 = same input element read repeatedly).
- transpose: out_dims = permuted in_dims; in_strides = input row-major strides
  permuted the same way.

Both reduce to: choose out_dims and, per output axis, the input stride to walk.

aux layout (docs/vmprogram.md, OP_GATHER_STRIDED):
    rank u32, out_dims i32[rank], in_strides i32[rank], src_off i32
"""
from __future__ import annotations

import math

from .. import lowering as L
from .. import opsem
from ..scheduler import Task, TILE_GATHER


def _row_major_strides(shape: tuple[int, ...]) -> list[int]:
    strides = [0] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return strides


def _emit_gather(ctx, out_shape, in_buf, out_strides, src_off=0, dtype=None):
    """Emit a GATHER_STRIDED producing out_shape from in_buf, walking in_buf with
    `out_strides` (one per output axis) starting at element `src_off`. Gather
    copies whole elements, so the output dtype = input dtype (must be passed for
    non-f32 — the element size drives the VM's copy width)."""
    n = math.prod(out_shape) if out_shape else 1
    rank = len(out_shape)
    aux_words = [rank] + list(out_shape) + list(out_strides) + [src_off]
    aux_off = ctx.add_aux(aux_words)
    dst = ctx.new_buffer(n, L.DT_F32 if dtype is None else dtype)
    ctx.emit(L.Instr(L.OP_GATHER_STRIDED, dst=dst, a=in_buf, n=n, aux=aux_off))
    return dst


@L.handles("stablehlo.broadcast_in_dim")
def _broadcast_in_dim(ctx, op):
    from jaxlib.mlir import ir
    in_shape, _, in_dt = L.tensor_info(op.operands[0].type)
    out_shape, _, _ = L.tensor_info(op.results[0].type)
    bcast_dims = [int(x) for x in ir.DenseI64ArrayAttr(
        op.attributes["broadcast_dimensions"])]
    if len(bcast_dims) != len(in_shape):
        raise L.LoweringError("broadcast_in_dim: dims/operand rank mismatch")
    in_strides = _row_major_strides(in_shape)
    # out_stride[axis] = input stride of the input axis mapped here, or 0 if the
    # input axis has size 1 (stretched) or this out axis is newly introduced.
    out_strides = [0] * len(out_shape)
    for in_axis, out_axis in enumerate(bcast_dims):
        if in_shape[in_axis] == 1:
            out_strides[out_axis] = 0            # stretch: read same element
        else:
            if out_shape[out_axis] != in_shape[in_axis]:
                raise L.LoweringError("broadcast_in_dim: non-1 axis size change")
            out_strides[out_axis] = in_strides[in_axis]
    dst = _emit_gather(ctx, out_shape, ctx.buf_for(op.operands[0]), out_strides,
                       dtype=in_dt)
    ctx.value_to_buf[op.results[0]] = dst
    # scalar-const broadcast (rank-0 f32 -> full shape): record the constant so
    # an elementwise consumer can fold `x*s`/`x+t` into OP_AFFINE_F32 and let the
    # now-dead gather be NOP'd (perf: kills scalar-broadcast materialization).
    src = op.operands[0]
    if src in ctx.const_scalar and all(s == 0 for s in out_strides):
        ctx.scalar_bcast[op.results[0]] = ctx.const_scalar[src]


@L.handles("stablehlo.reshape")
def _reshape(ctx, op):
    # Row-major flat storage makes reshape a pure relabel: the element order is
    # bit-identical, so alias the input's buffer under the new shape (no
    # instruction, like func.return / constants). stablehlo.reshape carries no
    # dimension-reorder attribute (that would be a transpose).
    in_n = L.tensor_info(op.operands[0].type)[1]
    out_n = L.tensor_info(op.results[0].type)[1]
    if in_n != out_n:
        raise L.LoweringError(
            f"reshape: element count {in_n} != {out_n} (bitcast/resize?)")
    ctx.value_to_buf[op.results[0]] = ctx.buf_for(op.operands[0])


@L.handles("stablehlo.slice")
def _slice(ctx, op):
    # dst[i] = a[ start + i*slice_stride ] per axis -> a strided gather.
    from jaxlib.mlir import ir
    in_shape, _, in_dt = L.tensor_info(op.operands[0].type)
    out_shape, _, _ = L.tensor_info(op.results[0].type)
    starts = [int(x) for x in
              ir.DenseI64ArrayAttr(op.attributes["start_indices"])]
    sstrides = [int(x) for x in ir.DenseI64ArrayAttr(op.attributes["strides"])]
    in_strides = _row_major_strides(in_shape)
    out_strides = [in_strides[d] * sstrides[d] for d in range(len(out_shape))]
    src_off = sum(starts[d] * in_strides[d] for d in range(len(in_shape)))
    dst = _emit_gather(ctx, out_shape, ctx.buf_for(op.operands[0]),
                       out_strides, src_off, dtype=in_dt)
    ctx.value_to_buf[op.results[0]] = dst


@L.handles("stablehlo.reverse")
def _reverse(ctx, op):
    # reversed axes walk with a negative stride, starting at the last element.
    from jaxlib.mlir import ir
    in_shape, _, in_dt = L.tensor_info(op.operands[0].type)
    dims = {int(x) for x in ir.DenseI64ArrayAttr(op.attributes["dimensions"])}
    in_strides = _row_major_strides(in_shape)
    out_strides = list(in_strides)
    src_off = 0
    for d in dims:
        out_strides[d] = -in_strides[d]
        src_off += (in_shape[d] - 1) * in_strides[d]
    dst = _emit_gather(ctx, in_shape, ctx.buf_for(op.operands[0]),
                       out_strides, src_off, dtype=in_dt)
    ctx.value_to_buf[op.results[0]] = dst


@L.handles("stablehlo.transpose")
def _transpose(ctx, op):
    from jaxlib.mlir import ir
    in_shape, _, in_dt = L.tensor_info(op.operands[0].type)
    out_shape, _, _ = L.tensor_info(op.results[0].type)
    perm = [int(x) for x in ir.DenseI64ArrayAttr(op.attributes["permutation"])]
    if len(perm) != len(in_shape):
        raise L.LoweringError("transpose: permutation/rank mismatch")
    in_strides = _row_major_strides(in_shape)
    # output axis k corresponds to input axis perm[k]; walk that input stride.
    out_strides = [in_strides[perm[k]] for k in range(len(out_shape))]
    dst = _emit_gather(ctx, out_shape, ctx.buf_for(op.operands[0]), out_strides,
                       dtype=in_dt)
    ctx.value_to_buf[op.results[0]] = dst


# --- tensor-opcode semantics for OP_GATHER_STRIDED --------------------------

def _gather_to_task(ins) -> Task:
    return Task(TILE_GATHER, dst=ins.dst, a=ins.a, b=0,
                p0=ins.aux, p1=ins.n, p2=0, p3=0)


def _gather_interp(ins, rt):
    import numpy as np
    base = ins.aux
    rank = rt.aux[base]
    out_dims = [rt.aux_i32(base + 1 + d) for d in range(rank)]
    in_strides = [rt.aux_i32(base + 1 + rank + d) for d in range(rank)]
    src_off = rt.aux_i32(base + 1 + 2 * rank)
    n = ins.n
    src = rt.view(ins.a)
    out = rt.view(ins.dst, n)
    if rank == 0:
        out[0] = src[src_off]
        return
    # vectorized affine gather: offset(i) = src_off + sum_d coord_d(i)*stride_d
    idx = np.arange(n)
    offs = np.full(n, src_off, dtype=np.int64)
    for d in range(rank - 1, -1, -1):
        coord = idx % out_dims[d]
        idx //= out_dims[d]
        offs += coord * in_strides[d]
    out[:] = src[offs]


def _gather_reads(ins) -> set[int]:
    return {ins.a}


def _gather_coords(base, rt, out_index):
    """offset into `a` for a batch of flat output indices (numpy int array)."""
    import numpy as np
    rank = rt.aux[base]
    out_dims = [rt.aux_i32(base + 1 + d) for d in range(rank)]
    in_strides = [rt.aux_i32(base + 1 + rank + d) for d in range(rank)]
    src_off = rt.aux_i32(base + 1 + 2 * rank)
    if rank == 0:
        return np.full(len(out_index), src_off, dtype=np.int64)
    idx = out_index.astype(np.int64).copy()
    offs = np.full(len(out_index), src_off, dtype=np.int64)
    for d in range(rank - 1, -1, -1):
        offs += (idx % out_dims[d]) * in_strides[d]
        idx //= out_dims[d]
    return offs


def _gather_tile_sim(task, entry, rt):
    """Validator b: gather output elements for tiles [tile_lo, tile_hi)."""
    import numpy as np
    n = task.p1
    lo = entry.tile_lo * rt.tile_size
    hi = min(entry.tile_hi * rt.tile_size, n)
    if lo >= hi:
        return
    out_index = np.arange(lo, hi)
    offs = _gather_coords(task.p0, rt, out_index)
    rt.view(task.dst)[lo:hi] = rt.view(task.a)[offs]


opsem.register(L.OP_GATHER_STRIDED, to_task=_gather_to_task,
               interp=_gather_interp, reads=_gather_reads)
opsem.register_tile_sim(TILE_GATHER, _gather_tile_sim)
