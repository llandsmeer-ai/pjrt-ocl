"""stablehlo.gather — general DATA-DEPENDENT gather (OP_GATHER_INDEX, §38).

Distinct from OP_GATHER_STRIDED (ops/shape.py), which is a compile-time-affine
view (broadcast/transpose/slice/reverse). Here the operand base offset of every
output element depends on index values read at RUNTIME from a `start_indices`
tensor — e.g. an embedding lookup `emb[ids]`.

stablehlo.gather semantics (docs/stablehlo-notes.md):
  operand[R_op], start_indices[R_si], dnums = {offset_dims, collapsed_slice_dims,
  start_index_map, index_vector_dim}, slice_sizes[R_op].

  The output splits into BATCH dims (output dims not in offset_dims, ascending)
  and OFFSET dims (offset_dims). Batch dims map 1:1 (in order) to the batch dims
  of start_indices (all its dims except index_vector_dim). Offset dims map 1:1
  (in order) to the NON-collapsed operand dims.

  For output element i:
    start_k = clamp(start_indices[batch_coords, k], 0, dim - slice_size)  for
              operand dim start_index_map[k]
    operand_index[d] = start[d] + offset_coord[d]   (offset_coord 0 for collapsed
                       / non-offset dims; start 0 for dims not in start_index_map)
    out[i] = operand[operand_index]

Reduced to a flat affine form the kernel evaluates per element (all row-major):
    op_off(i)  = Σ_e coord_e(i)·op_stride[e] + Σ_k clamp(S_k)·idx_op_stride[k]
    si_base(i) = Σ_e coord_e(i)·si_stride[e]
    S_k        = start_indices[si_base(i) + k·si_vec_stride]
  where op_stride[e]!=0 only on offset output dims, si_stride[e]!=0 only on batch
  output dims. See kernels/ops/gather.cl (vmo_gather_index_tile) for the twin.

aux layout (MUST match the kernel):
    out_rank, nidx, si_vec_stride, is64,
    idx_byteoff (placeholder, loader-patched), idx_bufid,
    out_dims[out_rank], op_stride[out_rank], si_stride[out_rank],
    idx_op_stride[nidx], clamp_max[nidx]

SUPPORTED: static shapes, i32/i64 indices, any Tier-1/2 element dtype (whole-
element copy). REJECTED: operand/start-indices batching dims (newer stablehlo
batched gather) — raised, not silently wrong.
"""
from __future__ import annotations

import numpy as np

from .. import lowering as L
from .. import opsem
from ..scheduler import Task, TILE_GATHER_INDEX


def _row_major_strides(shape):
    s = [0] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        s[i] = acc
        acc *= shape[i]
    return s


def _plan(op):
    """Compute the affine gather descriptor from the stablehlo op. Returns
    (out_shape, out_n, op_stride, si_stride, idx_op_stride, clamp_max,
     nidx, si_vec_stride, is64)."""
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import stablehlo
    op_shape, _, op_dt = L.tensor_info(op.operands[0].type)
    si_shape, _, si_dt = L.tensor_info(op.operands[1].type)
    out_shape, out_n, _ = L.tensor_info(op.results[0].type)

    dn = stablehlo.GatherDimensionNumbers(op.attributes["dimension_numbers"])
    offset_dims = list(dn.offset_dims)
    collapsed = list(dn.collapsed_slice_dims)
    start_index_map = list(dn.start_index_map)
    index_vector_dim = int(dn.index_vector_dim)
    if list(dn.operand_batching_dims) or list(dn.start_indices_batching_dims):
        raise L.LoweringError("gather: operand/indices batching dims unsupported")

    slice_sizes = [int(v) for v in
                   ir.DenseI64ArrayAttr(op.attributes["slice_sizes"])]

    if si_dt not in (L.DT_I32, L.DT_I64):
        raise L.LoweringError(f"gather: start-index dtype {si_dt} unsupported")
    is64 = 1 if si_dt == L.DT_I64 else 0

    R_op, R_si, R_out = len(op_shape), len(si_shape), len(out_shape)
    op_strides = _row_major_strides(op_shape)
    si_strides = _row_major_strides(si_shape)

    nidx = len(start_index_map)
    if index_vector_dim == R_si:            # implicit trailing size-1 vector
        if nidx != 1:
            raise L.LoweringError("gather: implicit index_vector_dim needs nidx==1")
        si_vec_stride = 0
        si_batch_dims = list(range(R_si))
    else:
        if si_shape[index_vector_dim] != nidx:
            raise L.LoweringError("gather: index_vector_dim size != len(start_index_map)")
        si_vec_stride = si_strides[index_vector_dim]
        si_batch_dims = [j for j in range(R_si) if j != index_vector_dim]

    # offset output dims -> non-collapsed operand dims (both in ascending order).
    offset_operand_dims = [d for d in range(R_op) if d not in collapsed]
    offset_sorted = sorted(offset_dims)
    if len(offset_sorted) != len(offset_operand_dims):
        raise L.LoweringError("gather: offset_dims / non-collapsed operand rank mismatch")
    off_out_to_opdim = {out_d: offset_operand_dims[k]
                        for k, out_d in enumerate(offset_sorted)}

    # batch output dims (ascending) -> start_indices batch dims (in order).
    batch_out_dims = [d for d in range(R_out) if d not in set(offset_dims)]
    if len(batch_out_dims) != len(si_batch_dims):
        raise L.LoweringError("gather: batch-dim rank mismatch (output vs indices)")
    batch_out_to_sidim = dict(zip(batch_out_dims, si_batch_dims))

    op_stride = [0] * R_out
    si_stride = [0] * R_out
    for out_d in range(R_out):
        if out_d in off_out_to_opdim:
            op_stride[out_d] = op_strides[off_out_to_opdim[out_d]]
        else:
            si_stride[out_d] = si_strides[batch_out_to_sidim[out_d]]

    idx_op_stride = [op_strides[start_index_map[k]] for k in range(nidx)]
    clamp_max = [op_shape[start_index_map[k]] - slice_sizes[start_index_map[k]]
                 for k in range(nidx)]

    return (out_shape, out_n, op_stride, si_stride, idx_op_stride, clamp_max,
            nidx, si_vec_stride, is64, op_dt)


@L.handles("stablehlo.gather")
def _gather(ctx, op):
    (out_shape, out_n, op_stride, si_stride, idx_op_stride, clamp_max,
     nidx, si_vec_stride, is64, op_dt) = _plan(op)
    idx_buf = ctx.buf_for(op.operands[1])
    aux = ([len(out_shape), nidx, si_vec_stride, is64, 0, idx_buf]
           + list(out_shape) + list(op_stride) + list(si_stride)
           + list(idx_op_stride) + list(clamp_max))
    aux_off = ctx.add_aux(aux)
    dst = ctx.new_buffer(out_n, op_dt)
    ctx.emit(L.Instr(L.OP_GATHER_INDEX, dst=dst,
                     a=ctx.buf_for(op.operands[0]), n=out_n, aux=aux_off,
                     reads_hint=(idx_buf,)))
    ctx.value_to_buf[op.results[0]] = dst


# --- tensor-opcode semantics (numpy reference + schedule simulator) ---------

def _read_aux(rt, base):
    out_rank = rt.aux[base]
    nidx = rt.aux[base + 1]
    si_vec_stride = rt.aux_i32(base + 2)
    idx_bufid = rt.aux[base + 5]
    p = base + 6
    out_dims = [rt.aux_i32(p + e) for e in range(out_rank)]
    op_stride = [rt.aux_i32(p + out_rank + e) for e in range(out_rank)]
    si_stride = [rt.aux_i32(p + 2 * out_rank + e) for e in range(out_rank)]
    q = p + 3 * out_rank
    idx_op_stride = [rt.aux_i32(q + k) for k in range(nidx)]
    clamp_max = [rt.aux_i32(q + nidx + k) for k in range(nidx)]
    return (out_rank, nidx, si_vec_stride, idx_bufid, out_dims, op_stride,
            si_stride, idx_op_stride, clamp_max)


def _offsets(base, rt, out_index):
    (out_rank, nidx, si_vec_stride, idx_bufid, out_dims, op_stride,
     si_stride, idx_op_stride, clamp_max) = _read_aux(rt, base)
    idx = out_index.astype(np.int64).copy()
    off = np.zeros(len(out_index), dtype=np.int64)
    si_base = np.zeros(len(out_index), dtype=np.int64)
    for e in range(out_rank - 1, -1, -1):
        coord = idx % out_dims[e]
        idx //= out_dims[e]
        off += coord * op_stride[e]
        si_base += coord * si_stride[e]
    starts = rt.view(idx_bufid)   # buffer's native int dtype (i32/i64)
    for k in range(nidx):
        s = starts[si_base + k * si_vec_stride].astype(np.int64)
        s = np.clip(s, 0, clamp_max[k])
        off += s * idx_op_stride[k]
    return off


def _interp(ins, rt):
    off = _offsets(ins.aux, rt, np.arange(ins.n))
    rt.view(ins.dst, ins.n)[:] = rt.view(ins.a)[off]


def _tile_sim(task, entry, rt):
    n = task.p1
    lo = entry.tile_lo * rt.tile_size
    hi = min(entry.tile_hi * rt.tile_size, n)
    if lo >= hi:
        return
    off = _offsets(task.p0, rt, np.arange(lo, hi))
    rt.view(task.dst)[lo:hi] = rt.view(task.a)[off]


def _reads(ins):
    return {ins.a} | set(ins.reads_hint)


def _to_task(ins) -> Task:
    return Task(TILE_GATHER_INDEX, dst=ins.dst, a=ins.a, b=0, p0=ins.aux,
                p1=ins.n)


opsem.register(L.OP_GATHER_INDEX, to_task=_to_task, interp=_interp, reads=_reads)
opsem.register_tile_sim(TILE_GATHER_INDEX, _tile_sim)
