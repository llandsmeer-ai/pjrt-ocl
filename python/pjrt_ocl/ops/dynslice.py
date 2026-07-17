"""dynamic_slice + dynamic_update_slice — gather/scatter with a RUNTIME offset.

stablehlo.dynamic_slice(operand, i0, i1, ...) reads a static-size slice whose
start offsets are RUNTIME scalar operands (not attributes). Unlike stablehlo.
slice (a plain strided gather whose src_off is a compile-time aux constant),
the base offset here is computed on device from the start scalars:

    base = sum_d clamp(start_d, 0, in_dim_d - slice_size_d) * in_stride_d
    out[i] = operand[base + affine(i)]

The start scalars' locations ride in the aux pool as idx_byteoff[rank] (for the
VM) + idx_bufid[rank] (for the numpy validators, which address by id). The
handler CANNOT fill idx_byteoff meaningfully: arena offsets are reassigned by
the _reuse_arena post-pass, and a start scalar that is a program input may live
in an I/O port (assigned at load time, runtime.cc). So idx_byteoff is written
as a placeholder 0 here and the loader patches it to elem_off(idx_bufid) —
arena byte offset or bit-31 port handle — which the kernels read through AP().

dynamic_update_slice(operand, update, i0, ...) is the mirror: copy operand into
the output, then scatter `update` at the runtime base offset. The copy is an
identity GATHER (dtype-agnostic); the WAW on the output buffer makes the
scheduler barrier the scatter after the copy.

SUPPORTED: static-shape operand, i32/i64 start scalars, any Tier-1/2 element
dtype (whole-element copy). REJECTED: nothing shaped — dynamic_slice/update are
fully general here (clamping matches stablehlo).

aux layout (both ops; MUST match kernels/ops/dynslice.cl):
    rank, dims[rank], strides[rank], clamp_max[rank],
    idx_byteoff[rank], idx_bufid[rank], is64
"""
from __future__ import annotations

import math

import numpy as np

from .. import lowering as L
from .. import opsem
from ..scheduler import Task, TILE_DYN_GATHER, TILE_DYN_SCATTER


def _row_major_strides(shape):
    s = [0] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        s[i] = acc
        acc *= shape[i]
    return s


def _index_info(ctx, index_operands):
    """Validate + collect the start-index scalar operands. Returns
    (buf_ids, byte_offset_placeholders, is64). The byte offsets are 0
    placeholders — the loader patches them from the buffer ids (see module
    docstring); only the ids are meaningful at lowering time."""
    buf_ids, byte_offs, is64 = [], [], None
    for iv in index_operands:
        shape, n, dt = L.tensor_info(iv.type)
        if n != 1:
            raise L.LoweringError("dynamic_slice: start index must be a scalar")
        if dt not in (L.DT_I32, L.DT_I64):
            raise L.LoweringError(
                f"dynamic_slice: start index dtype {dt} unsupported (i32/i64)")
        this64 = 1 if dt == L.DT_I64 else 0
        if is64 is None:
            is64 = this64
        elif is64 != this64:
            raise L.LoweringError("dynamic_slice: mixed i32/i64 start indices")
        buf_ids.append(ctx.buf_for(iv))
        byte_offs.append(0)
    return buf_ids, byte_offs, (is64 or 0)


def _emit_dyn(ctx, opcode, tile_dims, strides, clamp_max, idx_bufs, idx_offs,
              is64, dst, a, n):
    rank = len(tile_dims)
    aux = ([rank] + list(tile_dims) + list(strides) + list(clamp_max) +
           list(idx_offs) + list(idx_bufs) + [is64])
    aux_off = ctx.add_aux(aux)
    ctx.emit(L.Instr(opcode, dst=dst, a=a, n=n, aux=aux_off,
                     reads_hint=tuple(idx_bufs)))


@L.handles("stablehlo.dynamic_slice")
def _dynamic_slice(ctx, op):
    in_shape, _, in_dt = L.tensor_info(op.operands[0].type)
    out_shape, out_n, _ = L.tensor_info(op.results[0].type)
    rank = len(in_shape)
    index_ops = list(op.operands[1:])
    if len(index_ops) != rank:
        raise L.LoweringError("dynamic_slice: start-index count != operand rank")
    in_strides = _row_major_strides(in_shape)
    clamp_max = [in_shape[d] - out_shape[d] for d in range(rank)]
    idx_bufs, idx_offs, is64 = _index_info(ctx, index_ops)
    dst = ctx.new_buffer(out_n, in_dt)
    _emit_dyn(ctx, L.OP_DYNAMIC_SLICE, out_shape, in_strides, clamp_max,
              idx_bufs, idx_offs, is64, dst, ctx.buf_for(op.operands[0]), out_n)
    ctx.value_to_buf[op.results[0]] = dst


@L.handles("stablehlo.dynamic_update_slice")
def _dynamic_update_slice(ctx, op):
    in_shape, in_n, in_dt = L.tensor_info(op.operands[0].type)
    upd_shape, upd_n, _ = L.tensor_info(op.operands[1].type)
    rank = len(in_shape)
    index_ops = list(op.operands[2:])
    if len(index_ops) != rank:
        raise L.LoweringError(
            "dynamic_update_slice: start-index count != operand rank")
    out_strides = _row_major_strides(in_shape)
    clamp_max = [in_shape[d] - upd_shape[d] for d in range(rank)]
    idx_bufs, idx_offs, is64 = _index_info(ctx, index_ops)
    out_buf = ctx.new_buffer(in_n, in_dt)
    # 1) copy operand -> output via an identity gather (dtype-agnostic).
    copy_aux = ctx.add_aux([rank] + list(in_shape) + list(out_strides) + [0])
    ctx.emit(L.Instr(L.OP_GATHER_STRIDED, dst=out_buf,
                     a=ctx.buf_for(op.operands[0]), n=in_n, aux=copy_aux))
    # 2) scatter the update at the runtime base offset (WAW on out_buf barriers
    #    it after the copy).
    _emit_dyn(ctx, L.OP_DYNAMIC_UPDATE_SLICE, upd_shape, out_strides, clamp_max,
              idx_bufs, idx_offs, is64, out_buf, ctx.buf_for(op.operands[1]),
              upd_n)
    ctx.value_to_buf[op.results[0]] = out_buf


# --- tensor-opcode semantics ------------------------------------------------

def _read_aux(rt, base):
    rank = rt.aux[base]
    dims = [rt.aux_i32(base + 1 + d) for d in range(rank)]
    strides = [rt.aux_i32(base + 1 + rank + d) for d in range(rank)]
    clamp_max = [rt.aux_i32(base + 1 + 2 * rank + d) for d in range(rank)]
    idx_bufid = [rt.aux[base + 1 + 4 * rank + d] for d in range(rank)]
    return rank, dims, strides, clamp_max, idx_bufid


def _base_off(rt, rank, strides, clamp_max, idx_bufid):
    base = 0
    for d in range(rank):
        s = int(rt.view(idx_bufid[d])[0])
        s = 0 if s < 0 else (clamp_max[d] if s > clamp_max[d] else s)
        base += s * strides[d]
    return base


def _affine(base, rank, dims, strides, out_index):
    idx = out_index.astype(np.int64).copy()
    offs = np.full(len(out_index), base, dtype=np.int64)
    for d in range(rank - 1, -1, -1):
        offs += (idx % dims[d]) * strides[d]
        idx //= dims[d]
    return offs


def _dyn_slice_interp(ins, rt):
    rank, dims, strides, cmax, idxbuf = _read_aux(rt, ins.aux)
    base = _base_off(rt, rank, strides, cmax, idxbuf)
    offs = _affine(base, rank, dims, strides, np.arange(ins.n))
    rt.view(ins.dst, ins.n)[:] = rt.view(ins.a)[offs]


def _dyn_slice_tile_sim(task, entry, rt):
    n = task.p1
    lo = entry.tile_lo * rt.tile_size
    hi = min(entry.tile_hi * rt.tile_size, n)
    if lo >= hi:
        return
    rank, dims, strides, cmax, idxbuf = _read_aux(rt, task.p0)
    base = _base_off(rt, rank, strides, cmax, idxbuf)
    offs = _affine(base, rank, dims, strides, np.arange(lo, hi))
    rt.view(task.dst)[lo:hi] = rt.view(task.a)[offs]


def _dyn_update_interp(ins, rt):
    rank, dims, strides, cmax, idxbuf = _read_aux(rt, ins.aux)
    base = _base_off(rt, rank, strides, cmax, idxbuf)
    offs = _affine(base, rank, dims, strides, np.arange(ins.n))
    rt.view(ins.dst)[offs] = rt.view(ins.a, ins.n)


def _dyn_update_tile_sim(task, entry, rt):
    n = task.p1
    lo = entry.tile_lo * rt.tile_size
    hi = min(entry.tile_hi * rt.tile_size, n)
    if lo >= hi:
        return
    rank, dims, strides, cmax, idxbuf = _read_aux(rt, task.p0)
    base = _base_off(rt, rank, strides, cmax, idxbuf)
    offs = _affine(base, rank, dims, strides, np.arange(lo, hi))
    rt.view(task.dst)[offs] = rt.view(task.a)[lo:hi]


def _dyn_reads(ins):
    return {ins.a} | set(ins.reads_hint)


def _dyn_slice_to_task(ins) -> Task:
    return Task(TILE_DYN_GATHER, dst=ins.dst, a=ins.a, b=0, p0=ins.aux,
                p1=ins.n)


def _dyn_update_to_task(ins) -> Task:
    return Task(TILE_DYN_SCATTER, dst=ins.dst, a=ins.a, b=ins.a, p0=ins.aux,
                p1=ins.n)


opsem.register(L.OP_DYNAMIC_SLICE, to_task=_dyn_slice_to_task,
               interp=_dyn_slice_interp, reads=_dyn_reads)
opsem.register_tile_sim(TILE_DYN_GATHER, _dyn_slice_tile_sim)

opsem.register(L.OP_DYNAMIC_UPDATE_SLICE, to_task=_dyn_update_to_task,
               interp=_dyn_update_interp, reads=_dyn_reads)
opsem.register_tile_sim(TILE_DYN_SCATTER, _dyn_update_tile_sim)
