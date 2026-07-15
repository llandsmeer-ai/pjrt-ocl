"""concatenate + pad via the strided-scatter tile op (OP_SCATTER / TILE_SCATTER).

concatenate(inputs, dim): allocate the output, then scatter each input into its
disjoint slice along `dim`. pad: fill the output with the pad value, then scatter
the operand into the interior. Each source scatters `dst[out_off + affine(i)] =
a[i]`; multiple scatters write the same output buffer but disjoint regions (the
scheduler WAW-serializes them, which is correct).
"""
from __future__ import annotations

import math

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


def _emit_scatter(ctx, out_buf, in_buf, in_shape, out_strides, out_off):
    n = math.prod(in_shape) if in_shape else 1
    rank = len(in_shape)
    aux = [rank] + list(in_shape) + list(out_strides) + [out_off]
    aux_off = ctx.add_aux(aux)
    ctx.emit(L.Instr(L.OP_SCATTER, dst=out_buf, a=in_buf, n=n, aux=aux_off))


@L.handles("stablehlo.concatenate")
def _concatenate(ctx, op):
    from jaxlib.mlir import ir
    out_shape, out_n, dtype = L.tensor_info(op.results[0].type)
    dim = int(ir.IntegerAttr(op.attributes["dimension"]).value)
    out_strides = _row_major_strides(out_shape)
    out_buf = ctx.new_buffer(out_n, dtype)
    offset = 0                                   # running position along `dim`
    for operand in op.operands:
        in_shape, _, _ = L.tensor_info(operand.type)
        _emit_scatter(ctx, out_buf, ctx.buf_for(operand), in_shape,
                      out_strides, offset * out_strides[dim])
        offset += in_shape[dim]
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
