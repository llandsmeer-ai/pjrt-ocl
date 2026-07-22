"""stablehlo.scatter — general DATA-DEPENDENT scatter (OP_SCATTER_INDEX, §42).

The mirror of OP_GATHER_INDEX (ops/gather_index.py). Where the general gather
reads each output element's operand base offset from a runtime start_indices
tensor, the general scatter iterates over UPDATE elements and COMBINES each into
an operand location whose base offset is likewise read at runtime from a
scatter_indices tensor. This is what `jnp.zeros().at[idx].set/add/max/min(x)`
and MJX physics lower to.

Distinct from OP_SCATTER (ops/concat.py / making.py), a compile-time-affine
scatter used by concatenate/pad, and from OP_DYNAMIC_UPDATE_SLICE, a single
runtime base offset. Here every update element's target is data-dependent.

stablehlo.scatter semantics (docs/stablehlo-notes.md):
  inputs[R_op] (the operand), scatter_indices[R_si], updates[R_upd],
  dnums = {update_window_dims, inserted_window_dims, scatter_dims_to_operand_dims,
  index_vector_dim}, plus an update_computation region.

  The updates tensor splits into WINDOW dims (update_window_dims) and SCATTER
  dims (the rest, ascending). Window dims map 1:1 (ascending) to the NON-inserted
  operand dims. Scatter dims map 1:1 (ascending) to the batch dims of
  scatter_indices (all its dims except index_vector_dim).

  For update element i:
    start_k = clamp(scatter_indices[scatter_coords, k], 0, dim - window_size)  for
              operand dim scatter_dims_to_operand_dims[k]
    operand_index[d] = start[d] + window_coord[d]   (window_coord 0 for inserted /
                       non-window dims; start 0 for dims not in the index map)
    operand[operand_index] = update_computation(operand[operand_index], updates[i])

Reduced to a flat affine form the kernel evaluates per update element (row-major):
    op_off(i)  = Σ_e coord_e(i)·op_stride[e] + Σ_k clamp(S_k)·idx_op_stride[k]
    si_base(i) = Σ_e coord_e(i)·si_stride[e]
    S_k        = scatter_indices[si_base(i) + k·si_vec_stride]
  where op_stride[e]!=0 only on WINDOW update dims, si_stride[e]!=0 only on
  SCATTER update dims. See kernels/ops/scatter.cl (vmo_scatter_index_tile).

The operand is first copied into the output (an identity OP_GATHER_STRIDED); the
WAW on the output buffer barriers the scatter after the copy so it sees the full
operand. Update combine kinds: 0 set (overwrite, last-writer), 1 add, 2 max,
3 min. add/max/min run through GLOBAL ATOMICS so duplicate indices accumulate
exactly regardless of tiling (matches stablehlo for the commutative kinds); set
is a plain store (stablehlo leaves duplicate-index overwrite order unspecified).

aux layout (MUST match the kernel):
    out_rank(=R_upd), nidx, si_vec_stride, is64,
    idx_byteoff (placeholder, loader-patched), idx_bufid, kind,
    upd_dims[out_rank], op_stride[out_rank], si_stride[out_rank],
    idx_op_stride[nidx], clamp_max[nidx]

SUPPORTED: single operand/update/result, static shapes, i32/i64 indices; set for
any Tier-1/2 element dtype; add/max/min for f32/i32/u32 (the 4-byte types with
core atomics). REJECTED (raised, not silently wrong): variadic scatter,
operand/indices batching dims, non-(set/add/max/min) update-computation, and
add/max/min on non-4-byte dtypes.
"""
from __future__ import annotations

import numpy as np

from .. import lowering as L
from .. import opsem
from ..scheduler import Task, TILE_SCATTER_INDEX

# update-combine kinds (MUST match kernels/ops/scatter.cl)
SET, ADD, MAX, MIN = 0, 1, 2, 3

_KIND_BY_REGION_OP = {
    "stablehlo.add": ADD,
    "stablehlo.maximum": MAX,
    "stablehlo.minimum": MIN,
}


def _row_major_strides(shape):
    s = [0] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        s[i] = acc
        acc *= shape[i]
    return s


def _classify_update(op):
    """Return the combine kind (SET/ADD/MAX/MIN) from the scatter's
    update_computation region. `set` is a bare `return %update` (the second
    block arg); add/max/min are a single binary op of the two args."""
    rblk = op.regions[0].blocks[0]
    args = list(rblk.arguments)
    body = [x.operation for x in rblk.operations]
    compute = [o for o in body
               if o.name not in ("stablehlo.return", "func.return")]
    rets = [o for o in body if o.name in ("stablehlo.return", "func.return")]
    if len(rets) != 1 or len(args) < 2:
        raise L.LoweringError("scatter: unrecognized update-computation shape")
    retval = rets[0].operands[0]
    if not compute:
        # overwrite: the region just returns the incoming update (2nd arg).
        if retval == args[1]:
            return SET
        if retval == args[0]:
            raise L.LoweringError(
                "scatter: update-computation returns the existing value "
                "(no-op scatter) — unsupported")
        raise L.LoweringError("scatter: unsupported overwrite update-computation")
    if len(compute) == 1 and compute[0].name in _KIND_BY_REGION_OP:
        return _KIND_BY_REGION_OP[compute[0].name]
    names = [o.name for o in compute]
    raise L.LoweringError(
        f"scatter: unsupported update-computation body {names} "
        "(expected set / add / maximum / minimum)")


def _plan(op):
    """Compute the affine scatter descriptor. Returns (op_shape, op_n, op_dt,
    upd_shape, upd_n, op_stride, si_stride, idx_op_stride, clamp_max, nidx,
    si_vec_stride, is64, kind)."""
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import stablehlo

    if len(op.operands) != 3 or len(op.results) != 1:
        raise L.LoweringError(
            "scatter: only single-input scatter is supported "
            f"(got {len(op.operands)} operands, {len(op.results)} results)")

    op_shape, op_n, op_dt = L.tensor_info(op.operands[0].type)
    si_shape, _, si_dt = L.tensor_info(op.operands[1].type)
    upd_shape, upd_n, upd_dt = L.tensor_info(op.operands[2].type)

    dn = stablehlo.ScatterDimensionNumbers(
        op.attributes["scatter_dimension_numbers"])
    update_window_dims = list(dn.update_window_dims)
    inserted = list(dn.inserted_window_dims)
    sdo = list(dn.scattered_dims_to_operand_dims)
    index_vector_dim = int(dn.index_vector_dim)
    if list(dn.input_batching_dims) or list(dn.scatter_indices_batching_dims):
        raise L.LoweringError("scatter: operand/indices batching dims unsupported")

    if si_dt not in (L.DT_I32, L.DT_I64):
        raise L.LoweringError(f"scatter: scatter-index dtype {si_dt} unsupported")
    is64 = 1 if si_dt == L.DT_I64 else 0

    kind = _classify_update(op)
    if kind != SET and op_dt not in (L.DT_F32, L.DT_I32, L.DT_U32):
        raise L.LoweringError(
            f"scatter: add/max/min combine on dtype {op_dt} unsupported "
            "(only f32/i32/u32 have core atomics; use set or a 4-byte dtype)")

    R_op, R_si, R_upd = len(op_shape), len(si_shape), len(upd_shape)
    op_strides = _row_major_strides(op_shape)
    si_strides = _row_major_strides(si_shape)

    nidx = len(sdo)
    if index_vector_dim == R_si:            # implicit trailing size-1 vector
        if nidx != 1:
            raise L.LoweringError("scatter: implicit index_vector_dim needs nidx==1")
        si_vec_stride = 0
        si_batch_dims = list(range(R_si))
    else:
        if si_shape[index_vector_dim] != nidx:
            raise L.LoweringError(
                "scatter: index_vector_dim size != len(scatter_dims_to_operand_dims)")
        si_vec_stride = si_strides[index_vector_dim]
        si_batch_dims = [j for j in range(R_si) if j != index_vector_dim]

    # window update dims (ascending) -> non-inserted operand dims (ascending).
    window_operand_dims = [d for d in range(R_op) if d not in inserted]
    uw_sorted = sorted(update_window_dims)
    if len(uw_sorted) != len(window_operand_dims):
        raise L.LoweringError(
            "scatter: update_window_dims / non-inserted operand rank mismatch")

    # window size along each operand dim: inserted dims are size-1 windows;
    # non-inserted dims take the size of their mapped update window dim.
    window_size = [1] * R_op
    op_stride = [0] * R_upd
    for w, opd in enumerate(window_operand_dims):
        window_size[opd] = upd_shape[uw_sorted[w]]
        op_stride[uw_sorted[w]] = op_strides[opd]

    # scatter update dims (ascending) -> scatter_indices batch dims (in order).
    scatter_update_dims = [d for d in range(R_upd)
                           if d not in set(update_window_dims)]
    if len(scatter_update_dims) != len(si_batch_dims):
        raise L.LoweringError(
            "scatter: scatter-dim rank mismatch (updates vs indices)")
    si_stride = [0] * R_upd
    for su, sib in zip(scatter_update_dims, si_batch_dims):
        si_stride[su] = si_strides[sib]

    idx_op_stride = [op_strides[sdo[k]] for k in range(nidx)]
    clamp_max = [op_shape[sdo[k]] - window_size[sdo[k]] for k in range(nidx)]

    return (op_shape, op_n, op_dt, upd_shape, upd_n, op_stride, si_stride,
            idx_op_stride, clamp_max, nidx, si_vec_stride, is64, kind)


@L.handles("stablehlo.scatter")
def _scatter(ctx, op):
    (op_shape, op_n, op_dt, upd_shape, upd_n, op_stride, si_stride,
     idx_op_stride, clamp_max, nidx, si_vec_stride, is64, kind) = _plan(op)
    out_buf = ctx.new_buffer(op_n, op_dt)
    # 1) copy operand -> output via an identity gather (dtype-agnostic). WAW on
    #    out_buf barriers the scatter after the copy so it sees the full operand.
    in_strides = _row_major_strides(op_shape)
    copy_aux = ctx.add_aux([len(op_shape)] + list(op_shape) + list(in_strides) + [0])
    ctx.emit(L.Instr(L.OP_GATHER_STRIDED, dst=out_buf,
                     a=ctx.buf_for(op.operands[0]), n=op_n, aux=copy_aux))
    # 2) scatter the updates (combine) at the runtime base offsets.
    idx_buf = ctx.buf_for(op.operands[1])
    aux = ([len(upd_shape), nidx, si_vec_stride, is64, 0, idx_buf, kind]
           + list(upd_shape) + list(op_stride) + list(si_stride)
           + list(idx_op_stride) + list(clamp_max))
    aux_off = ctx.add_aux(aux)
    ctx.emit(L.Instr(L.OP_SCATTER_INDEX, dst=out_buf,
                     a=ctx.buf_for(op.operands[2]), n=upd_n, aux=aux_off,
                     reads_hint=(idx_buf,)))
    ctx.value_to_buf[op.results[0]] = out_buf


# --- tensor-opcode semantics (numpy reference + schedule simulator) ---------

def _read_aux(rt, base):
    out_rank = rt.aux[base]
    nidx = rt.aux[base + 1]
    si_vec_stride = rt.aux_i32(base + 2)
    idx_bufid = rt.aux[base + 5]
    kind = rt.aux[base + 6]
    p = base + 7
    upd_dims = [rt.aux_i32(p + e) for e in range(out_rank)]
    op_stride = [rt.aux_i32(p + out_rank + e) for e in range(out_rank)]
    si_stride = [rt.aux_i32(p + 2 * out_rank + e) for e in range(out_rank)]
    q = p + 3 * out_rank
    idx_op_stride = [rt.aux_i32(q + k) for k in range(nidx)]
    clamp_max = [rt.aux_i32(q + nidx + k) for k in range(nidx)]
    return (out_rank, nidx, si_vec_stride, idx_bufid, kind, upd_dims, op_stride,
            si_stride, idx_op_stride, clamp_max)


def _offsets(base, rt, upd_index):
    (out_rank, nidx, si_vec_stride, idx_bufid, kind, upd_dims, op_stride,
     si_stride, idx_op_stride, clamp_max) = _read_aux(rt, base)
    idx = upd_index.astype(np.int64).copy()
    off = np.zeros(len(upd_index), dtype=np.int64)
    si_base = np.zeros(len(upd_index), dtype=np.int64)
    for e in range(out_rank - 1, -1, -1):
        coord = idx % upd_dims[e]
        idx //= upd_dims[e]
        off += coord * op_stride[e]
        si_base += coord * si_stride[e]
    starts = rt.view(idx_bufid)   # buffer's native int dtype (i32/i64)
    for k in range(nidx):
        s = starts[si_base + k * si_vec_stride].astype(np.int64)
        s = np.clip(s, 0, clamp_max[k])
        off += s * idx_op_stride[k]
    return off, kind


def _apply(dst, off, upd, kind):
    if kind == SET:
        dst[off] = upd
    elif kind == ADD:
        np.add.at(dst, off, upd)
    elif kind == MAX:
        np.maximum.at(dst, off, upd)
    else:
        np.minimum.at(dst, off, upd)


def _interp(ins, rt):
    off, kind = _offsets(ins.aux, rt, np.arange(ins.n))
    _apply(rt.view(ins.dst), off, rt.view(ins.a, ins.n), kind)


def _tile_sim(task, entry, rt):
    n = task.p1
    lo = entry.tile_lo * rt.tile_size
    hi = min(entry.tile_hi * rt.tile_size, n)
    if lo >= hi:
        return
    off, kind = _offsets(task.p0, rt, np.arange(lo, hi))
    _apply(rt.view(task.dst), off, rt.view(task.a)[lo:hi], kind)


def _reads(ins):
    return {ins.a} | set(ins.reads_hint)


def _to_task(ins) -> Task:
    return Task(TILE_SCATTER_INDEX, dst=ins.dst, a=ins.a, b=0, p0=ins.aux,
                p1=ins.n)


opsem.register(L.OP_SCATTER_INDEX, to_task=_to_task, interp=_interp, reads=_reads)
opsem.register_tile_sim(TILE_SCATTER_INDEX, _tile_sim)
