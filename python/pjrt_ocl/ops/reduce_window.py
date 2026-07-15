"""reduce_window (pooling) via the windowed-reduction tile op (OP_REDUCE_WINDOW).

stablehlo.reduce_window slides a window over the input and reduces each window
with a reducer region (add -> sum-pool, maximum -> max-pool, minimum -> min-pool)
into one output element:

    out[o] = reduce_{w in window} in[o*window_stride + w - pad_low]

reducing only in-bounds positions (padding elements equal the reducer identity,
so they are skipped — correct because the init value is asserted to be that
identity, exactly like ops/reduce.py).

Coverage (intentionally narrow, matching kernels/ops/reduce_window.cl):

  SUPPORTED: single input/output; kind sum/max/min; no base or window dilation;
  VALID or explicit non-negative padding; element dtype f32 / i32.

  REJECTED (LoweringError): base_dilations or window_dilations != 1; negative
  (cropping) padding; a reducer body other than a single add/maximum/minimum;
  multiply (product) pooling; variadic reduce_window; other dtypes.

aux layout (MUST match reduce_window.cl):
    kind, rank, out_dims[rank], win_dims[rank], win_strides[rank],
    pad_low[rank], in_dims[rank], in_strides[rank]
"""
from __future__ import annotations

import numpy as np

from .. import lowering as L
from .. import opsem
from ..scheduler import Task, TILE_RED_WINDOW
from .reduce import _assert_identity, _read_scalar_const

# kind: 0 sum, 1 max, 2 min (aligns with reduce.py's SUM/MAX/MIN codes)
_KIND_BY_REGION_OP = {
    "stablehlo.add": 0,
    "stablehlo.maximum": 1,
    "stablehlo.minimum": 2,
}


def _row_major_strides(shape):
    s = [0] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        s[i] = acc
        acc *= shape[i]
    return s


def _opt_i64(op, name, rank, default):
    from jaxlib.mlir import ir
    try:
        attr = op.attributes[name]
    except KeyError:
        return [default] * rank
    return [int(x) for x in ir.DenseI64ArrayAttr(attr)]


def _padding(op, rank):
    from jaxlib.mlir import ir
    try:
        attr = op.attributes["padding"]
    except KeyError:
        return [0] * rank, [0] * rank
    pad = np.asarray(ir.DenseIntElementsAttr(attr)).reshape(rank, 2)
    return [int(pad[d, 0]) for d in range(rank)], \
           [int(pad[d, 1]) for d in range(rank)]


def _read_init(value, dt):
    """Read the reduce_window init scalar. jax feeds it through an identity
    broadcast_in_dim (scalar -> scalar), so unwrap single-operand shape wrappers
    back to the defining stablehlo.constant."""
    for _ in range(8):
        owner = value.owner
        op = owner.operation if hasattr(owner, "operation") else owner
        name = getattr(op, "name", None)
        if name == "stablehlo.constant":
            return _read_scalar_const(value, dt)
        if name in ("stablehlo.broadcast_in_dim", "stablehlo.reshape",
                    "stablehlo.convert") and len(op.operands) == 1:
            value = op.operands[0]
            continue
        break
    raise L.LoweringError(
        "reduce_window: init value is not a compile-time constant "
        "(cannot verify it is the reduction identity)")


def _classify(op):
    rblk = op.regions[0].blocks[0]
    compute = [x.operation for x in rblk.operations
               if x.operation.name not in ("stablehlo.return", "func.return")]
    if len(compute) != 1 or compute[0].name not in _KIND_BY_REGION_OP:
        raise L.LoweringError(
            f"reduce_window: unsupported reducer body "
            f"{[o.name for o in compute]} (expected one add/maximum/minimum)")
    return _KIND_BY_REGION_OP[compute[0].name]


@L.handles("stablehlo.reduce_window")
def _reduce_window(ctx, op):
    if len(op.operands) != 2 or len(op.results) != 1:
        raise L.LoweringError(
            "reduce_window: only single-input reductions are supported")
    in_shape, in_n, in_dt = L.tensor_info(op.operands[0].type)
    out_shape, out_n, _ = L.tensor_info(op.results[0].type)
    rank = len(in_shape)
    if in_dt not in (L.DT_F32, L.DT_I32, L.DT_U32):
        raise L.LoweringError(
            f"reduce_window: element dtype {in_dt} unsupported (f32/i32 only)")

    win = _opt_i64(op, "window_dimensions", rank, 1)
    wstr = _opt_i64(op, "window_strides", rank, 1)
    base_dil = _opt_i64(op, "base_dilations", rank, 1)
    win_dil = _opt_i64(op, "window_dilations", rank, 1)
    if any(v != 1 for v in base_dil) or any(v != 1 for v in win_dil):
        raise L.LoweringError(
            "reduce_window: base/window dilations unsupported (must be 1)")
    pad_low, pad_high = _padding(op, rank)
    if any(v < 0 for v in pad_low) or any(v < 0 for v in pad_high):
        raise L.LoweringError(
            "reduce_window: negative (cropping) padding unsupported")

    kind = _classify(op)
    _assert_identity(kind, _read_init(op.operands[1], in_dt), in_dt)

    in_strides = _row_major_strides(in_shape)
    aux = ([kind, rank] + list(out_shape) + list(win) + list(wstr) +
           list(pad_low) + list(in_shape) + list(in_strides))
    aux_off = ctx.add_aux(aux)
    dst = ctx.new_buffer(out_n, in_dt)
    ctx.emit(L.Instr(L.OP_REDUCE_WINDOW, dst=dst,
                     a=ctx.buf_for(op.operands[0]), n=out_n, aux=aux_off))
    ctx.value_to_buf[op.results[0]] = dst


# --- tensor-opcode semantics ------------------------------------------------

def _read_win_aux(rt, base):
    kind = rt.aux_i32(base)
    rank = rt.aux_i32(base + 1)
    o = base + 2

    def rd(off):
        return [rt.aux_i32(off + d) for d in range(rank)]

    return (kind, rank, rd(o), rd(o + rank), rd(o + 2 * rank),
            rd(o + 3 * rank), rd(o + 4 * rank), rd(o + 5 * rank))


def _redwin_range(rt, base, out_index, src, dt):
    kind, rank, odims, wdims, wstr, plow, idims, istr = _read_win_aux(rt, base)
    n = len(out_index)
    isint = np.issubdtype(dt, np.integer)
    if kind == 0:
        ident = 0
    elif kind == 1:
        ident = np.iinfo(dt).min if isint else -np.inf
    else:
        ident = np.iinfo(dt).max if isint else np.inf
    acc = np.full(n, ident, dtype=dt)

    rem = out_index.astype(np.int64).copy()
    ocoord = [None] * rank
    for d in range(rank - 1, -1, -1):
        ocoord[d] = rem % odims[d]
        rem //= odims[d]

    wcount = 1
    for d in range(rank):
        wcount *= wdims[d]
    for w in range(wcount):
        rw = w
        wc = [0] * rank
        for d in range(rank - 1, -1, -1):
            wc[d] = rw % wdims[d]
            rw //= wdims[d]
        off = np.zeros(n, dtype=np.int64)
        inb = np.ones(n, dtype=bool)
        for d in range(rank):
            ic = ocoord[d] * wstr[d] + wc[d] - plow[d]
            inb &= (ic >= 0) & (ic < idims[d])
            off += ic * istr[d]
        vals = src[np.where(inb, off, 0)]
        if kind == 0:
            acc = acc + np.where(inb, vals, 0).astype(dt)
        elif kind == 1:
            acc = np.where(inb, np.maximum(acc, vals), acc)
        else:
            acc = np.where(inb, np.minimum(acc, vals), acc)
    return acc.astype(dt)


def _redwin_interp(ins, rt):
    dst = rt.view(ins.dst, ins.n)
    dst[:] = _redwin_range(rt, ins.aux, np.arange(ins.n), rt.view(ins.a),
                           dst.dtype)


def _redwin_tile_sim(task, entry, rt):
    n = task.p1
    lo = entry.tile_lo * rt.tile_size
    hi = min(entry.tile_hi * rt.tile_size, n)
    if lo >= hi:
        return
    dst = rt.view(task.dst)
    dst[lo:hi] = _redwin_range(rt, task.p0, np.arange(lo, hi),
                               rt.view(task.a), dst.dtype)


def _redwin_to_task(ins) -> Task:
    return Task(TILE_RED_WINDOW, dst=ins.dst, a=ins.a, b=0, p0=ins.aux,
                p1=ins.n)


opsem.register(L.OP_REDUCE_WINDOW, to_task=_redwin_to_task,
               interp=_redwin_interp, reads=lambda ins: {ins.a})
opsem.register_tile_sim(TILE_RED_WINDOW, _redwin_tile_sim)
