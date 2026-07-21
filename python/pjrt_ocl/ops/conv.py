"""stablehlo.convolution — direct N-D convolution (OP_CONV / TILE_CONV, §39).

Covers the canonical XLA layout: input NHWC-style [b, spatial..., f], kernel
HWIO-style [spatial..., i, o], output NHWC-style [b, spatial..., f]. Spatial
rank 1..4. Supports window strides, explicit/SAME padding (non-negative), and
rhs (kernel) dilation. Each output element serially accumulates over the kernel
window and input channels:

    out[b, osp, oc] = sum_{win, ic}
        in[b, osp*stride + win*dil - pad_low, ic] * w[win, ic, oc]

taps landing in the (implicit-zero) padding halo are skipped.

REJECTED (LoweringError), so unsupported programs fail loudly rather than
miscompute: non-f32 dtype; feature_group_count / batch_group_count != 1; lhs
(base) dilation != 1 (transposed conv); window reversal; negative padding; a
non-canonical dimension_numbers layout (anything other than NHWC input / HWIO
kernel / NHWC output).

aux layout (MUST match conv.cl):
    sdim, Cin, Cout,
    out_spatial[sdim], win[sdim], stride[sdim], pad_low[sdim], dil[sdim],
    in_spatial[sdim]
"""
from __future__ import annotations

import numpy as np

from .. import lowering as L
from .. import opsem
from ..scheduler import Task, TILE_CONV


def _i64_array(op, name, sdim, default):
    from jaxlib.mlir import ir
    try:
        attr = op.attributes[name]
    except KeyError:
        return [default] * sdim
    return [int(x) for x in ir.DenseI64ArrayAttr(attr)]


def _bool_array(op, name, sdim):
    from jaxlib.mlir import ir
    try:
        attr = op.attributes[name]
    except KeyError:
        return [False] * sdim
    return [bool(x) for x in ir.DenseBoolArrayAttr(attr)]


def _int_attr(op, name, default):
    from jaxlib.mlir import ir
    try:
        return int(ir.IntegerAttr(op.attributes[name]))
    except KeyError:
        return default


def _padding(op, sdim):
    from jaxlib.mlir import ir
    try:
        attr = op.attributes["padding"]
    except KeyError:
        return [0] * sdim, [0] * sdim
    pad = np.asarray(ir.DenseIntElementsAttr(attr)).reshape(sdim, 2)
    return [int(pad[d, 0]) for d in range(sdim)], \
           [int(pad[d, 1]) for d in range(sdim)]


def _check_canonical(dn, rank, sdim):
    """Require input NHWC / kernel HWIO / output NHWC (the layout conv.cl
    assumes). Raise otherwise — coverage is intentionally this one layout."""
    ok = (
        dn.input_batch_dimension == 0
        and dn.input_feature_dimension == rank - 1
        and list(dn.input_spatial_dimensions) == list(range(1, sdim + 1))
        and list(dn.kernel_spatial_dimensions) == list(range(sdim))
        and dn.kernel_input_feature_dimension == sdim
        and dn.kernel_output_feature_dimension == sdim + 1
        and dn.output_batch_dimension == 0
        and dn.output_feature_dimension == rank - 1
        and list(dn.output_spatial_dimensions) == list(range(1, sdim + 1))
    )
    if not ok:
        raise L.LoweringError(
            "convolution: only canonical NHWC input / HWIO kernel / NHWC output "
            "layout is supported")


@L.handles("stablehlo.convolution")
def _convolution(ctx, op):
    from jaxlib.mlir.dialects import stablehlo as shlo

    if len(op.operands) != 2 or len(op.results) != 1:
        raise L.LoweringError("convolution: expected 2 operands, 1 result")
    in_shape, _, in_dt = L.tensor_info(op.operands[0].type)
    w_shape, _, w_dt = L.tensor_info(op.operands[1].type)
    out_shape, out_n, out_dt = L.tensor_info(op.results[0].type)
    if in_dt != L.DT_F32 or w_dt != L.DT_F32 or out_dt != L.DT_F32:
        raise L.LoweringError("convolution: only f32 is supported")

    rank = len(in_shape)
    sdim = rank - 2
    if sdim < 1 or sdim > 4:
        raise L.LoweringError(
            f"convolution: spatial rank {sdim} unsupported (1..4)")

    if _int_attr(op, "feature_group_count", 1) != 1:
        raise L.LoweringError("convolution: feature_group_count != 1 unsupported")
    if _int_attr(op, "batch_group_count", 1) != 1:
        raise L.LoweringError("convolution: batch_group_count != 1 unsupported")

    dn = shlo.ConvDimensionNumbers(op.attributes["dimension_numbers"])
    _check_canonical(dn, rank, sdim)

    lhs_dil = _i64_array(op, "lhs_dilation", sdim, 1)
    if any(v != 1 for v in lhs_dil):
        raise L.LoweringError(
            "convolution: lhs_dilation != 1 (transposed conv) unsupported")
    if any(_bool_array(op, "window_reversal", sdim)):
        raise L.LoweringError("convolution: window reversal unsupported")

    stride = _i64_array(op, "window_strides", sdim, 1)
    rhs_dil = _i64_array(op, "rhs_dilation", sdim, 1)
    pad_low, pad_high = _padding(op, sdim)
    if any(v < 0 for v in pad_low) or any(v < 0 for v in pad_high):
        raise L.LoweringError("convolution: negative padding unsupported")

    # canonical layouts: in [B, S..., Cin], w [W..., Cin, Cout], out [B, O..., Cout]
    in_spatial = [int(in_shape[1 + d]) for d in range(sdim)]
    Cin = int(in_shape[rank - 1])
    win = [int(w_shape[d]) for d in range(sdim)]
    Cout = int(w_shape[sdim + 1])
    out_spatial = [int(out_shape[1 + d]) for d in range(sdim)]

    aux = ([sdim, Cin, Cout] + out_spatial + win + list(stride) +
           list(pad_low) + list(rhs_dil) + in_spatial)
    aux_off = ctx.add_aux(aux)
    dst = ctx.new_buffer(out_n, L.DT_F32)
    ctx.emit(L.Instr(L.OP_CONV, dst=dst,
                     a=ctx.buf_for(op.operands[0]),
                     b=ctx.buf_for(op.operands[1]), n=out_n, aux=aux_off))
    ctx.value_to_buf[op.results[0]] = dst


# --- tensor-opcode semantics (numpy reference, mirrors conv.cl) --------------

def _read_conv_aux(rt, base):
    sdim = rt.aux_i32(base)
    Cin = rt.aux_i32(base + 1)
    Cout = rt.aux_i32(base + 2)
    o = base + 3

    def rd(k):
        return [rt.aux_i32(o + k * sdim + d) for d in range(sdim)]

    return (sdim, Cin, Cout, rd(0), rd(1), rd(2), rd(3), rd(4), rd(5))


def _conv_range(rt, base, out_index, inp, wts):
    """Compute output flat elements `out_index` (vectorized over that set)."""
    sdim, Cin, Cout, ospat, win, strd, plow, dil, ispat = _read_conv_aux(rt, base)
    n = len(out_index)

    # input strides (row-major over [B, S..., Cin])
    in_stride = [0] * sdim
    acc = Cin
    for d in range(sdim - 1, -1, -1):
        in_stride[d] = acc
        acc *= ispat[d]
    in_batch_stride = acc

    # decode output flat index over [B, O..., Cout]
    rem = out_index.astype(np.int64).copy()
    oc = rem % Cout
    rem //= Cout
    ocoord = [None] * sdim
    for d in range(sdim - 1, -1, -1):
        ocoord[d] = rem % ospat[d]
        rem //= ospat[d]
    b = rem  # batch

    wcount = 1
    for d in range(sdim):
        wcount *= win[d]

    accv = np.zeros(n, dtype=np.float64)
    for w in range(wcount):
        rw = w
        wc = [0] * sdim
        for d in range(sdim - 1, -1, -1):
            wc[d] = rw % win[d]
            rw //= win[d]
        inoff = b * in_batch_stride
        inb = np.ones(n, dtype=bool)
        for d in range(sdim):
            ic = ocoord[d] * strd[d] + wc[d] * dil[d] - plow[d]
            inb &= (ic >= 0) & (ic < ispat[d])
            inoff = inoff + ic * in_stride[d]
        wbase = w * Cin * Cout + oc  # w[win=w, ic=0, oc]
        for c in range(Cin):
            src = inp[np.where(inb, inoff + c, 0)]
            accv += np.where(inb, src * wts[wbase + c * Cout], 0.0)
    return accv.astype(np.float32)


def _conv_interp(ins, rt):
    dst = rt.view(ins.dst, ins.n)
    dst[:] = _conv_range(rt, ins.aux, np.arange(ins.n),
                         rt.view(ins.a), rt.view(ins.b))


def _conv_tile_sim(task, entry, rt):
    n = task.p1
    lo = entry.tile_lo * rt.tile_size
    hi = min(entry.tile_hi * rt.tile_size, n)
    if lo >= hi:
        return
    dst = rt.view(task.dst)
    dst[lo:hi] = _conv_range(rt, task.p0, np.arange(lo, hi),
                             rt.view(task.a), rt.view(task.b))


def _conv_to_task(ins) -> Task:
    return Task(TILE_CONV, dst=ins.dst, a=ins.a, b=ins.b, p0=ins.aux, p1=ins.n)


opsem.register(L.OP_CONV, to_task=_conv_to_task, interp=_conv_interp,
               reads=lambda ins: {ins.a, ins.b})
opsem.register_tile_sim(TILE_CONV, _conv_tile_sim)
