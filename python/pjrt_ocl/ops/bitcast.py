"""bitcast_convert — reinterpret the BITS of a value as another same-size dtype.

stablehlo.bitcast_convert (jax.lax.bitcast_convert_type) is NOT a numeric cast:
it relabels the raw bytes of each element as a different dtype of the SAME
element size (f32<->i32<->u32; f64<->i64; f16<->bf16 storage). Unlike convert
(which rounds/truncates the numeric value), bitcast copies the word unchanged.

Coverage: same element-size only. StableHLO also allows width-changing
bitcasts (e.g. i32 -> 4xi8) which add or drop a trailing dimension; those need a
gather-like repack and are rejected here (LoweringError) until i8/i16 land.

Lowering: OP_BITCAST, a new EW subop (SUB_BITCAST). The result buffer carries
the OUTPUT dtype; the scheduler sets task.dtype = output, task.adtype = input
(from the operand buffer). The VM path is a typed memcpy of the 2/4/8-byte word.
"""
from __future__ import annotations

from .. import lowering as L
from .. import opsem
from ..scheduler import Task, TILE_EW

# vm_common.cl SUB_BITCAST (subop after SUB_CONVERT=23). Must match the kernel.
SUB_BITCAST = 24


@L.handles("stablehlo.bitcast_convert")
def _bitcast(ctx, op):
    in_shape, n_in, in_dt = L.tensor_info(op.operands[0].type)
    out_shape, n_out, out_dt = L.tensor_info(op.results[0].type)
    in_sz = L.DTYPE_NUMPY[in_dt].itemsize
    out_sz = L.DTYPE_NUMPY[out_dt].itemsize
    if in_sz != out_sz or n_in != n_out:
        raise L.LoweringError(
            "bitcast_convert: only same-element-size bitcasts are supported "
            f"(got {L.DTYPE_NUMPY[in_dt]} -> {L.DTYPE_NUMPY[out_dt]}); "
            "width-changing bitcasts (add/drop a dim) are not yet implemented")
    dst = ctx.new_buffer(n_out, out_dt)
    ctx.emit(L.Instr(L.OP_BITCAST, dst=dst, a=ctx.buf_for(op.operands[0]),
                     n=n_out))
    ctx.value_to_buf[op.results[0]] = dst


def _bitcast_to_task(ins) -> Task:
    return Task(TILE_EW, dst=ins.dst, a=ins.a, b=0, p0=SUB_BITCAST, p1=ins.n)


def _bitcast_interp(ins, rt) -> None:
    # numpy reference: src.view(dst_dtype) — reinterpret bytes, no conversion.
    src = rt.view(ins.a, ins.n)
    dst = rt.view(ins.dst, ins.n)
    dst[:] = src.view(dst.dtype)


def _bitcast_ew_sim(a, b, task, rt, lo, hi):
    dst_dt = rt.view(task.dst).dtype
    return a.view(dst_dt)


opsem.register(L.OP_BITCAST, to_task=_bitcast_to_task, interp=_bitcast_interp,
               reads=lambda ins: {ins.a})
opsem.register_ew_sim(SUB_BITCAST, _bitcast_ew_sim)
