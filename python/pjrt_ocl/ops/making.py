"""iota + convert — the "making" op family: value-generating / dtype-changing
ops with no meaningful GATHER structure of their own.

--- iota (stablehlo.iota -> OP_IOTA_DIM) -----------------------------------

stablehlo.iota has NO operands: the output shape and the `iota_dimension`
(i64) attribute fully determine it. dst[i] = the i-th output element's
coordinate along axis `iota_dimension` (row-major unravel of the flat index),
as f32.

aux layout (docs/vmprogram.md OP_IOTA_DIM row; MUST match vm2.cl TOP_IOTA_DIM
exactly, which reads `x[0]`=rank, `x[1..rank]`=out_dims, `x[1+rank]`=dim):
    rank u32, out_dims i32[rank], dim i32

--- convert (stablehlo.convert -> OP_COPY_F32 / SUB_COPY) ------------------

Per the int-as-f32 policy (docs/vmprogram.md "Dtype policy"): the arena is
all f32; a real v3 would store integers as the exact float value they
represent, so an int<->f32 convert would be semantics-preserving metadata
only in the *identity* direction (int -> f32 is a copy) but NOT in the
*narrowing* direction (f32 -> int truncates toward zero, which changes the
stored value and needs a round-toward-zero op vm2.cl doesn't have — only
SUB_FLOOR/SUB_CEIL/SUB_SIGN, and floor != trunc for negative operands).

However this repo's current type system (`lowering.tensor_info`) accepts
ONLY F32Type ranked tensors, enforced uniformly at every boundary (function
arguments, constants, every other op's operands/results) — nothing in the
existing, unmodified pipeline can ever produce or consume a non-f32-typed
SSA value. Empirically (see NOTES in the PR/report), jax itself never emits
a same-dtype (f32 -> f32) `stablehlo.convert` through `jax.jit(f).lower(...)`
either — `x.astype(jnp.float32)` on an already-float32 array is elided at
the jax tracing level before stablehlo is ever produced, even via
`lax.convert_element_type` or a direct `convert_element_type_p.bind`. So:

  - f32 -> f32 (only reachable convert in a real program; also the only
    direction that is a true identity under any dtype policy): implemented,
    OP_COPY_F32 / SUB_COPY.
  - anything else (int/bool operand or result dtype, in either direction):
    NOT implemented — raises LoweringError with a specific message. This
    is not reachable from a real jax-compiled program today (blocked
    upstream by tensor_info on argument/producer types regardless of this
    module), so the check here is defensive/documentation rather than a
    live gap; revisit together with a real integer arena dtype (v3).
"""
from __future__ import annotations

import math

from .. import lowering as L
from .. import opsem
from ..scheduler import Task, TILE_EW, TILE_IOTA_DIM

# vm2.cl `SUB_COPY` (docs/vmprogram.md EW subop table: 0 add, 1 mul, 2 sub,
# 3 div, 4 max, 5 min, 6 pow, 7 copy, ...).
EW_SUB_COPY = 7


# --- stablehlo.iota -----------------------------------------------------

def _emit_iota(ctx, out_shape, dim: int, dtype: int = L.DT_F32) -> int:
    n = math.prod(out_shape) if out_shape else 1
    rank = len(out_shape)
    aux_words = [rank] + list(out_shape) + [dim]
    aux_off = ctx.add_aux(aux_words)
    dst = ctx.new_buffer(n, dtype)
    ctx.emit(L.Instr(L.OP_IOTA_DIM, dst=dst, n=n, aux=aux_off))
    return dst


@L.handles("stablehlo.iota")
def _iota(ctx, op):
    from jaxlib.mlir import ir
    # iota has no operands: shape comes only from the result type. The dtype
    # matters: integer iota (int32/uint32, e.g. the threefry RNG counter) must
    # write integer coordinates, not their f32 bit-pattern.
    out_shape, _, dtype = L.tensor_info(op.results[0].type)
    dim = int(ir.IntegerAttr(op.attributes["iota_dimension"]).value)
    if not (0 <= dim < len(out_shape)):
        raise L.LoweringError(
            f"stablehlo.iota: iota_dimension {dim} out of range for rank "
            f"{len(out_shape)}")
    dst = _emit_iota(ctx, out_shape, dim, dtype)
    ctx.value_to_buf[op.results[0]] = dst


def _iota_to_task(ins) -> Task:
    return Task(TILE_IOTA_DIM, dst=ins.dst, a=0, b=0, p0=ins.aux, p1=ins.n,
                p2=0, p3=0)


def _iota_coords(base: int, rt, out_index):
    """coordinate along `dim` for a batch of flat output indices (numpy int
    array); base is the aux word offset. Mirrors shape.py's _gather_coords
    unravel loop, but keeps the coordinate of the target axis instead of an
    affine-strided offset."""
    import numpy as np
    rank = rt.aux[base]
    out_dims = [rt.aux_i32(base + 1 + d) for d in range(rank)]
    dim = rt.aux_i32(base + 1 + rank)
    if rank == 0:
        return np.zeros(len(out_index), dtype=np.int64)
    rem = out_index.astype(np.int64).copy()
    val = np.zeros(len(out_index), dtype=np.int64)
    for d in range(rank - 1, -1, -1):
        coord = rem % out_dims[d]
        rem //= out_dims[d]
        if d == dim:
            val = coord
    return val


def _iota_interp(ins, rt):
    import numpy as np
    n = ins.n
    out = rt.view(ins.dst, n)
    val = _iota_coords(ins.aux, rt, np.arange(n))
    out[:] = val   # cast to the dst buffer's own dtype (f32 or int32/uint32)


def _iota_reads(ins) -> set[int]:
    return set()


def _iota_tile_sim(task, entry, rt):
    """Validator b: fill iota output elements for tiles [tile_lo, tile_hi)."""
    import numpy as np
    n = task.p1
    lo = entry.tile_lo * rt.tile_size
    hi = min(entry.tile_hi * rt.tile_size, n)
    if lo >= hi:
        return
    val = _iota_coords(task.p0, rt, np.arange(lo, hi))
    rt.view(task.dst)[lo:hi] = val   # dst buffer dtype (f32 or int32/uint32)


opsem.register(L.OP_IOTA_DIM, to_task=_iota_to_task, interp=_iota_interp,
               reads=_iota_reads)
opsem.register_tile_sim(TILE_IOTA_DIM, _iota_tile_sim)


# --- stablehlo.convert (f32 -> f32 identity only; see module docstring) ----

@L.handles("stablehlo.convert")
def _convert(ctx, op):
    # Real dtype cast: the VM reads the input as its dtype (task.adtype) and
    # writes the result dtype (task.dtype) with a C cast (float->int truncates
    # toward zero, matching stablehlo). f64 casts are device-gated at load.
    in_shape, n_elems, _ = L.tensor_info(op.operands[0].type)
    out_shape, _, _ = L.tensor_info(op.results[0].type)
    if in_shape != out_shape:
        raise L.LoweringError("convert: shape mismatch (not a pure cast)")
    dst = ctx.new_buffer(n_elems, L.tensor_info(op.results[0].type)[2])
    ctx.emit(L.Instr(L.OP_CONVERT, dst=dst, a=ctx.buf_for(op.operands[0]),
                     n=n_elems))
    ctx.value_to_buf[op.results[0]] = dst


def _convert_to_task(ins) -> Task:
    from ..scheduler import TILE_EW
    return Task(TILE_EW, dst=ins.dst, a=ins.a, b=ins.a, p0=23, p1=ins.n)


def _convert_interp(ins, rt) -> None:
    # dst view uses the result dtype; a view uses the operand dtype -> numpy
    # assignment performs the cast (float->int truncates toward zero via astype
    # after trunc? numpy astype truncates toward zero for float->int).
    import numpy as np
    src = rt.view(ins.a, ins.n)
    dst = rt.view(ins.dst, ins.n)
    if np.issubdtype(src.dtype, np.floating) and np.issubdtype(dst.dtype,
                                                              np.integer):
        dst[:] = np.trunc(src).astype(dst.dtype)
    else:
        dst[:] = src.astype(dst.dtype)


def _convert_ew_sim(a, b, task, rt, lo, hi):
    import numpy as np
    dst_dt = rt.view(task.dst).dtype
    if np.issubdtype(a.dtype, np.floating) and np.issubdtype(dst_dt, np.integer):
        return np.trunc(a).astype(dst_dt)
    return a.astype(dst_dt)


opsem.register(L.OP_CONVERT, to_task=_convert_to_task, interp=_convert_interp,
               reads=lambda ins: {ins.a})
opsem.register_ew_sim(23, _convert_ew_sim)   # SUB_CONVERT


def _copy_to_task(ins) -> Task:
    # p2 carries the operand-a VIEW aux-offset (0 = direct): _fuse_views may
    # fold a broadcast/transpose into a copy (SUB_COPY is in the device's
    # viewable range), e.g. a while carry initialized from a broadcast.
    return Task(TILE_EW, dst=ins.dst, a=ins.a, b=ins.a, p0=EW_SUB_COPY,
                p1=ins.n, p2=ins.imm, p3=ins.imm2)


def _copy_interp(ins, rt):
    rt.view(ins.dst, ins.n)[:] = rt.viewed(ins.a, ins.n, ins.imm)


def _copy_reads(ins) -> set[int]:
    return {ins.a}


opsem.register(L.OP_COPY_F32, to_task=_copy_to_task, interp=_copy_interp,
               reads=_copy_reads)
opsem.register_ew_sim(EW_SUB_COPY, lambda a, b, task, rt, lo, hi: a.copy())
