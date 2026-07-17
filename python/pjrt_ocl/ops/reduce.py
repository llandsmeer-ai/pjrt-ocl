"""Reductions via the flat two-phase REDUCE_PART/REDUCE_COMB tile ops.

stablehlo.reduce is a region op: ``op.regions[0]`` holds the reducer body whose
single compute op selects the reduction kind (add -> sum, maximum -> max,
minimum -> min, multiply -> prod). ``dimensions`` (i64 array) names the reduced
axes; operand[0] is the input tensor, operand[1] the init value (must be the
kind's identity).

Coverage (v1 — intentionally narrow, monotone increment):

  SUPPORTED: **full reductions** only — ``dimensions`` == every input axis, so
  the result is a scalar. That is exactly what the FLAT vm2.cl REDUCE model can
  do correctly: split the contiguous input into ``chunk``-element runs, reduce
  each run to one partial (REDUCE_PART), then fold all partials (REDUCE_COMB).

  REJECTED (LoweringError): any partial-axis reduction (``dimensions`` is a
  strict subset of the axes). The flat model cannot express a strided reduction
  over a non-contiguous sub-space; doing so needs a permuting GATHER first
  (another family) — deferred. See NOTES.md.

Decomposition — a reduction lowers to **two OP_REDUCE tensor instructions**:

  1. PART:  dst = partials buffer (n_parts elems), a = input.  Maps to a
            TILE_REDUCE_PART task (p0=n, p1=chunk, p2=kind).
  2. COMB:  dst = output scalar, a = partials.  Maps to a TILE_REDUCE_COMB task
            (p0=n_parts, p1=kind).

The scheduler's ``_instr_to_task`` returns ONE task per instruction and cannot
be edited, so the two phases MUST be two instructions. COMB reads the buffer
PART writes, so the dependency analysis puts them in different dataflow levels
and the scheduler drops a global BARRIER between them automatically — the
part-phase completes on every lane before any lane starts the comb-phase.

Both instructions share opcode OP_REDUCE (26); the phase + kind ride in
``Instr.imm`` (``imm = (phase << 2) | kind``) so the phase-free scheduler
mapper, the tensor interp and the tile simulators can all recover them from the
instruction alone. ``chunk`` is NOT transported: it is a deterministic function
of ``n`` (``_chunk_for``) recomputed identically by every consumer, so the
partials count derived by the handler, the mapper's ``p1`` and ``n_tiles()`` all
agree by construction.
"""
from __future__ import annotations

import math

from .. import lowering as L
from .. import opsem
from ..scheduler import (Task, TILE_REDUCE_PART, TILE_REDUCE_COMB, TILE_RED_SEG,
                         TILE_SOFTMAX_SEG, TILE_LAYERNORM_SEG, TILE_SIZE)

# reduction kinds (docs/vmprogram.md v2.1 REDUCE table)
SUM, MAX, MIN, PROD = 0, 1, 2, 3

# imm phase tag (low 2 bits carry kind, bit 2 carries phase)
PHASE_PART = 0
PHASE_COMB = 1

_KIND_BY_REGION_OP = {
    "stablehlo.add": SUM,
    "stablehlo.maximum": MAX,
    "stablehlo.minimum": MIN,
    "stablehlo.multiply": PROD,
}

# integer dtypes whose reduce uses integer accumulation + iinfo identities
# (matches reduce.cl's i32/u32 path). bool/f-types use the float path.
_INT_DTYPES = frozenset({L.DT_I32, L.DT_U32, L.DT_I64})


def _chunk_for(n: int) -> int:
    """Deterministic chunk size for a flat n-element reduction. Recomputed
    identically by the handler, the scheduler mapper, the tensor interp and the
    tile sims — so nothing has to be transported through the instruction."""
    if n <= 0:
        return 1
    return max(TILE_SIZE, math.ceil(n / 256))


def _n_parts(n: int) -> int:
    return max(1, math.ceil(n / _chunk_for(n)))


def _encode_imm(phase: int, kind: int) -> int:
    return (phase << 2) | kind


def _decode_imm(imm: int) -> tuple[int, int]:
    return (imm >> 2) & 1, imm & 3


# --- stablehlo handler ------------------------------------------------------

def _read_scalar_const(value, dt: int):
    """Read a scalar stablehlo.constant that defines `value` (int or float,
    per the reduction's dtype). Returns a python int/float."""
    import numpy as np
    from jaxlib.mlir import ir
    owner = value.owner
    op = owner.operation if hasattr(owner, "operation") else owner
    if getattr(op, "name", None) != "stablehlo.constant":
        raise L.LoweringError(
            "reduce: init value is not a compile-time constant "
            "(cannot verify it is the reduction identity)")
    attr = op.attributes["value"]
    if dt in _INT_DTYPES:
        vals = np.asarray(ir.DenseIntElementsAttr(attr)).reshape(-1)
    else:
        vals = np.asarray(ir.DenseFPElementsAttr(attr)).reshape(-1)
    if vals.size != 1:
        raise L.LoweringError("reduce: init value is not a scalar")
    return vals[0].item()


def _assert_identity(kind: int, init, dt: int) -> None:
    if dt in _INT_DTYPES:
        import numpy as np
        info = np.iinfo(L.DTYPE_NUMPY[dt])
        ok = ((kind == SUM and init == 0) or
              (kind == PROD and init == 1) or
              (kind == MAX and init == info.min) or
              (kind == MIN and init == info.max))
    else:
        ok = ((kind == SUM and init == 0.0) or
              (kind == PROD and init == 1.0) or
              (kind == MAX and math.isinf(init) and init < 0) or
              (kind == MIN and math.isinf(init) and init > 0))
    if not ok:
        raise L.LoweringError(
            f"reduce: init value {init} is not the identity for kind {kind} "
            "(non-identity inits are not supported)")


@L.handles("stablehlo.reduce")
def _reduce(ctx, op):
    from jaxlib.mlir import ir
    if len(op.operands) != 2 or len(op.results) != 1:
        raise L.LoweringError(
            "reduce: only single-input, single-output reductions are supported "
            f"(got {len(op.operands)} operands, {len(op.results)} results)")

    in_shape, n_in, in_dt = L.tensor_info(op.operands[0].type)
    in_rank = len(in_shape)
    dims = sorted(int(x) for x in ir.DenseI64ArrayAttr(op.attributes["dimensions"]))

    # classify the reduction kind from the single compute op in the region body
    rblk = op.regions[0].blocks[0]
    body_ops = [x.operation for x in rblk.operations]
    compute = [o for o in body_ops
               if o.name not in ("stablehlo.return", "func.return")]
    if len(compute) != 1 or compute[0].name not in _KIND_BY_REGION_OP:
        names = [o.name for o in compute]
        raise L.LoweringError(
            f"reduce: unsupported reducer body {names} (expected exactly one of "
            f"{sorted(_KIND_BY_REGION_OP)})")
    kind = _KIND_BY_REGION_OP[compute[0].name]

    _assert_identity(kind, _read_scalar_const(op.operands[1], in_dt), in_dt)

    in_buf = ctx.buf_for(op.operands[0])

    # Partial reduction over a CONTIGUOUS INNERMOST suffix of axes (softmax /
    # layernorm reduce the last axis): output element o = reduce of the seg
    # contiguous inputs at [o*seg, (o+1)*seg). One TILE_RED_SEG task, tiled over
    # the n_out outputs. Non-suffix axis sets still need a permuting transpose
    # first (deferred).
    if dims != list(range(in_rank)):
        if dims != list(range(in_rank - len(dims), in_rank)):
            raise L.LoweringError(
                f"reduce: only full or innermost-suffix reductions are "
                f"supported; got dimensions={dims} of rank-{in_rank} (a "
                "non-suffix axis set needs a transpose first — not yet done).")
        seg = 1
        for d in dims:
            seg *= in_shape[d]
        n_out = n_in // seg
        out = ctx.new_buffer(n_out, in_dt)
        ctx.emit(L.Instr(L.OP_REDUCE_SEG, dst=out, a=in_buf, b=in_buf,
                         n=n_out, imm=(kind << 28) | seg))
        ctx.value_to_buf[op.results[0]] = out
        return

    n_parts = _n_parts(n_in)
    n_parts = _n_parts(n_in)

    # PART: input -> n_parts partials (one per chunk). Partials + output carry
    # the INPUT dtype so the VM reduces/accumulates in that dtype.
    partials = ctx.new_buffer(n_parts, in_dt)
    ctx.emit(L.Instr(L.OP_REDUCE, dst=partials, a=in_buf, b=in_buf,
                     n=n_in, imm=_encode_imm(PHASE_PART, kind)))
    # COMB: partials -> scalar output
    out = ctx.new_buffer(1, in_dt)
    ctx.emit(L.Instr(L.OP_REDUCE, dst=out, a=partials, b=partials,
                     n=n_parts, imm=_encode_imm(PHASE_COMB, kind)))
    ctx.value_to_buf[op.results[0]] = out


# --- numpy reference semantics ----------------------------------------------

def _reduce_np(arr, kind):
    if kind == SUM:
        return arr.sum()
    if kind == MAX:
        return arr.max()
    if kind == MIN:
        return arr.min()
    return arr.prod()


def _reduce_interp(ins, rt):
    """Tensor validator (a). Each OP_REDUCE instr computes its own phase."""
    phase, kind = _decode_imm(ins.imm)
    if phase == PHASE_PART:
        n = ins.n
        chunk = _chunk_for(n)
        n_parts = _n_parts(n)
        src = rt.view(ins.a, n)
        out = rt.view(ins.dst, n_parts)
        for t in range(n_parts):
            lo = t * chunk
            hi = min(lo + chunk, n)
            out[t] = _reduce_np(src[lo:hi], kind)
    else:  # PHASE_COMB
        n_parts = ins.n
        src = rt.view(ins.a, n_parts)
        rt.view(ins.dst, 1)[0] = _reduce_np(src[:n_parts], kind)


def _reduce_reads(ins) -> set[int]:
    return {ins.a}


# --- scheduler mapping ------------------------------------------------------

def _reduce_to_task(ins) -> Task:
    phase, kind = _decode_imm(ins.imm)
    if phase == PHASE_PART:
        # p0=n, p1=chunk, p2=kind — matches vm2.cl TOP_RED_PART exactly
        return Task(TILE_REDUCE_PART, dst=ins.dst, a=ins.a, b=0,
                    p0=ins.n, p1=_chunk_for(ins.n), p2=kind, p3=0)
    # p0=n_parts, p1=kind — matches vm2.cl TOP_RED_COMB exactly
    return Task(TILE_REDUCE_COMB, dst=ins.dst, a=ins.a, b=0,
                p0=ins.n, p1=kind, p2=0, p3=0)


# --- schedule simulator (validator b) ---------------------------------------

def _reduce_part_sim(task, entry, rt):
    """Write one partial per tile in [tile_lo, tile_hi)."""
    n, chunk, kind = task.p0, task.p1, task.p2
    src = rt.view(task.a)
    out = rt.view(task.dst)
    for tile in range(entry.tile_lo, entry.tile_hi):
        lo = tile * chunk
        hi = min(lo + chunk, n)
        if lo >= hi:
            continue
        out[tile] = _reduce_np(src[lo:hi], kind)


def _reduce_comb_sim(task, entry, rt):
    """Fold all partials into the scalar output (single tile)."""
    if not (entry.tile_lo <= 0 < entry.tile_hi):
        return
    n_parts, kind = task.p0, task.p1
    src = rt.view(task.a)
    rt.view(task.dst)[0] = _reduce_np(src[:n_parts], kind)


opsem.register(L.OP_REDUCE, to_task=_reduce_to_task, interp=_reduce_interp,
               reads=_reduce_reads)
opsem.register_tile_sim(TILE_REDUCE_PART, _reduce_part_sim)
opsem.register_tile_sim(TILE_REDUCE_COMB, _reduce_comb_sim)


# --- segmented reduce (partial innermost-suffix reduction) -------------------

def _redseg_decode(imm: int) -> tuple[int, int]:
    return imm >> 28, imm & 0x0FFFFFFF          # kind, seg


def _redseg_to_task(ins) -> Task:
    kind, seg = _redseg_decode(ins.imm)
    # p0 = n_out (tiling), p1 = seg, p2 = kind
    return Task(TILE_RED_SEG, dst=ins.dst, a=ins.a, b=0,
                p0=ins.n, p1=seg, p2=kind, p3=0)


def _redseg_interp(ins, rt) -> None:
    kind, seg = _redseg_decode(ins.imm)
    n_out = ins.n
    src = rt.view(ins.a, n_out * seg).reshape(n_out, seg)
    rt.view(ins.dst, n_out)[:] = _reduce_np_axis(src, kind)


def _reduce_np_axis(arr, kind):
    if kind == SUM:
        return arr.sum(-1)
    if kind == MAX:
        return arr.max(-1)
    if kind == MIN:
        return arr.min(-1)
    return arr.prod(-1)


def _redseg_sim(task, entry, rt):
    # one segment per tile: entry covers segments [tile_lo, tile_hi)
    n_out, seg, kind = task.p0, task.p1, task.p2
    src = rt.view(task.a)
    out = rt.view(task.dst)
    for o in range(entry.tile_lo, min(entry.tile_hi, n_out)):
        out[o] = _reduce_np(src[o * seg:(o + 1) * seg], kind)


opsem.register(L.OP_REDUCE_SEG, to_task=_redseg_to_task, interp=_redseg_interp,
               reads=_reduce_reads)
opsem.register_tile_sim(TILE_RED_SEG, _redseg_sim)


# --- fused segmented norms (softmax / layernorm core), §19 -------------------
# Both are recognized in lowering (_fuse_norm) from the reduce->broadcast idiom
# and lowered to one fused local-memory op: imm = seg (innermost axis length),
# n = n_out (segment count); layernorm carries eps in imm2 (f32 bits). The numpy
# reference here MUST mirror the kernel exactly (vmo_softmax_seg / _layernorm_seg
# in reduce.cl) so the dual validators agree bit-for-bit on re-parsed bytecode.

def _f32_from_bits(bits: int):
    import numpy as np
    return np.frombuffer(np.uint32(bits & 0xFFFFFFFF).tobytes(), "<f4")[0]


def _softmax_np(x):   # x: (..., seg); stable softmax matching the kernel
    import numpy as np
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(-1, keepdims=True)


def _layernorm_np(x, eps):   # one-pass var = E[x^2] - E[x]^2, matching the kernel
    import numpy as np
    mu = x.mean(-1, keepdims=True)
    var = (x * x).mean(-1, keepdims=True) - mu * mu
    return (x - mu) / np.sqrt(var + eps)


def _softmax_to_task(ins) -> Task:
    return Task(TILE_SOFTMAX_SEG, dst=ins.dst, a=ins.a, b=0,
                p0=ins.n, p1=ins.imm)


def _layernorm_to_task(ins) -> Task:
    return Task(TILE_LAYERNORM_SEG, dst=ins.dst, a=ins.a, b=0,
                p0=ins.n, p1=ins.imm, p2=ins.imm2)   # p2 = eps f32 bits


def _softmax_interp(ins, rt) -> None:
    n_out, seg = ins.n, ins.imm
    src = rt.view(ins.a, n_out * seg).reshape(n_out, seg)
    rt.view(ins.dst, n_out * seg)[:] = _softmax_np(src).reshape(-1)


def _layernorm_interp(ins, rt) -> None:
    n_out, seg = ins.n, ins.imm
    eps = float(_f32_from_bits(ins.imm2))
    src = rt.view(ins.a, n_out * seg).reshape(n_out, seg)
    rt.view(ins.dst, n_out * seg)[:] = _layernorm_np(src, eps).reshape(-1)


def _norm_reads(ins) -> set:
    return {ins.a}


def _softmax_sim(task, entry, rt):
    import numpy as np
    n_out, seg = task.p0, task.p1
    src = rt.view(task.a)
    out = rt.view(task.dst)
    for o in range(entry.tile_lo, min(entry.tile_hi, n_out)):
        out[o * seg:(o + 1) * seg] = _softmax_np(src[o * seg:(o + 1) * seg])


def _layernorm_sim(task, entry, rt):
    n_out, seg = task.p0, task.p1
    eps = float(_f32_from_bits(task.p2))
    src = rt.view(task.a)
    out = rt.view(task.dst)
    for o in range(entry.tile_lo, min(entry.tile_hi, n_out)):
        out[o * seg:(o + 1) * seg] = _layernorm_np(src[o * seg:(o + 1) * seg], eps)


opsem.register(L.OP_SOFTMAX, to_task=_softmax_to_task, interp=_softmax_interp,
               reads=_norm_reads)
opsem.register(L.OP_LAYERNORM, to_task=_layernorm_to_task,
               interp=_layernorm_interp, reads=_norm_reads)
opsem.register_tile_sim(TILE_SOFTMAX_SEG, _softmax_sim)
opsem.register_tile_sim(TILE_LAYERNORM_SEG, _layernorm_sim)
