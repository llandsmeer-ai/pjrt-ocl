"""StableHLO -> VMProgram v3 TENSOR sections. Producer half of docs/vmprogram.md.

This module emits the tensor sections (the ISA of record). The v2.1 SCHEDULE
sections are produced by pjrt_ocl.scheduler from the VMProgram this returns;
`VMProgram.serialize(schedule)` appends them. A real v3 file always carries a
schedule (lower_service runs the scheduler); serialize(schedule=None) writes
tensor-only bytes for inspection / the reference-interpreter path.

Runs inside the compile-time subprocess (lower_service.py) spawned by the C++
plugin; also importable directly for tests/tooling. Uses jaxlib's bundled
StableHLO/MLIR python bindings, so it is version-matched to the host JAX by
construction (docs/decisions.md #2). Only numpy + jaxlib are imported, jaxlib
lazily (keeps subprocess startup light and lets vmreader stay jax-free).

VMProgram v1 binary layout (normative spec: docs/vmprogram.md; all integers
little-endian, file = header then sections in this order, each 8-byte aligned):

  header (40B):     magic u32 0x314D5056, version u32 =1, n_buffers u32,
                    n_instrs u32, n_consts u32, main_len u32, n_inputs u32,
                    n_outputs u32, arena_bytes u64
  buffer table:     n_buffers x { arena_byte_offset u64, size_bytes u64,
                    dtype u32 (0=f32), pad u32 }  — offsets 64B-aligned
  IO maps:          n_inputs x u32 buffer ids (PJRT argument order), then
                    n_outputs x u32 (result order); pad to 8B after each array
  IO shapes:        for each IO buffer (inputs then outputs):
                    { rank u32, pad u32, dims u64[rank] } — each entry 8B-aligned
  const pool:       n_consts x { buffer_id u32, byte_len u32, data[byte_len] },
                    each entry padded to 8B
  instructions:     n_instrs x 32B { op u32, dst u32, a u32, b u32, n u32,
                    imm u32, pad u32, pad u32 }; dst/a/b are BUFFER-TABLE
                    INDICES (except WHILE, whose a/b/n/imm are instruction
                    ranges); root list = instrs [0, main_len)

Supported ops: stablehlo.add / multiply / subtract / constant on static-shaped
f32 tensors (+ func.return / arg plumbing). Anything else raises LoweringError
listing the known ops. Per-op handlers live in OP_HANDLERS — add a dict entry
to support a new op.
"""
from __future__ import annotations

import dataclasses
import math
import os
import struct

import numpy as np

# --- format constants (docs/vmprogram.md) ----------------------------------

MAGIC = 0x314D5056  # b"VPM1" read little-endian
VERSION = 3         # v3 = v2 tensor sections + v2.1 schedule sections
ARENA_ALIGN = 64    # buffer arena offsets
SECTION_ALIGN = 8   # file sections / variable-length entries

# Dtype enum (docs/vmprogram.md). Tier 1 = 4-byte types the VM handles by
# bit-reinterpreting arena slots (as_int/as_float); the arena stays 4-byte
# slotted. Tier 2/3 (8-byte i64/f64, 2-byte f16/bf16, complex) need the
# byte-addressed arena and are added later.
DT_F32 = 0
DT_I32 = 1
DT_U32 = 2
DT_BOOL = 3          # stored as i32 0/1 in a 4-byte slot
# reserved for later tiers:
DT_I64 = 4
DT_F64 = 5
DT_F16 = 6
DT_BF16 = 7
DT_C64 = 8

import ml_dtypes as _ml_dtypes  # noqa: E402 (bf16 numpy dtype; ships with jax)

DTYPE_NUMPY = {
    DT_F32: np.dtype("<f4"),
    DT_I32: np.dtype("<i4"),
    DT_U32: np.dtype("<u4"),
    DT_BOOL: np.dtype("u1"),    # 1-byte 0/1, matching jax PRED
    DT_I64: np.dtype("<i8"),
    DT_F64: np.dtype("<f8"),
    DT_F16: np.dtype("<f2"),
    DT_BF16: np.dtype(_ml_dtypes.bfloat16),
}
# 4-byte dtypes (share tile size math); 8-byte need the wide arena path.
TIER1_DTYPES = frozenset({DT_F32, DT_I32, DT_U32, DT_BOOL})

OP_NOP = 0
OP_ADD_F32 = 1
OP_MUL_F32 = 2
OP_SUB_F32 = 3
OP_FILL_F32 = 4
OP_IOTA_F32 = 5
OP_LTS_F32 = 6
OP_WHILE = 7
# v2 tensor opcodes (docs/vmprogram.md "v2 deltas"). Binary/unary elementwise
# share the EW tile-op (subop chosen in the scheduler); the shaped ops carry
# metadata in the aux pool (Instr.aux = word offset).
OP_DIV_F32 = 8
OP_MAX_F32 = 9
OP_MIN_F32 = 10
OP_POW_F32 = 11
OP_COPY_F32 = 12
OP_NEG_F32 = 13
OP_EXP_F32 = 14
OP_LOG_F32 = 15
OP_SQRT_F32 = 16
OP_RSQRT_F32 = 17
OP_TANH_F32 = 18
OP_ABS_F32 = 19
OP_FLOOR_F32 = 20
OP_CEIL_F32 = 21
OP_SIGN_F32 = 22
OP_CMP_F32 = 23          # imm = predicate (0 EQ,1 NE,2 LT,3 LE,4 GT,5 GE)
OP_SELECT_F32 = 24       # imm = pred buffer id
OP_GATHER_STRIDED = 25   # aux: rank, out_dims[], in_strides[], src_off
OP_REDUCE = 26           # aux: kind,out_rank,out_dims[],kept_strides[],red_rank,red_dims[],red_strides[],src_off
OP_DOT = 27              # aux: M, N, K
OP_IOTA_DIM = 28         # aux: rank, out_dims[], dim
OP_IF = 29
OP_CONVERT = 30          # dtype cast: read a as operand dtype, write dst dtype
OP_BITCAST = 31          # bit reinterpret: copy a's bytes, relabel as dst dtype
# batch: elementwise coverage growth (docs/coverage-baseline.md "easy EW ops").
OP_LOG1P_F32 = 32        # stablehlo.log_plus_one
OP_EXPM1_F32 = 33        # stablehlo.exponential_minus_one
OP_CBRT_F32 = 34         # stablehlo.cbrt
OP_SIN_F32 = 35          # stablehlo.sine
OP_COS_F32 = 36          # stablehlo.cosine
OP_TAN_F32 = 37          # stablehlo.tan
OP_ROUND_NEAREST_EVEN_F32 = 38   # stablehlo.round_nearest_even
OP_ROUND_NEAREST_AFZ_F32 = 39    # stablehlo.round_nearest_afz
OP_ATAN2_F32 = 40        # stablehlo.atan2
OP_REMAINDER_F32 = 41    # stablehlo.remainder (C fmod semantics)
OP_AND = 42              # stablehlo.and (int32/bool)
OP_OR = 43               # stablehlo.or
OP_XOR = 44              # stablehlo.xor
OP_NOT = 45              # stablehlo.not
OP_IS_FINITE = 46        # stablehlo.is_finite (float -> bool)
OP_SCATTER = 47          # strided scatter (concatenate/pad); aux = rank,in_dims,out_strides,out_off
OP_DYNAMIC_SLICE = 48    # gather with a runtime base offset (aux carries start-scalar byte offsets)
OP_DYNAMIC_UPDATE_SLICE = 49  # scatter with a runtime base offset
OP_REDUCE_WINDOW = 50    # windowed reduction (pooling); aux = kind,rank,out/win/stride/pad/in dims+strides
OP_AFFINE_F32 = 51       # fused scalar affine d = a*s + t; imm = f32(s) bits, aux = f32(t) bits
OP_REDUCE_SEG = 52       # segmented reduce over the innermost `seg` elems; imm = (kind<<28)|seg
OP_FOR = 53              # fixed-trip loop: body = instrs [n, n+imm), b = trip count
OP_NAMES = {
    OP_NOP: "nop", OP_ADD_F32: "add_f32", OP_MUL_F32: "mul_f32",
    OP_SUB_F32: "sub_f32", OP_FILL_F32: "fill_f32", OP_IOTA_F32: "iota_f32",
    OP_LTS_F32: "lts_f32", OP_WHILE: "while",
    OP_DIV_F32: "div_f32", OP_MAX_F32: "max_f32", OP_MIN_F32: "min_f32",
    OP_POW_F32: "pow_f32", OP_COPY_F32: "copy_f32", OP_NEG_F32: "neg_f32",
    OP_EXP_F32: "exp_f32", OP_LOG_F32: "log_f32", OP_SQRT_F32: "sqrt_f32",
    OP_RSQRT_F32: "rsqrt_f32", OP_TANH_F32: "tanh_f32", OP_ABS_F32: "abs_f32",
    OP_FLOOR_F32: "floor_f32", OP_CEIL_F32: "ceil_f32", OP_SIGN_F32: "sign_f32",
    OP_CMP_F32: "cmp_f32", OP_SELECT_F32: "select_f32",
    OP_GATHER_STRIDED: "gather_strided", OP_REDUCE: "reduce", OP_DOT: "dot",
    OP_IOTA_DIM: "iota_dim", OP_IF: "if", OP_CONVERT: "convert",
    OP_BITCAST: "bitcast",
    OP_LOG1P_F32: "log1p_f32", OP_EXPM1_F32: "expm1_f32", OP_CBRT_F32: "cbrt_f32",
    OP_SIN_F32: "sin_f32", OP_COS_F32: "cos_f32", OP_TAN_F32: "tan_f32",
    OP_ROUND_NEAREST_EVEN_F32: "round_nearest_even_f32",
    OP_ROUND_NEAREST_AFZ_F32: "round_nearest_afz_f32",
    OP_ATAN2_F32: "atan2_f32", OP_REMAINDER_F32: "remainder_f32",
    OP_AND: "and", OP_OR: "or", OP_XOR: "xor", OP_NOT: "not",
    OP_IS_FINITE: "is_finite", OP_SCATTER: "scatter",
    OP_DYNAMIC_SLICE: "dynamic_slice",
    OP_DYNAMIC_UPDATE_SLICE: "dynamic_update_slice",
    OP_REDUCE_WINDOW: "reduce_window",
    OP_AFFINE_F32: "affine_f32",
    OP_REDUCE_SEG: "reduce_seg",
    OP_FOR: "for",
}

# v3 header: 48 bytes. After n_outputs, insert n_aux u32 + pad u32, then the
# arena_bytes u64 as before (docs/vmprogram.md "v2 deltas").
HEADER_STRUCT = struct.Struct("<IIIIIIIIIIQ")  # 48 bytes
BUFENT_STRUCT = struct.Struct("<QQII")         # 24 bytes
CONSTHDR_STRUCT = struct.Struct("<II")         # 8 bytes, then data
# instruction: { op, dst, a, b, n, imm, aux, pad1 } — pad0 renamed aux (v2).
INSTR_STRUCT = struct.Struct("<IIIIIIII")      # 32 bytes
assert HEADER_STRUCT.size == 48
assert BUFENT_STRUCT.size == 24
assert INSTR_STRUCT.size == 32


class LoweringError(NotImplementedError):
    """Valid program, but beyond current op/type coverage (service exit 2)."""


# --- in-memory model + writer ----------------------------------------------

@dataclasses.dataclass
class Buffer:
    arena_byte_offset: int
    size_bytes: int
    dtype: int = DT_F32


@dataclasses.dataclass
class Instr:
    op: int
    dst: int = 0
    a: int = 0
    b: int = 0
    n: int = 0
    imm: int = 0
    aux: int = 0        # v2: aux-pool word offset (pad0 renamed); 0 for EW ops
    imm2: int = 0       # second immediate (was pad1): OP_AFFINE_F32's t bits;
                        # 0 for every other op (kept a padding word there)
    # Extra buffer ids read besides a/b (e.g. dynamic_slice start-index scalars,
    # whose ids ride in the aux pool the scheduler can't inspect). Consumed by
    # the scheduler's dependency analysis only — NOT serialized.
    reads_hint: tuple = ()


def _pad8(out: bytearray) -> None:
    out += b"\0" * (-len(out) % SECTION_ALIGN)


@dataclasses.dataclass
class VMProgram:
    arena_bytes: int
    buffers: list[Buffer]
    inputs: list[int]                       # buffer ids, PJRT argument order
    outputs: list[int]                      # buffer ids, result order
    input_shapes: list[tuple[int, ...]]     # parallel to inputs
    output_shapes: list[tuple[int, ...]]    # parallel to outputs
    consts: list[tuple[int, bytes]]         # (buffer id, raw element bytes)
    instrs: list[Instr]                     # flat; root list = [0, main_len)
    main_len: int
    aux: list[int] = dataclasses.field(default_factory=list)  # aux pool, u32 words

    def serialize(self, schedule=None) -> bytes:
        """Serialize a v3 VMProgram: tensor sections, then (if given) the v2.1
        schedule sections. `schedule` is any object exposing
        `serialize_sections() -> bytes` (pjrt_ocl.scheduler.Schedule). A real
        v3 file always carries a schedule; None is allowed only for
        tensor-only inspection / the reference-interpreter path."""
        assert len(self.inputs) == len(self.input_shapes)
        assert len(self.outputs) == len(self.output_shapes)
        assert self.main_len <= len(self.instrs)
        out = bytearray()
        out += HEADER_STRUCT.pack(
            MAGIC, VERSION, len(self.buffers), len(self.instrs),
            len(self.consts), self.main_len, len(self.inputs),
            len(self.outputs), len(self.aux), 0, self.arena_bytes)
        # buffer table
        for b in self.buffers:
            assert b.arena_byte_offset % ARENA_ALIGN == 0, b
            out += BUFENT_STRUCT.pack(b.arena_byte_offset, b.size_bytes,
                                      b.dtype, 0)
        # IO maps: inputs, pad to 8B, outputs, pad to 8B
        for ids in (self.inputs, self.outputs):
            for buf_id in ids:
                out += struct.pack("<I", buf_id)
            _pad8(out)
        # IO shapes: inputs then outputs, {rank u32, pad u32, dims u64[rank]}
        for shape in (*self.input_shapes, *self.output_shapes):
            out += struct.pack("<II", len(shape), 0)
            for dim in shape:
                out += struct.pack("<Q", dim)
        # aux pool: n_aux x u32, padded to 8B (between IO shapes and const pool)
        for word in self.aux:
            out += struct.pack("<I", word & 0xFFFFFFFF)
        _pad8(out)
        # const pool
        for buf_id, data in self.consts:
            out += CONSTHDR_STRUCT.pack(buf_id, len(data))
            out += data
            _pad8(out)
        # instructions
        for ins in self.instrs:
            out += INSTR_STRUCT.pack(ins.op, ins.dst, ins.a, ins.b, ins.n,
                                     ins.imm, ins.aux, ins.imm2)
        # schedule sections (8B-aligned; instructions are 32B each => aligned)
        if schedule is not None:
            out += schedule.serialize_sections()
        return bytes(out)

    def dump(self) -> str:
        """Human-readable disassembly (debugging aid)."""
        lines = [f"VMProgram v{VERSION}: arena={self.arena_bytes}B "
                 f"buffers={len(self.buffers)} consts={len(self.consts)} "
                 f"instrs={len(self.instrs)} (main {self.main_len}) "
                 f"in={self.inputs} out={self.outputs}"]
        for i, b in enumerate(self.buffers):
            lines.append(f"  buf[{i}] off={b.arena_byte_offset} "
                         f"size={b.size_bytes} dt={b.dtype}")
        for pc, ins in enumerate(self.instrs):
            tag = " <main end>" if pc == self.main_len - 1 else ""
            lines.append(f"  [{pc:3d}] {OP_NAMES.get(ins.op, hex(ins.op)):8s} "
                         f"dst={ins.dst} a={ins.a} b={ins.b} n={ins.n} "
                         f"imm={ins.imm}{tag}")
        return "\n".join(lines)


# --- lowering: stablehlo ir.Module -> VMProgram -----------------------------

class _Ctx:
    """Per-module lowering state."""

    def __init__(self):
        self.buffers: list[Buffer] = []
        self.consts: list[tuple[int, bytes]] = []
        self.value_to_buf: dict = {}       # ir.Value -> buffer id
        self.instrs: list[Instr] = []
        self.inputs: list[int] = []
        self.outputs: list[int] = []
        self.input_shapes: list[tuple[int, ...]] = []
        self.output_shapes: list[tuple[int, ...]] = []
        self.aux: list[int] = []           # aux pool (u32 words)
        self.funcs: dict = {}              # sym_name -> func.func (call inlining)
        # deferred region lowering: stablehlo.while emits its cond/body sub-lists
        # AFTER the root list so all region instrs live at indices >= main_len
        # (docs/vmprogram.md root_len rule). Each item is a _WhileJob.
        self.region_queue: list = []
        # scalar-const folding (perf): a rank-0 f32 constant records its value
        # here; a broadcast_in_dim of it records the *result* value in
        # scalar_bcast. _elementwise_binop then folds `x * s` / `x + t` into a
        # single OP_AFFINE_F32 instead of materializing the broadcast + a full
        # binary op (see _compose_affines / _dce_nops post-passes).
        self.const_scalar: dict = {}       # ir.Value (rank-0 f32 const) -> float
        self.scalar_bcast: dict = {}       # ir.Value (broadcast result) -> float
        self._arena = 0

    def emit(self, instr: Instr) -> None:
        self.instrs.append(instr)

    def add_aux(self, words) -> int:
        """Append u32 words to the aux pool; return their starting word offset.
        Signed ints (strides, src_off) are stored two's-complement."""
        off = len(self.aux)
        self.aux.extend(w & 0xFFFFFFFF for w in words)
        return off

    def new_buffer(self, n_elems: int, dtype: int = DT_F32) -> int:
        size = n_elems * DTYPE_NUMPY[dtype].itemsize
        offset = self._arena
        # keep every offset 64B-aligned by advancing in 64B units
        self._arena += -(-size // ARENA_ALIGN) * ARENA_ALIGN
        self.buffers.append(Buffer(offset, size, dtype))
        return len(self.buffers) - 1

    def buf_for(self, value) -> int:
        try:
            return self.value_to_buf[value]
        except KeyError:
            raise LoweringError(f"no buffer for SSA value {value}") from None


def _elem_dtype(element_type) -> int:
    """MLIR element type -> our dtype enum. Raises for unsupported."""
    from jaxlib.mlir import ir
    if isinstance(element_type, ir.F32Type):
        return DT_F32
    if isinstance(element_type, ir.F64Type):
        return DT_F64                      # device-gated (cl_khr_fp64) at load
    if isinstance(element_type, ir.F16Type):
        return DT_F16                      # 2-byte, f32 compute (portable)
    if isinstance(element_type, ir.BF16Type):
        return DT_BF16                     # 2-byte, f32 compute (portable)
    if isinstance(element_type, ir.IntegerType):
        w = element_type.width
        if w == 1:
            return DT_BOOL                 # i1 predicate/mask
        if w == 32:
            # stablehlo integers are signless; treat as i32 (jax uses i32 for
            # indices/counters). unsigned is distinguished by the op, not type.
            return DT_I32
        if w == 64:
            return DT_I64                  # Tier 2
        raise LoweringError(
            f"unsupported integer width i{w} (have i1/i32/i64; i8/i16 later)")
    raise LoweringError(
        f"unsupported element type {element_type} "
        f"(have f32/f64/i32/i64/bool; f16/bf16/complex are later)")


def _tensor_info(mlir_type) -> tuple[tuple[int, ...], int, int]:
    """-> (shape, n_elems, dtype enum). Static-shape ranked tensors, Tier-1
    dtypes (f32/i32/bool)."""
    from jaxlib.mlir import ir
    # jaxlib 0.10.2 bindings auto-downcast types and lack Type.isinstance();
    # plain python isinstance() is the supported check (poc/03 NOTES #4).
    if not isinstance(mlir_type, ir.RankedTensorType):
        raise LoweringError(f"unsupported type (not a ranked tensor): {mlir_type}")
    t = mlir_type
    if any(t.is_dynamic_dim(i) for i in range(t.rank)):
        raise LoweringError(f"dynamic shapes unsupported: {mlir_type}")
    dtype = _elem_dtype(t.element_type)
    shape = tuple(t.shape)
    return shape, math.prod(shape) if t.rank else 1, dtype


# per-op handlers: stablehlo op name -> handler(ctx, op). Add entries to grow
# coverage; unknown ops raise LoweringError listing sorted(OP_HANDLERS).

OP_HANDLERS: dict = {}


def _handles(name):
    def deco(fn):
        OP_HANDLERS[name] = fn
        return fn
    return deco


# Public API for per-family op modules in pjrt_ocl.ops.* — register a
# stablehlo handler `fn(ctx, op)`. `ctx` exposes emit/new_buffer/buf_for/
# add_aux/value_to_buf; use tensor_info(type) for (shape, n_elems, dtype).
handles = _handles
tensor_info = _tensor_info


def _f32_bits(x: float) -> int:
    return int(np.float32(x).view(np.uint32))


def emit_affine(ctx: _Ctx, dst: int, x_buf: int, s: float, t: float,
                n: int) -> None:
    """Emit d = x*s + t (scalar immediates). b self-aliases a (unary convention:
    the scheduler's read-set + schedule simulator stay in-bounds); s/t ride in
    imm/imm2 as f32 bit patterns (OP_AFFINE_F32 to_task forwards them to p2/p3)."""
    ctx.emit(Instr(OP_AFFINE_F32, dst=dst, a=x_buf, b=x_buf, n=n,
                   imm=_f32_bits(s), imm2=_f32_bits(t)))


# multiply/add/subtract by a folded scalar constant -> (s, t) for `other`.
def _affine_st(opcode: int, other_is_rhs: bool, c: float):
    if opcode == OP_MUL_F32:
        return (c, 0.0)                       # x*c  (commutes)
    if opcode == OP_ADD_F32:
        return (1.0, c)                       # x+c  (commutes)
    # subtract does NOT commute: rhs scalar -> x - c ; lhs scalar -> c - x
    return (1.0, -c) if other_is_rhs else (-1.0, c)


def _elementwise_binop(opcode):
    def handler(ctx: _Ctx, op):
        _, n_elems, dtype = _tensor_info(op.results[0].type)
        for operand in op.operands:
            if _tensor_info(operand.type)[1:] != (n_elems, dtype):
                raise LoweringError(
                    f"{op.name}: operand/result shape mismatch "
                    f"(implicit broadcast?)")
        lhs, rhs = op.operands[0], op.operands[1]
        # Fold `tensor (op) scalar_const` into a single affine pass (f32 only).
        # Requires exactly one operand to be a folded scalar broadcast and the
        # other to be a genuine tensor (not itself a scalar const).
        if dtype == DT_F32:
            r_s = ctx.scalar_bcast.get(rhs)
            l_s = ctx.scalar_bcast.get(lhs)
            if r_s is not None and l_s is None:
                s, t = _affine_st(opcode, True, r_s)
                dst = ctx.new_buffer(n_elems, dtype)
                emit_affine(ctx, dst, ctx.buf_for(lhs), s, t, n_elems)
                ctx.value_to_buf[op.results[0]] = dst
                return
            if l_s is not None and r_s is None:
                s, t = _affine_st(opcode, False, l_s)
                dst = ctx.new_buffer(n_elems, dtype)
                emit_affine(ctx, dst, ctx.buf_for(rhs), s, t, n_elems)
                ctx.value_to_buf[op.results[0]] = dst
                return
        dst = ctx.new_buffer(n_elems, dtype)
        ctx.emit(Instr(opcode, dst=dst, a=ctx.buf_for(lhs),
                       b=ctx.buf_for(rhs), n=n_elems))
        ctx.value_to_buf[op.results[0]] = dst
    return handler


OP_HANDLERS["stablehlo.add"] = _elementwise_binop(OP_ADD_F32)
OP_HANDLERS["stablehlo.multiply"] = _elementwise_binop(OP_MUL_F32)
OP_HANDLERS["stablehlo.subtract"] = _elementwise_binop(OP_SUB_F32)


@_handles("stablehlo.constant")
def _lower_constant(ctx: _Ctx, op):
    from jaxlib.mlir import ir
    _, n_elems, dtype = _tensor_info(op.results[0].type)
    value = op.attributes["value"]
    npdt = DTYPE_NUMPY[dtype]
    # int/bool constants are DenseIntElementsAttr; float are DenseFPElementsAttr.
    if dtype in (DT_I32, DT_U32, DT_BOOL, DT_I64):
        arr = np.asarray(ir.DenseIntElementsAttr(value), dtype=npdt).reshape(-1)
    elif dtype in (DT_F16, DT_BF16):
        # MLIR's numpy interface can't convert bf16/f16 element attrs; extract
        # via f32 (splats are the common case: scalars, bias, scale).
        attr = ir.DenseFPElementsAttr(value)
        if attr.is_splat:
            v = float(ir.FloatAttr(attr.get_splat_value()))
            arr = np.full(n_elems, v, dtype=np.float32).astype(npdt)
        else:
            try:
                arr = np.array([float(x) for x in attr],
                               dtype=np.float32).astype(npdt)
            except Exception as e:  # noqa: BLE001
                raise LoweringError(
                    f"non-splat {npdt} constant not yet supported: {e}") from e
    else:
        arr = np.asarray(ir.DenseFPElementsAttr(value), dtype=npdt).reshape(-1)
    if arr.size == 1 and n_elems != 1:  # splat: expand into the const pool
        arr = np.broadcast_to(arr, (n_elems,)).copy()
    if arr.size != n_elems:
        raise LoweringError(
            f"stablehlo.constant: {arr.size} elements for type of {n_elems}")
    buf = ctx.new_buffer(n_elems, dtype)
    ctx.consts.append((buf, arr.astype(DTYPE_NUMPY[dtype]).tobytes()))
    ctx.value_to_buf[op.results[0]] = buf
    if n_elems == 1 and dtype == DT_F32:          # rank-0 f32 scalar: foldable
        ctx.const_scalar[op.results[0]] = float(np.asarray(arr).reshape(-1)[0])


@_handles("func.return")
def _lower_return(ctx: _Ctx, op):
    # v1 has no COPY opcode: results are whatever buffer produced the value.
    # Returning an argument/constant (or the same value twice) aliases that
    # buffer id in the outputs map — legal, the executor just reads the region.
    for operand in op.operands:
        shape, _, _ = _tensor_info(operand.type)
        ctx.outputs.append(ctx.buf_for(operand))
        ctx.output_shapes.append(shape)


_ops_registered = False


def _ensure_ops_registered() -> None:
    """Import pjrt_ocl.ops on first lowering (registers family handlers +
    opcode semantics). Deferred to call time so import-time cycles
    (lowering -> ops -> scheduler -> lowering) never form."""
    global _ops_registered
    if not _ops_registered:
        _ops_registered = True
        from . import ops  # noqa: F401


def _inline_call(ctx: _Ctx, o) -> None:
    """Inline a func.call to a private function: bind the callee's block args to
    the call's operand buffers, lower the callee body, and alias the callee's
    return values to the call's results. jax emits these for jnp.where / clip /
    some transcendentals."""
    from jaxlib.mlir import ir
    callee_name = ir.FlatSymbolRefAttr(o.attributes["callee"]).value
    callee = ctx.funcs.get(callee_name)
    if callee is None:
        raise LoweringError(f"func.call to unknown function @{callee_name}")
    block = callee.regions[0].blocks[0]
    for arg, operand in zip(block.arguments, o.operands):
        ctx.value_to_buf[arg] = ctx.buf_for(operand)
    for inner in block.operations:
        io = inner.operation
        if io.name == "func.return":
            for ret_val, call_res in zip(io.operands, o.results):
                ctx.value_to_buf[call_res] = ctx.buf_for(ret_val)
            return
        _lower_op(ctx, io)


@dataclasses.dataclass
class _WhileJob:
    """A deferred stablehlo.while region-lowering job. `while_idx` is the index
    of the WHILE placeholder Instr in ctx.instrs; carry is the list of
    (buf_id, n_elems, dtype) loop-carried buffers. `trip` is the compile-time
    trip count when the loop is a detected counted loop (OP_FOR; the cond
    region is then never lowered), or None for a genuine data-dependent WHILE."""
    while_idx: int
    cond_block: object
    body_block: object
    carry: list
    trip: int | None = None


# Opcodes whose result element i is a pure function of operand element i (same
# flat index) — safe to compute in place into a loop carry. Excludes cross-index
# ops (gather/reduce/dot/scatter/dynamic_slice/reduce_window/iota_dim).
_EW_INPLACE_SAFE = frozenset({
    OP_ADD_F32, OP_MUL_F32, OP_SUB_F32, OP_DIV_F32, OP_MAX_F32, OP_MIN_F32,
    OP_POW_F32, OP_ATAN2_F32, OP_REMAINDER_F32, OP_NEG_F32, OP_EXP_F32,
    OP_LOG_F32, OP_SQRT_F32, OP_RSQRT_F32, OP_TANH_F32, OP_ABS_F32, OP_FLOOR_F32,
    OP_CEIL_F32, OP_SIGN_F32, OP_LOG1P_F32, OP_EXPM1_F32, OP_CBRT_F32, OP_SIN_F32,
    OP_COS_F32, OP_TAN_F32, OP_ROUND_NEAREST_EVEN_F32, OP_ROUND_NEAREST_AFZ_F32,
    OP_COPY_F32, OP_CONVERT, OP_BITCAST, OP_CMP_F32, OP_SELECT_F32, OP_AND,
    OP_OR, OP_XOR, OP_NOT, OP_IS_FINITE, OP_AFFINE_F32,
})


def _defining_op(value):
    """The Operation that produces `value`, or None for block arguments."""
    owner = getattr(value, "owner", None)
    owner = getattr(owner, "operation", owner)
    return owner if getattr(owner, "name", None) else None


def _const_scalar_int_of(value) -> int | None:
    """If `value` is a scalar (1-element) integer stablehlo.constant, return
    its python int value; else None."""
    from jaxlib.mlir import ir
    op = _defining_op(value)
    if op is None or op.name != "stablehlo.constant":
        return None
    try:
        _, n_elems, dtype = _tensor_info(op.results[0].type)
    except LoweringError:
        return None
    if n_elems != 1 or dtype not in (DT_I32, DT_I64, DT_U32):
        return None
    arr = np.asarray(ir.DenseIntElementsAttr(op.attributes["value"]),
                     dtype=np.int64).reshape(-1)
    return int(arr[0])


def _detect_fixed_trip(op):
    """Detect the counted-loop pattern JAX emits for lax.scan / fori_loop:
    some carry k is a scalar int counter with a constant init, the cond region
    is exactly `compare LT (arg_k, constant)` and the body's k-th return is
    `add(arg_k, constant step)` with step > 0. Nobody writes data-dependent
    whiles by hand in JAX — scans dominate — so this catches nearly all loops.

    Returns (trip_count, k, init, step, counter_dtype) or None. The compare
    must not be UNSIGNED (init/limit/step are read as signed ints)."""
    cond_block = op.regions[0].blocks[0]
    body_block = op.regions[1].blocks[0]
    cmp = ret = None
    for inner in cond_block.operations:
        io = inner.operation
        if io.name == "stablehlo.constant":
            continue
        if io.name == "stablehlo.compare":
            if cmp is not None:
                return None
            cmp = io
        elif io.name == "stablehlo.return":
            ret = io
        else:
            return None              # cond computes more than a counter check
    if cmp is None or ret is None or ret.operands[0] != cmp.results[0]:
        return None
    try:
        direction = str(cmp.attributes["comparison_direction"])
    except KeyError:
        return None
    if "comparison_direction LT" not in direction:
        return None
    try:
        if "UNSIGNED" in str(cmp.attributes["compare_type"]):
            return None
    except KeyError:
        pass                             # compare_type is optional (NOTYPE)
    k = next((i for i, a in enumerate(cond_block.arguments)
              if cmp.operands[0] == a), None)
    if k is None:
        return None
    limit = _const_scalar_int_of(cmp.operands[1])
    init = _const_scalar_int_of(op.operands[k])
    if limit is None or init is None:
        return None
    body_ret = None
    for inner in body_block.operations:
        io = inner.operation
        if io.name == "stablehlo.return":
            body_ret = io
    if body_ret is None:
        return None
    upd = _defining_op(body_ret.operands[k])
    if upd is None or upd.name != "stablehlo.add":
        return None
    step = None
    if upd.operands[0] == body_block.arguments[k]:
        step = _const_scalar_int_of(upd.operands[1])
    elif upd.operands[1] == body_block.arguments[k]:
        step = _const_scalar_int_of(upd.operands[0])
    if step is None or step <= 0:
        return None
    trip = max(0, (limit - init + step - 1) // step)
    if trip >= 1 << 31:
        return None
    counter_dt = _tensor_info(op.operands[k].type)[2]
    return trip, k, init, step, counter_dt


def _while_mode() -> str:
    """PJRT_OCL_WHILE: auto (default; unroll small counted loops, OP_FOR the
    rest), for / unroll (force that path for counted loops), while (disable
    trip-count detection — the A/B baseline)."""
    m = os.environ.get("PJRT_OCL_WHILE", "auto").strip().lower()
    return m if m in ("auto", "for", "unroll", "while") else "auto"


def _unroll_trip_limit() -> int:
    """auto mode unrolls counted loops up to this trip count (PoC poc/12:
    bytecode size grows linearly and compile/upload time with it, while the
    per-iteration win over OP_FOR shrinks; see docs/decisions.md)."""
    try:
        return int(os.environ.get("PJRT_OCL_UNROLL_TRIPS", "32"))
    except ValueError:
        return 32


def _emit_const_scalar_int(ctx: _Ctx, val: int, dtype: int) -> int:
    """Materialize a scalar integer into the const pool; return its buffer."""
    buf = ctx.new_buffer(1, dtype)
    ctx.consts.append((buf, np.array([val], DTYPE_NUMPY[dtype]).tobytes()))
    return buf


def _unroll_while(ctx: _Ctx, op, trip: int, k: int, init: int, step: int,
                  counter_dt: int) -> None:
    """Inline a counted loop's body `trip` times into the current list (pure
    SSA — no carry buffers, no copies). The counter arg is rebound to a fresh
    const-pool scalar each iteration (its value is compile-time known), so the
    counter-add chain goes dead (DCE) and iterations don't serialize on it;
    dynamic_slice by the counter reads a constant. Constants in the body are
    lowered once (same ir.Value each iteration — the binding persists)."""
    n = len(op.operands)
    body = op.regions[1].blocks[0]
    cur = [ctx.buf_for(v) for v in op.operands]
    for it in range(trip):
        for j in range(n):
            ctx.value_to_buf[body.arguments[j]] = cur[j]
        ctx.value_to_buf[body.arguments[k]] = _emit_const_scalar_int(
            ctx, init + it * step, counter_dt)
        ret = None
        for inner in body.operations:
            io = inner.operation
            if io.name == "stablehlo.return":
                ret = [ctx.buf_for(v) for v in io.operands]
            elif (io.name == "stablehlo.constant"
                  and io.results[0] in ctx.value_to_buf):
                continue
            else:
                _lower_op(ctx, io)
        cur = ret
    # A result that aliases a buffer NO instruction writes (trip 0 -> the init
    # operand; a passthrough carry -> an input; a const-folded counter -> the
    # const pool) must be materialized: the executor binds outputs as I/O
    # ports, and a port only holds data if some instruction writes it.
    written = {ins.dst for ins in ctx.instrs if ins.op != OP_NOP}
    for j in range(n):
        if cur[j] not in written:
            _, n_elems, dtype = _tensor_info(op.operands[j].type)
            dst = ctx.new_buffer(n_elems, dtype)
            ctx.emit(Instr(OP_COPY_F32, dst=dst, a=cur[j], n=n_elems))
            cur[j] = dst
    for j in range(n):
        ctx.value_to_buf[op.results[j]] = cur[j]


@_handles("stablehlo.while")
def _lower_while(ctx: _Ctx, op):
    """Lower stablehlo.while to an OP_WHILE instruction over cond/body sub-lists.

    Model (docs/vmprogram.md WHILE + task prompt): N operands are the initial
    carried values. We allocate N *mutable* carry buffers, init-copy the
    operands into them (SSA-per-buffer is broken deliberately: the carries are
    updated in place across iterations). The cond region reads the carries and
    produces the loop scalar; the body reads the carries, computes fresh values,
    and copies them back into the carries so the next iteration sees the update.
    The N while results alias the carry buffers.

    The cond/body regions are lowered LATER (region_queue, drained after the
    root list) so their instrs land at indices >= main_len — the root walk only
    covers [0, main_len) and must never step into a sub-list (root_len rule).

    COUNTED LOOPS (poc/12): when the loop is a detected fixed-trip counter
    pattern (lax.scan / fori_loop — the overwhelmingly common case), the cond
    region never needs to run: small loops UNROLL inline (enables cross-
    iteration fusion + affine composition; no control at all), larger ones
    become OP_FOR (body sub-list + trip count; no cond compute, no cond-flag
    read, one barrier per iteration instead of two). PJRT_OCL_WHILE forces a
    path for A/B; default `auto` picks by trip count."""
    fixed = _detect_fixed_trip(op)
    mode = _while_mode()
    if fixed is not None and mode != "while":
        trip, ck, init, step, counter_dt = fixed
        if mode == "unroll" or (mode == "auto"
                                and trip <= _unroll_trip_limit()):
            _unroll_while(ctx, op, trip, ck, init, step, counter_dt)
            return
    else:
        fixed = None
    n = len(op.operands)
    carry: list = []
    for k in range(n):
        _, n_elems, dtype = _tensor_info(op.operands[k].type)
        cbuf = ctx.new_buffer(n_elems, dtype)
        carry.append((cbuf, n_elems, dtype))
        # init: operand -> carry (root list; runs once before the loop)
        ctx.emit(Instr(OP_COPY_F32, dst=cbuf, a=ctx.buf_for(op.operands[k]),
                       n=n_elems))
    while_idx = len(ctx.instrs)
    ctx.emit(Instr(OP_WHILE))          # placeholder; patched once regions lower
    ctx.region_queue.append(_WhileJob(
        while_idx, op.regions[0].blocks[0], op.regions[1].blocks[0], carry,
        trip=fixed[0] if fixed is not None else None))
    for k in range(n):                 # results alias the carry buffers
        ctx.value_to_buf[op.results[k]] = carry[k][0]


def _lower_while_regions(ctx: _Ctx, job: _WhileJob) -> None:
    """Lower a queued while's cond + body regions into ctx.instrs (appended
    after the root list) and patch the WHILE placeholder with the resulting
    sub-list ranges. Nested whiles enqueue further jobs (drained in turn)."""
    carry = job.carry
    n = len(carry)

    # --- cond sub-list: bind block args to carries, lower, read the scalar ----
    # A fixed-trip loop (job.trip) has NO cond sub-list: the trip count drives
    # iteration, so the counter compare/convert never executes.
    cond_start = len(ctx.instrs)
    cond_flag = 0
    if job.trip is None:
        for k in range(n):
            ctx.value_to_buf[job.cond_block.arguments[k]] = carry[k][0]
        cond_scalar = None
        for inner in job.cond_block.operations:
            io = inner.operation
            if io.name == "stablehlo.return":
                cond_scalar = ctx.buf_for(io.operands[0])
            else:
                _lower_op(ctx, io)
        if cond_scalar is None:
            raise LoweringError("stablehlo.while cond region has no return")
        # The device reads the loop scalar with a 4-byte atomic; the compare
        # result is a 1-byte i1. Convert it to an f32 0.0/1.0 flag so all 4
        # bytes are defined (a bare bool slot would leave 3 padding bytes in
        # the atomic read).
        cond_flag = ctx.new_buffer(1, DT_F32)
        ctx.emit(Instr(OP_CONVERT, dst=cond_flag, a=cond_scalar, n=1))
    cond_len = len(ctx.instrs) - cond_start

    # --- body sub-list: lower, then copy new values back into the carries -----
    body_start = len(ctx.instrs)
    for k in range(n):
        ctx.value_to_buf[job.body_block.arguments[k]] = carry[k][0]
    ret_bufs = None
    for inner in job.body_block.operations:
        io = inner.operation
        if io.name == "stablehlo.return":
            ret_bufs = [ctx.buf_for(v) for v in io.operands]
        else:
            _lower_op(ctx, io)
    if ret_bufs is None or len(ret_bufs) != n:
        raise LoweringError("stablehlo.while body region return arity mismatch")
    # Commit body results into the carries. Default: snapshot returns into temps,
    # then commit temps into carries — two phases (with a scheduler barrier
    # between) make the carry writes strictly later than every body read, so no
    # WAR/aliasing hazard survives even for swap/passthrough bodies (carries are
    # not SSA). PERF: when a carry's new value is produced by a single
    # elementwise (index-aligned) op that is the ONLY body reader of that carry
    # buffer, retarget the producer to write the carry IN PLACE and drop both
    # copies — this is the dominant while cost at large N (2 full-length
    # passes/iteration) and enables the carry to stay L2-resident across
    # iterations. Carries that don't qualify keep the safe two-phase snapshot.
    # collapse this body's affine chains first, so a scale/bias recurrence is a
    # single producer reading the carry directly (else the in-place test below
    # sees the pre-composition intermediate and bails).
    _compose_affines(ctx)
    body_end0 = len(ctx.instrs)
    producer: dict[int, int] = {}
    dup: set[int] = set()
    body_reads: dict[int, int] = {}
    for i in range(body_start, body_end0):
        bi = ctx.instrs[i]
        if bi.op == OP_NOP:
            continue
        if bi.dst in producer:
            dup.add(bi.dst)
        producer[bi.dst] = i
        for rbuf in _reads_of(bi):
            body_reads[rbuf] = body_reads.get(rbuf, 0) + 1
    carry_ids = {carry[k][0] for k in range(n)}
    # nested control (a WHILE/IF in this body) can make a carry's "producer" be a
    # nested region's init-copy; in-placing that corrupts the nested loop. Only
    # in-place flat elementwise bodies (the common fori/scan recurrence).
    has_nested = any(ctx.instrs[i].op in (OP_WHILE, OP_IF, OP_FOR)
                     for i in range(body_start, body_end0))
    inplace = [False] * n
    for k in ([] if has_nested else range(n)):
        cbuf = carry[k][0]
        rb = ret_bufs[k]
        pidx = producer.get(rb)
        if pidx is None or rb in dup or rb in carry_ids:
            continue                      # passthrough / reused / not body-produced
        prod = ctx.instrs[pidx]
        if prod.op not in _EW_INPLACE_SAFE:
            continue
        if body_reads.get(rb, 0) != 0:    # rb reused inside body (not terminal)
            continue
        # no body instr OTHER than the producer may read cbuf (else writing it in
        # place would clobber a sibling producer's operand).
        others = body_reads.get(cbuf, 0) - (1 if cbuf in _reads_of(prod) else 0)
        if others != 0:
            continue
        ctx.instrs[pidx] = dataclasses.replace(prod, dst=cbuf)
        inplace[k] = True
    temps: dict[int, int] = {}
    for k in range(n):
        if inplace[k]:
            continue
        _, n_elems, dtype = carry[k]
        t = ctx.new_buffer(n_elems, dtype)
        ctx.emit(Instr(OP_COPY_F32, dst=t, a=ret_bufs[k], n=n_elems))
        temps[k] = t
    for k in range(n):
        if inplace[k]:
            continue
        cbuf, n_elems, _ = carry[k]
        ctx.emit(Instr(OP_COPY_F32, dst=cbuf, a=temps[k], n=n_elems))
    body_len = len(ctx.instrs) - body_start

    if job.trip is not None:
        ctx.instrs[job.while_idx] = Instr(OP_FOR, b=job.trip,
                                          n=body_start, imm=body_len)
    else:
        ctx.instrs[job.while_idx] = Instr(OP_WHILE, dst=cond_flag,
                                          a=cond_start, b=cond_len,
                                          n=body_start, imm=body_len)


def _lower_op(ctx: _Ctx, o) -> None:
    if o.name == "func.call":
        _inline_call(ctx, o)
        return
    handler = OP_HANDLERS.get(o.name)
    if handler is None:
        raise LoweringError(f"unsupported op: {o.name} "
                            f"(known: {sorted(OP_HANDLERS)})")
    handler(ctx, o)


def _reads_of(ins: Instr) -> set[int]:
    """Over-approximate the buffer ids an instruction reads (safe for both DCE
    liveness and the compose single-use test: never under-counts). Uses the
    opsem read registry for shaped ops; defaults to {a, b} (+ select pred +
    reads_hint scalars) otherwise."""
    from . import opsem
    op = ins.op
    if op == OP_NOP:
        return set()
    base = set(ins.reads_hint)
    if op in (OP_WHILE, OP_IF, OP_FOR):
        return base                      # region instrs are separately live
    if op in opsem.READS:
        return base | opsem.reads_of(ins)
    base |= {ins.a, ins.b}
    if op == OP_SELECT_F32:
        base |= {ins.imm}                # select predicate rides in imm
    return base


def _compose_affines(ctx: _Ctx) -> None:
    """Fuse affine∘affine into one op: (x*s1+t1)*s2+t2 = x*(s1 s2) + (t1 s2+t2).
    Only when the inner result is written once and read once (by the outer), so
    NOP'ing the inner is safe. Index-stable (dead inner -> OP_NOP). Iterates to a
    fixpoint to collapse whole scale/bias chains into a single in-place pass."""
    instrs = ctx.instrs
    # buffers written exactly once -> the writer index (multi-write carries excl.)
    once: dict[int, int] = {}
    multi: set[int] = set()
    for idx, ins in enumerate(instrs):
        if ins.op == OP_NOP:
            continue
        d = ins.dst
        if d in once or d in multi:
            once.pop(d, None)
            multi.add(d)
        else:
            once[d] = idx
    # read counts (over-approx) for the single-use test
    reads_count: dict[int, int] = {}
    for ins in instrs:
        for b in _reads_of(ins):
            reads_count[b] = reads_count.get(b, 0) + 1
    outs = set(ctx.outputs)

    changed = True
    while changed:
        changed = False
        for idx, ins in enumerate(instrs):
            if ins.op != OP_AFFINE_F32:
                continue
            inner_idx = once.get(ins.a)
            if inner_idx is None:
                continue
            inner = instrs[inner_idx]
            if inner.op != OP_AFFINE_F32:
                continue
            if reads_count.get(ins.a, 0) != 1 or ins.a in outs:
                continue                 # inner result reused / is an output
            s1 = np.float32(np.array(inner.imm, np.uint32).view(np.float32))
            t1 = np.float32(np.array(inner.imm2, np.uint32).view(np.float32))
            s2 = np.float32(np.array(ins.imm, np.uint32).view(np.float32))
            t2 = np.float32(np.array(ins.imm2, np.uint32).view(np.float32))
            instrs[idx] = Instr(OP_AFFINE_F32, dst=ins.dst, a=inner.a,
                                b=inner.a, n=ins.n,
                                imm=_f32_bits(float(s1 * s2)),
                                imm2=_f32_bits(float(t1 * s2 + t2)))
            instrs[inner_idx] = Instr(OP_NOP)
            # bookkeeping: outer now reads inner.a instead of the old inner dst
            reads_count[ins.a] = reads_count.get(ins.a, 1) - 1
            reads_count[inner.a] = reads_count.get(inner.a, 0) + 1
            once.pop(ins.dst, None)
            once[ins.dst] = idx
            changed = True


def _dce_nops(ctx: _Ctx) -> None:
    """Dead-code eliminate to OP_NOP (index-stable): drop instructions whose
    result is never read by a live instruction and is not an output. Control ops
    (WHILE/IF) are always live and pin their cond_flag; liveness then propagates
    through the loop-carry chain, so region sub-lists survive while folded-away
    scalar broadcasts / intermediates do not."""
    instrs = ctx.instrs
    live_buf = set(ctx.outputs) | set(ctx.inputs)
    live_instr = [False] * len(instrs)
    # seed: control ops are always live (side effects / device-read cond_flag)
    for idx, ins in enumerate(instrs):
        if ins.op in (OP_WHILE, OP_IF, OP_FOR):
            live_instr[idx] = True
            if ins.dst and ins.op != OP_FOR:   # FOR has no cond flag
                live_buf.add(ins.dst)
    changed = True
    while changed:
        changed = False
        for idx, ins in enumerate(instrs):
            if ins.op == OP_NOP or live_instr[idx]:
                if live_instr[idx]:
                    for b in _reads_of(ins):
                        if b not in live_buf:
                            live_buf.add(b)
                            changed = True
                continue
            if ins.dst in live_buf:
                live_instr[idx] = True
                changed = True
    for idx, ins in enumerate(instrs):
        if ins.op != OP_NOP and not live_instr[idx]:
            instrs[idx] = Instr(OP_NOP)


# Elementwise f32 ops that read operand a (and b) at the output index and leave
# task fields p2/p3 free — so a strided VIEW descriptor can ride there. Excludes
# cmp/select/affine/fill (which use p2/p3) and non-f32 paths.
_VIEWABLE_EW = frozenset({
    OP_ADD_F32, OP_SUB_F32, OP_MUL_F32, OP_DIV_F32, OP_MAX_F32, OP_MIN_F32,
    OP_POW_F32, OP_ATAN2_F32, OP_REMAINDER_F32, OP_NEG_F32, OP_EXP_F32,
    OP_LOG_F32, OP_SQRT_F32, OP_RSQRT_F32, OP_TANH_F32, OP_ABS_F32, OP_FLOOR_F32,
    OP_CEIL_F32, OP_SIGN_F32, OP_LOG1P_F32, OP_EXPM1_F32, OP_CBRT_F32, OP_SIN_F32,
    OP_COS_F32, OP_TAN_F32, OP_ROUND_NEAREST_EVEN_F32, OP_ROUND_NEAREST_AFZ_F32,
    OP_COPY_F32,
})


def _fuse_views(ctx: _Ctx) -> None:
    """The general access-map fusion: fold a shape op (broadcast/transpose/slice/
    reshape/reverse — all OP_GATHER_STRIDED) into its consuming elementwise
    operands as a strided VIEW instead of materializing it. An EW op then reads
    `src[view_index(i)]` — output element i's read is a static function of i, so
    the producer never occupies memory and no barrier crosses the (former) edge.

    A gather G (dst=g, src=s, aux=desc) folds iff every reader of g is a viewable
    f32 EW op that reads g as operand a/b (that operand not already a view) and g
    is not a program output. Each reader's operand is retargeted to (s, view=desc)
    — the view aux-offset rides in imm (operand a) / imm2 (operand b), +1 so 0
    means direct. G becomes NOP (DCE'd). Readers that aren't viewable EW (a
    matmul, another gather, a reduce) keep the gather materialized."""
    instrs = ctx.instrs
    outs = set(ctx.outputs)
    changed = True
    while changed:
        changed = False
        writer_idxs: dict[int, list[int]] = {}
        for idx, ins in enumerate(instrs):
            if ins.op != OP_NOP:
                writer_idxs.setdefault(ins.dst, []).append(idx)
        for gi, g in enumerate(instrs):
            if g.op != OP_GATHER_STRIDED or g.dst in outs:
                continue
            if ctx.buffers[g.dst].dtype != DT_F32:
                continue
            # The gather's dst must be a pure view of its source: skip if any
            # OTHER instr writes g.dst (read-modify-write, e.g. the identity
            # copy + dyn_scatter pair of dynamic_update_slice — folding it
            # orphans the scatter's write), or if g.a is written after the
            # gather (loop carries are multi-write: the folded readers would
            # see the NEXT iteration's value).
            if len(writer_idxs.get(g.dst, ())) != 1:
                continue
            if any(w > gi for w in writer_idxs.get(g.a, ())):
                continue
            gbuf = g.dst
            readers = [(j, ins) for j, ins in enumerate(instrs)
                       if ins.op != OP_NOP and gbuf in _reads_of(ins)]
            if not readers:
                continue
            def foldable(r):
                if r.op not in _VIEWABLE_EW:
                    return False
                if ctx.buffers[r.dst].dtype != DT_F32:
                    return False
                if r.a == gbuf and r.imm != 0:      # operand a already viewed
                    return False
                if r.b == gbuf and r.imm2 != 0:     # operand b already viewed
                    return False
                return r.a == gbuf or r.b == gbuf
            if not all(foldable(r) for _, r in readers):
                continue
            for j, r in readers:
                if r.a == gbuf:
                    r = dataclasses.replace(r, a=g.a, imm=g.aux + 1)
                if r.b == gbuf:
                    r = dataclasses.replace(r, b=g.a, imm2=g.aux + 1)
                instrs[j] = r
            instrs[gi] = Instr(OP_NOP)
            changed = True


def lower_module(module) -> VMProgram:
    """Lower a deserialized stablehlo module's public @main to a VMProgram."""
    from jaxlib.mlir import ir
    _ensure_ops_registered()
    funcs = {}          # sym_name -> func.func operation (for call inlining)
    main = None
    for op in module.body.operations:
        o = op.operation
        if o.name == "func.func":
            name = ir.StringAttr(o.attributes["sym_name"]).value
            funcs[name] = o
            if name == "main":
                main = o
    if main is None:
        raise LoweringError("no func.func @main in module")

    ctx = _Ctx()
    ctx.funcs = funcs
    entry_block = main.regions[0].blocks[0]
    for arg in entry_block.arguments:
        shape, n_elems, dtype = _tensor_info(arg.type)
        buf = ctx.new_buffer(n_elems, dtype)
        ctx.value_to_buf[arg] = buf
        ctx.inputs.append(buf)
        ctx.input_shapes.append(shape)

    for op in entry_block.operations:
        _lower_op(ctx, op.operation)

    # The root list is exactly the entry-block lowering; region sub-lists
    # (cond/body of every while, nested included) are appended after it so the
    # root walk [0, main_len) never enters a sub-range (docs/vmprogram.md).
    main_len = len(ctx.instrs)
    while ctx.region_queue:
        _lower_while_regions(ctx, ctx.region_queue.pop(0))

    # perf peepholes (index-stable, NOP-substituting): collapse scale/bias chains
    # into one in-place affine pass, fold shape ops (broadcast/transpose/slice)
    # into consuming elementwise operands as strided views, then DCE the dead.
    _compose_affines(ctx)
    _fuse_views(ctx)
    _dce_nops(ctx)

    return VMProgram(
        arena_bytes=ctx._arena,
        buffers=ctx.buffers,
        inputs=ctx.inputs,
        outputs=ctx.outputs,
        input_shapes=ctx.input_shapes,
        output_shapes=ctx.output_shapes,
        consts=ctx.consts,
        instrs=ctx.instrs,
        aux=ctx.aux,
        main_len=main_len,
    )


def deserialize_artifact(artifact: bytes):
    """VHLO portable artifact bytes -> stablehlo ir.Module (auto-upgraded)."""
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import stablehlo
    # No explicit dialect registration needed (poc/03 NOTES #5).
    return stablehlo.deserialize_portable_artifact(ir.Context(), artifact)


def lower_artifact(artifact: bytes) -> VMProgram:
    """PJRT_Client_Compile program bytes (VHLO artifact) -> VMProgram."""
    return lower_module(deserialize_artifact(artifact))
