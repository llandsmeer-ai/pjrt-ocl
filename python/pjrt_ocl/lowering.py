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
OP_SOFTMAX = 54          # fused softmax over the innermost `seg` elems; imm = seg (§19)
OP_LAYERNORM = 55        # fused layernorm core over innermost `seg`; imm = seg, imm2 = eps bits
OP_GELU = 56             # fused GELU tanh-approx unary EW op (§19b/§24); a = input, no imm
OP_MAP_REGION = 57       # §27/§28 register-resident fused map-region: a=in0, b=in1,
                         # n=elems, aux=descriptor word-offset (slots + micro-program)
OP_FLASH_ATTN = 58       # §34 fused flash-attention (online softmax): a=Q b=K src,
                         # imm2=V src, n=H, imm=T, aux=descriptor word-offset
                         # ([H,T,C,hd,scale,causal,qv,kv,vv]); dst=out
OP_REDUCE_STRIDED = 59   # partial-axis reduce over a CONTIGUOUS interior/prefix
                         # axis block (inner stride > 1): input viewed (outer,
                         # red, inner); out[o*inner+i]=reduce_r in[(o*red+r)*inner+i].
                         # n = n_out (outer*inner), imm = (kind<<28)|red, imm2 = inner
OP_GATHER_INDEX = 60     # §38 general data-dependent gather (stablehlo.gather):
                         # a=operand, aux=descriptor word-offset, reads start_indices
                         # (its buffer id rides in aux + reads_hint). (60: 59 = REDUCE_STRIDED)
OP_SHL = 61              # stablehlo.shift_left (int32/uint32) (61-63: 59/60 taken)
OP_SHR_L = 62            # stablehlo.shift_right_logical (zero-fill)
OP_SHR_A = 63            # stablehlo.shift_right_arithmetic (sign-fill)
OP_CONV = 64             # stablehlo.convolution (direct N-D NHWC/HWIO conv, §39):
                         # a=input, b=weights, aux=descriptor word-offset
                         # (sdim,Cin,Cout, out/win/stride/pad_low/dil/in spatial dims)
OP_SCATTER_INDEX = 65    # §42 general data-dependent scatter (stablehlo.scatter):
                         # a=updates, dst=operand result, aux=descriptor word-offset,
                         # reads scatter_indices (id in aux + reads_hint). kind in aux
                         # (0 set / 1 add / 2 max / 3 min). (65: 64 = OP_CONV)
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
    OP_SOFTMAX: "softmax",
    OP_LAYERNORM: "layernorm",
    OP_GELU: "gelu",
    OP_MAP_REGION: "map_region",
    OP_FLASH_ATTN: "flash_attn",
    OP_REDUCE_STRIDED: "reduce_strided", OP_GATHER_INDEX: "gather_index",
    OP_SHL: "shl", OP_SHR_L: "shr_l", OP_SHR_A: "shr_a",
    OP_CONV: "convolution",
    OP_SCATTER_INDEX: "scatter_index",
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
    # OP_DOT operand VIEW aux-offsets (+1; 0 = contiguous) set by
    # _fuse_matmul_views when a transpose/reshape/broadcast folds into the
    # matmul operand read (§13). NOT serialized: the scheduler reads these
    # in-memory (-> task p4/p5); validator a reads the mirror aux header at
    # ins.aux (written by _finalize_matmul_views, so it survives serialization).
    aview: int = 0
    bview: int = 0
    # OP_DOT matmul epilogue (§33 R2c), set by _fuse_mma_epilogue. NOT serialized:
    # the scheduler reads these in-memory -> task p6/p7; the epilogue descriptor
    # itself lives in the (serialized) aux pool and the kernel finds it via p6.
    epi: int = 0        # epilogue descriptor aux word-offset (+1; 0 = none)
    epi_res: int = 0    # epilogue second-input (residual/bias) buffer id
    # OP_MAP_REGION multi-input handles (§28 follow-up): the FULL ordered list of
    # region input buffer ids (len ≤ 8). Rides task fields a,b,p2..p7 (all
    # loader-patched). NOT serialized: the scheduler reads it in-memory to build
    # the task + its dependency read-set; the tensor validator reads it too.
    region_inputs: tuple = ()


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
        # complex64/128 SSA values are represented split into a (real, imag) pair
        # of f32 buffers entirely in the lowering (the arena stays 4-byte-slotted
        # f32; no complex arena dtype). ops/complex_fft.py populates this;
        # buf_pair/_alias_value below propagate it across call/return like
        # value_to_buf does for scalar values. See docs/decisions.md §43.
        self.cbuf: dict = {}               # ir.Value (complex) -> (re_id, im_id)
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
            # StableHLO integers are signless (i32) EXCEPT genuinely unsigned
            # values, which carry the `ui32` type (element_type.is_unsigned).
            # Reporting the correct signedness matters at the host boundary:
            # PJRT_Buffer_ElementType is derived from this dtype, so a ui32
            # device array (e.g. a threefry RNG key) must round-trip to numpy
            # as uint32 — otherwise np.asarray(key) yields int32 and JAX
            # materialises the key constant as tensor<2xi32>, mismatching its
            # own ui32 @_threefry_split signature (verifier error; blocks any
            # closed-over-key jit on this platform, e.g. brax reset+step).
            # U32 is a 4-byte Tier-1 slot; elementwise/bitwise/shift/bitcast ops
            # are bit-identical to I32 (the threefry/uniform RNG path uses only
            # shl/shr_l/xor/add/iota + bitcast_convert, all bit-exact — verified
            # vs JAX-CPU golden). KNOWN GAP (pre-existing, orthogonal): the
            # DEVICE convert (u32->f32) and integer compare still run the SIGNED
            # path, so u32 values with the high bit set convert/compare wrong vs
            # JAX-CPU. Not hit by RNG/brax; fixing needs unsigned device kernels
            # (tracked with reduce(and) as the next brax op gap, §42).
            return DT_U32 if element_type.is_unsigned else DT_I32
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


def _lower_sdy_identity(ctx: _Ctx, op):
    """Shardy sharding hints are no-ops on our single OpenCL device: every
    sdy op that carries a value through (sharding_constraint / reshard) is the
    identity here, so alias each result to its like-typed operand. `sdy.mesh`
    (module-level, no results) and other result-less sdy ops simply do nothing.
    General infra: this is what makes ANY sharded JAX program lowerable — the
    dialect is registered in deserialize_artifact, and its value-carrying ops
    collapse to identity here (brax/MJX emit sdy.sharding_constraint)."""
    for res, operand in zip(op.results, op.operands):
        ctx.value_to_buf[res] = ctx.buf_for(operand)


OP_HANDLERS["sdy.sharding_constraint"] = _lower_sdy_identity
OP_HANDLERS["sdy.reshard"] = _lower_sdy_identity
OP_HANDLERS["sdy.mesh"] = _lower_sdy_identity          # module-level, no results


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
        _alias_value(ctx, arg, operand)
    for inner in block.operations:
        io = inner.operation
        if io.name == "func.return":
            for ret_val, call_res in zip(io.operands, o.results):
                _alias_value(ctx, call_res, ret_val)
            return
        _lower_op(ctx, io)


def _alias_value(ctx: _Ctx, dst, src) -> None:
    """Alias SSA value `dst` to `src` across a call/return boundary, carrying
    a split-complex (real, imag) pair when `src` is complex, else the scalar
    buffer id."""
    if src in ctx.cbuf:
        ctx.cbuf[dst] = ctx.cbuf[src]
    else:
        ctx.value_to_buf[dst] = ctx.buf_for(src)


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
    OP_SHL, OP_SHR_L, OP_SHR_A,
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
    """auto mode unrolls counted loops up to this trip count (poc/12: unroll
    wins by 10-20x at small sizes and stays ahead of OP_FOR well past 100
    trips, but bytecode size and compile time grow linearly)."""
    try:
        return int(os.environ.get("PJRT_OCL_UNROLL_TRIPS", "64"))
    except ValueError:
        return 64


def _unroll_arena_cap() -> int:
    """auto mode's byte budget for unrolled intermediates. The arena allocator
    is a bump allocator (no SSA liveness reuse yet), so unrolling allocates
    fresh buffers per iteration: a 512-trip loop over 1M-element carries wants
    gigabytes and dies (poc/12 measured). PJRT_OCL_UNROLL_ARENA_MB."""
    try:
        return int(os.environ.get("PJRT_OCL_UNROLL_ARENA_MB", "256")) << 20
    except ValueError:
        return 256 << 20


def _unroll_bytes_estimate(ctx: _Ctx, block, depth: int = 0) -> int:
    """Rough per-iteration arena bytes if this body block were unrolled: the
    sum of every op result's tensor bytes (constants excluded — they lower
    once), func.call bodies included. Over-approximates (DCE/fusion drop some)
    but scales correctly with body size x carry size."""
    from jaxlib.mlir import ir
    if depth > 4:
        return 0
    total = 0
    for inner in block.operations:
        io = inner.operation
        if io.name in ("stablehlo.return", "func.return", "stablehlo.constant"):
            continue
        if io.name == "func.call":
            callee = ctx.funcs.get(
                ir.FlatSymbolRefAttr(io.attributes["callee"]).value)
            if callee is not None:
                total += _unroll_bytes_estimate(
                    ctx, callee.regions[0].blocks[0], depth + 1)
            continue
        for res in io.results:
            try:
                _, n_elems, dtype = _tensor_info(res.type)
            except LoweringError:
                continue
            total += n_elems * DTYPE_NUMPY[dtype].itemsize
    return total


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
        # auto: unroll when the trip count is modest AND the unrolled
        # intermediates fit the arena budget (bump allocator, no reuse —
        # poc/12: a 512-trip 1M-element unroll wants ~4 GB and dies).
        if mode == "unroll" or (
                mode == "auto" and trip <= _unroll_trip_limit()
                and trip * _unroll_bytes_estimate(ctx, op.regions[1].blocks[0])
                <= _unroll_arena_cap()):
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


def _is_identity_gather_aux(ctx: _Ctx, aux_off: int, n_elems: int) -> bool:
    """True iff a GATHER_STRIDED aux block ([rank, out_dims, in_strides,
    src_off]) describes a plain contiguous copy of n_elems elements."""
    rank = ctx.aux[aux_off]
    if aux_off + 2 + 2 * rank > len(ctx.aux):
        return False
    dims = ctx.aux[aux_off + 1:aux_off + 1 + rank]
    strides = ctx.aux[aux_off + 1 + rank:aux_off + 1 + 2 * rank]
    if ctx.aux[aux_off + 1 + 2 * rank] != 0:
        return False                      # nonzero source offset
    total, acc = 1, 1
    for d in range(rank - 1, -1, -1):
        if strides[d] != acc:
            return False                  # not row-major contiguous
        acc *= dims[d]
        total *= dims[d]
    return total == n_elems


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
    writers: dict[int, list[int]] = {}
    body_reads: dict[int, int] = {}
    for i in range(body_start, body_end0):
        bi = ctx.instrs[i]
        if bi.op == OP_NOP:
            continue
        if bi.dst in producer:
            dup.add(bi.dst)
        producer[bi.dst] = i
        writers.setdefault(bi.dst, []).append(i)
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
    # IN-PLACE dynamic_update_slice into the carry (scan's ys stacking). The
    # DUS lowering is a pair: identity-gather copy carry->fresh out_buf, then
    # scatter the update row into out_buf. When the carry's ONLY body use is
    # feeding that copy and out_buf is only ever returned, the pure semantics
    # collapse to a mutation: NOP the full-length copy and scatter straight
    # into the carry buffer. This turns scan's O(T*n) per-iteration ys traffic
    # into O(n) — the dominant large-scan cost (decisions.md §15). The
    # scatter's runtime index reads (the counter carry) ride in reads_hint, so
    # body_reads already sees them and the counter keeps its safe snapshot.
    for k in ([] if has_nested else range(n)):
        if inplace[k]:
            continue
        cbuf, n_elems, _dt = carry[k]
        rb = ret_bufs[k]
        if rb in carry_ids or ret_bufs.count(rb) != 1:
            continue
        w = writers.get(rb, [])
        if len(w) != 2:
            continue
        g, s = ctx.instrs[w[0]], ctx.instrs[w[1]]
        if g.op != OP_GATHER_STRIDED or s.op != OP_DYNAMIC_UPDATE_SLICE:
            continue
        if g.a != cbuf or g.n != n_elems:
            continue
        if not _is_identity_gather_aux(ctx, g.aux, n_elems):
            continue
        if body_reads.get(rb, 0) != 0:    # out_buf read inside body: keep copy
            continue
        # cbuf may be read ONLY by the identity gather, and written by nothing
        # else in the body (in-place mutation must not be observable).
        if body_reads.get(cbuf, 0) != 1 or cbuf in writers:
            continue
        ctx.instrs[w[0]] = Instr(OP_NOP)
        ctx.instrs[w[1]] = dataclasses.replace(s, dst=cbuf)
        inplace[k] = True
    # IDENTITY passthrough (ret == this carry's own buffer, e.g. scan's xs):
    # the "new" value already IS the carry — both snapshot copies are pure
    # waste (2 full-length passes/iteration over the largest carry in a scan).
    # Nothing writes cbuf in the body (each in-place commit targets its own
    # carry), so skipping is safe. NOT the same as `rb in carry_ids` with
    # rb != cbuf (a swap) — that one keeps the two-phase snapshot.
    skip = [ret_bufs[k] == carry[k][0] for k in range(n)]
    temps: dict[int, int] = {}
    for k in range(n):
        if inplace[k] or skip[k]:
            continue
        _, n_elems, dtype = carry[k]
        t = ctx.new_buffer(n_elems, dtype)
        ctx.emit(Instr(OP_COPY_F32, dst=t, a=ret_bufs[k], n=n_elems))
        temps[k] = t
    for k in range(n):
        if inplace[k] or skip[k]:
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


# --- stablehlo.if / stablehlo.case : N-way region control (OP_IF) ------------

_CMP_EQ = 0   # OP_CMP_F32 predicate for ==  (mirrors ops.elementwise pred table)


@dataclasses.dataclass
class _IfJob:
    """A deferred region-lowering job for one OP_IF instruction (used for both
    stablehlo.if and each arm of a lowered stablehlo.case). `if_idx` is the
    index of the OP_IF placeholder in ctx.instrs; `cond_buf` is the f32 0/1
    branch flag the device reads atomically; `results` is the shared list of
    (buf, n_elems, dtype) carries every taken branch writes into (the if/case
    results alias these). `else_block` is None for a case arm (empty else)."""
    if_idx: int
    then_block: object
    else_block: object
    cond_buf: int
    results: list


def _lower_branch_into(ctx: _Ctx, block, results: list) -> None:
    """Lower a region block (no block arguments; captures enclosing SSA values)
    and commit its returns into the shared result carries via COPY. Emits into
    ctx.instrs at the current position (a region sub-list)."""
    ret = None
    for inner in block.operations:
        io = inner.operation
        if io.name in ("stablehlo.return", "func.return"):
            ret = [ctx.buf_for(v) for v in io.operands]
        else:
            _lower_op(ctx, io)
    if ret is None or len(ret) != len(results):
        raise LoweringError("case/if branch return arity mismatch")
    for (rbuf, n_elems, _dt), rb in zip(results, ret):
        ctx.emit(Instr(OP_COPY_F32, dst=rbuf, a=rb, n=n_elems))


def _lower_if_regions(ctx: _Ctx, job: _IfJob) -> None:
    """Lower a queued OP_IF's then/else sub-lists (appended after the root, at
    indices >= main_len — root_len rule) and patch the placeholder. Nested
    regions inside a branch enqueue further jobs (drained in turn)."""
    then_start = len(ctx.instrs)
    _lower_branch_into(ctx, job.then_block, job.results)
    then_len = len(ctx.instrs) - then_start
    else_start = len(ctx.instrs)
    if job.else_block is not None:
        _lower_branch_into(ctx, job.else_block, job.results)
    else_len = len(ctx.instrs) - else_start
    ctx.instrs[job.if_idx] = Instr(OP_IF, dst=job.cond_buf,
                                   a=then_start, b=then_len,
                                   n=else_start, imm=else_len)


def _alloc_results(ctx: _Ctx, op) -> list:
    """Allocate one result carry buffer per op result and alias the results to
    them. Returns [(buf, n_elems, dtype), ...]."""
    results = []
    for res in op.results:
        _, n_elems, dtype = _tensor_info(res.type)
        rbuf = ctx.new_buffer(n_elems, dtype)
        results.append((rbuf, n_elems, dtype))
        ctx.value_to_buf[res] = rbuf
    return results


@_handles("stablehlo.if")
def _lower_if(ctx: _Ctx, op):
    """stablehlo.if(pred) -> OP_IF over then/else region sub-lists. pred is an
    i1 scalar; convert to an f32 0/1 flag the device reads atomically (same as
    the while cond)."""
    cond = ctx.buf_for(op.operands[0])
    cond_f32 = ctx.new_buffer(1, DT_F32)
    ctx.emit(Instr(OP_CONVERT, dst=cond_f32, a=cond, n=1))
    results = _alloc_results(ctx, op)
    if_idx = len(ctx.instrs)
    ctx.emit(Instr(OP_IF))                 # placeholder; patched once regions lower
    ctx.region_queue.append(_IfJob(if_idx, op.regions[0].blocks[0],
                                   op.regions[1].blocks[0], cond_f32, results))


@_handles("stablehlo.case")
def _lower_case(ctx: _Ctx, op):
    """stablehlo.case(index) -> an N-branch region op. StableHLO selects branch
    `index`, clamping index<0 or index>=N to the LAST branch. We express the
    N-way switch as N flat sibling OP_IFs sharing one set of result carries:
    branch k runs iff its selection flag sel_k is 1. For k<N-1, sel_k =
    (index==k) (exact); the default (out-of-range OR ==N-1) is sel_{N-1} =
    1 - Σ_{k<N-1} sel_k (at most one earlier flag is set, so this is 0/1).
    Guarding EVERY branch (incl. the last) keeps branches that never run — a
    nested while, an expensive cone — from executing, unlike a select-all
    lowering; frame depth stays 1 (siblings, not nested)."""
    n_branches = len(op.regions)
    if n_branches == 0:
        raise LoweringError("stablehlo.case with no branches")

    # N==1: index is irrelevant (always clamps to branch 0). Inline the single
    # branch with no control, aliasing the case results to its returns.
    if n_branches == 1:
        block = op.regions[0].blocks[0]
        ret = None
        for inner in block.operations:
            io = inner.operation
            if io.name in ("stablehlo.return", "func.return"):
                ret = [ctx.buf_for(v) for v in io.operands]
            else:
                _lower_op(ctx, io)
        if ret is None or len(ret) != len(op.results):
            raise LoweringError("stablehlo.case branch return arity mismatch")
        for res, rb in zip(op.results, ret):
            ctx.value_to_buf[res] = rb
        return

    results = _alloc_results(ctx, op)

    # index -> f32 (exact for small branch indices) for scalar == compares.
    _, idx_n, idx_dt = _tensor_info(op.operands[0].type)
    idx_f32 = ctx.buf_for(op.operands[0])
    if idx_dt != DT_F32:
        idx_f32 = ctx.new_buffer(idx_n, DT_F32)
        ctx.emit(Instr(OP_CONVERT, dst=idx_f32, a=ctx.buf_for(op.operands[0]),
                       n=idx_n))

    sels = []                              # f32 0/1 flags for branches 0..N-2
    for k in range(n_branches - 1):
        kconst = ctx.new_buffer(1, DT_F32)
        ctx.emit(Instr(OP_FILL_F32, dst=kconst, imm=_f32_bits(float(k)), n=1))
        cmpb = ctx.new_buffer(1, DT_BOOL)
        ctx.emit(Instr(OP_CMP_F32, dst=cmpb, a=idx_f32, b=kconst, n=1,
                       imm=_CMP_EQ))
        selk = ctx.new_buffer(1, DT_F32)
        ctx.emit(Instr(OP_CONVERT, dst=selk, a=cmpb, n=1))
        sels.append(selk)
    # sel_last = 1 - Σ sel_k  (n_branches >= 2 here, so `sels` is non-empty)
    acc = sels[0]
    for s in sels[1:]:
        nacc = ctx.new_buffer(1, DT_F32)
        ctx.emit(Instr(OP_ADD_F32, dst=nacc, a=acc, b=s, n=1))
        acc = nacc
    one = ctx.new_buffer(1, DT_F32)
    ctx.emit(Instr(OP_FILL_F32, dst=one, imm=_f32_bits(1.0), n=1))
    sel_last = ctx.new_buffer(1, DT_F32)
    ctx.emit(Instr(OP_SUB_F32, dst=sel_last, a=one, b=acc, n=1))
    all_sels = sels + [sel_last]

    for k in range(n_branches):
        if_idx = len(ctx.instrs)
        ctx.emit(Instr(OP_IF))             # placeholder; patched once regions lower
        ctx.region_queue.append(_IfJob(if_idx, op.regions[k].blocks[0], None,
                                       all_sels[k], results))


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


def _reader_index(instrs: list) -> dict:
    """buf_id -> list of live instr indices that read it. Built once per fusion
    round so the view-fold passes look up a gather's readers in O(readers)
    instead of rescanning the whole stream per gather (O(n²)→O(n) per round;
    matters on large graphs like the brax step, ~19k instrs)."""
    idx: dict[int, list[int]] = {}
    for j, ins in enumerate(instrs):
        if ins.op == OP_NOP:
            continue
        for b in _reads_of(ins):
            idx.setdefault(b, []).append(j)
    return idx


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
        rmap = _reader_index(instrs)
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
            # rmap is built once per round; re-filter live (an earlier fold this
            # round may have NOP'd/retargeted a listed reader).
            readers = [(j, instrs[j]) for j in rmap.get(gbuf, ())
                       if instrs[j].op != OP_NOP and gbuf in _reads_of(instrs[j])]
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


# --- fused segmented norms (softmax / layernorm), §19 -----------------------

# Max segment size the collaborative in-local-memory kernel can stage (one row
# in __local As, sized MMA_ASZ = 1024 floats in the portable build). Larger
# suffix reductions fall back to the decomposed reduce+broadcast lowering.
_NORM_SEG_MAX = 1024
_RSEG_SUM, _RSEG_MAX = 0, 1        # OP_REDUCE_SEG imm kind field (imm >> 28)


def _f32_from_bits(bits: int) -> float:
    return float(np.frombuffer(struct.pack("<I", bits & 0xFFFFFFFF), "<f4")[0])


def _fuse_norm(ctx: _Ctx) -> None:
    """Recognize the softmax / layernorm-core reduce→broadcast idioms in OUR
    lowered VM-instr stream and collapse each into a SINGLE fused op
    (OP_SOFTMAX / OP_LAYERNORM) done in local memory by one workgroup-per-segment
    kernel — one global read + one write instead of 5–7 cross-workgroup phases
    (§19). Matches on our own instrs (robust to jaxlib/StableHLO variation: both
    idioms funnel through OP_REDUCE_SEG + viewed EW ops).

    GATED HARD: only fires on the exact innermost-suffix dataflow chain with the
    right op kinds and producer→consumer linkage; on ANY mismatch it leaves the
    decomposed lowering untouched (an unfused norm is correct, only slower).
    PJRT_OCL_FUSE_NORM=0 disables it (A/B + revert lever). Runs before
    _reuse_arena; the following _dce_nops removes the now-dead intermediates."""
    if os.environ.get("PJRT_OCL_FUSE_NORM", "1") == "0":
        return
    instrs = ctx.instrs
    outs = set(ctx.outputs)

    # unique writer per buffer (None if 0 or >1 non-NOP writers: multi-write
    # carries and RMW pairs must not be traced through).
    def _writer_map() -> dict:
        seen: dict = {}
        multi: set = set()
        for idx, ins in enumerate(instrs):
            if ins.op == OP_NOP:
                continue
            d = ins.dst
            if d in seen or d in multi:
                seen.pop(d, None)
                multi.add(d)
            else:
                seen[d] = idx
        return seen

    # f32 scalar/broadcast consts (all elements equal) -> python float.
    const_f: dict = {}
    for bid, data in ctx.consts:
        if bid < len(ctx.buffers) and ctx.buffers[bid].dtype == DT_F32:
            arr = np.frombuffer(data, "<f4")
            if arr.size >= 1 and np.all(arr == arr[0]):
                const_f[bid] = float(arr[0])

    def _redseg(imm: int):
        return imm >> 28, imm & 0x0FFFFFFF        # kind, seg

    changed = True
    while changed:
        changed = False
        writers = _writer_map()

        def bcast_src(buf: int) -> int:
            """Follow a chain of OP_GATHER_STRIDED (reshape/broadcast) producers
            back to the ultimate source buffer id."""
            seen: set = set()
            while (buf in writers and buf not in seen
                   and instrs[writers[buf]].op == OP_GATHER_STRIDED):
                seen.add(buf)
                buf = instrs[writers[buf]].a
            return buf

        def find(pred):
            for j, ins in enumerate(instrs):
                if ins.op != OP_NOP and pred(j, ins):
                    return j, ins
            return None

        def is_op(buf: int, op: int):
            w = writers.get(buf)
            return None if w is None else (instrs[w] if instrs[w].op == op else None)

        for r1i, R1 in enumerate(instrs):
            if R1.op != OP_REDUCE_SEG:
                continue
            kind1, seg = _redseg(R1.imm)
            if seg == 0 or seg > _NORM_SEG_MAX:
                continue
            X, n_out = R1.a, R1.n
            total = n_out * seg

            # ---- softmax: max -> sub(x,·) -> exp -> sum -> div(exp,·) --------
            if kind1 == _RSEG_MAX:
                mSUB = find(lambda j, s: s.op == OP_SUB_F32 and s.a == X
                            and s.n == total and bcast_src(s.b) == R1.dst)
                if mSUB is None:
                    continue
                _, SUB = mSUB
                mEXP = find(lambda j, s: s.op == OP_EXP_F32 and s.a == SUB.dst)
                if mEXP is None:
                    continue
                _, EXP = mEXP
                mR2 = find(lambda j, s: s.op == OP_REDUCE_SEG and s.a == EXP.dst
                           and _redseg(s.imm) == (_RSEG_SUM, seg) and s.n == n_out)
                if mR2 is None:
                    continue
                _, R2 = mR2
                mDIV = find(lambda j, s: s.op == OP_DIV_F32 and s.a == EXP.dst
                            and s.n == total and bcast_src(s.b) == R2.dst)
                if mDIV is None:
                    continue
                di, DIV = mDIV
                if DIV.dst in outs and X == DIV.dst:
                    continue
                instrs[di] = Instr(OP_SOFTMAX, dst=DIV.dst, a=X, b=X,
                                   n=n_out, imm=seg)
                changed = True
                break

            # ---- layernorm core ---------------------------------------------
            if kind1 != _RSEG_SUM:
                continue
            # mu = mean(x) = sum(x) / seg
            mMU = find(lambda j, s: s.op == OP_DIV_F32 and s.n == n_out
                       and bcast_src(s.a) == R1.dst
                       and const_f.get(s.b) == float(seg))
            if mMU is None:
                continue
            _, MU = mMU
            muv = MU.dst

            def is_x_minus_mu(buf: int) -> bool:
                s = is_op(buf, OP_SUB_F32)
                return bool(s and s.a == X and s.n == total
                            and bcast_src(s.b) == muv)

            # (x-mu)^2 -> sum -> var = mean - mu^2 chain
            mSQ = find(lambda j, s: s.op == OP_MUL_F32 and s.a == s.b
                       and s.n == total and is_x_minus_mu(s.a))
            if mSQ is None:
                continue
            _, SQ = mSQ
            mR2 = find(lambda j, s: s.op == OP_REDUCE_SEG and s.a == SQ.dst
                       and _redseg(s.imm) == (_RSEG_SUM, seg) and s.n == n_out)
            if mR2 is None:
                continue
            _, R2 = mR2
            mVAR = find(lambda j, s: s.op == OP_DIV_F32 and s.n == n_out
                        and bcast_src(s.a) == R2.dst
                        and const_f.get(s.b) == float(seg))
            if mVAR is None:
                continue
            _, VAR = mVAR
            # var + eps (affine, scale 1) -> ^-0.5 (pow) -> (x-mu)*rsqrt
            mAFF = find(lambda j, s: s.op == OP_AFFINE_F32 and s.a == VAR.dst
                        and s.n == n_out and _f32_from_bits(s.imm) == 1.0)
            if mAFF is None:
                continue
            _, AFF = mAFF
            eps_bits = AFF.imm2
            mPOW = find(lambda j, s: s.op == OP_POW_F32 and s.a == AFF.dst
                        and const_f.get(s.b) == -0.5)
            if mPOW is None:
                continue
            _, POW = mPOW
            mOUT = find(lambda j, s: s.op == OP_MUL_F32 and s.n == total
                        and is_x_minus_mu(s.a) and bcast_src(s.b) == POW.dst)
            if mOUT is None:
                continue
            oi, OUT = mOUT
            if OUT.dst in outs and X == OUT.dst:
                continue
            instrs[oi] = Instr(OP_LAYERNORM, dst=OUT.dst, a=X, b=X,
                               n=n_out, imm=seg, imm2=eps_bits)
            changed = True
            break


# GPT-2 tanh-approx GELU constants, compared as f32 (jax folds these exact
# literals into OP_AFFINE_F32 scales; see _fuse_gelu).
_GELU_C_CUBE = float(np.float32(0.044715))      # inner cubic coefficient
_GELU_C_SQRT = float(np.float32(0.7978845608))  # sqrt(2/pi)


def _capprox(v, ref: float, rtol: float = 1e-6) -> bool:
    return v is not None and abs(v - ref) <= rtol * abs(ref) + 1e-12


def _fuse_gelu(ctx: _Ctx) -> None:
    """Recognize the GPT-2 tanh-approx GELU idiom
    `0.5*x*(1+tanh(0.7978845608*(x + 0.044715*x^3)))` in OUR lowered VM-instr
    stream and collapse it into a SINGLE dedicated unary op (OP_GELU) that
    computes the whole thing per element in registers — one global read + one
    write instead of ~8 EW ops each round-tripping an intermediate through the
    arena (§19b/§24: dedicated OP_GELU ~4× on its component at base, where the
    general region-op manages only 1.3×).

    Pure elementwise (no reduce/segment): this is a plain EW-unary subop on the
    existing TILE_EW path — no new SLM, no tile-op family.

    Both spellings of the idiom reach us with an identical backbone (all ops
    reuse the input buffer X):
        x2  = x*x                        (MUL, a==b==X)
        x3  = x2*x                       (MUL, {x2, X})
        c   = 0.044715 * x3              (AFFINE s=0.044715, t=0)
        s   = x + c                      (ADD, {X, c})
        i   = 0.7978845608 * s           (AFFINE s=0.7978845608, t=0)
        t   = tanh(i)                    (TANH)
    …and differ ONLY in how the `0.5*x*(1+tanh)` tail is factored:
      - jax.nn.gelu:      out = x * (0.5*t + 0.5)      (TF s=t=0.5;  x direct)
      - manual `0.5*x*(1+tanh)`: out = (0.5*x) * (1*t + 1)  (TF s=t=1; XF s=0.5)
    Generalize the tail: the final MUL is `(s_x·X) · (s_t·(t+1))` where the tanh
    factor is an affine with bias==scale (= k·(tanh+1)) and the x factor is X
    directly (s_x=1) or `affine(X, s_x, 0)`, gated on `s_x·s_t == 0.5`. This
    matches any mathematically-equivalent factoring (correctness rests on the
    algebra, not a specific spelling), while the backbone constants
    0.044715 / 0.7978845608 stay exact-checked and X stays reused throughout.

    GATED HARD on every op kind + the exact backbone constants + producer→
    consumer linkage + the reused-X thread + the tail scale product; ANY
    mismatch leaves the decomposed chain untouched (never wrong, only
    sometimes-unfused). PJRT_OCL_FUSE_GELU=0 disables it. Runs after _fuse_norm
    on the cleaned stream (post view-fold / affine-compose / DCE); the following
    _dce_nops removes the dead chain."""
    if os.environ.get("PJRT_OCL_FUSE_GELU", "1") == "0":
        return
    instrs = ctx.instrs
    outs = set(ctx.outputs)

    def _writer_map() -> dict:
        seen: dict = {}
        multi: set = set()
        for idx, ins in enumerate(instrs):
            if ins.op == OP_NOP:
                continue
            d = ins.dst
            if d in seen or d in multi:
                seen.pop(d, None)
                multi.add(d)
            else:
                seen[d] = idx
        return seen

    changed = True
    while changed:
        changed = False
        writers = _writer_map()

        def producer(buf: int, op: int):
            """The unique writer of `buf` if it is `op`, else None."""
            w = writers.get(buf)
            return None if w is None else (instrs[w] if instrs[w].op == op else None)

        def affine_c(buf: int, s_ref: float, t_ref: float):
            """producer of `buf` iff OP_AFFINE_F32 with matching scale/bias."""
            a = producer(buf, OP_AFFINE_F32)
            if a is None or not _capprox(_f32_from_bits(a.imm), s_ref):
                return None
            return a if _capprox(_f32_from_bits(a.imm2), t_ref) else None

        def backbone(tanh_in: int, total: int):
            """Verify tanh_in = 0.7978845608*(X + 0.044715*X^3) and return X,
            else None. Tries both add-operand orderings to locate X."""
            AFFI = affine_c(tanh_in, _GELU_C_SQRT, 0.0)
            if AFFI is None or AFFI.n != total:
                return None
            ADD = producer(AFFI.a, OP_ADD_F32)
            if ADD is None or ADD.n != total or ADD.imm or ADD.imm2:
                return None
            for X, cbuf in ((ADD.a, ADD.b), (ADD.b, ADD.a)):
                AFFC = affine_c(cbuf, _GELU_C_CUBE, 0.0)
                if AFFC is None or AFFC.n != total:
                    continue
                MUL3 = producer(AFFC.a, OP_MUL_F32)
                if MUL3 is None or MUL3.n != total or MUL3.imm or MUL3.imm2:
                    continue
                x2 = (MUL3.b if MUL3.a == X else
                      MUL3.a if MUL3.b == X else None)
                if x2 is None:
                    continue
                MUL2 = producer(x2, OP_MUL_F32)
                if (MUL2 is None or MUL2.n != total or MUL2.imm or MUL2.imm2
                        or MUL2.a != X or MUL2.b != X):
                    continue
                return X
            return None

        for oi, OUT in enumerate(instrs):
            # anchor: the final `out = (s_x·x) · (s_t·(1+tanh))`
            if OUT.op != OP_MUL_F32 or OUT.imm or OUT.imm2:
                continue          # a direct (unviewed) multiply only
            total = OUT.n
            done = False
            # one operand is the tanh factor, the other the x factor
            for tf_buf, xf_buf in ((OUT.a, OUT.b), (OUT.b, OUT.a)):
                # tanh factor: affine(tanh, s_t, t_t) with t_t == s_t (= k*(1+t))
                TF = producer(tf_buf, OP_AFFINE_F32)
                if TF is None or TF.n != total:
                    continue
                s_t = _f32_from_bits(TF.imm)
                if not _capprox(_f32_from_bits(TF.imm2), s_t):
                    continue
                TANH = producer(TF.a, OP_TANH_F32)
                if TANH is None or TANH.n != total:
                    continue
                X = backbone(TANH.a, total)
                if X is None:
                    continue
                # x factor: X directly (s_x=1) or affine(X, s_x, 0)
                if xf_buf == X:
                    s_x = 1.0
                else:
                    XF = producer(xf_buf, OP_AFFINE_F32)
                    if (XF is None or XF.a != X or XF.n != total
                            or not _capprox(_f32_from_bits(XF.imm2), 0.0)):
                        continue
                    s_x = _f32_from_bits(XF.imm)
                # combined scale must be exactly 0.5: (s_x·X)·(s_t·(1+t))
                if not _capprox(s_x * s_t, 0.5):
                    continue
                # in-place self-write guard (mirror _fuse_norm).
                if OUT.dst in outs and X == OUT.dst:
                    continue
                instrs[oi] = Instr(OP_GELU, dst=OUT.dst, a=X, b=X, n=total)
                changed = done = True
                break
            if done:
                break


# --- general register-resident map-region fusion (§23/§27/§28) ---------------
#
# The culmination of the §23 arc: collapse a maximal run of pure-map f32 EW ops
# whose intermediates round-trip through the arena into ONE register-resident
# OP_MAP_REGION. K ops + K−1 barrier phases + K global round-trips → 1 phase, 1
# load per input + 1 store, intermediates kept in per-thread float4 slots. This
# GENERALIZES the hand-fusions (§11 chain, §19 norms, §26 gelu) into one
# mechanism; the dedicated OP_SOFTMAX/LAYERNORM/GELU are single ops = REGION
# BOUNDARIES (not in _REGION_KIND), as are all cross-lane ops (reduce, matmul,
# gather/scatter, dynamic index) — the region is bounded by anything non-map.
#
# Occupancy is FREE inside the one megakernel (§27, measured): per-thread float4
# slots overlap the matmul case's registers (max-not-sum), no SLM. So no VM
# split, no relaunch.

# tensor opcode -> (region micro-op SUB_* kind, arity). SUB_* MUST match
# vm_common.cl / ops/region.cl (the kernel switches on these). arity picks how
# operand/imm fields map to the {a_slot,b_slot,s,t} micro-op word.
_RSUB_ADD, _RSUB_MUL, _RSUB_SUB, _RSUB_DIV, _RSUB_MAX, _RSUB_MIN = 0, 1, 2, 3, 4, 5
_RSUB_NEG, _RSUB_EXP, _RSUB_LOG, _RSUB_SQRT, _RSUB_RSQRT = 8, 9, 10, 11, 12
_RSUB_TANH, _RSUB_ABS, _RSUB_AFFINE = 13, 14, 40
_REGION_NONE = 0xFFFF

_REGION_KIND = {
    OP_ADD_F32: (_RSUB_ADD, "bin"), OP_SUB_F32: (_RSUB_SUB, "bin"),
    OP_MUL_F32: (_RSUB_MUL, "bin"), OP_DIV_F32: (_RSUB_DIV, "bin"),
    OP_MAX_F32: (_RSUB_MAX, "bin"), OP_MIN_F32: (_RSUB_MIN, "bin"),
    OP_NEG_F32: (_RSUB_NEG, "un"), OP_EXP_F32: (_RSUB_EXP, "un"),
    OP_LOG_F32: (_RSUB_LOG, "un"), OP_SQRT_F32: (_RSUB_SQRT, "un"),
    OP_RSQRT_F32: (_RSUB_RSQRT, "un"), OP_TANH_F32: (_RSUB_TANH, "un"),
    OP_ABS_F32: (_RSUB_ABS, "un"), OP_AFFINE_F32: (_RSUB_AFFINE, "affine"),
}

# Kernel per-thread slot budget (ops/region.cl REGION_NSLOTS). The recognizer's
# budget may be set lower via env to force the over-budget split in tests.
_REGION_NSLOTS = 8
# Max region inputs (ops/region.cl REGION_MAXIN): a, b, p2..p7 task fields.
_REGION_MAXIN = 8


def _region_budget() -> int:
    try:
        b = int(os.environ.get("PJRT_OCL_REGION_SLOTS", str(_REGION_NSLOTS)))
    except ValueError:
        b = _REGION_NSLOTS
    return max(2, min(b, _REGION_NSLOTS))


def _region_operands(ins: Instr) -> tuple[int, ...]:
    """Ordered (a[, b]) operand buffers of an eligible EW instr — the values the
    micro-op reads. Binary reads a,b; unary/affine read a only."""
    _, arity = _REGION_KIND[ins.op]
    return (ins.a,) if arity != "bin" else (ins.a, ins.b)


def _region_eligible(ctx: _Ctx, ins: Instr) -> bool:
    """A pure-map f32 EW op the region interpreter can execute. Gated HARD:
    result + every operand f32, and (for viewable binary/unary ops) operands
    read DIRECTLY — a folded strided VIEW (imm/imm2 != 0) is not fused in v1
    (the kernel reads region inputs contiguously). Affine's imm/imm2 are its
    s/t immediates, not views, so it is always direct."""
    kind = _REGION_KIND.get(ins.op)
    if kind is None:
        return False
    if ctx.buffers[ins.dst].dtype != DT_F32:
        return False
    for b in _region_operands(ins):
        if b >= len(ctx.buffers) or ctx.buffers[b].dtype != DT_F32:
            return False
    if kind[1] != "affine" and (ins.imm != 0 or ins.imm2 != 0):
        return False                    # viewed operand — not fused in v1
    return True


def _region_slots(members: list[int], instrs, inputs: list[int],
                  out_buf: int) -> dict[int, int] | None:
    """Linear-scan slot allocation over a region (members in SSA order). Values =
    external `inputs` (live from the start) + each member's dst. Returns
    {buffer_id: slot} whose max slot < budget, or None if it needs more slots
    than the budget allows. `out_buf` lives past the region (used at the end)."""
    budget = _region_budget()
    k = len(members)
    produced = {instrs[m].dst for m in members}   # region-internal buffer ids
    in_set = set(inputs)
    # last use position of each value (member index that last reads it); out_buf
    # is read at the end (k). Inputs/temps: last member reading them.
    last: dict[int, int] = {}
    for i, m in enumerate(members):
        for b in _reads_of(instrs[m]):
            if b in produced or b in in_set:
                last[b] = max(last.get(b, -1), i)
    for b in (*inputs, *produced):
        last.setdefault(b, -1)
    last[out_buf] = k                    # externally live: survives the region

    active: dict[int, int] = {}          # value -> slot
    free: list[int] = []
    nxt = [0]

    def alloc() -> int:
        if free:
            return free.pop()
        s = nxt[0]
        nxt[0] += 1
        return s

    slot: dict[int, int] = {}
    for b in inputs:                     # inputs live from step 0
        slot[b] = alloc()
        active[b] = slot[b]
    for i, m in enumerate(members):
        # expire strictly-dead values (last use < i) so their slots free up
        for v in [v for v in list(active) if last.get(v, -1) < i]:
            free.append(active.pop(v))
        d = instrs[m].dst
        s = alloc()
        slot[d] = s
        active[d] = s
        if nxt[0] > budget:
            return None                  # peak distinct slots exceeds budget
    return slot if nxt[0] <= budget else None


def _region_phase_map(ctx: _Ctx, main_len: int) -> dict[int, int]:
    """Instr index -> scheduler phase number (only for root compute instrs).
    Mirrors _reuse_arena: two ops fuse into a region ONLY if the scheduler
    co-locates them in one barrier phase, so a lane-local map chain fuses but a
    dependency threading the whole program (the residual stream, whose links sit
    in different phases separated by the attention/FFN/norm boundaries between
    them) never does."""
    from . import scheduler as S

    class _PV:
        pass
    pv = _PV()
    pv.instrs = ctx.instrs
    pv.buffers = ctx.buffers
    pv.main_len = main_len
    sc = S._Scheduler(pv, S.DeviceConfig.from_env(), 1)
    levels = sc._build_levels(list(range(main_len)))
    tphase: dict[int, int] = {}
    for ph, (kind, payload) in enumerate(levels):
        if kind == "compute":
            for idx in payload:
                tphase[idx] = ph
    return tphase


def _region_phase_map_scoped(ctx: _Ctx, main_len: int) -> dict[int, int]:
    """Like _region_phase_map, but also phases every loop-body sub-list so
    region fusion can fire INSIDE a while/for body (§28's "next step": the
    laggard scan/loop workloads run a long serial per-iteration EW chain that
    never leaves the body sub-range, so root-only fusion never sees it).

    Each scope (root + every FOR/WHILE body, nested included) is phased with the
    SAME scheduler `_build_levels` that drives the barrier layout, then given a
    DISJOINT phase-number band so the within-phase gate in _fuse_region can only
    connect ops that are co-scheduled in ONE barrier phase of the SAME sub-list
    — never a root op to a body op, never across two bodies. Loop carries stay
    excluded by the global multi-writer test in _fuse_region (a carry is written
    by its root init-copy AND its body commit ⇒ multi ⇒ never a region member or
    the fused read's definition), so a region only ever produces fresh in-body
    SSA temporaries and reads carries/inputs read-only — the previous iteration's
    value, exactly as the decomposed chain does."""
    from . import scheduler as S

    class _PV:
        pass
    pv = _PV()
    pv.instrs = ctx.instrs
    pv.buffers = ctx.buffers
    pv.main_len = main_len
    sc = S._Scheduler(pv, S.DeviceConfig.from_env(), 1)
    tphase: dict[int, int] = {}
    base = 0

    def _do(indices: list[int]) -> None:
        nonlocal base
        levels = sc._build_levels(indices)
        for ph, (kind, payload) in enumerate(levels):
            if kind == "compute":
                for idx in payload:
                    tphase[idx] = base + ph
        base += len(levels) + 1          # disjoint band per scope

    _do(list(range(main_len)))
    for ins in ctx.instrs:
        if ins.op in (OP_FOR, OP_WHILE) and ins.imm:
            _do(list(range(ins.n, ins.n + ins.imm)))
    return tphase


def _fuse_region(ctx: _Ctx, main_len: int) -> None:
    """Recognize maximal map-regions — WITHIN-PHASE connected runs of pure-map
    f32 EW ops (elementwise + affine) bounded by cross-lane ops and the
    dedicated fused ops — and collapse each single-output region into ONE
    OP_MAP_REGION whose aux descriptor carries a slot map + straight-line
    micro-program (one global load per input, one store; intermediates stay in
    per-thread float4 registers). Over-budget or >2-input regions are SPLIT into
    budget-sized single-output sub-regions (still one kernel; each sub-region's
    boundary tensor is materialized in the arena and read by the next). GATED
    HARD: co-scheduled in one phase, single externally-live output, ≤2 inputs
    per (sub)region, ≤budget slots, all-f32, no viewed operands; any mismatch
    leaves the decomposed chain untouched. A split that yields only singletons
    (no op-count reduction) is skipped. PJRT_OCL_FUSE_REGION=0 reverts. Runs
    after the other fusion passes; the following _dce_nops drops the now-dead
    members."""
    if os.environ.get("PJRT_OCL_FUSE_REGION", "1") == "0":
        return
    instrs = ctx.instrs
    outs = set(ctx.outputs)
    budget = _region_budget()
    # PJRT_OCL_FUSE_REGION_LOOP (default on): also fuse inside while/for bodies —
    # the serial per-iteration EW chains of scan/loop workloads live in the body
    # sub-range and root-only fusion never reaches them (§28 follow-up).
    loop_fuse = os.environ.get("PJRT_OCL_FUSE_REGION_LOOP", "1") != "0"
    if not loop_fuse:
        phase = _region_phase_map(ctx, main_len)
    else:
        phase = _region_phase_map_scoped(ctx, main_len)
    # Loop-body index ranges. A region inside a body is barrier-ISOLATED (a
    # map-region is a phase boundary), so it only pays off when it collapses a
    # LONG lane-local chain — a body's whole EW chain runs in one barrier-free
    # phase, and replacing a short run of it with a region that needs its own
    # phase can add barriers for little op-count win (rk4's 3–7-op cones). Gate
    # loop-body cones on a minimum size; the root has no such penalty (its
    # regions sit between shaped-op phase boundaries that exist anyway) and is
    # ungated. PJRT_OCL_REGION_LOOP_MIN overrides the threshold.
    body_ranges = [(ins.n, ins.n + ins.imm) for ins in instrs
                   if ins.op in (OP_FOR, OP_WHILE) and ins.imm] if loop_fuse else []
    try:
        loop_min = int(os.environ.get("PJRT_OCL_REGION_LOOP_MIN", "8"))
    except ValueError:
        loop_min = 8

    def _in_loop_body(idx: int) -> bool:
        return any(lo <= idx < hi for lo, hi in body_ranges)

    # unique writer of each buffer (multi-write buffers, e.g. loop carries, are
    # excluded from regions — the fused reads must see one definition).
    writer: dict[int, int] = {}
    multi: set[int] = set()
    for idx, ins in enumerate(instrs):
        if ins.op == OP_NOP:
            continue
        if ins.dst in writer or ins.dst in multi:
            writer.pop(ins.dst, None)
            multi.add(ins.dst)
        else:
            writer[ins.dst] = idx

    elig = [idx for idx, ins in enumerate(instrs)
            if _region_eligible(ctx, ins) and ins.dst not in multi
            and idx in phase]
    if len(elig) < 2:
        return
    eset = set(elig)

    # union-find: connect two eligible members in the SAME phase when one reads
    # the other's dst (a lane-local map dep — the scheduler runs them on one lane
    # already; the region additionally keeps the intermediate off the arena).
    parent = {i: i for i in elig}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for j in elig:
        for b in _reads_of(instrs[j]):
            w = writer.get(b)
            if (w is not None and w in eset and phase[w] == phase[j]
                    and find(w) != find(j)):
                parent[find(w)] = find(j)

    comps: dict[int, list[int]] = {}
    for i in elig:
        comps.setdefault(find(i), []).append(i)

    for members in comps.values():
        if len(members) < 2:
            continue
        members = sorted(members)                 # SSA (topological) order
        mset = set(members)
        n = instrs[members[0]].n
        if any(instrs[m].n != n for m in members):
            continue                              # all one element count
        produced = {instrs[m].dst for m in members}
        # externally-live members: dst is a program output, or read by a
        # non-member live instr, or written more than once.
        readers_out: dict[int, int] = {}          # member dst -> external reads
        for j, ins in enumerate(instrs):
            if ins.op == OP_NOP or j in mset:
                continue
            for b in _reads_of(ins):
                if b in produced:
                    readers_out[b] = readers_out.get(b, 0) + 1
        live_outs = [m for m in members
                     if instrs[m].dst in outs or readers_out.get(instrs[m].dst)]
        live_dsts = {instrs[m].dst for m in live_outs}
        # ROOT regions keep the pre-existing behavior (single live output, ≤2
        # inputs) so tuned root scheduling is byte-unchanged. LOOP-BODY regions
        # (§28 follow-up) get the new lever: split a multi-output connected
        # component into one single-output sub-region per live output = its
        # backward fan-in cone (stopping at — and reading as inputs — other live
        # outputs and members SHARED by ≥2 cones, which stay decomposed). Every
        # emitted region still has ONE writer/output, so the scheduler/arena/
        # liveness are untouched; up to REGION_MAXIN inputs let a carry's whole
        # rate chain collapse (HH neuron's new_m/new_h/new_n/new_V).
        in_body = _in_loop_body(members[0])
        maxin = _REGION_MAXIN if in_body else 2
        if in_body:
            cones = _output_cones(instrs, members, mset, produced, live_dsts)
        elif len(live_dsts) == 1:
            cones = [(members, next(iter(live_dsts)))]
        else:
            continue                              # root single-output gate
        for cone_members, out_buf in cones:
            if len(cone_members) < 2:
                continue
            # loop-body cones must be long enough to earn their barrier phase.
            if in_body and len(cone_members) < loop_min:
                continue
            subregions = _split_region(instrs, cone_members, out_buf, budget, maxin)
            if subregions is None:
                continue                          # can't fit/split → decomposed
            # a split into only singletons replaces plain EW ops with regions
            # 1:1 — no phase/round-trip win; leave that cone decomposed.
            if max(len(s) for s, _ in subregions) < 2:
                continue
            for sub_members, sub_out in subregions:
                _emit_region(ctx, sub_members, sub_out)


def _output_cones(instrs, members: list[int], mset: set, produced: set,
                  live_dsts: set):
    """Split a connected map-region component into one single-output sub-region
    per live output. Each is the output's backward fan-in cone within the
    component, stopping at (reading as inputs) other live outputs and any member
    SHARED by ≥2 output cones. Shared members stay decomposed EW ops producing
    real buffers, so every emitted region has exactly one writer/output — the
    scheduler/arena/liveness never see a multi-write. Returns [(cone_members
    (SSA order), out_buf), ...]."""
    dstof = {instrs[m].dst: m for m in members}   # unique writer within comp
    live_out_members = [m for m in members if instrs[m].dst in live_dsts]

    def cone_of(om: int, stop_dsts: set) -> set:
        seen: set = set()
        stack = [om]
        while stack:
            m = stack.pop()
            if m in seen:
                continue
            seen.add(m)
            for b in _region_operands(instrs[m]):
                if b in produced and b not in stop_dsts:
                    w = dstof.get(b)
                    if w is not None and w in mset and w != m:
                        stack.append(w)
        return seen

    # raw cones (stop only at OTHER live outputs) to find shared members
    fanin: dict[int, int] = {}
    for om in live_out_members:
        O = instrs[om].dst
        for m in cone_of(om, live_dsts - {O}):
            fanin[m] = fanin.get(m, 0) + 1
    shared = {instrs[m].dst for m, c in fanin.items()
              if c > 1 and instrs[m].dst not in live_dsts}

    result = []
    for om in sorted(live_out_members):
        O = instrs[om].dst
        cm = sorted(cone_of(om, (live_dsts - {O}) | shared))
        result.append((cm, O))
    return result


def _split_region(instrs, members: list[int], out_buf: int, budget: int,
                  maxin: int = 2):
    """Partition `members` (SSA order) into consecutive single-output sub-regions
    each ≤maxin inputs / ≤budget slots. Returns [(sub_members, out_buf), ...] in
    order, or None if it cannot be split cleanly (→ fall back to decomposed).
    A sub-region's output is the member whose dst is read outside the sub-region
    (later members or externally); a clean cut needs exactly one such value."""
    produced_here: set[int] = set()      # boundary buffers already emitted
    later_reads: list[set[int]] = []     # reads by members strictly after i
    acc: set[int] = set()
    for m in reversed(members):
        later_reads.append(set(acc))
        for b in _reads_of(instrs[m]):
            acc.add(b)
    later_reads.reverse()                # later_reads[i] = reads of members[i+1:]

    def analyze(sub: list[int]):
        prod = {instrs[m].dst for m in sub}
        inputs = []
        seen = set()
        for m in sub:
            for b in _region_operands(instrs[m]):
                if b not in prod and b not in seen:
                    seen.add(b)
                    inputs.append(b)
        return prod, inputs

    subs: list[tuple[list[int], int]] = []
    cur: list[int] = []
    for i, m in enumerate(members):
        trial = cur + [m]
        _, inputs = analyze(trial)
        slot = _region_slots(trial, instrs, inputs, out_buf)
        fits = len(inputs) <= maxin and slot is not None
        if fits:
            cur = trial
            continue
        if not cur:
            return None                  # a single op doesn't fit → give up
        cut = _finalize_sub(instrs, cur, members, i, out_buf)
        if cut is None:
            return None
        subs.append(cut)
        produced_here.add(cut[1])
        cur = [m]
        _, inputs = analyze(cur)
        if (len(inputs) > maxin
                or _region_slots(cur, instrs, inputs, out_buf) is None):
            return None
    if cur:
        subs.append((cur, out_buf))
    # a single sub covering everything is the no-split case (still valid).
    return subs


def _finalize_sub(instrs, sub: list[int], members: list[int], next_i: int,
                  out_buf: int):
    """Close a sub-region `sub` (a prefix of `members`, members[next_i:] remain).
    Its output = the single member dst read by a later member or externally;
    returns (sub, out_buf) or None if the live-out is not exactly one value."""
    subset = set(sub)
    later = set(members[next_i:])
    live: set[int] = set()
    for m in sub:
        d = instrs[m].dst
        # read by a member NOT in this sub (later member or the whole-region out)
        read_later = any(d in _reads_of(instrs[o]) for o in later)
        if read_later or d == out_buf:
            live.add(d)
    if len(live) != 1:
        return None
    return (sub, next(iter(live)))


def _emit_region(ctx: _Ctx, members: list[int], out_buf: int) -> None:
    """Encode one single-output sub-region into the aux pool + emit OP_MAP_REGION
    at the output-producing member's index; NOP the rest. members: SSA order,
    ≤REGION_MAXIN external inputs, fits the slot budget (guaranteed by
    _split_region). Inputs ride task fields a,b,p2..p7 (region_inputs, ordered)."""
    instrs = ctx.instrs
    produced = {instrs[m].dst for m in members}
    inputs: list[int] = []
    seen: set[int] = set()
    for m in members:
        for b in _region_operands(instrs[m]):
            if b not in produced and b not in seen:
                seen.add(b)
                inputs.append(b)
    slot = _region_slots(members, instrs, inputs, out_buf)
    assert slot is not None and len(inputs) <= _REGION_MAXIN

    n = instrs[members[0]].n
    # descriptor: [n_in, out_slot, n_micro, in_slot×n_in] + n_micro×6 words
    #   + in_handle×n_in (buffer ids, for the tensor validator after re-parse;
    #   the device/schedule-sim read the handles from the task's a,b,p2..p7).
    desc = [len(inputs), slot[out_buf], len(members)]
    desc += [slot[b] for b in inputs]
    for m in members:
        ins = instrs[m]
        kind, arity = _REGION_KIND[ins.op]
        a_slot = slot[ins.a]
        if arity == "bin":
            b_slot = slot[ins.b]
            s_bits, t_bits = 0, 0
        elif arity == "affine":
            b_slot = a_slot
            s_bits, t_bits = ins.imm, ins.imm2
        else:                            # unary
            b_slot = a_slot
            s_bits, t_bits = 0, 0
        desc += [kind, slot[ins.dst], a_slot, b_slot, s_bits, t_bits]
    desc += list(inputs)                 # trailing input handles (validator a)
    aux_off = ctx.add_aux(desc)

    out_idx = max(m for m in members if instrs[m].dst == out_buf)
    in0 = inputs[0]
    b_field = inputs[1] if len(inputs) >= 2 else in0
    for m in members:
        instrs[m] = Instr(OP_NOP)
    instrs[out_idx] = Instr(OP_MAP_REGION, dst=out_buf, a=in0, b=b_field, n=n,
                            aux=aux_off, region_inputs=tuple(inputs))


def _fuse_matmul_views(ctx: _Ctx) -> None:
    """Access-map fusion for matmul (§13, §14): fold a shape op (transpose/
    reshape/broadcast/slice/reverse — all OP_GATHER_STRIDED) that feeds a
    matmul operand into the matmul's strided operand READ instead of
    materializing the whole tensor + a barrier phase.

    Attention does q.reshape(B,T,H,hd).transpose(0,2,1,3) then a batched dot;
    the transpose is a gather feeding the dot. Because the dot treats its
    operand as a contiguous [G,M,K] tensor (element (g,m,k) at flat index
    g*M*K+m*K+k) and the gather output IS that operand row-major, the dot can
    read the PRE-transpose source at src[view_index(flat)] using the SAME
    descriptor the gather already carries — no new math, just the flat [G,M,K]/
    [G,K,N] index passed through vmo_view_idx.

    A gather g folds iff every reader of its result is an OP_DOT reading it as
    operand a or b on a not-yet-viewed slot (mirrors _fuse_views' all-readers
    gate). Each such dot's operand is retargeted to the gather SOURCE with the
    gather's descriptor as its view; g becomes NOP (DCE'd). A gather with any
    non-dot reader (or already-viewed slot) stays materialized.

    PJRT_OCL_MM_VIEWFOLD=0 disables the fold (A/B + revert lever): the strided
    operand read replaces a materialize+contiguous read, a win only when the
    eliminated gather phase costs more than the extra per-load index math."""
    if os.environ.get("PJRT_OCL_MM_VIEWFOLD", "1") == "0":
        return
    instrs = ctx.instrs
    outs = set(ctx.outputs)
    changed = True
    while changed:
        changed = False
        rmap = _reader_index(instrs)
        for gi, g in enumerate(instrs):
            if g.op != OP_GATHER_STRIDED or g.dst in outs:
                continue
            if ctx.buffers[g.dst].dtype != DT_F32:
                continue
            gbuf = g.dst
            # rmap is built once per round; re-filter live (an earlier fold this
            # round may have NOP'd/retargeted a listed reader).
            readers = [(j, instrs[j]) for j in rmap.get(gbuf, ())
                       if instrs[j].op != OP_NOP and gbuf in _reads_of(instrs[j])]
            if not readers:
                continue

            def foldable(r):
                if r.op != OP_DOT:
                    return False
                # gbuf must NOT already be a materialized view SOURCE of this dot:
                # if another gather already folded onto a slot whose source is
                # gbuf (r.a==gbuf & aview, or r.b==gbuf & bview), gbuf's output is
                # read strided and must stay materialized — folding it away would
                # leave that view reading an unwritten buffer. (Hit when the SAME
                # tensor feeds one operand directly and the other via a transpose,
                # e.g. self-attention q @ q.T.)
                if (r.a == gbuf and r.aview != 0) or (r.b == gbuf and r.bview != 0):
                    return False
                # foldable only if gbuf is read on a direct (unviewed) slot.
                if r.a == gbuf and r.aview == 0:
                    return True
                if r.b == gbuf and r.bview == 0:
                    return True
                return False
            if not all(foldable(r) for _, r in readers):
                continue
            for j, r in readers:
                if r.a == gbuf and r.aview == 0:
                    r = dataclasses.replace(r, a=g.a, aview=g.aux + 1)
                if r.b == gbuf and r.bview == 0:
                    r = dataclasses.replace(r, b=g.a, bview=g.aux + 1)
                instrs[j] = r
            instrs[gi] = Instr(OP_NOP)
            changed = True


def _finalize_matmul_views(ctx: _Ctx) -> None:
    """Mirror each folded DOT's (aview, bview) AND its §33 R2c epilogue (epi
    descriptor offset + residual buf) into a 4-word aux header, pointing ins.aux
    at it, so the tensor-interpreter validator (which runs on the re-parsed,
    serialized bytecode and cannot see the in-memory aview/bview/epi) can recover
    them. The scheduler uses the in-memory fields directly (-> task p4/p5/p6/p7);
    the device uses the resulting task fields. Runs AFTER _fuse_mma_epilogue so
    the epilogue fields are settled."""
    for idx, ins in enumerate(ctx.instrs):
        if ins.op == OP_DOT and (ins.aview or ins.bview or ins.epi):
            hdr = ctx.add_aux([ins.aview, ins.bview, ins.epi, ins.epi_res])
            ctx.instrs[idx] = dataclasses.replace(ins, aux=hdr)


# --- §33 R2c: matmul-inclusive epilogue fusion --------------------------------
#
# A matmul computes its output TILE in registers/accumulators before the store
# (ops/mma.cl vmo_mma_tile). Fold a following pure-map EW chain
# (scale/bias/gelu/+residual) into that store so `matmul → {scale,gelu,+residual}`
# collapses from ≥2 barrier phases into ONE. The micro-program rides in the aux
# pool (task.p6) and the kernel runs the shared vmo_region_micro on each
# accumulator value before storing. Attacks the §29/§32 phase-count wall at the
# matmul boundary the §19/§26/§28 EW fusion never touched.

# SUB_* opcodes the epilogue micro-program uses (MUST match vm_common.cl).
_EPI_ADD, _EPI_AFFINE, _EPI_GELU = 0, 40, 41
# src: 0 = unary on the accumulator; 1 = binary reading p7 per-element
# (residual: p7[g*M*N + gr*N + gc]); 2 = binary reading p7 per-column (bias).
_EPI_SELF, _EPI_ELEM, _EPI_COL = 0, 1, 2


def _fuse_mma_epilogue(ctx: _Ctx, main_len: int) -> None:
    """Fold each matmul's following pure-map EW chain into the matmul store
    (§33 R2c). For every OP_DOT that will use TILE_MMA (skip the N==1 gemv route
    and viewed-output cases), greedily walk the SINGLE consumer of the matmul
    output while it is a fusible epilogue op with matching element count and is
    not a program output:
      OP_GELU            → gelu on the accumulator          (unary)
      OP_AFFINE_F32      → x*s+t on the accumulator          (unary; QKᵀ scale)
      OP_ADD_F32 + a full-size external `res` → +residual    (binary, one per DOT)
    Stop at the first non-eligible/multi-consumer/second-binary/size-mismatch.
    Retarget the DOT dst to the last folded op's dst, NOP the folded ops, encode
    the micro-program into aux, set DOT.epi / epi_res / reads_hint. Gated HARD;
    any mismatch leaves the decomposed chain untouched.  PJRT_OCL_FUSE_MMA_EPI=0
    reverts. Runs after _fuse_norm/_fuse_gelu/_fuse_region (so GELU/scale are
    already single ops) and before _reuse_arena."""
    if os.environ.get("PJRT_OCL_FUSE_MMA_EPI", "1") == "0":
        return
    # §36 hybrid: the standalone TF32 kernel (mm_tc) has no store-epilogue path,
    # so a matmul with a fused epilogue cannot be routed to it. When the hybrid
    # is enabled we keep the big FFN/projection matmuls epilogue-free (the GELU/
    # residual run as their own cheap VM phases) so they route to mm_tc — the
    # 1.4x large-config win depends on it. One switch: MM_HYBRID implies this.
    if os.environ.get("PJRT_OCL_MM_HYBRID", "0") not in ("", "0"):
        return
    instrs = ctx.instrs

    def _elems(buf: int) -> int:
        return ctx.buffers[buf].size_bytes // 4

    # unique writer + total reader count per buffer (over live, non-NOP instrs).
    writer: dict[int, int] = {}
    multi: set[int] = set()
    readers: dict[int, list[int]] = {}
    for idx, ins in enumerate(instrs):
        if ins.op == OP_NOP:
            continue
        d = ins.dst
        if d in writer or d in multi:
            writer.pop(d, None)
            multi.add(d)
        else:
            writer[d] = idx
        for b in _reads_of(ins):
            readers.setdefault(b, []).append(idx)
    outs = set(ctx.outputs)

    for di, DOT in enumerate(instrs):
        if DOT.op != OP_DOT:
            continue
        M, N = DOT.n, DOT.imm >> 16
        G = max(1, DOT.imm2)
        total = M * N * G
        # N==1 routes to the segmented-reduce gemv path (no epilogue) — skip.
        # A viewed OUTPUT never happens (dot writes contiguous); operand views
        # (aview/bview) are fine and independent of the epilogue.
        if N == 1 and G == 1:
            continue

        micros: list[tuple[int, int, int, int]] = []   # (kind, src, s, t)
        consumed: list[int] = []
        res_buf = 0
        binary_used = False
        cur = DOT.dst
        last_dst = DOT.dst
        while True:
            live_rs = [j for j in readers.get(cur, [])
                       if instrs[j].op != OP_NOP and j != di]
            if len(live_rs) != 1 or cur in outs or cur in multi:
                break
            ci = live_rs[0]
            c = instrs[ci]
            if c.dst in multi or c.n != total:
                break
            if c.op == OP_GELU and c.a == cur:
                micros.append((_EPI_GELU, _EPI_SELF, 0, 0))
            elif c.op == OP_AFFINE_F32 and c.a == cur:
                micros.append((_EPI_AFFINE, _EPI_SELF, c.imm, c.imm2))
            elif (c.op == OP_ADD_F32 and not binary_used and not c.imm
                  and not c.imm2 and (c.a == cur or c.b == cur)):
                res = c.b if c.a == cur else c.a
                # residual: a full-size external buffer, distinct from the fused
                # output, single-writer (not a loop carry), read directly.
                if res in (cur, c.dst) or res in multi or _elems(res) != total:
                    break
                # CRITICAL ordering gate: folding a binary makes the matmul DEPEND
                # on `res`, but the matmul keeps its (early) position in the instr
                # stream. `res` must therefore already be produced BEFORE the DOT
                # (else the tensor interp / same-phase schedule read it stale).
                # Program inputs (no writer) are available from the start. The
                # transformer's residual x is always the earlier block input, so
                # this holds; a `q+matmul` where q is computed later is rejected.
                wr = writer.get(res)
                if wr is not None and wr > di:
                    break
                micros.append((_EPI_ADD, _EPI_ELEM, 0, 0))
                res_buf = res
                binary_used = True
            else:
                break
            consumed.append(ci)
            last_dst = c.dst
            cur = c.dst

        if not micros:
            continue
        # encode: [n_micro] then n_micro × {kind, src, s_bits, t_bits}
        desc = [len(micros)]
        for kind, src, s, t in micros:
            desc += [kind, src, s & 0xFFFFFFFF, t & 0xFFFFFFFF]
        off = ctx.add_aux(desc)
        for ci in consumed:
            instrs[ci] = Instr(OP_NOP)
        new_hint = tuple(DOT.reads_hint) + ((res_buf,) if res_buf else ())
        instrs[di] = dataclasses.replace(
            DOT, dst=last_dst, epi=off + 1, epi_res=res_buf,
            reads_hint=new_hint)


def _gather_is_identity(ctx: _Ctx, aux_off: int, n: int) -> bool:
    """True iff the OP_GATHER_STRIDED descriptor at `aux_off` is a pure
    contiguous reshape (element i ↦ i): src_off == 0 and in_strides equal the
    row-major strides of out_dims, over exactly `n` elements. Such a gather can
    be skipped in the flash-attention walk (it only re-labels axes)."""
    aux = ctx.aux
    if aux_off + 1 > len(aux):
        return False
    rank = aux[aux_off]
    if rank <= 0 or aux_off + 1 + 2 * rank + 1 > len(aux):
        return False
    out_dims = [aux[aux_off + 1 + i] for i in range(rank)]
    in_strides = [_as_i32(aux[aux_off + 1 + rank + i]) for i in range(rank)]
    src_off = _as_i32(aux[aux_off + 1 + 2 * rank])
    total = 1
    for d in out_dims:
        total *= d
    if src_off != 0 or total != n:
        return False
    # element i ↦ i iff every axis with dim>1 carries its row-major stride
    # (size-1 axes never contribute — their index is always 0 — so their stride
    # is irrelevant; a leading size-1 axis is exactly why we can't require exact
    # equality). src_off already checked to be 0.
    acc, want = 1, [0] * rank
    for i in range(rank - 1, -1, -1):
        want[i] = acc
        acc *= out_dims[i]
    return all(out_dims[i] == 1 or in_strides[i] == want[i]
               for i in range(rank))


def _as_i32(u: int) -> int:
    return u - (1 << 32) if u >= (1 << 31) else u


def _epi_scale(ctx: _Ctx, epi: int):
    """If a DOT epilogue (aux at epi-1) is EXACTLY one AFFINE(x·s + 0) with a
    unary self source, return its scale s (float bits); else None (bail). Used to
    recover the QKᵀ ×(hd**-0.5) that _fuse_mma_epilogue folded into the DOT."""
    if not epi:
        return 0  # no epilogue → scale contribution is identity (handled = 1.0)
    aux = ctx.aux
    off = epi - 1
    if off < 0 or off + 1 > len(aux):
        return None
    nm = aux[off]
    if nm != 1 or off + 1 + 4 > len(aux):
        return None
    kind, src, s_bits, t_bits = (aux[off + 1], aux[off + 2],
                                 aux[off + 3], aux[off + 4])
    if kind != _EPI_AFFINE or src != _EPI_SELF or _bits_to_f32(t_bits) != 0.0:
        return None
    return s_bits


def _bits_to_f32(bits: int) -> float:
    import numpy as np
    return float(np.frombuffer(np.uint32(bits & 0xFFFFFFFF).tobytes(), "<f4")[0])


def _fuse_attention(ctx: _Ctx, main_len: int) -> None:
    """Recognize the batched per-head attention idiom  DOT(QKᵀ)[·scale] →
    softmax(-1) → DOT(AV)  and collapse it into ONE OP_FLASH_ATTN (online
    softmax), so the (T×C) score matrix never materializes and the two attention
    matmuls + the softmax reduce fold from 3 barriered phases into 1 (§34).

    Anchored on OP_SOFTMAX. Walks BACKWARD through an optional identity-reshape
    gather and an optional scale-affine to reach DOT1 (the QKᵀ matmul, whose
    ×scale may instead sit in its §33 epilogue), and FORWARD to the single
    consumer DOT2 (the AV matmul). Q/K/V and their FOLDED views (aview/bview) are
    carried verbatim into the fused op, which reads them through the SAME strided
    descriptors the matmuls used — so the result is byte-addressed identically to
    the decomposed path (decode: kv only; prefill: qv,kv,vv all fold).

    Gated HARD on every shape relation + single-consumer linkage + hd ≤ 256
    (local staging). Any mismatch leaves the decomposed DOT→softmax→DOT chain
    untouched (never wrong, only sometimes-unfused).

    DEFAULT OFF (`PJRT_OCL_FLASH=1` enables it): the scalar online-softmax kernel
    is a MEASURED regression on this workload (§34) — it replaces two TF32
    tensor-core matmuls with a scalar streaming kernel (prefill: up to 11× slower)
    and, at decode T=1, runs only H workgroups (severe underutilization). Kept,
    gated off, as the correct substrate for a future split-KV + tensor-core
    version. Runs after _fuse_mma_epilogue/_dce_nops (so the scale is a single
    epilogue or a single affine and softmax is one op) and before
    _finalize_matmul_views / _reuse_arena (so liveness sees the fused Q/K/V reads)."""
    if os.environ.get("PJRT_OCL_FLASH", "0") != "1":
        return
    instrs = ctx.instrs
    outs = set(ctx.outputs)
    _HD_MAX = 256

    def _elems(buf: int) -> int:
        return ctx.buffers[buf].size_bytes // 4

    # unique-writer + live-reader maps (over non-NOP instrs).
    writer: dict[int, int] = {}
    multi: set[int] = set()
    readers: dict[int, list[int]] = {}
    for idx, ins in enumerate(instrs):
        if ins.op == OP_NOP:
            continue
        d = ins.dst
        if d in writer or d in multi:
            writer.pop(d, None)
            multi.add(d)
        else:
            writer[d] = idx
        for b in _reads_of(ins):
            readers.setdefault(b, []).append(idx)

    def live_readers(buf: int) -> list[int]:
        return [j for j in readers.get(buf, []) if instrs[j].op != OP_NOP]

    import numpy as np

    for si, SM in enumerate(instrs):
        if SM.op != OP_SOFTMAX:
            continue
        C = SM.imm            # softmax segment length (= attention key count)
        n_out = SM.n          # G*M
        if C <= 0 or C > 0xFFFF:
            continue

        # --- backward: softmax.a → [reshape] → [affine·scale] → DOT1 -----------
        walked_scale = np.float32(1.0)
        consumed_mid: list[int] = []
        cur = SM.a
        DOT1i = None
        for _ in range(4):
            if cur in outs or cur in multi:
                break
            w = writer.get(cur)
            if w is None:
                break
            P = instrs[w]
            # the producer must be single-consumer (only the chain reads it) so
            # NOP'ing it later is safe.
            if len(live_readers(cur)) != 1:
                break
            if P.op == OP_DOT:
                DOT1i = w
                break
            if (P.op == OP_AFFINE_F32 and P.a == P.b and not P.imm2 and
                    _elems(P.dst) == _elems(P.a)):
                walked_scale = walked_scale * np.float32(_bits_to_f32(P.imm))
                consumed_mid.append(w)
                cur = P.a
                continue
            if (P.op == OP_GATHER_STRIDED and _elems(P.dst) == _elems(P.a) and
                    _gather_is_identity(ctx, P.aux, _elems(P.dst))):
                consumed_mid.append(w)
                cur = P.a
                continue
            break
        if DOT1i is None:
            continue
        DOT1 = instrs[DOT1i]
        if len(live_readers(DOT1.dst)) != 1:      # its only reader is the chain
            continue

        G = max(1, DOT1.imm2)
        M = DOT1.n
        N1 = DOT1.imm >> 16
        hd = DOT1.imm & 0xFFFF
        if N1 != C or n_out != G * M or hd <= 0 or hd > _HD_MAX:
            continue
        epi_s = _epi_scale(ctx, DOT1.epi)
        if epi_s is None:                          # epilogue present but not a pure scale
            continue
        scale = walked_scale
        if DOT1.epi:
            scale = scale * np.float32(_bits_to_f32(epi_s))

        # --- forward: softmax.dst → DOT2 (AV) ---------------------------------
        fwd = live_readers(SM.dst)
        if SM.dst in outs or len(fwd) != 1:
            continue
        DOT2i = fwd[0]
        DOT2 = instrs[DOT2i]
        if (DOT2.op != OP_DOT or DOT2.a != SM.dst or DOT2.aview != 0
                or DOT2.epi != 0):
            continue
        G2 = max(1, DOT2.imm2)
        M2 = DOT2.n
        N2 = DOT2.imm >> 16
        K2 = DOT2.imm & 0xFFFF
        if G2 != G or M2 != M or K2 != C or N2 != hd:
            continue

        Q, qv = DOT1.a, DOT1.aview
        K, kv = DOT1.b, DOT1.bview
        V, vv = DOT2.b, DOT2.bview
        out = DOT2.dst
        # sanity: the fused op reads Q/K/V/out with the sizes the matmuls used.
        if (_elems(out) != G * M * hd or _elems(Q) < G * M * hd):
            continue

        hdr = ctx.add_aux([G, M, C, hd, int(np.float32(scale).view(np.uint32)),
                           0, qv, kv, vv])   # causal = 0 (idiom has no mask)
        for j in consumed_mid + [DOT1i, si]:
            instrs[j] = Instr(OP_NOP)
        # emit the fused op in DOT2's slot (after Q/K/V are produced; its output
        # buffer + downstream consumers are unchanged).
        instrs[DOT2i] = Instr(OP_FLASH_ATTN, dst=out, a=Q, b=K,
                              n=G, imm=M, imm2=V, aux=hdr,
                              reads_hint=(V,))


def _reuse_arena(ctx: _Ctx, main_len: int) -> None:
    """Reassign arena byte offsets by live interval (offline linear-scan /
    register-allocation) so the arena is bounded by PEAK concurrent liveness,
    not the SUM of every intermediate ever emitted. The old bump allocator
    (`offset = _arena; _arena += size`) never reused a slot, so a multi-layer
    transformer's temporaries accumulated past the 2^31 offset cap (§16).

    Buffer IDs are UNCHANGED — only Buffer.arena_byte_offset moves. Everything
    downstream keys on IDs: the scheduler patches offsets into task fields from
    the buffer table, the runtime/validators read the table.

    LIVENESS IS MEASURED IN SCHEDULER PHASE TIME, not program-instruction order.
    The scheduler runs independent ops in parallel across lanes and inserts a
    global barrier only BETWEEN phases (scheduler._build_levels/_phases). Two
    buffers may share a slot only when a barrier is GUARANTEED between the last
    use of one and the first def of the other — i.e. their phase intervals are
    disjoint. Instruction-order liveness would be UNSAFE: an independent
    producer/consumer pair in the SAME phase runs concurrently on different
    lanes, so a recycled slot's write could clobber a still-live read. The
    scheduler assumes SSA (single-assignment) and adds NO WAR edge for the
    alias, so phase time is the correct granularity. We recompute the phase
    partition here from the SAME instrs + fuse flag the real scheduler uses
    (offsets don't affect it), so it matches the schedule that will execute.

    Correctness pins (an early free = SILENT memory corruption):
      * inputs  — non-port inputs are copied into the arena before phase 0, so
                  they are live from phase 0; zero-copy PORTS ignore the arena
                  offset entirely (runtime overrides with a port handle), so
                  pinning + not-relocating them is automatic/harmless.
      * outputs — read out (D2H) after the program ends: live to the very end.
      * consts  — uploaded once at load into their slot: live the whole program.
      * regions — a WHILE/FOR's entire sub-list (every iteration, nested regions
                  included) and its carries collapse to the region op's single
                  phase, so nothing a region touches is reused within OR across
                  the region. Conservative but safe (region arenas are tiny).
      * views   — a folded gather source (§13/§14a) is read by its viewer via the
                  operand's a/b field, so _reads_of already counts it as a read
                  of the SOURCE buffer; its interval extends to its last viewer.
      * in-place — dst==a ops contribute both a read and a write at the same
                  phase to the one (shared) buffer id; no cross-id aliasing.
    """
    from . import scheduler as S
    instrs = ctx.instrs
    n_buf = len(ctx.buffers)
    if n_buf == 0:
        return

    # --- phase partition (mirror the scheduler: same instrs, same fuse flag;
    # arena offsets don't affect _build_levels, so this equals the real one) ---
    class _ProgView:
        pass
    pv = _ProgView()
    pv.instrs = instrs
    pv.buffers = ctx.buffers
    pv.main_len = main_len
    sc = S._Scheduler(pv, S.DeviceConfig.from_env(), 1)
    levels = sc._build_levels(list(range(main_len)))
    n_phases = len(levels)

    tphase: dict[int, int] = {}                 # instr idx -> phase number
    region_ops: list[tuple[int, int]] = []      # (root region op idx, phase)
    for ph, (kind, payload) in enumerate(levels):
        if kind == "compute":
            for idx in payload:
                tphase[idx] = ph
        else:                                   # "while": one region op == a phase
            tphase[payload] = ph
            region_ops.append((payload, ph))

    def _subranges(ins: Instr) -> list[int]:
        if ins.op in (OP_WHILE, OP_IF):
            return (list(range(ins.a, ins.a + ins.b))       # cond / then sub-list
                    + list(range(ins.n, ins.n + ins.imm)))  # body / else sub-list
        if ins.op == OP_FOR:
            return list(range(ins.n, ins.n + ins.imm))      # body sub-list
        return []

    # collapse each region's whole (transitively nested) sub-list to its phase
    for r, ph in region_ops:
        stack = _subranges(instrs[r])
        while stack:
            j = stack.pop()
            tphase[j] = ph
            stack.extend(_subranges(instrs[j]))

    # --- per-buffer live interval in phase time ------------------------------
    INF = n_phases + 1
    lo = [INF] * n_buf
    hi = [-1] * n_buf

    def touch(b: int, t: int) -> None:
        if 0 <= b < n_buf:
            if t < lo[b]:
                lo[b] = t
            if t > hi[b]:
                hi[b] = t

    for idx, ins in enumerate(instrs):
        if ins.op == OP_NOP:
            continue
        t = tphase.get(idx)
        if t is None:
            continue                            # dead root instr not in any phase
        for b in _reads_of(ins):
            touch(b, t)
        if ins.op in (OP_WHILE, OP_IF, OP_FOR):
            if ins.op != OP_FOR and ins.dst:
                touch(ins.dst, t)               # WHILE cond flag: device-read/iter
        else:
            touch(ins.dst, t)                   # data write

    TEND = n_phases                             # strictly after phases 0..n_phases-1
    for b in set(ctx.inputs):
        lo[b] = 0                               # copied in before phase 0 (non-port)
        if hi[b] < 0:
            hi[b] = 0
    for b in set(ctx.outputs):
        if lo[b] == INF:
            lo[b] = 0
        hi[b] = TEND                            # read out after the program
    for b, _data in ctx.consts:
        lo[b] = 0
        hi[b] = TEND                            # uploaded once; never overwritten

    # --- offline greedy placement: biggest buffer first, lowest offset whose
    # [off, off+size) misses every already-placed buffer with an overlapping
    # phase interval. Inclusive overlap (l..h): buffers sharing a phase never
    # share a slot. Sizes 64B-aligned -> all offsets stay 64B-aligned. ---------
    def aligned(size: int) -> int:
        return -(-size // ARENA_ALIGN) * ARENA_ALIGN

    live = [b for b in range(n_buf) if hi[b] >= 0]
    live.sort(key=lambda b: (-aligned(ctx.buffers[b].size_bytes),
                             -(hi[b] - lo[b]), b))
    placed: list[tuple[int, int, int, int]] = []   # (lo, hi, off, end)
    arena_end = 0
    for b in live:
        size = aligned(ctx.buffers[b].size_bytes)
        l, h = lo[b], hi[b]
        conflicts = sorted((off, end) for (pl, ph_, off, end) in placed
                           if not (h < pl or ph_ < l))
        off = 0
        for c0, c1 in conflicts:
            if off + size <= c0:
                break                           # fits in the gap before c0
            if c1 > off:
                off = c1
        ctx.buffers[b].arena_byte_offset = off
        placed.append((l, h, off, off + size))
        if off + size > arena_end:
            arena_end = off + size

    # never-referenced buffers (DCE'd dead temps): no task ever touches them, so
    # any in-range offset is safe. Park at 0 (still counts toward arena_bytes so
    # the runtime's offset+size <= arena_bytes check passes).
    for b in range(n_buf):
        if hi[b] < 0:
            ctx.buffers[b].arena_byte_offset = 0
            arena_end = max(arena_end, aligned(ctx.buffers[b].size_bytes))

    ctx._arena = arena_end


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

    _TP = os.environ.get("PJRT_OCL_TIME_PASSES")
    if _TP:
        import sys as _sys, time as _time
        _t = [_time.time()]

        def _tick(name):
            now = _time.time()
            print(f"[time] {name}: {now - _t[0]:.2f}s "
                  f"(instrs={len(ctx.instrs)} bufs={len(ctx.buffers)})",
                  file=_sys.stderr, flush=True)
            _t[0] = now
    else:
        def _tick(name):
            pass

    for op in entry_block.operations:
        _lower_op(ctx, op.operation)
    _tick("root-lower")

    # The root list is exactly the entry-block lowering; region sub-lists
    # (cond/body of every while, nested included) are appended after it so the
    # root walk [0, main_len) never enters a sub-range (docs/vmprogram.md).
    main_len = len(ctx.instrs)
    while ctx.region_queue:
        job = ctx.region_queue.pop(0)
        if isinstance(job, _IfJob):
            _lower_if_regions(ctx, job)
        else:
            _lower_while_regions(ctx, job)
    _tick("region-lower")

    # perf peepholes (index-stable, NOP-substituting): collapse scale/bias chains
    # into one in-place affine pass, fold shape ops (broadcast/transpose/slice)
    # into consuming matmul operands (strided read) then elementwise operands
    # (strided views), then DCE the dead. Matmul fold runs first: a gather feeds
    # EITHER a dot or EW readers (disjoint by _fuse_views' viewable-EW gate), so
    # order only matters for a gather read by both — those don't fold either way.
    _compose_affines(ctx)
    _tick("compose_affines")
    _fuse_matmul_views(ctx)
    _tick("fuse_matmul_views")
    _fuse_views(ctx)
    _tick("fuse_views")
    _dce_nops(ctx)
    _tick("dce_nops")

    # Recognize softmax/layernorm reduce→broadcast idioms and collapse each into
    # one fused local-memory op (§19), then DCE the now-dead intermediates. Runs
    # on the cleaned stream (after view-fold/affine-compose/DCE) so the idiom is
    # in its canonical form; before _reuse_arena so liveness sees the fused op.
    _fuse_norm(ctx)
    _tick("fuse_norm")
    _fuse_gelu(ctx)
    _tick("fuse_gelu")
    # General register-resident map-region fusion (§23/§27/§28): collapse the
    # remaining pure-map EW chains (bounded by the dedicated fused ops above and
    # all cross-lane ops) into one OP_MAP_REGION each — K phases → 1. Runs last
    # so softmax/layernorm/gelu are already single ops (= region boundaries).
    _fuse_region(ctx, main_len)
    _tick("fuse_region")
    # DCE first so the dead gelu/norm backbones (_fuse_gelu leaves them for later
    # DCE) are gone — the epilogue recognizer's single-consumer test must see the
    # matmul output's ONLY live reader (the fused GELU/affine/residual op).
    _dce_nops(ctx)
    # §33 R2c: fold post-matmul map chains (scale/gelu/+residual) into the DOT
    # store-epilogue, collapsing the matmul→EW barrier boundary. Runs after the
    # EW/norm/gelu fusions so those are single ops = epilogue candidates, and
    # before _reuse_arena so liveness sees the retargeted DOT + residual read.
    _fuse_mma_epilogue(ctx, main_len)
    _tick("fuse_mma_epilogue")
    _dce_nops(ctx)
    # §34 flash-attention: collapse DOT(QKᵀ)·scale → softmax → DOT(AV) into ONE
    # online-softmax op (no materialized score matrix). Runs after the epilogue
    # fold (so QKᵀ's ×scale is a single epilogue/affine and softmax is one op)
    # and before _finalize_matmul_views/_reuse_arena. Reads Q/K/V through the
    # DOTs' own folded views, so it must see aview/bview BEFORE they are consumed.
    _fuse_attention(ctx, main_len)
    _tick("fuse_attention")
    _dce_nops(ctx)
    # Mirror DOT views + epilogue into the serialized aux header now that both
    # are settled, so the reparsed tensor validator can recover them.
    _finalize_matmul_views(ctx)
    _tick("finalize_matmul_views")

    # Liveness-reuse: rewrite arena offsets so the arena is bounded by peak
    # concurrent liveness, not the sum of every intermediate (§16). Runs AFTER
    # fusion/DCE (they NOP out buffers) and BEFORE the cap check (the backstop).
    _bump_arena = ctx._arena
    _reuse_arena(ctx, main_len)
    _tick("reuse_arena")
    if os.environ.get("PJRT_OCL_ARENA_DEBUG"):
        import sys
        print(f"[pjrt_ocl arena] bump={_bump_arena} "
              f"({_bump_arena / (1 << 20):.1f} MiB) -> reuse={ctx._arena} "
              f"({ctx._arena / (1 << 20):.1f} MiB)", file=sys.stderr)

    # Arena offsets are patched into u32 task fields and bit 31 is the I/O-port
    # flag (VMO_IO_BIT): an arena >= 2 GiB silently addresses the wrong memory
    # (poc/12: a force-unrolled 512-trip loop over 1M-element carries returned
    # inf). Fail the compile cleanly instead.
    if ctx._arena >= 1 << 31:
        raise LoweringError(
            f"arena {ctx._arena} bytes exceeds the 31-bit offset space "
            f"(u32 offsets, bit 31 = I/O-port flag); reduce the program's "
            f"working set (e.g. PJRT_OCL_WHILE=for instead of unroll)")

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


def _register_optional_dialects(context) -> None:
    """Register dialects that ride along in a portable artifact as no-ops on a
    single device but whose ops the deserializer must still be able to parse.

    Shardy ('sdy') is the one that matters in practice: any sharded JAX program
    (jax.jit under a mesh, shard_map, with_sharding_constraint) serializes its
    sharding hints as sdy ops (sdy.sharding_constraint / mesh / manual_computation)
    into the VHLO artifact. On our single OpenCL device these are identity — but
    without the dialect registered, deserialize_portable_artifact aborts with
    "dialect 'sdy' is unknown" before we ever see the compute. brax/MuJoCo-MJX
    hit exactly this. Register the dialect if jaxlib ships its bindings (it does
    since the Shardy migration); the later stablehlo walk skips sdy ops via the
    handlers in pjrt_ocl.ops (they carry no tensor result we consume)."""
    try:
        from jaxlib.mlir._mlir_libs import _sdy
    except Exception:  # noqa: BLE001 — older jaxlib without Shardy: nothing to do
        return
    try:
        _sdy.register_dialect(context, load=True)
    except Exception:  # noqa: BLE001 — best-effort; deserialize reports if needed
        pass


def deserialize_artifact(artifact: bytes):
    """VHLO portable artifact bytes -> stablehlo ir.Module (auto-upgraded)."""
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import stablehlo
    # No explicit dialect registration needed for stablehlo (poc/03 NOTES #5);
    # sdy (and any other sharding dialects) must be registered up front so the
    # portable-artifact deserializer can parse their ops (identity on one device).
    context = ir.Context()
    _register_optional_dialects(context)
    return stablehlo.deserialize_portable_artifact(context, artifact)


def lower_artifact(artifact: bytes) -> VMProgram:
    """PJRT_Client_Compile program bytes (VHLO artifact) -> VMProgram."""
    return lower_module(deserialize_artifact(artifact))
