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
import struct

import numpy as np

# --- format constants (docs/vmprogram.md) ----------------------------------

MAGIC = 0x314D5056  # b"VPM1" read little-endian
VERSION = 3         # v3 = v2 tensor sections + v2.1 schedule sections
ARENA_ALIGN = 64    # buffer arena offsets
SECTION_ALIGN = 8   # file sections / variable-length entries

DT_F32 = 0
DTYPE_NUMPY = {DT_F32: np.dtype("<f4")}

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
    OP_IOTA_DIM: "iota_dim", OP_IF: "if",
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
                                     ins.imm, ins.aux, 0)
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


def _tensor_info(mlir_type) -> tuple[tuple[int, ...], int, int]:
    """-> (shape, n_elems, dtype enum). Static rank-N f32 tensors only."""
    from jaxlib.mlir import ir
    # jaxlib 0.10.2 bindings auto-downcast types and lack Type.isinstance();
    # plain python isinstance() is the supported check (poc/03 NOTES #4).
    if not isinstance(mlir_type, ir.RankedTensorType):
        raise LoweringError(f"unsupported type (not a ranked tensor): {mlir_type}")
    t = mlir_type
    if any(t.is_dynamic_dim(i) for i in range(t.rank)):
        raise LoweringError(f"dynamic shapes unsupported: {mlir_type}")
    if not isinstance(t.element_type, ir.F32Type):
        raise LoweringError(
            f"unsupported element type {t.element_type} (v1 is f32-only)")
    shape = tuple(t.shape)
    return shape, math.prod(shape) if t.rank else 1, DT_F32


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


def _elementwise_binop(opcode):
    def handler(ctx: _Ctx, op):
        _, n_elems, dtype = _tensor_info(op.results[0].type)
        for operand in op.operands:
            if _tensor_info(operand.type)[1:] != (n_elems, dtype):
                raise LoweringError(
                    f"{op.name}: operand/result shape mismatch "
                    f"(implicit broadcast?)")
        dst = ctx.new_buffer(n_elems, dtype)
        ctx.emit(Instr(opcode, dst=dst, a=ctx.buf_for(op.operands[0]),
                       b=ctx.buf_for(op.operands[1]), n=n_elems))
        ctx.value_to_buf[op.results[0]] = dst
    return handler


OP_HANDLERS["stablehlo.add"] = _elementwise_binop(OP_ADD_F32)
OP_HANDLERS["stablehlo.multiply"] = _elementwise_binop(OP_MUL_F32)
OP_HANDLERS["stablehlo.subtract"] = _elementwise_binop(OP_SUB_F32)


@_handles("stablehlo.constant")
def _lower_constant(ctx: _Ctx, op):
    from jaxlib.mlir import ir
    _, n_elems, dtype = _tensor_info(op.results[0].type)
    attr = ir.DenseFPElementsAttr(op.attributes["value"])
    arr = np.asarray(attr, dtype=DTYPE_NUMPY[dtype]).reshape(-1)
    if arr.size == 1 and n_elems != 1:  # splat: expand into the const pool
        arr = np.broadcast_to(arr, (n_elems,)).copy()
    if arr.size != n_elems:
        raise LoweringError(
            f"stablehlo.constant: {arr.size} elements for type of {n_elems}")
    buf = ctx.new_buffer(n_elems, dtype)
    ctx.consts.append((buf, arr.astype(DTYPE_NUMPY[dtype]).tobytes()))
    ctx.value_to_buf[op.results[0]] = buf


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


def lower_module(module) -> VMProgram:
    """Lower a deserialized stablehlo module's public @main to a VMProgram."""
    from jaxlib.mlir import ir
    _ensure_ops_registered()
    main = None
    for op in module.body.operations:
        o = op.operation
        if (o.name == "func.func"
                and ir.StringAttr(o.attributes["sym_name"]).value == "main"):
            main = o
            break
    if main is None:
        raise LoweringError("no func.func @main in module")

    ctx = _Ctx()
    entry_block = main.regions[0].blocks[0]
    for arg in entry_block.arguments:
        shape, n_elems, dtype = _tensor_info(arg.type)
        buf = ctx.new_buffer(n_elems, dtype)
        ctx.value_to_buf[arg] = buf
        ctx.inputs.append(buf)
        ctx.input_shapes.append(shape)

    for op in entry_block.operations:
        o = op.operation
        handler = OP_HANDLERS.get(o.name)
        if handler is None:
            raise LoweringError(f"unsupported op: {o.name} "
                                f"(known: {sorted(OP_HANDLERS)})")
        handler(ctx, o)

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
        main_len=len(ctx.instrs),
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
