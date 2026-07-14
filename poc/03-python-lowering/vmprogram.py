#!/usr/bin/env python
"""VMProgram v0 — strawman flat bytecode emitted by the Python lowerer, executed by the
C++/OpenCL VM. Strictly linear instruction lists, no jumps (see docs/decisions.md #1).

BINARY LAYOUT (all little-endian)
=================================

  [header] [buffer table] [constant pool] [instruction-list table]

Header (32 bytes):
  off  0  magic       u32   0x30504D56  (bytes b"VMP0")
  off  4  version     u32   0
  off  8  arena_size  u64   bytes of the single device arena (all buffers live here)
  off 16  n_buffers   u32
  off 20  n_consts    u32
  off 24  n_lists     u32   number of instruction lists (list 0 = entry point)
  off 28  reserved    u32   0

Dtype enum (u8) — only F32 is implemented; the rest are reserved:
  0=F32  1=F16  2=BF16  3=S32  4=S64  5=PRED

Buffer kind enum (u8):
  0=TEMP  1=ARG  2=RESULT  3=CONST

Buffer table (n_buffers entries, 24 bytes each; buffer id = table index).
Single-arena offsets model: the executor allocates ONE device allocation of
arena_size bytes; every buffer is (offset, size) into it. 64-byte aligned.
  off  0  offset   u64   byte offset into the arena
  off  8  size     u64   byte size
  off 16  n_elems  u32   logical element count
  off 20  dtype    u8
  off 21  kind     u8
  off 22  index    u16   position within its kind: ARG -> parameter index,
                         RESULT -> result index, CONST -> constant-pool index, TEMP -> 0

Constant pool (n_consts records, in CONST-index order, no padding):
  { buf_id u32, byte_len u32, data[byte_len] }
  data = raw little-endian element bytes; the executor memcpys each into the arena
  at its buffer's offset once, before first execution.

Instruction-list table (n_lists lists concatenated in id order; list 0 = entry):
  list := { n_instrs u32, n_instrs x instr }
  instr (24 bytes fixed):
    off  0  opcode   u16
    off  2  flags    u16   0
    off  4  out      u32   output buffer id
    off  8  a        u32   input buffer id (opcode-specific: may be a sub-list id)
    off 12  b        u32   input buffer id (opcode-specific: may be a sub-list id)
    off 16  n_elems  u32   iteration space for the grid-stride loop
    off 20  aux      u32   opcode-specific extra field

Opcodes (u16):
  0x0001 ADD    out[i] = a[i] + b[i]
  0x0002 MUL    out[i] = a[i] * b[i]
  0x0003 SUB    out[i] = a[i] - b[i]
  0x0004 COPY   out[i] = a[i]                      (b, aux unused)
  0x0040 WHILE  region-carrying op, spec'd for M4 (not yet emitted by this lowerer):
                a = cond sub-list id, b = body sub-list id, aux = predicate buffer id,
                out unused. Loop-carried values are ordinary buffers written in place.
                The VM alternates: run cond list, read pred buffer (scalar), if nonzero
                run body list, repeat. Sub-lists are linear lists in the same table —
                "linear lists all the way down", never a jump.

LOWERING (this file also implements stablehlo -> VMProgram for the elementwise subset)
======================================================================================
Supported today: stablehlo.add / multiply / subtract / constant, func args/results,
rank-N f32 tensors. Everything else raises NotImplementedError (per-op handlers live
in OP_HANDLERS — add a dict entry to support a new op).
Buffer plan is naive one-buffer-per-SSA-value; liveness-based reuse is M1 work.
"""
from __future__ import annotations

import dataclasses
import math
import struct

import numpy as np

MAGIC = 0x30504D56  # b"VMP0"
VERSION = 0
ALIGN = 64

# dtype enum
F32, F16, BF16, S32, S64, PRED = range(6)
DTYPE_NUMPY = {F32: np.dtype("<f4")}

# buffer kinds
TEMP, ARG, RESULT, CONST = range(4)
KIND_NAMES = {TEMP: "temp", ARG: "arg", RESULT: "result", CONST: "const"}

# opcodes
OP_ADD, OP_MUL, OP_SUB, OP_COPY = 0x0001, 0x0002, 0x0003, 0x0004
OP_WHILE = 0x0040
OP_NAMES = {OP_ADD: "add", OP_MUL: "mul", OP_SUB: "sub", OP_COPY: "copy",
            OP_WHILE: "while"}

_HEADER = struct.Struct("<IIQIIII")
_BUFENT = struct.Struct("<QQIBBH")
_CONSTHDR = struct.Struct("<II")
_INSTR = struct.Struct("<HHIIIII")


# ---------------------------------------------------------------------------
# In-memory model + writer/reader
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Buffer:
    offset: int
    size: int
    n_elems: int
    dtype: int
    kind: int
    index: int


@dataclasses.dataclass
class Instr:
    opcode: int
    out: int
    a: int
    b: int
    n_elems: int
    aux: int = 0
    flags: int = 0


@dataclasses.dataclass
class VMProgram:
    arena_size: int
    buffers: list[Buffer]
    consts: list[tuple[int, bytes]]        # (buf_id, raw bytes), in CONST-index order
    lists: list[list[Instr]]               # lists[0] = entry

    def serialize(self) -> bytes:
        out = bytearray()
        out += _HEADER.pack(MAGIC, VERSION, self.arena_size, len(self.buffers),
                            len(self.consts), len(self.lists), 0)
        for b in self.buffers:
            out += _BUFENT.pack(b.offset, b.size, b.n_elems, b.dtype, b.kind, b.index)
        for buf_id, data in self.consts:
            out += _CONSTHDR.pack(buf_id, len(data))
            out += data
        for lst in self.lists:
            out += struct.pack("<I", len(lst))
            for i in lst:
                out += _INSTR.pack(i.opcode, i.flags, i.out, i.a, i.b, i.n_elems, i.aux)
        return bytes(out)

    def dump(self) -> str:
        """Human-readable disassembly."""
        lines = [f"VMProgram v{VERSION}: arena={self.arena_size}B "
                 f"buffers={len(self.buffers)} consts={len(self.consts)} "
                 f"lists={len(self.lists)}"]
        for i, b in enumerate(self.buffers):
            lines.append(f"  buf[{i}] {KIND_NAMES[b.kind]}#{b.index} "
                         f"off={b.offset} size={b.size} n={b.n_elems} dt={b.dtype}")
        for li, lst in enumerate(self.lists):
            lines.append(f"  list[{li}]{' (entry)' if li == 0 else ''}:")
            for i in lst:
                lines.append(f"    {OP_NAMES.get(i.opcode, hex(i.opcode)):5s} "
                             f"out=%{i.out} a=%{i.a} b=%{i.b} n={i.n_elems} aux={i.aux}")
        return "\n".join(lines)


def parse(data: bytes) -> VMProgram:
    """Reader: the mirror image of VMProgram.serialize (reference for the C++ parser)."""
    magic, version, arena_size, n_buffers, n_consts, n_lists, _ = \
        _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic:#x}")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    pos = _HEADER.size

    buffers = []
    for _ in range(n_buffers):
        buffers.append(Buffer(*_BUFENT.unpack_from(data, pos)))
        pos += _BUFENT.size

    consts = []
    for _ in range(n_consts):
        buf_id, byte_len = _CONSTHDR.unpack_from(data, pos)
        pos += _CONSTHDR.size
        consts.append((buf_id, bytes(data[pos:pos + byte_len])))
        pos += byte_len

    lists = []
    for _ in range(n_lists):
        (n_instrs,) = struct.unpack_from("<I", data, pos)
        pos += 4
        lst = []
        for _ in range(n_instrs):
            opcode, flags, out, a, b, n_elems, aux = _INSTR.unpack_from(data, pos)
            pos += _INSTR.size
            lst.append(Instr(opcode, out, a, b, n_elems, aux, flags))
        lists.append(lst)

    if pos != len(data):
        raise ValueError(f"trailing bytes: parsed {pos} of {len(data)}")
    return VMProgram(arena_size, buffers, consts, lists)


# ---------------------------------------------------------------------------
# Lowering: stablehlo ir.Module -> VMProgram
# ---------------------------------------------------------------------------

class LoweringError(NotImplementedError):
    pass


class _Ctx:
    """Per-module lowering state."""

    def __init__(self):
        self.buffers: list[Buffer] = []
        self.consts: list[tuple[int, bytes]] = []
        self.value_to_buf: dict = {}      # ir.Value -> buffer id
        self.lists: list[list[Instr]] = [[]]
        self.cur_list = 0
        self._arena = 0

    def emit(self, instr: Instr):
        self.lists[self.cur_list].append(instr)

    def new_buffer(self, n_elems: int, dtype: int, kind: int = TEMP,
                   index: int = 0) -> int:
        size = n_elems * DTYPE_NUMPY[dtype].itemsize
        offset = self._arena
        self._arena += (size + ALIGN - 1) // ALIGN * ALIGN
        self.buffers.append(Buffer(offset, size, n_elems, dtype, kind, index))
        return len(self.buffers) - 1

    def buf_for(self, value) -> int:
        try:
            return self.value_to_buf[value]
        except KeyError:
            raise LoweringError(f"no buffer for SSA value {value}") from None


def _tensor_info(mlir_type) -> tuple[int, int]:
    """-> (n_elems, dtype enum). Only static rank-N f32 tensors are supported."""
    from jaxlib.mlir import ir
    # NB: jaxlib 0.10.2 bindings auto-downcast types and lack Type.isinstance();
    # plain python isinstance() is the supported check (see NOTES.md).
    if not isinstance(mlir_type, ir.RankedTensorType):
        raise LoweringError(f"unsupported type (not a ranked tensor): {mlir_type}")
    t = mlir_type
    if any(t.is_dynamic_dim(i) for i in range(t.rank)):
        raise LoweringError(f"dynamic shapes unsupported: {mlir_type}")
    if not isinstance(t.element_type, ir.F32Type):
        raise LoweringError(f"unsupported element type {t.element_type} "
                            f"(only f32 in poc/03)")
    return math.prod(t.shape) if t.rank else 1, F32


# --- per-op handlers: op name -> handler(ctx, op). Add entries to grow coverage. ---

OP_HANDLERS: dict = {}


def _handles(name):
    def deco(fn):
        OP_HANDLERS[name] = fn
        return fn
    return deco


def _elementwise_binop(opcode):
    def handler(ctx: _Ctx, op):
        n_elems, dtype = _tensor_info(op.results[0].type)
        for operand in op.operands:
            if _tensor_info(operand.type) != (n_elems, dtype):
                raise LoweringError(
                    f"{op.name}: operand/result shape mismatch (implicit broadcast?)")
        out = ctx.new_buffer(n_elems, dtype)
        ctx.emit(Instr(opcode, out, ctx.buf_for(op.operands[0]),
                       ctx.buf_for(op.operands[1]), n_elems))
        ctx.value_to_buf[op.results[0]] = out
    return handler


OP_HANDLERS["stablehlo.add"] = _elementwise_binop(OP_ADD)
OP_HANDLERS["stablehlo.multiply"] = _elementwise_binop(OP_MUL)
OP_HANDLERS["stablehlo.subtract"] = _elementwise_binop(OP_SUB)


@_handles("stablehlo.constant")
def _lower_constant(ctx: _Ctx, op):
    from jaxlib.mlir import ir
    n_elems, dtype = _tensor_info(op.results[0].type)
    attr = ir.DenseFPElementsAttr(op.attributes["value"])
    arr = np.asarray(attr, dtype=DTYPE_NUMPY[dtype]).reshape(-1)
    if arr.size == 1 and n_elems != 1:  # splat
        arr = np.broadcast_to(arr, (n_elems,)).copy()
    assert arr.size == n_elems, (arr.size, n_elems)
    buf = ctx.new_buffer(n_elems, dtype, kind=CONST, index=len(ctx.consts))
    ctx.consts.append((buf, arr.astype(DTYPE_NUMPY[dtype]).tobytes()))
    ctx.value_to_buf[op.results[0]] = buf


@_handles("func.return")
def _lower_return(ctx: _Ctx, op):
    for res_index, operand in enumerate(op.operands):
        buf_id = ctx.buf_for(operand)
        buf = ctx.buffers[buf_id]
        if buf.kind == TEMP:
            buf.kind, buf.index = RESULT, res_index
        else:
            # returning an arg/const/already-returned value: materialize a copy
            out = ctx.new_buffer(buf.n_elems, buf.dtype, kind=RESULT, index=res_index)
            ctx.emit(Instr(OP_COPY, out, buf_id, 0, buf.n_elems))


def lower_module(module) -> VMProgram:
    """Lower a deserialized stablehlo module's public @main to a VMProgram."""
    from jaxlib.mlir import ir
    main = None
    for op in module.body.operations:
        o = op.operation
        if o.name == "func.func" and ir.StringAttr(o.attributes["sym_name"]).value == "main":
            main = o
            break
    if main is None:
        raise LoweringError("no func.func @main in module")

    ctx = _Ctx()
    entry_block = main.regions[0].blocks[0]
    for arg_index, arg in enumerate(entry_block.arguments):
        n_elems, dtype = _tensor_info(arg.type)
        ctx.value_to_buf[arg] = ctx.new_buffer(n_elems, dtype, kind=ARG,
                                               index=arg_index)

    for op in entry_block.operations:
        o = op.operation
        handler = OP_HANDLERS.get(o.name)
        if handler is None:
            raise LoweringError(f"unsupported op: {o.name} "
                                f"(known: {sorted(OP_HANDLERS)})")
        handler(ctx, o)

    return VMProgram(ctx._arena, ctx.buffers, ctx.consts, ctx.lists)


def lower_artifact(artifact: bytes) -> VMProgram:
    """VHLO portable artifact bytes (as received by PJRT_Client_Compile) -> VMProgram."""
    import walk
    return lower_module(walk.deserialize(artifact))


if __name__ == "__main__":
    import sys
    import dump_stablehlo
    example = sys.argv[1] if len(sys.argv) > 1 else "fma_const"
    module = dump_stablehlo.lower_to_stablehlo_module(example)
    artifact = dump_stablehlo.serialize_as_plugin_would_receive(module)
    prog = lower_artifact(artifact)
    print(prog.dump())
    blob = prog.serialize()
    print(f"serialized: {len(blob)} bytes; reparse ok: {parse(blob) == prog}")
