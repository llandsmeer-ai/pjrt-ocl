"""VMProgram v1 reader + numpy reference interpreter.

Consumer-side mirror of lowering.py: parses the binary format back into python
objects (the reference for the C++ parser in pjrt_plugin/runtime) and executes
it with numpy exactly as the OpenCL VM will — single arena, consts uploaded
once, inputs written per execute, root instruction list [0, main_len) run,
outputs read back. WHILE is interpreted per spec via plain recursion (the C++
executor uses a frame stack, depth <= 8).

No jax/jaxlib imports here — only numpy + stdlib.
"""
from __future__ import annotations

import dataclasses
import struct

import numpy as np

from .lowering import (
    ARENA_ALIGN, BUFENT_STRUCT, CONSTHDR_STRUCT, DT_F32, DTYPE_NUMPY,
    HEADER_STRUCT, INSTR_STRUCT, MAGIC, OP_ADD_F32, OP_FILL_F32, OP_IOTA_F32,
    OP_LTS_F32, OP_MUL_F32, OP_NAMES, OP_NOP, OP_SUB_F32, OP_WHILE,
    SECTION_ALIGN, VERSION,
)

MAX_WHILE_DEPTH = 8


@dataclasses.dataclass
class BufferEntry:
    arena_byte_offset: int
    size_bytes: int
    dtype: int


@dataclasses.dataclass
class Instr:
    op: int
    dst: int
    a: int
    b: int
    n: int
    imm: int


@dataclasses.dataclass
class Program:
    arena_bytes: int
    buffers: list[BufferEntry]
    inputs: list[int]
    outputs: list[int]
    input_shapes: list[tuple[int, ...]]
    output_shapes: list[tuple[int, ...]]
    consts: list[tuple[int, bytes]]
    instrs: list[Instr]
    main_len: int


class FormatError(ValueError):
    """Malformed VMProgram bytes (the C++ side rejects with INVALID_ARGUMENT)."""


def parse(data: bytes) -> Program:
    """Strict parse; validates magic/version/alignment/bounds/trailing bytes."""
    if len(data) < HEADER_STRUCT.size:
        raise FormatError(f"short file: {len(data)} bytes")
    (magic, version, n_buffers, n_instrs, n_consts, main_len, n_inputs,
     n_outputs, arena_bytes) = HEADER_STRUCT.unpack_from(data, 0)
    if magic != MAGIC:
        raise FormatError(f"bad magic {magic:#010x} (want {MAGIC:#010x})")
    if version != VERSION:
        raise FormatError(f"unsupported version {version} (want {VERSION})")
    if main_len > n_instrs:
        raise FormatError(f"main_len {main_len} > n_instrs {n_instrs}")
    pos = HEADER_STRUCT.size

    def check_aligned(what: str) -> None:
        if pos % SECTION_ALIGN:
            raise FormatError(f"{what} not {SECTION_ALIGN}B-aligned: {pos}")

    # buffer table
    check_aligned("buffer table")
    buffers: list[BufferEntry] = []
    for i in range(n_buffers):
        off, size, dtype, pad = BUFENT_STRUCT.unpack_from(data, pos)
        pos += BUFENT_STRUCT.size
        if off % ARENA_ALIGN:
            raise FormatError(f"buf[{i}] offset {off} not {ARENA_ALIGN}B-aligned")
        if off + size > arena_bytes:
            raise FormatError(f"buf[{i}] [{off},{off + size}) outside arena "
                              f"of {arena_bytes}")
        if dtype not in DTYPE_NUMPY:
            raise FormatError(f"buf[{i}] unknown dtype {dtype}")
        if pad != 0:
            raise FormatError(f"buf[{i}] nonzero pad {pad}")
        buffers.append(BufferEntry(off, size, dtype))

    # IO maps (each array padded to 8B)
    def read_io_map(count: int, what: str) -> list[int]:
        nonlocal pos
        check_aligned(f"{what} map")
        ids = list(struct.unpack_from(f"<{count}I", data, pos))
        pos += 4 * count
        pos += -pos % SECTION_ALIGN
        for buf_id in ids:
            if buf_id >= n_buffers:
                raise FormatError(f"{what} buffer id {buf_id} out of range")
        return ids

    inputs = read_io_map(n_inputs, "inputs")
    outputs = read_io_map(n_outputs, "outputs")

    # IO shapes: {rank u32, pad u32, dims u64[rank]} per IO buffer
    def read_shape(what: str) -> tuple[int, ...]:
        nonlocal pos
        check_aligned(f"{what} shape entry")
        rank, pad = struct.unpack_from("<II", data, pos)
        pos += 8
        if pad != 0:
            raise FormatError(f"{what} shape entry nonzero pad {pad}")
        dims = struct.unpack_from(f"<{rank}Q", data, pos)
        pos += 8 * rank
        return tuple(dims)

    input_shapes = [read_shape(f"input[{i}]") for i in range(n_inputs)]
    output_shapes = [read_shape(f"output[{i}]") for i in range(n_outputs)]

    # const pool (each entry padded to 8B)
    consts: list[tuple[int, bytes]] = []
    for i in range(n_consts):
        check_aligned(f"const[{i}]")
        buf_id, byte_len = CONSTHDR_STRUCT.unpack_from(data, pos)
        pos += CONSTHDR_STRUCT.size
        if buf_id >= n_buffers:
            raise FormatError(f"const[{i}] buffer id {buf_id} out of range")
        if byte_len > buffers[buf_id].size_bytes:
            raise FormatError(f"const[{i}] byte_len {byte_len} > buffer size "
                              f"{buffers[buf_id].size_bytes}")
        consts.append((buf_id, bytes(data[pos:pos + byte_len])))
        pos += byte_len
        pos += -pos % SECTION_ALIGN

    # instructions
    check_aligned("instructions")
    instrs: list[Instr] = []
    for i in range(n_instrs):
        op, dst, a, b, n, imm, pad0, pad1 = INSTR_STRUCT.unpack_from(data, pos)
        pos += INSTR_STRUCT.size
        if op not in OP_NAMES:
            raise FormatError(f"instr[{i}] unknown opcode {op}")
        if (pad0, pad1) != (0, 0):
            raise FormatError(f"instr[{i}] nonzero padding")
        if op == OP_WHILE:
            if a + b > n_instrs or n + imm > n_instrs:
                raise FormatError(f"instr[{i}] WHILE sub-list out of range")
            if dst >= n_buffers:
                raise FormatError(f"instr[{i}] WHILE cond buffer out of range")
        else:
            for name, buf_id in (("dst", dst), ("a", a), ("b", b)):
                # NOP/FILL/IOTA leave unused fields 0; a 0 index is always valid
                # when buffers exist, so only range-check.
                if buf_id >= n_buffers and not (op == OP_NOP and buf_id == 0):
                    raise FormatError(
                        f"instr[{i}] {name}={buf_id} out of range")
        instrs.append(Instr(op, dst, a, b, n, imm))

    if pos != len(data):
        raise FormatError(f"trailing bytes: parsed {pos} of {len(data)}")

    return Program(arena_bytes, buffers, inputs, outputs, input_shapes,
                   output_shapes, consts, instrs, main_len)


# --- numpy reference interpreter --------------------------------------------

def _f32_from_bits(imm: int) -> np.float32:
    return np.frombuffer(struct.pack("<I", imm), dtype="<f4")[0]


def execute(prog: Program, args: list[np.ndarray]) -> list[np.ndarray]:
    """Run the program on numpy; mirrors the executor contract in the spec."""
    if len(args) != len(prog.inputs):
        raise ValueError(f"expected {len(prog.inputs)} args, got {len(args)}")
    arena = np.zeros(prog.arena_bytes, dtype=np.uint8)

    def view(buf_id: int, n: int | None = None) -> np.ndarray:
        b = prog.buffers[buf_id]
        dt = DTYPE_NUMPY[b.dtype]
        count = b.size_bytes // dt.itemsize if n is None else n
        end = b.arena_byte_offset + count * dt.itemsize
        if end > b.arena_byte_offset + b.size_bytes:
            raise FormatError(f"instr element count {n} exceeds buf[{buf_id}]")
        return arena[b.arena_byte_offset:end].view(dt)

    # program load: upload consts once
    for buf_id, data in prog.consts:
        b = prog.buffers[buf_id]
        arena[b.arena_byte_offset:b.arena_byte_offset + len(data)] = \
            np.frombuffer(data, dtype=np.uint8)
    # execute: write inputs into their arena regions
    for buf_id, shape, arg in zip(prog.inputs, prog.input_shapes, args):
        flat = np.ascontiguousarray(arg, dtype=np.float32).ravel()
        if flat.nbytes != prog.buffers[buf_id].size_bytes:
            raise ValueError(f"arg for buf[{buf_id}] has {flat.nbytes} bytes, "
                             f"buffer is {prog.buffers[buf_id].size_bytes}")
        view(buf_id)[:] = flat

    def run_range(start: int, length: int, depth: int = 0) -> None:
        if depth > MAX_WHILE_DEPTH:
            raise FormatError(f"WHILE nesting exceeds {MAX_WHILE_DEPTH}")
        for pc in range(start, start + length):
            ins = prog.instrs[pc]
            op = ins.op
            if op == OP_NOP:
                pass
            elif op == OP_ADD_F32:
                view(ins.dst, ins.n)[:] = view(ins.a, ins.n) + view(ins.b, ins.n)
            elif op == OP_MUL_F32:
                view(ins.dst, ins.n)[:] = view(ins.a, ins.n) * view(ins.b, ins.n)
            elif op == OP_SUB_F32:
                view(ins.dst, ins.n)[:] = view(ins.a, ins.n) - view(ins.b, ins.n)
            elif op == OP_FILL_F32:
                view(ins.dst, ins.n)[:] = _f32_from_bits(ins.imm)
            elif op == OP_IOTA_F32:
                view(ins.dst, ins.n)[:] = np.arange(ins.n, dtype=np.float32)
            elif op == OP_LTS_F32:
                view(ins.dst, 1)[0] = np.float32(
                    1.0 if view(ins.a, 1)[0] < view(ins.b, 1)[0] else 0.0)
            elif op == OP_WHILE:
                # cond list = [a, a+b), body = [n, n+imm); loop while dst[0] != 0
                while True:
                    run_range(ins.a, ins.b, depth + 1)
                    if view(ins.dst, 1)[0] == np.float32(0.0):
                        break
                    run_range(ins.n, ins.imm, depth + 1)
            else:  # unreachable: parse() rejects unknown opcodes
                raise FormatError(f"unknown opcode {op}")

    run_range(0, prog.main_len)

    return [view(buf_id).copy().reshape(shape)
            for buf_id, shape in zip(prog.outputs, prog.output_shapes)]
