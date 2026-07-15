"""VMProgram v3 reader + two reference validators.

Consumer-side mirror of lowering.py + scheduler.py: parses the binary format
(tensor sections AND v2.1 schedule sections) back into python objects — the
reference for the C++ parser in pjrt_plugin/runtime — and provides two
independent executors that MUST agree:

(a) `execute`   : numpy interpreter over the TENSOR sections (source of truth),
                  single arena, consts uploaded once, inputs written per
                  execute, root instruction list [0, main_len) run, outputs
                  read back. WHILE is interpreted per spec via plain recursion.
(b) `execute_schedule` : a LANE SIMULATOR over the SCHEDULE sections. It runs
                  the per-lane streams barrier-phase by barrier-phase; within a
                  phase, entries of different lanes may run in any order, so it
                  runs them in two different orders and asserts the results are
                  identical (independence). It also asserts every tile of every
                  task is covered exactly once, that all lanes have identical
                  barrier counts, and that all tile ranges are in bounds. Its
                  final outputs must equal (a)'s.

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
from . import scheduler as S

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
    aux: int = 0


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
    aux: list[int] = dataclasses.field(default_factory=list)
    schedule: "ParsedSchedule | None" = None


@dataclasses.dataclass
class ParsedTask:
    tile_op: int
    dst: int
    a: int
    b: int
    p0: int
    p1: int
    p2: int
    p3: int

    def n_tiles(self) -> int:
        return S.Task(self.tile_op, self.dst, self.a, self.b,
                      self.p0, self.p1, self.p2, self.p3).n_tiles()


@dataclasses.dataclass
class ParsedEntry:
    task: int
    tile_lo: int
    tile_hi: int
    wait_flag: int
    wait_count: int
    signal_flag: int
    slots: int
    pad: int


@dataclasses.dataclass
class ParsedSchedule:
    n_flags: int
    n_lanes: int
    tasks: list[ParsedTask]
    lane_streams: list[list[ParsedEntry]]


class FormatError(ValueError):
    """Malformed VMProgram bytes (the C++ side rejects with INVALID_ARGUMENT)."""


def parse(data: bytes) -> Program:
    """Strict parse; validates magic/version/alignment/bounds/trailing bytes."""
    if len(data) < HEADER_STRUCT.size:
        raise FormatError(f"short file: {len(data)} bytes")
    (magic, version, n_buffers, n_instrs, n_consts, main_len, n_inputs,
     n_outputs, n_aux, hpad, arena_bytes) = HEADER_STRUCT.unpack_from(data, 0)
    if magic != MAGIC:
        raise FormatError(f"bad magic {magic:#010x} (want {MAGIC:#010x})")
    if version != VERSION:
        raise FormatError(f"unsupported version {version} (want {VERSION})")
    if hpad != 0:
        raise FormatError(f"header pad nonzero: {hpad}")
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

    # aux pool: n_aux x u32, padded to 8B (between IO shapes and const pool)
    check_aligned("aux pool")
    aux = list(struct.unpack_from(f"<{n_aux}I", data, pos)) if n_aux else []
    pos += 4 * n_aux
    pos += -pos % SECTION_ALIGN

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

    # instructions ({op,dst,a,b,n,imm,aux,pad1}; pad0 renamed aux in v2)
    check_aligned("instructions")
    instrs: list[Instr] = []
    for i in range(n_instrs):
        op, dst, a, b, n, imm, aux_off, pad1 = INSTR_STRUCT.unpack_from(data, pos)
        pos += INSTR_STRUCT.size
        if op not in OP_NAMES:
            raise FormatError(f"instr[{i}] unknown opcode {op}")
        if pad1 != 0:
            raise FormatError(f"instr[{i}] nonzero padding")
        if aux_off > n_aux:
            raise FormatError(f"instr[{i}] aux offset {aux_off} > n_aux {n_aux}")
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
        instrs.append(Instr(op, dst, a, b, n, imm, aux_off))

    # schedule sections (v3, 8B-aligned). Present in real files; a tensor-only
    # file (schedule=None at serialize time) simply has no trailing bytes.
    schedule = None
    if pos < len(data):
        schedule = _parse_schedule(data, pos, n_buffers)
        pos = len(data)   # _parse_schedule validates its own trailing-byte end

    if pos != len(data):
        raise FormatError(f"trailing bytes: parsed {pos} of {len(data)}")

    return Program(arena_bytes, buffers, inputs, outputs, input_shapes,
                   output_shapes, consts, instrs, main_len, aux, schedule)


def _parse_schedule(data: bytes, pos: int, n_buffers: int) -> ParsedSchedule:
    """Parse the v2.1 schedule sections beginning at `pos`. Validates the
    sched header, task/lane-tab/entry arrays, offsets and bounds."""
    if pos % SECTION_ALIGN:
        raise FormatError(f"schedule header not {SECTION_ALIGN}B-aligned: {pos}")
    if pos + S.SCHED_HDR_STRUCT.size > len(data):
        raise FormatError("truncated schedule header")
    n_tasks, n_entries, n_flags, n_lanes = S.SCHED_HDR_STRUCT.unpack_from(data, pos)
    pos += S.SCHED_HDR_STRUCT.size

    tasks: list[ParsedTask] = []
    for i in range(n_tasks):
        fields = S.TASK_STRUCT.unpack_from(data, pos)
        pos += S.TASK_STRUCT.size
        tile_op, dst, a, b, p0, p1, p2, p3 = fields
        # dst/a/b are buffer ids for compute tasks; range-check defensively.
        if tile_op == S.TILE_EW:
            for name, bid in (("dst", dst), ("a", a), ("b", b)):
                if bid >= n_buffers:
                    raise FormatError(
                        f"task[{i}] {name}={bid} out of range ({n_buffers})")
        tasks.append(ParsedTask(tile_op, dst, a, b, p0, p1, p2, p3))

    lane_tab: list[tuple[int, int]] = []
    for i in range(n_lanes):
        off, count, root_len, _pad = S.LANETAB_STRUCT.unpack_from(data, pos)
        pos += S.LANETAB_STRUCT.size
        if off + count > n_entries:
            raise FormatError(
                f"lane[{i}] stream [{off},{off + count}) exceeds n_entries "
                f"{n_entries}")
        if root_len > count:
            raise FormatError(f"lane[{i}] root_len {root_len} > count {count}")
        lane_tab.append((off, count))

    entries: list[ParsedEntry] = []
    for i in range(n_entries):
        fields = S.ENTRY_STRUCT.unpack_from(data, pos)
        pos += S.ENTRY_STRUCT.size
        task, tile_lo, tile_hi, wf, wc, sf, slots, epad = fields
        is_sentinel = task in (S.TASK_NOP, S.TASK_BARRIER,
                               S.TASK_WHILE, S.TASK_IF)
        if not is_sentinel and task >= n_tasks:
            raise FormatError(f"entry[{i}] task {task} out of range ({n_tasks})")
        entries.append(ParsedEntry(task, tile_lo, tile_hi, wf, wc, sf,
                                   slots, epad))

    if pos != len(data):
        raise FormatError(
            f"schedule trailing bytes: parsed {pos} of {len(data)}")

    lane_streams = [[entries[off + k] for k in range(count)]
                    for off, count in lane_tab]
    return ParsedSchedule(n_flags, n_lanes, tasks, lane_streams)


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


# --- schedule lane simulator (validator b) ----------------------------------

def _split_phases(sched: ParsedSchedule) -> list[list[tuple[int, ParsedEntry]]]:
    """Split each lane's stream by BARRIER into phases. Asserts every lane has
    the same barrier count and returns a list of phases; phase p holds
    [(lane, entry)] for the entries between barrier p-1 and barrier p. Control
    entries (WHILE/IF) are not produced by the current scheduler and are
    rejected here (seam)."""
    counts = [sum(1 for e in s if e.task == S.TASK_BARRIER)
              for s in sched.lane_streams]
    if len(set(counts)) > 1:
        raise AssertionError(f"barrier counts differ across lanes: {counts}")
    n_phases = counts[0] if counts else 0
    phases: list[list[tuple[int, ParsedEntry]]] = [[] for _ in range(n_phases)]
    for lane, stream in enumerate(sched.lane_streams):
        p = 0
        for e in stream:
            if e.task == S.TASK_BARRIER:
                p += 1
            elif e.task in (S.TASK_WHILE, S.TASK_IF):
                raise NotImplementedError(
                    "schedule simulator: WHILE/IF entries not supported yet "
                    "(structured seam — see python/NOTES.md)")
            elif e.task == S.TASK_NOP:
                pass
            else:
                phases[p].append((lane, e))
    return phases


def _check_coverage(sched: ParsedSchedule) -> None:
    """Every tile of every task covered exactly once; all ranges in-bounds."""
    per_task: dict[int, list[tuple[int, int]]] = {}
    for stream in sched.lane_streams:
        for e in stream:
            if e.task in (S.TASK_BARRIER, S.TASK_NOP, S.TASK_WHILE, S.TASK_IF):
                continue
            per_task.setdefault(e.task, []).append((e.tile_lo, e.tile_hi))
    for tid, task in enumerate(sched.tasks):
        nt = task.n_tiles()
        ranges = sorted(per_task.get(tid, []))
        expect = 0
        for lo, hi in ranges:
            if not (0 <= lo <= hi <= nt):
                raise AssertionError(
                    f"task {tid} range [{lo},{hi}) out of bounds (tiles={nt})")
            if lo != expect:
                raise AssertionError(
                    f"task {tid} tiles not covered exactly once near {expect} "
                    f"(got range start {lo}); ranges={ranges}")
            expect = hi
        if expect != nt:
            raise AssertionError(
                f"task {tid} covers {expect}/{nt} tiles; ranges={ranges}")


def execute_schedule(prog: Program, args: list[np.ndarray],
                     tile_size: int = S.TILE_SIZE) -> list[np.ndarray]:
    """Validator (b): execute the SCHEDULE sections with a lane simulator.

    Runs barrier-phase by barrier-phase; within a phase, executes the entries
    in the given order and in reverse order on a fresh copy of the arena and
    asserts the two results are identical (entries of different lanes in a
    phase must be order-independent). Also asserts tile coverage and barrier
    uniformity (via `_check_coverage` / `_split_phases`). Returns the outputs,
    which the caller compares against `execute` (validator a)."""
    if prog.schedule is None:
        raise ValueError("program has no schedule sections")
    if len(args) != len(prog.inputs):
        raise ValueError(f"expected {len(prog.inputs)} args, got {len(args)}")
    sched = prog.schedule
    _check_coverage(sched)
    phases = _split_phases(sched)

    arena = np.zeros(prog.arena_bytes, dtype=np.uint8)

    def view(buf_id: int) -> np.ndarray:
        b = prog.buffers[buf_id]
        dt = DTYPE_NUMPY[b.dtype]
        return arena[b.arena_byte_offset:
                     b.arena_byte_offset + b.size_bytes].view(dt)

    for buf_id, data in prog.consts:
        b = prog.buffers[buf_id]
        arena[b.arena_byte_offset:b.arena_byte_offset + len(data)] = \
            np.frombuffer(data, dtype=np.uint8)
    for buf_id, shape, arg in zip(prog.inputs, prog.input_shapes, args):
        flat = np.ascontiguousarray(arg, dtype=np.float32).ravel()
        if flat.nbytes != prog.buffers[buf_id].size_bytes:
            raise ValueError(f"arg for buf[{buf_id}] has {flat.nbytes} bytes, "
                             f"buffer is {prog.buffers[buf_id].size_bytes}")
        view(buf_id)[:] = flat

    def run_entry(e: ParsedEntry) -> None:
        task = sched.tasks[e.task]
        if task.tile_op != S.TILE_EW:
            raise NotImplementedError(
                f"schedule simulator: tile_op {task.tile_op} not supported")
        n = task.p1
        lo = e.tile_lo * tile_size
        hi = min(e.tile_hi * tile_size, n)
        if lo >= hi:
            return
        dst = view(task.dst)
        subop = task.p0
        if subop in (S.EW_ADD, S.EW_MUL, S.EW_SUB):
            a = view(task.a)[lo:hi]
            b = view(task.b)[lo:hi]
            dst[lo:hi] = (a + b if subop == S.EW_ADD else
                          a * b if subop == S.EW_MUL else a - b)
        elif subop == S.EW_FILL:
            dst[lo:hi] = _f32_from_bits(task.p2)
        else:
            raise NotImplementedError(
                f"schedule simulator: EW subop {subop} not supported")

    def run_order(entries: list[tuple[int, ParsedEntry]], snapshot: np.ndarray):
        arena[:] = snapshot
        for _lane, e in entries:
            run_entry(e)
        return arena.copy()

    for phase in phases:
        base = arena.copy()
        forward = run_order(phase, base)
        reverse = run_order(list(reversed(phase)), base)
        if not np.array_equal(forward, reverse):
            raise AssertionError(
                "schedule phase is order-dependent: lanes conflict within a "
                "barrier phase")
        arena[:] = forward   # both orders agree; adopt the result

    return [view(buf_id).copy().reshape(shape)
            for buf_id, shape in zip(prog.outputs, prog.output_shapes)]
