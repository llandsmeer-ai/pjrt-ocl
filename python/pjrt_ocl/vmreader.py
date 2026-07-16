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
    HEADER_STRUCT, INSTR_STRUCT, MAGIC, OP_ADD_F32, OP_AFFINE_F32, OP_DOT,
    OP_FILL_F32, OP_FOR, OP_IOTA_F32, OP_LTS_F32, OP_MUL_F32, OP_NAMES, OP_NOP,
    OP_SUB_F32, OP_WHILE, SECTION_ALIGN, VERSION,
)
from . import scheduler as S
from . import opsem

MAX_WHILE_DEPTH = 8


def _is_view_subop(sub: int) -> bool:
    """Plain float binary/unary EW subops (match ops/ew.cl's ew_is_bin/ew_is_un):
    these carry a/b strided-view aux-offsets in task p2/p3. cmp/select/affine/
    fill/bitwise reuse p2/p3 for other data, so they never view."""
    return sub <= 6 or (25 <= sub <= 26) or (7 <= sub <= 17) or (27 <= sub <= 34)


class _InterpRT:
    """Facade passed to opsem.INTERP handlers (numpy reference semantics)."""

    def __init__(self, prog, view):
        self._prog = prog
        self.view = view          # view(buf_id, n=None) -> f32 ndarray
        self.aux = prog.aux       # list[int] (u32 words)

    def aux_i32(self, off: int) -> int:
        """Read aux word `off` as a signed int32 (strides/offsets)."""
        v = self.aux[off]
        return v - 0x100000000 if v >= 0x80000000 else v

    def viewed(self, buf: int, n: int, view: int) -> np.ndarray:
        """Read n elements of an operand, applying a strided VIEW if `view` != 0
        (view = aux-offset + 1). Mirrors the device: element i reads
        src[src_off + Σ coord_e(i)*stride_e], the same descriptor gather uses."""
        if not view:
            return self.view(buf, n)
        off = view - 1
        rank = self.aux[off]
        dims = [self.aux[off + 1 + e] for e in range(rank)]
        strides = [self.aux_i32(off + 1 + rank + e) for e in range(rank)]
        src_off = self.aux_i32(off + 1 + 2 * rank)
        idx = np.zeros(n, dtype=np.int64) + src_off
        rem = np.arange(n, dtype=np.int64)
        for e in range(rank - 1, -1, -1):
            idx += (rem % dims[e]) * strides[e]
            rem //= dims[e]
        return self.view(buf)[idx]

    @staticmethod
    def f32_from_bits(imm: int) -> np.float32:
        return _f32_from_bits(imm)


class _SchedRT(_InterpRT):
    """Facade for schedule-simulator (validator b) tile ops. Adds tile_size;
    view() takes no element count (full-buffer view)."""

    def __init__(self, prog, view, tile_size):
        self.view = view
        self.aux = prog.aux
        self._prog = prog
        self.tile_size = tile_size


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
    imm2: int = 0       # OP_AFFINE_F32's t bits (8th instr word); 0 otherwise


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
    tile_op: int          # base op (dtype masked off)
    dst: int
    a: int
    b: int
    p0: int
    p1: int
    p2: int
    p3: int
    dtype: int = 0        # result dtype (tile_op bits 8-15)
    adtype: int = 0       # operand dtype (tile_op bits 16-23)

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
    root_lens: list[int] = dataclasses.field(default_factory=list)


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
        op, dst, a, b, n, imm, aux_off, imm2 = INSTR_STRUCT.unpack_from(data, pos)
        pos += INSTR_STRUCT.size
        if op not in OP_NAMES:
            raise FormatError(f"instr[{i}] unknown opcode {op}")
        # the 8th word is a general second immediate: OP_AFFINE_F32's t bits,
        # OP_DOT's batch count, or an elementwise op's operand-b view aux-offset.
        # (Control/nop never use it.)
        if imm2 != 0 and op in (OP_NOP, OP_WHILE, OP_FOR):
            raise FormatError(f"instr[{i}] nonzero padding")
        if aux_off > n_aux:
            raise FormatError(f"instr[{i}] aux offset {aux_off} > n_aux {n_aux}")
        if op == OP_WHILE:
            if a + b > n_instrs or n + imm > n_instrs:
                raise FormatError(f"instr[{i}] WHILE sub-list out of range")
            if dst >= n_buffers:
                raise FormatError(f"instr[{i}] WHILE cond buffer out of range")
        elif op == OP_FOR:
            # body = [n, n+imm), b = trip count; dst/a unused (no cond).
            if n + imm > n_instrs:
                raise FormatError(f"instr[{i}] FOR sub-list out of range")
            if dst != 0 or a != 0:
                raise FormatError(f"instr[{i}] FOR nonzero dst/a")
        else:
            for name, buf_id in (("dst", dst), ("a", a), ("b", b)):
                # NOP/FILL/IOTA leave unused fields 0; a 0 index is always valid
                # when buffers exist, so only range-check.
                if buf_id >= n_buffers and not (op == OP_NOP and buf_id == 0):
                    raise FormatError(
                        f"instr[{i}] {name}={buf_id} out of range")
        instrs.append(Instr(op, dst, a, b, n, imm, aux_off, imm2))

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
        packed, dst, a, b, p0, p1, p2, p3 = fields
        tile_op = packed & 0xFF            # base op
        dtype = (packed >> 8) & 0xFF       # result dtype
        adtype = (packed >> 16) & 0xFF     # operand dtype
        # dst/a/b are buffer ids for compute tasks; range-check defensively.
        if tile_op == S.TILE_EW:
            for name, bid in (("dst", dst), ("a", a), ("b", b)):
                if bid >= n_buffers:
                    raise FormatError(
                        f"task[{i}] {name}={bid} out of range ({n_buffers})")
        tasks.append(ParsedTask(tile_op, dst, a, b, p0, p1, p2, p3,
                                dtype=dtype, adtype=adtype))

    lane_tab: list[tuple[int, int]] = []
    root_lens: list[int] = []
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
        root_lens.append(root_len)

    entries: list[ParsedEntry] = []
    for i in range(n_entries):
        fields = S.ENTRY_STRUCT.unpack_from(data, pos)
        pos += S.ENTRY_STRUCT.size
        task, tile_lo, tile_hi, wf, wc, sf, slots, epad = fields
        is_sentinel = task in (S.TASK_NOP, S.TASK_BARRIER,
                               S.TASK_WHILE, S.TASK_IF, S.TASK_FOR)
        if not is_sentinel and task >= n_tasks:
            raise FormatError(f"entry[{i}] task {task} out of range ({n_tasks})")
        entries.append(ParsedEntry(task, tile_lo, tile_hi, wf, wc, sf,
                                   slots, epad))

    if pos != len(data):
        raise FormatError(
            f"schedule trailing bytes: parsed {pos} of {len(data)}")

    lane_streams = [[entries[off + k] for k in range(count)]
                    for off, count in lane_tab]
    return ParsedSchedule(n_flags, n_lanes, tasks, lane_streams, root_lens)


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
    # execute: write inputs into their arena regions (in the buffer's dtype, so
    # integer/other-typed inputs round-trip; f32 inputs are unchanged).
    for buf_id, shape, arg in zip(prog.inputs, prog.input_shapes, args):
        dt = DTYPE_NUMPY[prog.buffers[buf_id].dtype]
        flat = np.ascontiguousarray(arg, dtype=dt).ravel()
        if flat.nbytes != prog.buffers[buf_id].size_bytes:
            raise ValueError(f"arg for buf[{buf_id}] has {flat.nbytes} bytes, "
                             f"buffer is {prog.buffers[buf_id].size_bytes}")
        view(buf_id)[:] = flat

    irt = _InterpRT(prog, view)      # for viewed operand reads (imm/imm2)

    def run_range(start: int, length: int, depth: int = 0) -> None:
        if depth > MAX_WHILE_DEPTH:
            raise FormatError(f"WHILE nesting exceeds {MAX_WHILE_DEPTH}")
        for pc in range(start, start + length):
            ins = prog.instrs[pc]
            op = ins.op
            if op == OP_NOP:
                pass
            elif op == OP_ADD_F32:
                view(ins.dst, ins.n)[:] = (irt.viewed(ins.a, ins.n, ins.imm)
                                           + irt.viewed(ins.b, ins.n, ins.imm2))
            elif op == OP_MUL_F32:
                view(ins.dst, ins.n)[:] = (irt.viewed(ins.a, ins.n, ins.imm)
                                           * irt.viewed(ins.b, ins.n, ins.imm2))
            elif op == OP_SUB_F32:
                view(ins.dst, ins.n)[:] = (irt.viewed(ins.a, ins.n, ins.imm)
                                           - irt.viewed(ins.b, ins.n, ins.imm2))
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
            elif op == OP_FOR:
                # body = [n, n+imm) run b times (fixed trip; no cond list)
                for _ in range(ins.b):
                    run_range(ins.n, ins.imm, depth + 1)
            elif op in opsem.INTERP:
                opsem.INTERP[op](ins, _InterpRT(prog, view))
            else:  # unreachable: parse() rejects unknown opcodes
                raise FormatError(f"unknown opcode {op}")

    run_range(0, prog.main_len)

    return [view(buf_id).copy().reshape(shape)
            for buf_id, shape in zip(prog.outputs, prog.output_shapes)]


# --- schedule lane simulator (validator b) ----------------------------------

def _split_phases(sched: ParsedSchedule) -> list[list[tuple[int, ParsedEntry]]]:
    """Split each lane's ROOT walk by BARRIER into phases. Asserts every lane
    has the same barrier count and returns a list of phases; phase p holds
    [(lane, entry)] for the entries between barrier p-1 and barrier p. Only
    valid for control-flow-free schedules (WHILE/IF drive `_run_control`)."""
    counts = [sum(1 for e in s if e.task == S.TASK_BARRIER)
              for s in sched.lane_streams]
    if len(set(counts)) > 1:
        raise AssertionError(f"barrier counts differ across lanes: {counts}")
    # B barriers separate B+1 phases (the last phase has no trailing barrier).
    n_phases = (counts[0] if counts else 0) + 1
    phases: list[list[tuple[int, ParsedEntry]]] = [[] for _ in range(n_phases)]
    for lane, stream in enumerate(sched.lane_streams):
        p = 0
        for e in stream:
            if e.task == S.TASK_BARRIER:
                p += 1
            elif e.task in (S.TASK_WHILE, S.TASK_IF, S.TASK_FOR):
                raise NotImplementedError(
                    "_split_phases: control entries present (use _run_control)")
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
            if e.task in (S.TASK_BARRIER, S.TASK_NOP, S.TASK_WHILE, S.TASK_IF,
                          S.TASK_FOR):
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
    has_control = any(e.task in (S.TASK_WHILE, S.TASK_IF, S.TASK_FOR)
                      for s in sched.lane_streams for e in s)

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
        dt = DTYPE_NUMPY[prog.buffers[buf_id].dtype]
        flat = np.ascontiguousarray(arg, dtype=dt).ravel()
        if flat.nbytes != prog.buffers[buf_id].size_bytes:
            raise ValueError(f"arg for buf[{buf_id}] has {flat.nbytes} bytes, "
                             f"buffer is {prog.buffers[buf_id].size_bytes}")
        view(buf_id)[:] = flat

    sim_rt = _SchedRT(prog, view, tile_size)

    def run_entry(e: ParsedEntry) -> None:
        task = sched.tasks[e.task]
        if task.tile_op == S.TILE_EW:
            n = task.p1
            lo = e.tile_lo * tile_size
            hi = min(e.tile_hi * tile_size, n)
            if lo >= hi:
                return
            dst = view(task.dst)
            subop = task.p0
            # plain binary/unary subops carry a/b VIEW aux-offsets in p2/p3;
            # cmp/select/affine/fill use p2/p3 for their own data (no views).
            vw = _is_view_subop(subop)

            def rd(buf, vo):
                return (sim_rt.viewed(buf, n, vo)[lo:hi] if vw and vo
                        else view(buf)[lo:hi])
            if subop in (S.EW_ADD, S.EW_MUL, S.EW_SUB):
                a = rd(task.a, task.p2)
                b = rd(task.b, task.p3)
                dst[lo:hi] = (a + b if subop == S.EW_ADD else
                              a * b if subop == S.EW_MUL else a - b)
            elif subop == S.EW_FILL:
                dst[lo:hi] = _f32_from_bits(task.p2)
            elif subop in opsem.EW_SIM:
                a = rd(task.a, task.p2)
                # binary subops read b; unary/select handlers ignore extras.
                b = (rd(task.b, task.p3) if task.b < len(prog.buffers) else None)
                dst[lo:hi] = opsem.EW_SIM[subop](a, b, task, sim_rt, lo, hi)
            else:
                raise NotImplementedError(
                    f"schedule simulator: EW subop {subop} not supported")
            return
        sim = opsem.TILE_SIM.get(task.tile_op)
        if sim is None:
            raise NotImplementedError(
                f"schedule simulator: tile_op {task.tile_op} not supported")
        sim(task, e, sim_rt)

    def run_order(lane_ids, by_lane, snapshot: np.ndarray):
        # Each lane runs its OWN entries in sequence (a lane's chain of
        # elementwise ops is ordered); only the interleaving BETWEEN lanes is
        # varied. Running lane-by-lane in `lane_ids` order is one valid
        # interleaving that respects every lane's internal order.
        arena[:] = snapshot
        for lane in lane_ids:
            for e in by_lane[lane]:
                run_entry(e)
        return arena.copy()

    def run_batch(batch: list[tuple[int, ParsedEntry]]) -> None:
        """Execute a barrier phase. Entries within a lane are ordered (a lane may
        now carry a dependent elementwise CHAIN — same-index deps chain on one
        lane, no barrier); entries ACROSS lanes must be independent. Assert that
        by running the lanes in two opposite orders (each preserving intra-lane
        order) and requiring agreement — a mismatch means a genuine cross-lane
        write/read landed in one phase (a scheduler bug)."""
        if not batch:
            return
        by_lane: dict[int, list[ParsedEntry]] = {}
        order: list[int] = []
        for lane, e in batch:
            if lane not in by_lane:
                by_lane[lane] = []
                order.append(lane)
            by_lane[lane].append(e)
        base = arena.copy()
        forward = run_order(order, by_lane, base)
        reverse = run_order(list(reversed(order)), by_lane, base)
        if not np.array_equal(forward, reverse):
            raise AssertionError(
                "schedule phase is order-dependent: lanes conflict within a "
                "barrier phase")
        arena[:] = forward   # both lane orders agree; adopt the result

    if has_control:
        _run_control(sched, arena, view, run_batch)
    else:
        for phase in _split_phases(sched):
            run_batch(phase)

    return [view(buf_id).copy().reshape(shape)
            for buf_id, shape in zip(prog.outputs, prog.output_shapes)]


def _run_control(sched: ParsedSchedule, arena: np.ndarray, view,
                 run_batch) -> None:
    """Lane simulator with a frame stack, mirroring vm2.cl's vm2 interpreter,
    for schedules that contain WHILE control entries. All lanes step in lockstep
    between global barriers: each driver iteration walks every lane forward to
    its next global barrier, collecting the compute entries into one batch;
    `run_batch` runs it order-independently; then the (uniform) barrier is
    resolved — an explicit BARRIER just advances, a WHILE cond/body boundary
    reads the shared loop scalar and transitions every lane's frame identically.
    Control is uniform across lanes (same structure, same cond) so every lane
    yields the same barrier token each iteration; a mismatch is a scheduler bug."""
    n = sched.n_lanes
    streams = sched.lane_streams
    ROOT = -1
    # per-lane frame stack: each frame is [pc, end, widx, phase]
    stacks = [[[0, sched.root_lens[lane], ROOT, 0]] for lane in range(n)]

    def advance(lane: int, batch: list) -> str:
        stream = streams[lane]
        while True:
            f = stacks[lane][-1]
            if f[0] >= f[1]:                        # frame range exhausted
                if f[2] == ROOT:
                    return "DONE"
                w = stream[f[2]]
                if w.task == S.TASK_IF:             # branch done: pop, advance
                    stacks[lane].pop()
                    stacks[lane][-1][0] += 1
                    continue
                if w.task == S.TASK_FOR:            # iteration done
                    return "FOR_BODY"
                return "WHILE_COND" if f[3] == 0 else "WHILE_BODY"
            e = stream[f[0]]
            if e.task == S.TASK_BARRIER:
                return "BAR"
            if e.task == S.TASK_WHILE:
                if len(stacks[lane]) > MAX_WHILE_DEPTH:
                    raise FormatError(f"WHILE nesting exceeds {MAX_WHILE_DEPTH}")
                stacks[lane].append([e.tile_lo, e.tile_lo + e.tile_hi, f[0], 0])
                continue
            if e.task == S.TASK_FOR:
                if e.wait_flag == 0:                # trip 0: skip the loop
                    f[0] += 1
                    continue
                if len(stacks[lane]) > MAX_WHILE_DEPTH:
                    raise FormatError(f"WHILE nesting exceeds {MAX_WHILE_DEPTH}")
                # f[3] counts REMAINING iterations for a FOR frame
                stacks[lane].append([e.tile_lo, e.tile_lo + e.tile_hi, f[0],
                                     e.wait_flag])
                continue
            if e.task == S.TASK_IF:
                raise NotImplementedError("schedule simulator: IF entries")
            if e.task != S.TASK_NOP:
                batch.append((lane, e))
            f[0] += 1

    while True:
        batch: list = []
        tokens = [advance(lane, batch) for lane in range(n)]
        if all(t == "DONE" for t in tokens):
            # a lane's final compute level (no trailing root barrier) is
            # collected into `batch` by the same advance() that reaches the
            # frame end and returns DONE — run it before terminating, else the
            # last level is dropped.
            if batch:
                run_batch(batch)
            break
        if len(set(tokens)) != 1:
            raise AssertionError(
                f"non-uniform control across lanes: {sorted(set(tokens))}")
        run_batch(batch)
        kind = tokens[0]
        if kind == "BAR":
            for lane in range(n):
                stacks[lane][-1][0] += 1            # step past the barrier
        elif kind == "WHILE_COND":
            for lane in range(n):
                f = stacks[lane][-1]
                w = streams[lane][f[2]]
                if view(w.signal_flag)[0] != 0:     # loop continues -> body
                    f[0] = w.wait_flag
                    f[1] = w.wait_flag + w.wait_count
                    f[3] = 1
                else:                               # loop exits -> pop
                    stacks[lane].pop()
                    stacks[lane][-1][0] += 1
        elif kind == "FOR_BODY":                    # fixed-trip iteration done
            for lane in range(n):
                f = stacks[lane][-1]
                w = streams[lane][f[2]]
                f[3] -= 1
                if f[3] > 0:                        # more iterations -> body
                    f[0] = w.tile_lo
                    f[1] = w.tile_lo + w.tile_hi
                else:                               # done -> pop
                    stacks[lane].pop()
                    stacks[lane][-1][0] += 1
        else:                                       # WHILE_BODY -> recheck cond
            for lane in range(n):
                f = stacks[lane][-1]
                w = streams[lane][f[2]]
                f[0] = w.tile_lo
                f[1] = w.tile_lo + w.tile_hi
                f[3] = 0
