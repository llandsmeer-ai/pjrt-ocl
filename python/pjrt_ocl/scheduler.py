"""Tensor VMProgram -> v2.1 schedule sections (the VLIW engine's input).

Producer half of the schedule half of docs/vmprogram.md ("VMProgram v2.1 —
schedule sections"). Runs inside lower_service after lowering.py has produced
the tensor program, using device config from env (PJRT_OCL_NLANES,
PJRT_OCL_COST_TABLE). Pure stdlib + the lowering module — no jax/jaxlib, no
numpy required here.

Pipeline (docs/tile-isa.md, docs/roadmap.md Phase 1.3):

  tensor instrs (SSA order)
    -> dataflow LEVELS   (maximal sets of mutually independent instrs)
    -> TASK descriptors  (one per compute instr; tile counts from shapes)
    -> LPT cost-based lane packing within each level
    -> per-lane STREAMS with a global BARRIER entry after every level

Region-carrying ops (WHILE/IF) recurse: their region instruction lists are
scheduled into per-lane sub-ranges appended after the main stream, with a
WHILE/IF control entry emitted (uniformly) into every lane. Current lowering
never emits region ops, so that path is a structured seam (see NOTES.md);
`schedule_program` raises a clear error if one appears.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import struct

from . import lowering as L
from . import opsem

# --- tile-op vocabulary + sentinels (docs/vmprogram.md v2.1 table) ----------

TILE_SIZE = 16384          # EW tile size (TS)
MMA_T = 16                 # MMA output tile edge

TILE_EW = 0
TILE_MMA = 1
TILE_GATHER = 2
TILE_REDUCE_PART = 3
TILE_REDUCE_COMB = 4
TILE_IOTA_DIM = 5

# EW subops (docs/vmprogram.md)
EW_ADD = 0
EW_MUL = 1
EW_SUB = 2
EW_FILL = 18

# task-field sentinels
TASK_NOP = 0xFFFFFFFF
TASK_BARRIER = 0xFFFFFFFE
TASK_WHILE = 0xFFFFFFFD
TASK_IF = 0xFFFFFFFC
FLAG_NONE = 0xFFFFFFFF

# tensor opcode -> (tile_op, ew_subop) for the compute instrs we cover today.
# constants are NOT instructions (they live in the const pool) => no task.
_TENSOR_TO_EW = {
    L.OP_ADD_F32: EW_ADD,
    L.OP_MUL_F32: EW_MUL,
    L.OP_SUB_F32: EW_SUB,
    L.OP_FILL_F32: EW_FILL,
}

# region ops (need recursion; current lowering never emits them)
_REGION_OPS = {L.OP_WHILE}

SCHED_HDR_STRUCT = struct.Struct("<IIII")      # n_tasks,n_entries,n_flags,n_lanes
TASK_STRUCT = struct.Struct("<IIIIIIII")       # 32B
LANETAB_STRUCT = struct.Struct("<IIII")        # 16B: off, count, root_len, pad
ENTRY_STRUCT = struct.Struct("<IIIIIIII")      # 32B
assert SCHED_HDR_STRUCT.size == 16
assert TASK_STRUCT.size == 32
assert LANETAB_STRUCT.size == 16
assert ENTRY_STRUCT.size == 32


class ScheduleError(NotImplementedError):
    """Program valid but beyond the scheduler's current coverage (service exit 2)."""


# --- device config ----------------------------------------------------------

# cost-table keys (µs per tile of each tile-op class)
_COST_KEYS = {
    TILE_EW: "ew_tile_us",
    TILE_MMA: "mma_tile_us",
    TILE_GATHER: "gather_tile_us",
    TILE_REDUCE_PART: "reduce_tile_us",
    TILE_REDUCE_COMB: "reduce_tile_us",
    TILE_IOTA_DIM: "ew_tile_us",
}


@dataclasses.dataclass
class DeviceConfig:
    nlanes: int = 8
    costs: dict = dataclasses.field(default_factory=dict)   # key -> µs/tile

    @classmethod
    def from_env(cls, environ=None) -> "DeviceConfig":
        environ = os.environ if environ is None else environ
        nlanes = int(environ.get("PJRT_OCL_NLANES", "8") or "8")
        if nlanes < 1:
            nlanes = 1
        costs: dict = {}
        path = environ.get("PJRT_OCL_COST_TABLE", "")
        if path:
            try:
                with open(path, "r") as f:
                    raw = json.load(f)
                for k in ("ew_tile_us", "mma_tile_us", "gather_tile_us",
                          "reduce_tile_us"):
                    if k in raw:
                        costs[k] = float(raw[k])
            except (OSError, ValueError):
                costs = {}   # missing file / bad JSON -> all 1.0 (below)
        return cls(nlanes=nlanes, costs=costs)

    def unit_cost(self, tile_op: int) -> float:
        return self.costs.get(_COST_KEYS.get(tile_op, "ew_tile_us"), 1.0)


# --- in-memory schedule + writer --------------------------------------------

@dataclasses.dataclass
class Task:
    tile_op: int
    dst: int = 0
    a: int = 0
    b: int = 0
    p0: int = 0
    p1: int = 0
    p2: int = 0
    p3: int = 0

    def n_tiles(self) -> int:
        if self.tile_op in (TILE_EW, TILE_GATHER, TILE_IOTA_DIM):
            return max(1, math.ceil(self.p1 / TILE_SIZE))
        if self.tile_op == TILE_MMA:
            return math.ceil(self.p0 / MMA_T) * math.ceil(self.p1 / MMA_T)
        if self.tile_op == TILE_REDUCE_PART:
            return max(1, math.ceil(self.p0 / self.p1)) if self.p1 else 1
        if self.tile_op == TILE_REDUCE_COMB:
            return 1
        raise ScheduleError(f"n_tiles: unknown tile_op {self.tile_op}")


@dataclasses.dataclass
class Entry:
    task: int
    tile_lo: int = 0
    tile_hi: int = 0
    wait_flag: int = FLAG_NONE
    wait_count: int = 0
    signal_flag: int = FLAG_NONE
    slots: int = 0
    pad: int = 0


def _barrier_entry() -> Entry:
    return Entry(TASK_BARRIER, 0, 0, FLAG_NONE, 0, FLAG_NONE, 0, 0)


@dataclasses.dataclass
class Schedule:
    n_flags: int
    n_lanes: int
    tasks: list[Task]
    lane_streams: list[list[Entry]]         # len == n_lanes

    def serialize_sections(self) -> bytes:
        assert len(self.lane_streams) == self.n_lanes
        flat: list[Entry] = []
        lane_tab: list[tuple[int, int, int]] = []
        for stream in self.lane_streams:
            # root_len == len(stream): no control flow yet (region ops raise
            # ScheduleError). WHILE/IF sub-ranges will live at [root_len:count).
            lane_tab.append((len(flat), len(stream), len(stream)))
            flat.extend(stream)
        out = bytearray()
        out += SCHED_HDR_STRUCT.pack(len(self.tasks), len(flat),
                                     self.n_flags, self.n_lanes)
        for t in self.tasks:
            out += TASK_STRUCT.pack(t.tile_op, t.dst, t.a, t.b,
                                    t.p0, t.p1, t.p2, t.p3)
        for off, count, root_len in lane_tab:
            out += LANETAB_STRUCT.pack(off, count, root_len, 0)
        for e in flat:
            out += ENTRY_STRUCT.pack(e.task, e.tile_lo, e.tile_hi,
                                     e.wait_flag, e.wait_count, e.signal_flag,
                                     e.slots, e.pad)
        return bytes(out)


# --- dependency analysis + levels -------------------------------------------

def _reads(ins: L.Instr) -> set[int]:
    """Buffer ids read by an instruction. Built-in fast paths; other ops
    declare their read set in opsem.READS."""
    op = ins.op
    if op in (L.OP_ADD_F32, L.OP_MUL_F32, L.OP_SUB_F32):
        return {ins.a, ins.b}
    if op in (L.OP_FILL_F32, L.OP_IOTA_F32):
        return set()
    if op == L.OP_NOP:
        return set()
    if op in opsem.READS:
        return opsem.reads_of(ins)
    return {ins.a, ins.b}


def _writes(ins: L.Instr) -> set[int]:
    if ins.op == L.OP_NOP:
        return set()
    return {ins.dst}


def _depends(instrs, j: int, i: int) -> bool:
    """Does instr j (later) depend on instr i (earlier)?  RAW on j's reads +
    WAW on writes (docs/roadmap.md, task spec). WAR is intentionally omitted —
    the tensor program is SSA so buffers are single-assignment; recorded in
    NOTES.md."""
    a, b = instrs[i], instrs[j]
    aw = _writes(a)
    if not aw:
        return False
    return bool((_reads(b) & aw) or (_writes(b) & aw))


def _levels(instrs, indices: list[int]) -> list[list[int]]:
    """Greedy maximal levels over `indices` (SSA order): an instr joins the
    current level iff it depends on no instr already in it, else starts a new
    level."""
    levels: list[list[int]] = []
    cur: list[int] = []
    for j in indices:
        if cur and any(_depends(instrs, j, i) for i in cur):
            levels.append(cur)
            cur = [j]
        else:
            cur.append(j)
    if cur:
        levels.append(cur)
    return levels


# --- instruction -> task mapping --------------------------------------------

def _instr_to_task(ins: L.Instr) -> Task:
    """Map a compute tensor instruction to its tile task. Built-in EW fast
    path; other ops register a mapper in opsem.TO_TASK."""
    if ins.op in _TENSOR_TO_EW:
        subop = _TENSOR_TO_EW[ins.op]
        p2 = ins.imm if ins.op == L.OP_FILL_F32 else 0
        return Task(TILE_EW, dst=ins.dst, a=ins.a, b=ins.b,
                    p0=subop, p1=ins.n, p2=p2, p3=0)
    mapper = opsem.TO_TASK.get(ins.op)
    if mapper is not None:
        return mapper(ins)
    raise ScheduleError(
        f"scheduler: no task mapping for opcode {ins.op} "
        f"({L.OP_NAMES.get(ins.op, hex(ins.op))})")


# --- lane packing within a level (LPT by cost) ------------------------------

def _allocate_lanes(costs: list[float], tiles: list[int],
                    n_lanes: int) -> list[int]:
    """Lanes-per-task, proportional to cost share via LPT top-up: seed 1 lane
    each, then repeatedly hand a lane to the task with the highest per-lane
    cost that can still absorb one (lanes < tiles). Each task gets >=1 and
    <= tiles lanes; total <= n_lanes. Precondition: len(costs) <= n_lanes."""
    n = len(costs)
    lanes = [1] * n                     # tiles >= 1 always, so 1 lane min is valid
    remaining = n_lanes - sum(lanes)
    while remaining > 0:
        best, best_val = -1, -1.0
        for i in range(n):
            if lanes[i] < tiles[i]:
                v = costs[i] / lanes[i]
                if v > best_val:
                    best, best_val = i, v
        if best < 0:
            break                       # every task capped at its tile count
        lanes[best] += 1
        remaining -= 1
    return lanes


def _pack_level(level_tasks: list[tuple[int, Task]], n_lanes: int,
                config: DeviceConfig) -> list[tuple[int, Entry]]:
    """Return [(lane, Entry)] for one dataflow level.

    Primary regime (n_tasks <= n_lanes): each task owns a disjoint contiguous
    block of lanes (count proportional to cost); its tiles split evenly and
    contiguously across those lanes; one entry per (task, lane).

    Overflow regime (n_tasks > n_lanes, not reachable from current lowering):
    LPT bin-pack whole tasks onto lanes (each task one entry covering all its
    tiles on the least-loaded lane) — keeps "one entry per (task, lane-range)"
    valid and every task on >=1 lane. Recorded in NOTES.md."""
    infos = []                          # (task_id, tiles, cost)
    for tid, task in level_tasks:
        tiles = task.n_tiles()
        infos.append((tid, tiles, tiles * config.unit_cost(task.tile_op)))
    n = len(infos)
    result: list[tuple[int, Entry]] = []
    if n == 0:
        return result
    if n <= n_lanes:
        lanes_for = _allocate_lanes([c for _, _, c in infos],
                                    [t for _, t, _ in infos], n_lanes)
        cursor = 0
        for (tid, tiles, _), k in zip(infos, lanes_for):
            for j in range(k):
                lo = tiles * j // k
                hi = tiles * (j + 1) // k
                result.append((cursor + j, Entry(tid, lo, hi)))
            cursor += k
        return result
    # overflow: LPT bin-pack (cost desc onto least-loaded lane)
    order = sorted(range(n), key=lambda i: -infos[i][2])
    loads = [0.0] * n_lanes
    for i in order:
        tid, tiles, cost = infos[i]
        lane = min(range(n_lanes), key=lambda l: loads[l])
        result.append((lane, Entry(tid, 0, tiles)))
        loads[lane] += cost
    return result


# --- top-level scheduling ---------------------------------------------------

def schedule_program(prog: L.VMProgram,
                     config: DeviceConfig | None = None) -> Schedule:
    """Schedule the tensor VMProgram's root instruction list into per-lane
    streams. Returns a Schedule (v0 contract: BARRIER between levels; WAIT/
    SIGNAL unused; n_flags = 0)."""
    config = config or DeviceConfig.from_env()
    n_lanes = config.nlanes
    tasks: list[Task] = []
    lane_streams: list[list[Entry]] = [[] for _ in range(n_lanes)]

    main = prog.instrs[:prog.main_len]
    for ins in main:
        if ins.op in _REGION_OPS:
            raise ScheduleError(
                "scheduler: region ops (while/if) not yet scheduled "
                "(structured seam — see python/NOTES.md)")

    # compute instrs get a task; NOP (and future no-task ops) do not.
    instr_task: dict[int, int] = {}
    for idx, ins in enumerate(main):
        if ins.op == L.OP_NOP:
            continue
        instr_task[idx] = len(tasks)
        tasks.append(_instr_to_task(ins))

    sched_indices = [i for i in range(len(main)) if i in instr_task]
    for level in _levels(main, sched_indices):
        level_tasks = [(instr_task[i], tasks[instr_task[i]]) for i in level]
        for lane, entry in _pack_level(level_tasks, n_lanes, config):
            lane_streams[lane].append(entry)
        # v0: global BARRIER after every level (including the last)
        for lane in range(n_lanes):
            lane_streams[lane].append(_barrier_entry())

    # a program with no compute instrs still gets one barrier phase so lane
    # streams are uniform (and the executor has a defined shape)
    if not sched_indices:
        for lane in range(n_lanes):
            lane_streams[lane].append(_barrier_entry())

    return Schedule(n_flags=0, n_lanes=n_lanes, tasks=tasks,
                    lane_streams=lane_streams)


def lower_and_schedule(artifact: bytes,
                       config: DeviceConfig | None = None) -> bytes:
    """VHLO artifact -> serialized v3 VMProgram (tensor + schedule sections)."""
    prog = L.lower_artifact(artifact)
    sched = schedule_program(prog, config)
    return prog.serialize(sched)
