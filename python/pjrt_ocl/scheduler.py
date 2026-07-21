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

# EW tile size (TS). The plugin bakes the device-tuned value into the kernels
# (-DEW_TS, runtime.cc) and advertises it here via env; both sides must agree
# on the tile -> element-range mapping. Standalone imports (unit tests) get
# the CPU/default 16384.
TILE_SIZE = int(os.environ.get("PJRT_OCL_EW_TS", "16384") or "16384")
# MMA output tile edge (must equal the kernel's MMA_TM/MMA_TN). Default 64; the
# runtime advertises 128 via PJRT_OCL_MMA_T when the TF32 megakernel is built
# with -DVMO_MEGA_BIGTILE (§31 go-to-188). Tile counts (Task.n_tiles) MUST match
# the kernel's tile->(tr,tc) mapping, so the two sides read the same value.
MMA_T = int(os.environ.get("PJRT_OCL_MMA_T", "64") or "64")

TILE_EW = 0
TILE_MMA = 1
TILE_GATHER = 2
TILE_REDUCE_PART = 3
TILE_REDUCE_COMB = 4
TILE_IOTA_DIM = 5
TILE_SCATTER = 6      # strided scatter: dst[out_off + affine(i)] = a[i]
TILE_DYN_GATHER = 7   # dynamic_slice: gather with a runtime base offset
TILE_DYN_SCATTER = 8  # dynamic_update_slice: scatter with a runtime base offset
TILE_RED_WINDOW = 9   # windowed reduction (pooling)
TILE_RED_SEG = 10     # segmented reduce: out[o] = reduce(in[o*seg : (o+1)*seg])
TILE_SOFTMAX_SEG = 11    # fused softmax over the innermost seg elems (§19)
TILE_LAYERNORM_SEG = 12  # fused layernorm core over the innermost seg elems (§19)
TILE_MAP_REGION = 13     # §27/§28 register-resident fused map-region (one phase)
TILE_FLASH_ATTN = 14     # §34 fused flash-attention (online softmax; one WG per head,query)
TILE_RED_STRIDED = 15    # partial-axis reduce over interior/prefix axis block:
                         # out[o*inner+i]=reduce_r in[(o*red+r)*inner+i] (p0=n_out,
                         # p1=red, p2=inner, p3=kind); EW-style output tiling
TILE_GATHER_INDEX = 16   # §38 general data-dependent gather (16: 15 = TILE_RED_STRIDED)

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
TASK_FOR = 0xFFFFFFFB   # fixed-trip loop: body in tile_lo/tile_hi, trips in wait_flag
FLAG_NONE = 0xFFFFFFFF

# tensor opcode -> (tile_op, ew_subop) for the compute instrs we cover today.
# constants are NOT instructions (they live in the const pool) => no task.
_TENSOR_TO_EW = {
    L.OP_ADD_F32: EW_ADD,
    L.OP_MUL_F32: EW_MUL,
    L.OP_SUB_F32: EW_SUB,
    L.OP_FILL_F32: EW_FILL,
}

# region ops: scheduled recursively into per-lane control entries + sub-streams
_REGION_OPS = {L.OP_WHILE, L.OP_FOR}

SCHED_HDR_STRUCT = struct.Struct("<IIII")      # n_tasks,n_entries,n_flags,n_lanes
TASK_STRUCT = struct.Struct("<IIIIIIIIIIII")   # 48B (p4/p5 view, p6/p7 §33 epi)
LANETAB_STRUCT = struct.Struct("<IIII")        # 16B: off, count, root_len, pad
ENTRY_STRUCT = struct.Struct("<IIIIIIII")      # 32B
assert SCHED_HDR_STRUCT.size == 16
assert TASK_STRUCT.size == 48
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
    TILE_GATHER_INDEX: "gather_tile_us",
    TILE_REDUCE_PART: "reduce_tile_us",
    TILE_REDUCE_COMB: "reduce_tile_us",
    TILE_IOTA_DIM: "ew_tile_us",
}


@dataclasses.dataclass
class DeviceConfig:
    nlanes: int = 8
    costs: dict = dataclasses.field(default_factory=dict)   # key -> µs/tile
    fuse: bool = True     # fuse lane-local elementwise chains (no barrier)

    @classmethod
    def from_env(cls, environ=None) -> "DeviceConfig":
        environ = os.environ if environ is None else environ
        nlanes = int(environ.get("PJRT_OCL_NLANES", "8") or "8")
        if nlanes < 1:
            nlanes = 1
        fuse = environ.get("PJRT_OCL_FUSE", "1") != "0"
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
        return cls(nlanes=nlanes, costs=costs, fuse=fuse)

    def unit_cost(self, tile_op: int) -> float:
        return self.costs.get(_COST_KEYS.get(tile_op, "ew_tile_us"), 1.0)


# --- in-memory schedule + writer --------------------------------------------

@dataclasses.dataclass
class Task:
    tile_op: int          # base op; dtype packed into bits 8-15 at serialize
    dst: int = 0
    a: int = 0
    b: int = 0
    p0: int = 0
    p1: int = 0
    p2: int = 0
    p3: int = 0
    p4: int = 0           # MMA operand-a VIEW aux-offset (+1; 0 = contiguous)
    p5: int = 0           # MMA operand-b VIEW aux-offset (+1; 0 = contiguous)
    p6: int = 0           # §33 R2c: matmul epilogue descriptor aux-offset (+1; 0=none)
    p7: int = 0           # §33 R2c: epilogue second-input (residual/bias) buffer id
    dtype: int = 0        # DT_* result dtype (how the VM writes the output)
    adtype: int = 0       # DT_* operand dtype (compare/convert differ from dtype)

    def n_tiles(self) -> int:
        if self.tile_op in (TILE_EW, TILE_GATHER, TILE_IOTA_DIM, TILE_SCATTER,
                             TILE_DYN_GATHER, TILE_DYN_SCATTER, TILE_RED_WINDOW,
                             TILE_MAP_REGION, TILE_GATHER_INDEX):
            return max(1, math.ceil(self.p1 / TILE_SIZE))
        if self.tile_op == TILE_RED_STRIDED:
            return max(1, math.ceil(self.p0 / TILE_SIZE))   # p0 = n_out, EW-style
        if self.tile_op == TILE_MMA:
            return (math.ceil(self.p0 / MMA_T) * math.ceil(self.p1 / MMA_T)
                    * max(1, self.p3))          # p3 = batch count
        if self.tile_op == TILE_REDUCE_PART:
            return max(1, math.ceil(self.p0 / self.p1)) if self.p1 else 1
        if self.tile_op == TILE_REDUCE_COMB:
            return 1
        if self.tile_op in (TILE_RED_SEG, TILE_SOFTMAX_SEG, TILE_LAYERNORM_SEG):
            return max(1, self.p0)            # ONE segment per tile (p0 = n_out)
        if self.tile_op == TILE_FLASH_ATTN:
            return max(1, self.p1 * self.p2)  # ONE workgroup per (head, query row)
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
    # per-lane root_len (top-level walk length). None => whole stream is root
    # (no control flow). WHILE sub-ranges live at [root_len, len(stream)).
    root_lens: list[int] | None = None

    def serialize_sections(self) -> bytes:
        assert len(self.lane_streams) == self.n_lanes
        flat: list[Entry] = []
        lane_tab: list[tuple[int, int, int]] = []
        for lane, stream in enumerate(self.lane_streams):
            root_len = (len(stream) if self.root_lens is None
                        else self.root_lens[lane])
            lane_tab.append((len(flat), len(stream), root_len))
            flat.extend(stream)
        out = bytearray()
        out += SCHED_HDR_STRUCT.pack(len(self.tasks), len(flat),
                                     self.n_flags, self.n_lanes)
        for t in self.tasks:
            out += TASK_STRUCT.pack(
            t.tile_op | (t.dtype << 8) | (t.adtype << 16), t.dst, t.a, t.b,
            t.p0, t.p1, t.p2, t.p3, t.p4, t.p5, t.p6, t.p7)
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
    WAW on writes (docs/roadmap.md, task spec) + WAR on i's reads. WAR never
    fires in the SSA bulk of a program (nothing is written twice), but loop
    carries are NOT SSA: copy-backs and in-place carry commits (elementwise
    and DUS folds) rewrite a buffer that earlier body instrs read — e.g. the
    in-place scatter's runtime-index read of the counter carry vs the counter
    copy-back. Without the WAR edge those can share a barrier phase and race
    across lanes."""
    a, b = instrs[i], instrs[j]
    aw = _writes(a)
    bw = _writes(b)
    return bool((_reads(b) & aw) or (bw & aw) or (_reads(a) & bw))


def _cross_lane_dep(instrs, j: int, i: int, is_map) -> bool:
    """Does j depend on i via an edge that CROSSES lanes — i.e. one that a
    barrier must publish? A dependency is lane-LOCAL (no barrier) iff both ops
    are elementwise (map-type) with the same element count: then output tile T
    reads only input tile T, so a lane that owns tile T produces everything it
    needs and the two ops chain on that lane. Every other real dependency is
    cross-lane: a shaped producer (matmul/reduce/gather — its output tiling
    differs) or a shaped consumer (reads all tiles), or a change of element
    count (reshape/broadcast). WAW is treated cross-lane to be safe."""
    if not _depends(instrs, j, i):
        return False
    return not (is_map(i) and is_map(j) and instrs[i].n == instrs[j].n
                and _writes(instrs[i]).isdisjoint(_writes(instrs[j])))


def _phases(instrs, indices: list[int], is_map, fuse: bool = True
            ) -> list[list[int]]:
    """Group `indices` (SSA order) into PHASES separated by barriers. A new phase
    starts only when an instr has a CROSS-LANE dependency on a phase member;
    same-index elementwise deps stay in the phase and chain per-lane (no
    barrier). This coarsens the dataflow graph: a whole elementwise fork/join
    (a+b, a-b, then their product) is ONE phase, while a matmul or reduction
    forces a phase boundary because its consumers read across lanes.

    `fuse=False` (PJRT_OCL_FUSE=0) reverts to one phase per dataflow level — every
    dependency is a barrier — for A/B comparison and debugging."""
    def boundary(j, i):
        return (_depends(instrs, j, i) if not fuse
                else _cross_lane_dep(instrs, j, i, is_map))
    phases: list[list[int]] = []
    cur: list[int] = []
    for j in indices:
        if cur and any(boundary(j, i) for i in cur):
            phases.append(cur)
            cur = [j]
        else:
            cur.append(j)
    if cur:
        phases.append(cur)
    return phases


# --- instruction -> task mapping --------------------------------------------

def _instr_to_task(ins: L.Instr, buffers) -> Task:
    """Map a compute tensor instruction to its tile task. Built-in EW fast
    path; other ops register a mapper in opsem.TO_TASK. The task dtype is the
    result buffer's dtype (how the VM interprets its arena slots)."""
    dtype = buffers[ins.dst].dtype
    # operand dtype: for compare/convert the inputs differ from the bool/output
    # dtype. `a` is the representative operand for every current op.
    adtype = buffers[ins.a].dtype if ins.a < len(buffers) else dtype
    if ins.op in _TENSOR_TO_EW:
        subop = _TENSOR_TO_EW[ins.op]
        # add/mul/sub: p2/p3 carry the a/b VIEW aux-offsets (0 = direct);
        # fill: p2 carries the fill value bits (no operands to view).
        if ins.op == L.OP_FILL_F32:
            p2, p3 = ins.imm, 0
        else:
            p2, p3 = ins.imm, ins.imm2
        task = Task(TILE_EW, dst=ins.dst, a=ins.a, b=ins.b,
                    p0=subop, p1=ins.n, p2=p2, p3=p3)
    else:
        mapper = opsem.TO_TASK.get(ins.op)
        if mapper is None:
            raise ScheduleError(
                f"scheduler: no task mapping for opcode {ins.op} "
                f"({L.OP_NAMES.get(ins.op, hex(ins.op))})")
        task = mapper(ins)
    task.dtype = dtype
    task.adtype = adtype
    return task


# --- lane packing within a level (LPT by cost) ------------------------------

def _pack_units(units: list[tuple[list[int], int, float]], n_lanes: int
                ) -> list[tuple[int, Entry]]:
    """Return [(lane, Entry)] for one dataflow phase. A UNIT is a chain of task
    ids that must run together on one lane over a shared tile range: a shaped op
    is a singleton chain `[tid]`; a fused elementwise chain is `[tid0, tid1, …]`
    in topological order (its tile T of every op stays on the same lane, so the
    chain runs with no barrier and no fence).

    Chunk + LPT packing (one regime): each unit's `tiles` are split into
    k = min(tiles, ceil(n_lanes * cost_share)) contiguous chunks — an expensive
    unit fans out over many (up to all) lanes, a cheap one stays a single chunk —
    then every chunk is list-scheduled (LPT, cost-descending) onto the
    least-loaded lane, emitting the unit's WHOLE chain over that chunk's tile
    range. A lane may carry several units: that lets the scheduler sequentialize
    cheap units behind chunks of an expensive one instead of dedicating whole
    lanes to them (with measured costs the diamond packs the matmul on all lanes
    and stacks the elementwise chain behind it — docs/decisions.md #1)."""
    result: list[tuple[int, Entry]] = []
    if not units:
        return result
    total = sum(c for _, _, c in units) or 1.0
    chunks = []                         # (chunk_cost, tids, tile_lo, tile_hi)
    for uid, (tids, tiles, cost) in enumerate(units):
        k = min(tiles, max(1, math.ceil(n_lanes * cost / total)))
        for j in range(k):
            lo = tiles * j // k
            hi = tiles * (j + 1) // k
            chunks.append((cost * (hi - lo) / tiles, uid, tids, lo, hi))
    chunks.sort(key=lambda c: (-c[0], c[1], c[3]))
    loads = [0.0] * n_lanes
    for chunk_cost, _uid, tids, lo, hi in chunks:
        lane = min(range(n_lanes), key=lambda l: loads[l])
        for tid in tids:                # the whole chain over this tile range
            result.append((lane, Entry(tid, lo, hi)))
        loads[lane] += chunk_cost
    return result


# --- top-level scheduling ---------------------------------------------------

@dataclasses.dataclass
class _WhileJob:
    """A deferred region-scheduling job for one WHILE instruction: the per-lane
    WHILE Entry objects to patch, plus the cond/body instruction index lists."""
    while_entries: list          # one Entry per lane (patched after scheduling)
    cond_indices: list
    body_indices: list


class _Scheduler:
    """Recursive per-lane stream builder. The root list schedules into each
    lane's stream; every WHILE emits a uniform control Entry into all lanes and
    queues its cond/body sub-lists, which are appended AFTER the root (and after
    any enclosing region) so they live at stream indices >= root_len — the
    device frame-walk never steps into a sub-range it did not push (root_len
    rule, docs/vmprogram.md, validated by runtime_test B)."""

    def __init__(self, prog: L.VMProgram, config: DeviceConfig, n_lanes: int):
        self.prog = prog
        self.config = config
        self.n_lanes = n_lanes
        self.lanes: list[list[Entry]] = [[] for _ in range(n_lanes)]
        self.tasks: list[Task] = []
        self.instr_task: dict[int, int] = {}
        self.region_queue: list[_WhileJob] = []

    def _task_for(self, idx: int) -> int:
        tid = self.instr_task.get(idx)
        if tid is None:
            tid = len(self.tasks)
            self.instr_task[idx] = tid
            self.tasks.append(_instr_to_task(self.prog.instrs[idx],
                                             self.prog.buffers))
        return tid

    def _add_barrier(self) -> None:
        for lane in range(self.n_lanes):
            self.lanes[lane].append(_barrier_entry())

    def _is_map(self, i: int) -> bool:
        """Elementwise (map-type): output tile T reads only input tile T, so it
        chains on a lane. Everything that reshuffles tiles across lanes
        (matmul/reduce/gather/scatter/broadcast/…) is a non-map TILE op."""
        return self.tasks[self._task_for(i)].tile_op == TILE_EW

    def _build_levels(self, indices: list[int]):
        """Ordered phases over `indices` (SSA order): a compute run splits into
        barrier-separated PHASES (lane-local elementwise deps stay together, see
        _phases); each WHILE is its own phase (an all-lanes sync point). Yields
        ("compute", [idx...]) or ("while", idx). NOPs are dropped."""
        levels: list = []
        seg: list[int] = []

        def flush():
            if seg:
                for ph in _phases(self.prog.instrs, seg, self._is_map,
                                  self.config.fuse):
                    levels.append(("compute", ph))
                seg.clear()

        for i in indices:
            op = self.prog.instrs[i].op
            if op in _REGION_OPS:
                flush()
                levels.append(("while", i))
            elif op == L.OP_NOP:
                continue
            else:
                seg.append(i)
        flush()
        return levels

    def _map_chains(self, map_indices: list[int]) -> list[list[int]]:
        """Connected components of map (elementwise) ops under data dependencies
        = fused chains. Independent ops (no dep) stay in separate chains so they
        can run on different lanes in parallel; a dependent fork/join collapses
        into one chain that runs on a shared lane per tile. Within a phase every
        buffer has a unique writer (WAW is cross-lane ⇒ a phase boundary)."""
        parent = {i: i for i in map_indices}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        idxset = set(map_indices)
        writer: dict[int, int] = {}
        for i in map_indices:
            for b in _writes(self.prog.instrs[i]):
                writer[b] = i
        for j in map_indices:
            for b in _reads(self.prog.instrs[j]):
                w = writer.get(b)
                if w is not None and w != j:
                    parent[find(w)] = find(j)
        comps: dict[int, list[int]] = {}
        for i in map_indices:
            comps.setdefault(find(i), []).append(i)
        return [sorted(v) for v in comps.values()]

    def _emit_phase(self, indices: list[int]) -> None:
        """Emit one compute phase. Build UNITS — each shaped op is a singleton
        chain; each connected component of elementwise ops is a fused chain (topo
        = SSA order) — then chunk+LPT them onto lanes (_pack_units). Dependent
        elementwise ops thus chain on one lane per tile (no barrier, no fence:
        thread `lid` writes then re-reads the same elements across ops), while
        independent units fan out across lanes in parallel."""
        shaped: list[int] = []
        map_indices: list[int] = []
        for i in sorted(indices):                 # SSA order == topological
            (map_indices if self._is_map(i) else shaped).append(i)

        units: list[tuple[list[int], int, float]] = []
        for i in shaped:
            t = self.tasks[self._task_for(i)]
            tiles = t.n_tiles()
            units.append(([self._task_for(i)], tiles,
                          tiles * self.config.unit_cost(t.tile_op)))
        ew_cost = self.config.unit_cost(TILE_EW)
        for comp in self._map_chains(map_indices):
            comp.sort()                            # topological (SSA order)
            tids = [self._task_for(i) for i in comp]
            tiles = self.tasks[tids[0]].n_tiles()  # all same element count
            units.append((tids, tiles, len(tids) * tiles * ew_cost))

        for lane, entry in _pack_units(units, self.n_lanes):
            self.lanes[lane].append(entry)

    def schedule_range(self, indices: list[int], trailing_barrier: bool) -> None:
        """Append entries for a linear instruction sub-list to every lane, with
        a global BARRIER between phases. `trailing_barrier` controls the last
        phase's barrier: True for the root (barrier after every phase); False
        for cond/body sub-lists, whose closing barrier the WHILE machinery in
        the kernel supplies (after the cond scalar read / after the body)."""
        levels = self._build_levels(indices)
        for li, (kind, payload) in enumerate(levels):
            if kind == "compute":
                self._emit_phase(payload)
            else:
                self._emit_while(payload)
            if trailing_barrier or li != len(levels) - 1:
                self._add_barrier()

    def _emit_while(self, idx: int) -> None:
        ins = self.prog.instrs[idx]
        is_for = ins.op == L.OP_FOR
        while_entries = []
        for lane in range(self.n_lanes):
            # WHILE: tile_lo/tile_hi (cond range) + wait_flag/wait_count (body
            # range) are patched once the sub-lists are scheduled; signal_flag
            # carries the cond BUFFER id (executor patches to a byte offset at
            # load). FOR: body range in tile_lo/tile_hi (patched), trip count
            # in wait_flag, no cond buffer.
            if is_for:
                e = Entry(TASK_FOR, tile_lo=0, tile_hi=0, wait_flag=ins.b,
                          wait_count=0, signal_flag=FLAG_NONE)
            else:
                e = Entry(TASK_WHILE, tile_lo=0, tile_hi=0, wait_flag=0,
                          wait_count=0, signal_flag=ins.dst)
            self.lanes[lane].append(e)
            while_entries.append(e)
        self.region_queue.append(_WhileJob(
            while_entries,
            [] if is_for else list(range(ins.a, ins.a + ins.b)),  # cond range
            list(range(ins.n, ins.n + ins.imm))))     # body instr range

    def schedule_region(self, job: _WhileJob) -> None:
        """Schedule one while's cond then body sub-lists contiguously into every
        lane, then patch each lane's WHILE entry with its own (per-lane) cond/
        body entry ranges (a FOR entry has no cond range: body goes in
        tile_lo/tile_hi, its trip count is already in wait_flag). Nested whiles
        enqueue further jobs (drained later, so their sub-ranges land beyond
        this body)."""
        cond_start = [len(l) for l in self.lanes]
        self.schedule_range(job.cond_indices, trailing_barrier=False)
        body_start = [len(l) for l in self.lanes]
        self.schedule_range(job.body_indices, trailing_barrier=False)
        body_end = [len(l) for l in self.lanes]
        for lane in range(self.n_lanes):
            e = job.while_entries[lane]
            if e.task == TASK_FOR:
                e.tile_lo = body_start[lane]
                e.tile_hi = body_end[lane] - body_start[lane]    # body_len
            else:
                e.tile_lo = cond_start[lane]
                e.tile_hi = body_start[lane] - cond_start[lane]  # cond_len
                e.wait_flag = body_start[lane]
                e.wait_count = body_end[lane] - body_start[lane]  # body_len


def schedule_program(prog: L.VMProgram,
                     config: DeviceConfig | None = None,
                     allow_multilane_while: bool = True) -> Schedule:
    """Schedule the tensor VMProgram into per-lane streams. Root instrs schedule
    with a global BARRIER between levels; each WHILE becomes a uniform control
    entry whose cond/body sub-lists are appended after the root (root_len rule).
    v0 contract: WAIT/SIGNAL unused; n_flags = 0.

    WHILE + cross-lane data: loop-carried buffers written by one lane and read by
    another in a later iteration used to race — the barrier published the atomic
    cond flag but NOT non-atomic data (work-group-scoped fence; docs/decisions.md
    #1, quantified in poc/07). RESOLVED by the device-scope acquire/release fence
    barrier (poc/07 test E): plain cross-lane reads are now coherent, so while
    bodies schedule across ALL lanes like any other op. `allow_multilane_while`
    is retained (now default True) only for callers that pinned it; it no longer
    gates anything. (Multi-lane on PoCL still deadlocks on the LIVENESS axis —
    the spin-barrier needs co-resident workgroups — but that affects every
    multi-lane program, not just while; CPU needs the host-dispatch engine.)"""
    config = config or DeviceConfig.from_env()
    n_lanes = config.nlanes
    sc = _Scheduler(prog, config, n_lanes)

    # Root schedules with a global BARRIER only BETWEEN levels (trailing_barrier
    # =False): the barrier after the last root level synchronizes nothing
    # (nothing reads it before the kernel ends + clFinish), so omit it. A
    # single-level program then needs NO cross-workgroup barrier at all —
    # important on devices where the persistent-thread barrier doesn't co-reside
    # (docs/decisions.md #1). WHILE cond/body sub-lists still get their internal
    # barriers (the while machinery supplies them); they live at indices
    # >= root_len and are entered mid-root via a frame push, so the smaller
    # root_len is irrelevant to them.
    sc.schedule_range(list(range(prog.main_len)), trailing_barrier=False)
    # a program with no root entries still gets one barrier phase so lane
    # streams are uniform (and the executor has a defined shape)
    if not any(sc.lanes):
        sc._add_barrier()
    root_lens = [len(l) for l in sc.lanes]

    # drain region jobs (BFS): cond/body — and any nested whiles they contain —
    # append after the root, all at indices >= root_len.
    while sc.region_queue:
        sc.schedule_region(sc.region_queue.pop(0))

    return Schedule(n_flags=0, n_lanes=n_lanes, tasks=sc.tasks,
                    lane_streams=sc.lanes, root_lens=root_lens)


def lower_and_schedule(artifact: bytes,
                       config: DeviceConfig | None = None) -> bytes:
    """VHLO artifact -> serialized v3 VMProgram (tensor + schedule sections)."""
    prog = L.lower_artifact(artifact)
    sched = schedule_program(prog, config)
    return prog.serialize(sched)
