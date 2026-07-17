# python/ NOTES — productionizing poc/03 into pjrt_ocl (2026-07-14)

For merge into `docs/decisions.md`. Everything observed on jax/jaxlib 0.10.2, python 3.12.

## What the C++ side must know (the contract)

- `initialize()` registers platform **'opencl'**, priority 500, and passes
  `register_plugin(..., options={'python_exe': sys.executable,
  'lower_service': <abs path to python/pjrt_ocl/lower_service.py>})`. These arrive as
  `PJRT_Client_Create` create_options (both plain strings).
- Lowering invocation: `<python_exe> <lower_service>` with the VHLO portable artifact on
  **stdin**, VMProgram v1 on **stdout**. Exit codes: 0 ok; **2 unsupported program** (valid
  input, beyond op coverage → surface a clean PJRT error, UNIMPLEMENTED-ish); **3 internal
  error**. On failure stdout is empty and stderr carries one JSON object
  `{"error": <exception class>, "message": <str>}`. (poc/03 used exit 1 for internal errors;
  the packaged service uses 3 per the M1 task spec — update any C++ code written against 1.)
- Plugin .so discovery (python side): env `PJRT_OCL_PLUGIN_PATH`, else
  `<repo>/pjrt_plugin/build/libpjrt_ocl.so` (repo root = two dirs above the editable-installed
  package). Missing .so → `initialize()` logs a warning and **skips registration** (verified:
  `import jax; jax.devices()` stays healthy, cpu backend selected).

## VMProgram v1 — spec readings recorded (docs/vmprogram.md)

Implemented byte-exactly as specified plus the coordinator's IO-shapes amendment. Points where
the spec text left room, and the reading implemented (golden test `test_golden_layout_exact_bytes`
pins all of these):

1. **Section order** = the order the spec lists them: header, buffer table, IO maps,
   **IO shapes** (spec update 2026-07-14: `{rank u32, pad u32, dims u64[rank]}` per IO buffer,
   inputs in argument order then outputs), const pool, instructions. Every section start is
   8B-aligned by construction (all entries are 8B multiples after the mandated padding), so no
   inter-section padding is ever emitted; the reader still asserts alignment defensively.
2. **IO-shapes count is implicit** (`n_inputs + n_outputs`) — the header has no field for it.
   The amendment didn't add one; readers must derive it.
3. **"padded to 8B after each array"** (IO maps) read as: pad after the inputs array AND after
   the outputs array, separately. Rank-0 IO shapes are a bare `{rank=0, pad}` 8-byte entry.
4. **No COPY opcode in v1** (poc/03's strawman had one). Returning an argument/constant
   unchanged, or the same value twice, is lowered by **aliasing**: the outputs map simply
   references the producing buffer id. Executor consequence: an output region may coincide with
   an input/const region; reading outputs after the run is always correct since v1 is
   synchronous. If the C++ side ever wants distinct result allocations it must copy on D2H.
5. **arena_bytes includes the 64B tail padding** of the last buffer (allocation advances in
   64B units), so the arena size is always a multiple of 64. Harmless either way; recorded so
   the golden numbers make sense (3×f32[8] → arena 192, not 160).
6. **Splat constants are expanded** into the const pool (N copies) rather than lowered to
   `FILL_F32`. Chosen for executor simplicity (one const-upload path); FILL emission is a
   cheap follow-up if const-pool bloat ever matters. FILL/IOTA/LTS/WHILE are still fully
   supported by vmreader's parser+interpreter (hand-built-program test) so the C++ executor
   has a reference for all 8 opcodes.
7. **WHILE evaluation order**: cond list runs first, then `dst[0]` is tested; body runs only
   while it is nonzero (i.e. `while cond() { body(); }`, cond re-evaluated after each body).
   Matches poc/01 test4. The numpy interpreter uses plain recursion with the spec's depth
   cap (8) enforced.
8. Reader strictness (`vmreader.parse`): rejects bad magic/version, non-8B-aligned sections,
   non-64B-aligned buffer offsets, buffers outside the arena, unknown dtype/opcode, nonzero pad
   fields, out-of-range buffer indices / WHILE sub-list ranges, const byte_len > buffer size,
   and trailing bytes. Mirror this in the C++ loader (spec says reject loudly).

## Tried / failed / measured

- **XLA CPU contracts `a*b - c` into an FMA under `jax.jit`** — bit-exact comparison against
  jax.jit is NOT achievable for general data with mul-feeding-add/sub patterns. Measured
  4.66e-9 (1 ULP) divergence on f32 standard-normal data; jit result equals the
  f64-compute-then-round value, i.e. a hardware FMA. Tried and failed to disable it:
  `--xla_allow_excess_precision=false`, `--xla_cpu_enable_fast_math=false`,
  `--xla_cpu_use_fusion_emitters=false` (no effect),
  `--xla_cpu_disable_new_fusion_emitters=true`, `--xla_cpu_disable_tiled_emitter=true`
  (flags don't exist → crash). **Eager (non-jit) jax executes one XLA op at a time and matches
  our per-op VM exactly** (diff 0.0 on the same data). Test strategy adopted: e2e cases use
  integer-valued f32 inputs (all intermediates exactly representable ⇒ contraction cannot
  change results ⇒ bit-exact vs jax.jit), plus one real-valued case compared bit-exactly
  against eager jax. poc/03's atol=0 pass on `(a + b*c) - K` was seed luck.
  **M2/M3 consequence**: e2e GPU tests comparing against jax.jit CPU need the same policy
  (integer-valued data, or a 1-ULP tolerance on fusable patterns).
- **Entry-point discovery works as documented**: with `pjrt-ocl` pip-installed (editable),
  plain `import jax` already imports `pjrt_ocl` and calls `initialize()` (visible via the
  skip-registration warning). No `jax_plugins/` namespace dir needed — poc/02's dir approach
  is superseded; `initialize()` still guards `"opencl" in xb._backend_factories` in case both
  are ever present.
- jax's `discover_pjrt_plugins()` wraps `initialize()` in a bare `except:` with
  `logger.exception`, so raising on a missing .so wouldn't crash jax either — but it would
  print a full traceback on every import in a source checkout. Log-and-skip chosen instead;
  `find_plugin_library()` raises the clear FileNotFoundError for anyone calling it directly.
- `register_plugin` contract re-read from jax 0.10.2 source (`jax/_src/xla_bridge.py:596`):
  `options` may be a dict or a zero-arg callable returning one; values go through
  `xla_client._NameValueMapping` (str/int/float/bool/lists). We pass a plain dict of two strs.
  Also confirmed: passing both `library_path` and `c_api` is an error; we pass only
  `library_path`.
- **pyproject dependencies**: declared `jax` + `numpy` unpinned (jaxlib comes with jax).
  Deliberately no version pins — a plugin must not fight the host env's jax pin; we develop
  against 0.10.2 (docs/decisions.md #4b). No other deps.
- `pip install -e python/` with `--cache-dir third_party/pip-cache` works; `pytest` was not in
  the venv and was installed the same way (pytest 9.1.1).
- `lower_service.py` runs both as `python -m pjrt_ocl.lower_service` and as a direct script
  path (what the C++ plugin does). Direct-script mode falls back to inserting `python/` on
  sys.path if the package isn't installed, so a bare source checkout works too. Both modes
  produce byte-identical VMProgram output (asserted during bring-up).
- Artifact-byte instability across call sites (poc/03 NOTES #3) reconfirmed implicitly: tests
  never compare artifact bytes, only lowered results.

## Open questions / follow-ups

- `python_exe`/`lower_service` option **key names** are now load-bearing for the C++ side —
  if M2 renames them, update `pjrt_ocl/__init__.py` and this note together.
- Buffer plan is still naive one-buffer-per-SSA-value (no liveness reuse) — M1 leftover,
  fine for current program sizes.
- Should the service cap artifact size / add a timeout guard? Currently reads stdin to EOF
  unbounded; the C++ side should enforce a subprocess timeout.

## VMProgram v3 (tensor v2 + schedule v2.1) — Phase 1.3 (2026-07-15)

Upgraded the python producer from v1 to **version 3** (v2 tensor sections + v2.1 schedule
sections). Files: `lowering.py` (v3 tensor writer), new `scheduler.py` (tensor→schedule),
`vmreader.py` (parses v3 + two validators), `lower_service.py` (runs the scheduler),
`tests/test_lowering.py` (updated goldens + schedule tests). Consumer is the C++ VLIW engine
being built in parallel against the SAME spec (docs/vmprogram.md).

### Spec readings implemented byte-exactly (goldens pin these)

1. **`version = 3`.** docs/vmprogram.md v2.1 says "Header version becomes 3"; v2 tensor-only files
   are never emitted (one repo, no compat). Magic is unchanged (`0x314D5056` / "VPM1") — the spec
   only bumps the version field. Reader rejects any version != 3.
2. **48-byte header**: after `n_outputs`, inserted `n_aux u32, pad u32`, then `arena_bytes u64`
   as before → `<IIIIIIIIIIQ>`. `pad` asserted zero by the reader.
3. **Aux pool** section is between IO shapes and const pool: `n_aux × u32`, then pad to 8B. For
   the current EW-only op set `n_aux == 0`, so the section is empty (0 bytes). Instructions carry
   the aux word offset in the field formerly `pad0`, now **`aux`**; `pad1` stays reserved and is
   asserted zero. EW ops emit `aux = 0`.
4. **Schedule sections** follow the instruction array (already 8B-aligned since instrs are 32B).
   Layout exactly per spec: `sched header {n_tasks,n_entries,n_flags,n_lanes}` (16B), tasks
   (32B each), **lane table** (n_lanes × {entry_off u32, entry_count u32}), entries (32B each,
   `{task,tile_lo,tile_hi,wait_flag,wait_count,signal_flag,slots,pad}`). Entries are stored as one
   flat array; the lane table indexes into it (mirrors poc/04's host layout).
5. **v0 scheduling contract**: BARRIER (`task=0xFFFFFFFE`) appended to EVERY lane's stream after
   EVERY dataflow level, **including the last** (per task spec). Consequence: all lanes have an
   identical barrier count (= number of levels) and every lane's stream ends with a BARRIER.
   WAIT/SIGNAL unused: `wait_flag = signal_flag = 0xFFFFFFFF`, `wait_count = 0`. `slots = 0`.
   `n_flags = 0`.

### Ambiguities found + the reading chosen (recorded per hard-rule; no silent deviation)

- **A1 — Levels vs. barrier count / "one barrier phase for empty programs".** The spec defines a
  BARRIER after each level but doesn't say what an instruction-free root list produces. Chosen:
  emit a single BARRIER on every lane (one empty phase) so lane streams stay uniform and the
  executor always has a defined shape. (Current lowering always has ≥1 compute instr, so this is
  only a defensive corner.)
- **A2 — Dependency model.** Task spec: "B depends on A if B reads (a/b, +p3 for select later) or
  writes a buffer that A writes (WAW too)." Implemented RAW (B.reads ∩ A.writes) + WAW
  (B.writes ∩ A.writes) + **WAR (B.writes ∩ A.reads) — added 2026-07-17**. WAR was originally
  omitted (the SSA argument below), but loop CARRIES are not SSA: copy-backs and the in-place
  carry commits (elementwise + DUS folds) rewrite buffers earlier body instrs read. The concrete
  bug: the in-place DUS scatter reads the counter carry (runtime index) and shared a barrier
  phase with the counter copy-back — a cross-lane race the schedule simulator's order-independence
  check caught. WAR never fires in the SSA bulk of a program (nothing is written twice), so it
  costs no phases there (verified: transformer-ish block 15 phases either way). Note the arena
  liveness-reuse pass (§16) aliases SLOTS, not buffer IDs, and guarantees a barrier between
  intervals — it does not rely on WAR edges.
- **A3 — Greedy level grouping is order-sensitive but correctness-safe.** "an instr joins the
  current level iff it has no dependency on any instr in the current level; else new level" is a
  single forward pass (not optimal bin-of-levels). It never places an instr in the same level as
  a dependency, so barriers always separate producer/consumer → correct. It can be *suboptimal*
  (an instr that only depends on an *earlier* level still starts a new level if it conflicts with
  something in the current one). Matches the spec's literal wording; good enough for v0.
- **A4 — Lane packing: `n_tasks > n_lanes`.** The spec's packing ("≥1 lane/task, ≤tiles lanes,
  one entry per (task,lane-range)") implicitly assumes `n_tasks ≤ n_lanes` (else you can't give
  every task its own disjoint lane block). Current lowering can't produce that (small graphs,
  default 8 lanes), but for robustness I added an **overflow regime**: LPT bin-pack whole tasks
  onto lanes (cost-descending onto the least-loaded lane), each task = ONE entry covering all its
  tiles on a single lane. This keeps "one entry per (task,lane-range)" valid and every task on
  ≥1 lane, at the cost of multiple entries per lane in one phase (which the async engine already
  supports — poc/04 test E). Primary regime unchanged. Reader/simulator validate both.
- **A5 — "LPT by cost" + "proportional to cost share".** Read as: seed 1 lane per task, then hand
  each remaining lane to the task with the highest current *per-lane* cost (`total_cost/lanes`)
  that can still absorb one (`lanes < tiles`) — an LPT top-up that converges to proportional
  shares. A task's tiles then split into contiguous even ranges across its lanes
  (`lo = tiles*j//k`, poc/04 pattern). NOTE: for the *current* op set every task is EW with the
  SAME unit cost, so lane bias comes only from **tile count** (n_elems); the cost table only
  starts to matter once MMA/GATHER/REDUCE tasks (different unit costs) exist. Verified: 8-tile vs
  2-tile co-scheduled adds → 6 vs 2 lanes; equal tiles → even 4/4 split.
- **A6 — Cost table units / keys.** `PJRT_OCL_COST_TABLE` JSON keys are exactly
  `ew_tile_us, mma_tile_us, gather_tile_us, reduce_tile_us`. REDUCE_PART and REDUCE_COMB both map
  to `reduce_tile_us`; IOTA_DIM reuses `ew_tile_us` (no dedicated key). Missing env, missing file,
  or unparseable JSON → all unit costs default to 1.0 (per spec). Absolute µs values don't matter
  for packing — only cost *ratios* — so 1.0-everywhere degrades to tile-count balancing.

### Control-flow scheduling (WHILE/IF) — structured SEAM, not implemented

Current lowering never emits region ops (splat consts are expanded, no control-flow lowering yet).
The scheduler is structured to recurse on region lists but the recursion body is a seam:
`schedule_program` raises `ScheduleError` (→ service exit 2) if a WHILE/IF instruction appears,
and `vmreader._split_phases` / the simulator reject WHILE/IF entries. Tests:
`test_scheduler_while_loop` (skip skeleton documenting the intended encoding — cond/body region
lists scheduled into per-lane sub-ranges after the main stream, WHILE entry `task=0xFFFFFFFD`
uniform in every lane, ranges are entry-index offsets within the lane's OWN stream) and
`test_scheduler_rejects_region_op_for_now` (asserts the clean rejection). The v1 tensor WHILE
still round-trips through the reader's **semantic** interpreter (tensor-only serialize,
`schedule=None`) — `test_vmreader_while_loop`.

### Two reference validators (both run on every e2e case; must agree)

- (a) `vmreader.execute`: numpy interpreter over the TENSOR sections (source of truth), updated for
  the 48B header + aux field.
- (b) `vmreader.execute_schedule`: a LANE SIMULATOR over the SCHEDULE sections. Runs barrier-phase
  by barrier-phase; within a phase it executes the entries in forward AND reverse order on a fresh
  arena copy and asserts identical results (proves lane order-independence within a phase). Also
  asserts: every tile of every task covered exactly once (contiguous, in-bounds), identical
  barrier counts across lanes, task/entry/range bounds. Final outputs must equal (a). This is the
  python mirror of the C++ VLIW engine and pins the schedule semantics the C++ side must match.

### Verified

`.venv/bin/python -m pytest tests/test_lowering.py -q` → **21 passed, 1 skipped** (PYTHONPATH set
to the worktree's python/, confirmed `pjrt_ocl.__file__` resolves into the worktree). Coverage:
v3 golden byte layout (tensor exact bytes + schedule header/task/barrier unpack), jax e2e via the
lower_service subprocess through BOTH validators, multi-op independence
(`lambda a,b,c,d:(a+b,c*d)` → both tasks in one barrier phase on 2 distinct lanes), dependent-chain
serialization (a*b-c → 2 phases), lane allocation (proportional / even-split), device-config &
cost-table parsing, region-op rejection.

### Follow-ups / open

- Fill the WHILE/IF scheduling seam (Phase 1.5 / M4): needs per-lane sub-stream layout + the
  entry-relative range encoding; the reader simulator needs a matching frame-stack executor.
- WAIT/SIGNAL per-op completion counters (tile-isa.md) are the refinement past v0's barrier-per-
  level; `n_flags` and the wait/signal entry fields are already carried through (as FLAG_NONE/0).
- Once real int dtypes / non-EW tile ops land, revisit the cost table's role (it becomes load-
  bearing) and the aux-pool wiring (GATHER/REDUCE/DOT/IOTA_DIM populate it — n_aux > 0 path is
  written and reader-validated but currently unexercised).

## ops/dot.py — dot_general (plain 2D matmul) — Phase 2 (2026-07-15)

New op-family module `pjrt_ocl/ops/dot.py` (+ `tests/test_ops_dot.py`, `from . import dot` in
`ops/__init__.py`). Targets the EXISTING `vm2.cl` `mma_tile` (plain row-major SGEMM), not the
future fast MMA kernel.

### Supported vs rejected (dot_dimension_numbers)

Only the canonical plain 2D matmul `C[M,N] = A[M,K] @ B[K,N]`:
- SUPPORTED: lhs rank 2, rhs rank 2, `lhs_contracting=[1]`, `rhs_contracting=[0]`,
  `lhs_batching=[]`, `rhs_batching=[]`. `a @ b` on 2D jax arrays lowers to exactly these.
- REJECTED with `LoweringError` (message names the numbers): any batching dims, non-canonical
  contracting axes (e.g. `[0]x[1]`, which would need an operand transpose — a separate GATHER
  family), or rank != 2. Verified: rank-3 `jnp.matmul` lowers to `batching_dims=[0]x[0],
  contracting_dims=[2]x[1]` → rejected; a rank-3 `tensordot` also rejected.

### M/N/K encoding decision (Instr fields)

The device `mma_tile` reads dims as LITERALS from the scheduled task (`M=t.p0, N=t.p1, K=t.p2`),
so `to_task(ins)` must yield literal M,N,K. But `to_task` gets only the `Instr` — no `rt`, so it
CANNOT dereference the aux pool; and `vmreader.parse` bounds-checks `Instr.aux <= n_aux`, so a raw
K cannot ride in `aux` either (would fail parse when n_aux < K). An `Instr` has just two free
scalar fields once dst/a/b carry the buffers: `n` and `imm`. Three dims → two fields → pack:

    Instr.n = M ;  Instr.imm = (N << 16) | K ;  Instr.aux = 0
    decode:  M = ins.n ;  N = ins.imm >> 16 ;  K = ins.imm & 0xFFFF

`N, K > 0xFFFF` raise LoweringError (never silently corrupt the packing); M is a full u32.
This deviates from docs/vmprogram.md's DOT row (aux=[M,N,K], n=M*N) — that form is unusable here
because to_task has no pool access, and the C++ engine never reads the DOT *tensor* Instr anyway
(it consumes the scheduled Task p0/p1/p2). The Instr encoding therefore only has to satisfy our
own numpy interp + scheduler, and it is the single source of truth for both. Chose NOT to touch
vmreader/scheduler/docs (scope: one family module only).

### Validators

- interp (validator a): `C = (A.reshape(M,K) @ B.reshape(K,N)).ravel()`.
- tile_sim (validator b, TILE_MMA): mirrors `mma_tile` tile indexing exactly —
  `tiles_n = ceil(N/16)`, `tr = tile//tiles_n`, `tc = tile%tiles_n`, fills
  `C[tr*16:.., tc*16:..]` (clipped to M,N) `= A_rows @ B_cols` for tiles in `[tile_lo, tile_hi)`.

### Verified

`.venv/bin/python -m pytest tests/test_ops_dot.py -q` → **17 passed**; full `tests/ -q` → **49
passed, 2 skipped** (the 2 skips are the plugin-not-built e2e cases, pre-existing). Ran with the
worktree's python/ on PYTHONPATH (main venv's jaxlib — the worktree's own `.venv` has a partial,
broken jaxlib copy; `pjrt_ocl.__file__` confirmed resolving into the worktree). Non-16-multiple
correctness explicitly covered: shapes (3,4,5),(17,17,17),(33,17,3),(17,33,64),(3,64,3), etc. —
ragged M, N, and K edges all match the CPU backend through BOTH validators (integer-valued f32,
K<=64, so products are exact). Also: matmul→add, matmul→scale (const pool), matmul-by-identity,
chained `(a@b)@c` (two MMA tasks across a barrier level).
