# Roadmap — overnight execution plan (2026-07-15, user-approved)

User directive: implement the new execution model, then most-common ops test-driven, then
performance mode. Work autonomously. This file is the plan of record for continuation.

## STATUS 2026-07-15 (overnight)

- **Phase 1 ✅ COMPLETE**: VLIW per-lane-stream engine is the plugin executor (vm2.cl + runtime
  v3). jax.jit runs end-to-end on NVIDIA + PoCL; on-device while validated (runtime_test).
- **Phase 2 ✅ FIRST WAVE COMPLETE**: op-family registry infra (opsem: TO_TASK/INTERP/READS/
  TILE_SIM/EW_SIM; pjrt_ocl/ops/ package; tests/oputil.py + conftest.py forcing CPU oracle).
  5 families landed & merged: shape (broadcast/transpose), elementwise (div/max/min/pow/10
  unary/cmp/select), making (iota/convert), reduce (full sum/max/min/prod), dot (2D matmul).
  112 pytest pass through BOTH validators; dot/reduce/iota/fused-relu(matmul) verified on
  NVIDIA + PoCL hardware. Scoreboard: tests/SCOREBOARD.md. Next ops: reshape/slice/concat
  (GATHER variants), partial-axis reduce, if/case, batched dot.
- **Phase 3 (perf) STARTED**: ✅ fast MMA adopted into vm2 (64×64/4×4/BK16, ~17 TFLOPS
  pure-compute, 4× over naive; vregs 377→545 = accepted ceiling-1 tax). Findings recorded in
  tile-isa.md: (a) shared-megakernel register bloat → typed lanes is the real fix; (b) a lone
  big matmul is overhead/occupancy-bound end-to-end (~2 TFLOPS @4096) → wants the streamed-launch
  engine + buffer donation. REMAINING: engine routing (VLIW for heterogeneous segments,
  streamed-launch for single-big-op), buffer donation (cut H2D/D2H), calibration slope-fit,
  slot-file fusion, typed lanes integration (poc/05), then 128×128 MMA behind typed lanes.
- Merged PoCs this session: poc/04 (VLIW), poc/05 (typed-lanes viable), poc/06 (fast MMA).

## Phase 3 perf directives (user, 2026-07-15)

1. **Fix transfers — keep data ON DEVICE.** Investigate H2D/D2H per execute; PJRT input buffers
   should be device-resident cl_mem and stay on device across executes (device→device or direct
   VM read, not host round-trips). Buffer donation. This is the #1 next item.
2. **Per-op perf characterization** (every op): (a) does it actually parallelize? — measure
   scaling as lanes/execution-units increase; (b) compare to JAX's own CPU + GPU backends
   (python-level benchmark). Fix any perf bug found.

## Phase 1 — VLIW engine in the real plugin (main session, critical path)

1. **VMProgram v2.1 format**: v2 tensor sections (aux pool, docs/vmprogram.md) PLUS schedule
   sections: task descriptors, per-lane streams, n_flags. Python emits both; C++ consumes
   schedule sections; tensor sections remain source of truth / for reference interpreter.
2. **Device→lowering config**: C++ passes `PJRT_OCL_NLANES`, `PJRT_OCL_COST_TABLE` (JSON path)
   env vars to the lower_service subprocess. Calibration JSON produced by C++ on first client
   create, cached at `third_party/calib/<device-driver-hash>.json` (multi-K slope fit — poc/04
   NOTES: 1-tile calibration is contaminated by launch overhead).
3. **Python scheduler** (`pjrt_ocl/scheduler.py`): tensor instrs → dataflow levels → task
   descriptors (tile counts from shapes + device tile params) → LPT cost-based lane packing →
   per-lane streams with BARRIER entries at level joins (v0: barrier every level; WAIT/SIGNAL
   refinement later).
4. **C++ engine**: `kernels/vm2.cl` = poc/04 `vliw_async` + `exec_tiles` grown to the tile-op
   vocabulary (EW_TILE with subop covering v2 elementwise set incl. unary/cmp/select,
   GATHER_TILE, REDUCE_PARTIAL/COMBINE, MMA_TILE, FILL/IOTA). Executor: upload tasks+streams at
   load; one kernel launch per execute. Instrumentation under PJRT_OCL_VM_STATS=1 (barrier
   arrival ranks).
5. **Control flow on device**: WHILE/IF stream entries; per-lane frame stack over its OWN
   stream (poc/01 mechanics per lane); cond read atomically; barrier before cond eval.
   Exit criterion Phase 1: current 15-test suite green on VLIW engine, both platforms + a
   while-loop e2e test.

## Phase 2 — op coverage, test-driven (fan out; Sonnet agents for well-specified handlers)

- Registry split of lowering into per-family modules FIRST (so agents don't collide):
  ops_elementwise.py, ops_shape.py (broadcast/transpose/slice/reverse/reshape via GATHER),
  ops_reduce.py, ops_dot.py, ops_control.py. vmreader numpy interpreter grows in ONE file (me).
- Coverage targets (jax-driven order): iota, broadcast_in_dim, convert (int-as-f32 policy),
  compare+select, reduce (sum/max/min/prod), dot_general (2D + batched reject), transpose,
  reshape, slice, concatenate (as N gathers), exp/log/tanh/sqrt/neg/abs/floor/ceil/sign/pow,
  while, if. Scoreboard test: tests/test_coverage.py, one jax function per op, vs CPU backend
  (integer-valued f32 for jit-exactness — FMA policy in decisions 5b).
- Each agent: one family module + its tests; verify with pytest; NOTES for the decision log.

## Phase 3 — performance mode (after basics green)

- MMA_TILE rewrite: 128×128 tiles, 8×8 register blocks/thread, float4 loads, double buffering.
  Track with poc/04 BENCH_MMA vs clpeak 105.9 TFLOPS; target ≥30 TFLOPS first, then 40–70.
- Calibration slope-fit + bubble% reporting wired into plugin; profile-guided repacking later.
- poc/05 (parallel, delegated): cross-kernel co-residency of spinning groups (typed-lanes
  enabler — ceiling 1 in docs/tile-isa.md).
- EW throughput: vectorized loads; GATHER coalescing.

## Known facts to not re-derive

- clpeak NVIDIA FP32 peak 105.9 TFLOPS; current MMA 4.3 TFLOPS (376 lanes).
- Validated: per-lane streams + barriers + WAIT/SIGNAL (poc/04 E), inter-group barrier ≤564
  groups, atomic cond reads (poc/01), PJRT surface incl. CHECK-crash contracts (decisions #4).
- jax 0.10.2 pin, PJRT C API 0.112, VHLO artifact ingest, subprocess lowering interface
  (exit 0/2/3, options python_exe/lower_service).
- Env: `. ./env.sh` ALWAYS (root overlay full); 32 GB RAM; PoCL needs POCL_CACHE_DIR.
