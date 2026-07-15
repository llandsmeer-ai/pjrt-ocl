# Roadmap — overnight execution plan (2026-07-15, user-approved)

User directive: implement the new execution model, then most-common ops test-driven, then
performance mode. Work autonomously. This file is the plan of record for continuation.

## STATUS 2026-07-15 (overnight)

- **Phase 1 ✅ COMPLETE**: VLIW per-lane-stream engine is the plugin executor (vm2.cl + runtime
  v3). jax.jit runs end-to-end on NVIDIA + PoCL; on-device while validated (runtime_test).
- **Phase 2 IN PROGRESS**: op-family registry infra landed (opsem: TO_TASK/INTERP/READS/
  TILE_SIM/EW_SIM; pjrt_ocl/ops/ package; tests/oputil.py check()). Reference family
  ops/shape.py (broadcast_in_dim, transpose via GATHER) verified both validators + hardware.
  4 families fanned out to agents: elementwise (div/max/min/pow/unary/cmp/select), making
  (iota/convert), reduce (full reductions), dot (2D matmul). Each = one ops/<f>.py + one
  tests/test_ops_<f>.py; only shared edit is a 1-line import in ops/__init__.py (merge-resolve).
- **Phase 3 (perf) pending**: adopt poc/06 fast MMA (26 TFLOPS) into vm2 MMA_TILE; calibration
  slope-fit; slot-file fusion; typed lanes (poc/05).
- Merged PoCs this session: poc/04 (VLIW), poc/05 (typed-lanes viable), poc/06 (fast MMA).

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
