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

## SCOPE CORRECTION (user, 2026-07-15): full dtype coverage is a goal, NOT f32-only

f32-only was the M1/M2 *starting* milestone, wrongly carried forward as a permanent
constraint. Target = full JAX coverage for complex workloads ⇒ multiple dtypes. The whole
stack is currently f32-hardcoded (VM `__global float* arena`, loader patches f32-element
offsets, tile-ops compute in `float`, `tensor_info` rejects non-f32). Making it dtype-aware is
the new keystone workstream.

**Architecture change**: buffers already carry a `dtype` field. Extend the enum; make the arena
byte-addressed (patch BYTE offsets, not ÷4); each tile-op reinterprets via typed pointers and
dispatches on a per-task dtype. Alignment is fine (arena buffers are 64B-aligned).

**Dtype tiers by cost (OpenCL reality):**
- **Tier 1 — native OpenCL, needed by *default* JAX** (32-bit, x64 off): `i32`, `bool`/pred,
  `u32`. Unlock indexing/gather/scatter, real boolean masks, integer counters, argmax/argmin.
  Highest value, tractable. Do first.
- **Tier 2 — native but device-conditional / x64**: `i64`, `f64` (needs `cl_khr_fp64`; ABSENT
  on Intel Xe2 consumer & others — feature-detect, error clearly if used on a device without it),
  `f16` (needs `cl_khr_fp16`). jax x64 mode (`jax_enable_x64`) needs i64/f64.
- **Tier 3 — no native OpenCL type, needs emulation**: `bf16` (store u16, up-convert to f32 for
  math), `complex64/128` (pairs of f32/f64 — every op splits into real/imag). Significant.

### TODO — refresh README.md once coverage work settles (user, 2026-07-15)

README currently says "dtypes in progress / f32 works today". After the dtype system landed
(i32/u32/i64/bool/f64/f16/bf16 + convert/compare/select) and the in-flight op batch merges,
update the README: the supported-ops table, the dtype line (now a full 1/2/4/8-byte matrix, f16/
bf16 portable via f32-compute), and the limitations (drop f32-only framing; keep the PoCL-barrier
+ tensor-core-ceiling caveats). Do this after the op-coverage agents are merged and the corpus is
re-measured — not before, so the numbers/tables are accurate.

### TODO — native f16 via cl_khr_fp16 on supported platforms (user, 2026-07-15)

f16/bf16 currently do 2-byte storage + **f32 compute** (portable, no extension: vload_half/
vstore_half, bf16 bit-shift). This is correct everywhere but leaves performance on the table on
devices with native half arithmetic. TODO: when `cl_khr_fp16` is present (feature-detect at init,
like fp64), enable `#pragma OPENCL EXTENSION cl_khr_fp16` and a native-`half` compute path for
f16 (and vectorized half ops), selected per-device behind the same kernel-table/override
mechanism. Keep the portable f32-compute path as the fallback for devices without it (incl.
NVIDIA OpenCL, which exposes fp16 storage but not always compute). bf16 stays emulated (no
hardware bf16 in OpenCL). This is a perf optimization, not a correctness gap — sequence it after
op coverage.

**Plan**: (1) dtype-aware format + loader + VM foundation; (2) Tier 1 (i32, bool) end-to-end
through both validators + jax tests; (3) Tier 2 behind feature-detection; (4) Tier 3 emulation.
Mixed-dtype ops (convert, compare→bool, select(bool pred), bitcast) come with Tier 1.

**Design decisions (user, 2026-07-15):**
- ✅ **Modular kernel files** (DONE): the VM is split into kernels/vm_common.cl + ops/*.cl +
  vm_main.cl, concatenated into one program at build (CMakeLists VM_CL_SOURCES). One op family
  = one .cl file with a static tile function; parallel-safe like python/pjrt_ocl/ops/. No
  clLinkProgram (functions inline). This is where per-dtype/per-op work now lands.
- **f64 IS in scope** ⇒ the arena must be **byte-addressed** (`__global uchar*`, byte offsets,
  typed-pointer casts) — the 4-byte-slot shortcut can't hold 8-byte types. So do the
  byte-addressed refactor FIRST (behavior-preserving for f32), then dtypes are additive.
  f64 (and f16) **gated behind cl_khr_fp64 / cl_khr_fp16** feature-detection at init (both our
  dev devices have fp64+int64+byte_addressable_store; Intel Xe2 consumer lacks fp64 → clean
  error, not a crash). Enable the extension pragma conditionally in vm_common.cl.
- **Bit-recast via `union { float f; int i; uint u; }`** (slot_t, already in vm_common.cl) — not
  as_int/as_float — for bitcast_convert and NaN-safe integer handling.
- Loader: patch **byte** offsets (currently f32-element ÷4). Task carries a **dtype** (pack into
  tile_op high bits or a task field). exec_tiles dispatches per (tile_op, dtype).

## Phase 3 perf directives (user, 2026-07-15)

1. ✅ **Transfers fixed — data stays ON DEVICE.** PJRT_Buffer holds a device cl_mem;
   BufferFromHostBuffer uploads once, Execute copies inputs device→device into the arena and
   leaves outputs on device, ToHostBuffer is the only (lazy) D2H. Sequential/chained jit calls
   no longer round-trip host. (ef11206)
2. ✅ **Per-op perf characterized** (docs/perf-findings.md, tools/bench_ops.*): all ops
   parallelize with lanes (monotone speedup to saturation); beat JAX CPU 1.75×–19×; 43µs
   per-execute floor. Bug found: memory-bound ops at 13–26% HBM peak due to arena copies.
3. **NEXT perf item — zero-copy buffer binding**: pass input/output cl_mems as VM kernel args;
   loader marks buffer ids as input-slot/output-slot/arena so the VM reads inputs & writes
   outputs directly (no arena device→device copy). ~2× on memory-bound ops. Architectural
   (vm2 ABI + loader); deferred to avoid destabilizing the verified engine.
4. **NEXT reliability item — PoCL barrier (decisions #1)**: Plan-B host-dispatch or typed-lane
   kernel split so PoCL (and the shared-megakernel local-mem tax) stops deadlocking under
   iteration. NVIDIA unaffected.

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
