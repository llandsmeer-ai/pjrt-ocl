# Decision log (tree)

Institutional memory of the project. Every design exploration gets a node: **TRIED** (with the
actual error/measurement), **FAILED/OK**, **CHOSEN** and why. Update in the same session as the
exploration. Nested bullets = sub-decisions opened by a parent choice.

Legend: ✅ chosen · ❌ tried & rejected (keep the evidence!) · 🔬 open, needs PoC · 🅱️ fallback kept viable

## 1. Execution model

- ✅ **MEMORY-VISIBILITY HALF OF #1 RISK — SOLVED 2026-07-15 (poc/07): device-scope
  acquire/release fences.** The barrier bug has TWO independent axes; poc/07 separates them and
  fixes the visibility one. **Axis 1 (memory visibility):** the old `vmo_barrier` used
  `mem_fence(CLK_GLOBAL_MEM_FENCE)` — work-group-scoped, so a producer lane's *non-atomic* data
  writes were NOT published to a consumer lane. The atomic phase flag is L2-coherent (lanes agree
  on *when*), but the data sat in the producer SM's L1. Measured on NVIDIA Blackwell: a persistent
  loop reading a neighbour's cell across the barrier is **~100% stale** (1599968/200000; test B).
  Single-shot two-level programs escaped it by cold-L1 accident; iteration (while) keeps L1 warm →
  the while agent's forced `n_lanes=1`. **Fix (test E): OpenCL-2.0 device-scope acquire/release
  fences** around the atomic handshake (`atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE,
  memory_order_release/acquire, memory_scope_device)`) → **0 stale**, plain reads coherent, in
  spec, megakernel intact. **`clinfo` under-reports:** NVIDIA advertises only work-group atomic
  scope yet the compiler accepts device scope AND the hardware honours it — so capability must be
  RUNTIME-PROBED, never trusted from `CL_DEVICE_ATOMIC_FENCE_CAPABILITIES`. Native on PoCL/AMD/
  Intel. Rejected alternatives kept for the record: `volatile` cross-lane reads (test C) also work
  on NVIDIA via L1-bypass but kill L1 reuse; no `cl_khr_*` extension provides a grid barrier.
  Applied to `pjrt_plugin/kernels/vm_common.cl`; runtime_test A+B still PASS on NVIDIA. **DONE:
  multi-lane while shipped** — with the gate lifted, the while e2e on the OLD barrier returned 23
  (want 17: the real cross-lane race); on the device-scope barrier it returns 17, 20/20 e2e runs
  deterministic on NVIDIA. **Axis 2 (liveness) — ROOT-CAUSED below; device-scope fences do NOT
  help it.**

- ✅ **LIVENESS ROOT CAUSE NAILED 2026-07-15 (poc/07 part 2): imbalance + spin + PoCL
  non-preemption.** Earlier this was hand-waved as "PoCL doesn't co-schedule workgroups." poc/07's
  `barrier_starvation.c` bisects it exactly. The spin-barrier PRIMITIVE is fine on PoCL — poc/07
  runs 200k barrier iterations at G=8..32 (even G>CUs) with zero hangs. The megakernel deadlocks
  only when lanes are **imbalanced**: a variant that adds, one at a time, vm2's unstructured
  `for(;;)`+break loop, its 8 KB `__local`, and its private frame/register footprint STILL runs
  clean — until only lane 0 does pre-barrier work, then it **DEADLOCKS at every G (incl. G=4 with
  24 threads free)**. Mechanism: PoCL's worker pool is NON-PREEMPTIVE; a workgroup that reaches the
  barrier first spins on the arrival counter holding its thread and never yields, starving the slow
  workgroup that still owes an arrival. Real schedules are imbalanced by construction (tile on one
  lane, EW on others, idle lanes) → guaranteed starvation. **Co-residency is NOT the lever** (fails
  at G=4≪24 threads); balance is, and the scheduler can't guarantee it. OpenCL C has **no yield
  primitive**, so an in-kernel spin-barrier CANNOT be made imbalance-robust on a non-preemptive CPU
  runtime — this is why nobody ships CPU cross-workgroup sync as a spin. **The correct CPU barrier
  is the KERNEL BOUNDARY** (poc/07 test D: host relaunch per phase, 46 µs/phase on PoCL, immune
  because a finished workgroup EXITS and frees its thread). Decision: **host-dispatch engine for CPU
  (non-GPU) devices; GPUs keep the device-scope megakernel.**
  - **Literature confirms (2026-07-15 web check).** Canonical portable inter-workgroup barrier
    (Sorensen/Donaldson, OOPSLA 2016): "we must ensure that all workgroups are resident on the
    device at the same time. We size our launches accordingly to guarantee full occupancy" — an
    Occupancy-Bound Execution assumption non-preemptive CPU schedulers violate ("if a single
    work-group is blocked by the OBE model, the barrier deadlocks due to starvation"). OpenCL model
    itself: "OpenCL does not support synchronization across work-groups inside a kernel; instead
    multiple kernels must be launched" — the kernel boundary IS the sanctioned cross-group barrier.
    PoCL's own CPU pipeline uses Continuation-Based Synchronization, which "defines kernel entries
    and exits as barriers" — literally the host-dispatch model. Only in-kernel escape is
    *cooperative kernels* (need a bespoke scheduler that context-switches a waiting group; not
    available to portable OpenCL C or PoCL). Not a PoCL bug, not out-engineerable in-kernel — it is
    the established state of the art.
  - ✅ **SHIPPED 2026-07-15: host-dispatch engine (`vm2_seg` + `LaunchHostDispatch`).** The runtime
    now carries TWO engines behind one bytecode. GPU (is_gpu) keeps the persistent device-scope
    megakernel; non-GPU (CPU) defaults to host-dispatch — `PJRT_OCL_ENGINE=host|mega|auto` overrides.
    Host-dispatch mirrors vm2's per-lane frame walk ON THE HOST and launches the barrier-free
    `vm2_seg` kernel once per phase, using `clFinish` as the barrier (workgroups run their tile
    entries and EXIT — no co-residency, immune to the starvation deadlock). Key invariant that keeps
    it simple: the scheduler puts a barrier at every level boundary and gives WHILE its own level, so
    each inter-barrier segment is a CONTIGUOUS entry range per lane ({off,count}); while-cond scalars
    are read host-side between phases. Verified: runtime_test A+B pass 5/5 on PoCL (was 100% deadlock);
    `(a+b)*a` 300 iters, all 7 while programs, matmul/reduce/broadcast all correct on PoCL via the
    plugin; NVIDIA megakernel path unchanged; forced host-dispatch on NVIDIA also passes; PoCL+mega
    still deadlocks (flag works). 197 pytest + 3 e2e pass. **PoCL is deadlock-free for the first time.**
  - Supersedes the vaguer fix-options in
  the node below.

- ✅ **SHIPPED 2026-07-15: execution-trace instrumentation + timeline plots**
  (`PJRT_OCL_VM_TRACE=<file>` + `tools/plot_schedule.py`) — delivers the spec-level
  instrumentation item (bubble % now visible per lane, plotted planned-vs-measured).
  Design: OpenCL gives per-COMMAND timestamps only (no portable in-kernel clock), so
  per-entry timing requires one launch per entry → trace mode forces the host-dispatch
  engine and runs every schedule entry as its own single-workgroup `vm2_one` launch on a
  per-lane `CL_QUEUE_PROFILING_ENABLE` queue; `clFinish` over the lane queues is the
  phase barrier; one JSON line (task table + per-entry device-clock start/end) appended
  per Execute. **Pre-verified assumption: lanes stay concurrent across queues** — 8
  spin kernels on 8 queues take 1.06× one kernel on PoCL (events on a common timebase),
  and NVIDIA maps queues to streams; without that the traced timeline would be fiction.
  Caveats (recorded in README + tool docstring): (a) per-entry launches add ~tens of µs
  each — it's a timeline, not a benchmark; (b) the GPU megakernel is NOT per-entry
  observable from the host (only the existing barrier arrival-rank stats), so traces
  always measure the host-dispatch engine. Findings from the `diamond` example
  (matmul ∥ EW chain, then join): PoCL runs level 0 with lanes 5–7 (EW) 97–98% idle —
  an MMA tile costs ~25× an EW tile there vs the unit-cost default (~50% of lane-time
  idle overall); NVIDIA's level 0 is nearly flat (ratio ≈ 1). Same schedule, opposite
  balance — reconfirms measure-don't-assume; the cost-table (`PJRT_OCL_COST_TABLE`)
  is the rebalancing lever. Verified: runtime_test PoCL+NVIDIA PASS, 197 pytest pass,
  traced diamond output matches numpy (max |err| 4.8e-7 — f32 matmul accumulation).
- ✅ **SHIPPED 2026-07-16: measured per-device cost model + sequentializing lane packer**
  — closes the "cost model is MEASURED, not assumed" spec item (was designed 2026-07-14,
  validated in poc/04, then never wired into the tree: DeviceConfig defaulted every
  tile-op to 1.0 and nothing generated PJRT_OCL_COST_TABLE). Trigger: the trace-mode
  diamond plot — unit costs made the scheduler give the matmul 5–6 lanes and dedicate
  lanes to two cheap EW ops, which then sat 97–99% idle at the barrier (user diagnosis:
  they should have been sequentialized onto shared lanes).
  - **Calibration (runtime.cc `CalibrateCosts`, runs at client init):** per tile-op
    family (ew/mma/reduce/gather), execute a hand-built single-lane program (fill
    inputs → barrier → K op tiles) at K and 2K tiles; **µs/tile = slope** — fills and
    launch overhead cancel (poc/04's contamination lesson). Cached as JSON at
    `${XDG_CACHE_HOME:-~/.cache}/pjrt-ocl/costs-<fnv(platform|device|driver)>.json`;
    `PJRT_OCL_CALIBRATE=0|1` disables/forces; a user `PJRT_OCL_COST_TABLE` supersedes;
    plugin.cc forwards the resolved path to the lowering subprocess, so every compile
    is cost-aware with zero user action. Trace mode is suppressed during calibration
    (per-entry launches would distort the measurement). Measured: PoCL ew=310
    mma=5073 reduce=89 gather=201 µs/tile (MMA:EW ≈ 16×); NVIDIA ew=15 mma=27
    reduce=13 gather=21 (≈1.8×) — reconfirms poc/04's "same graph, different balance"
    at the ratio level. (The trace-mode ~25× estimate was under 8-lane contention;
    calibration is single-lane. Ratios, not absolutes, drive packing.)
  - **Packer (scheduler.py `_pack_level`): chunk + LPT, one regime.** Each task splits
    into k = min(tiles, ceil(n_lanes·cost_share)) contiguous chunks; all chunks LPT
    onto least-loaded lanes; a lane may carry MULTIPLE entries per level. Replaces the
    old primary/overflow pair, whose ≥1-dedicated-lane-per-task invariant made
    sequentialization impossible. Diamond with measured costs: matmul chunks on ALL 8
    lanes, add/mul stacked behind lanes 0–1; model makespan 75 → 56 cost units.
  - **Validated:** 199 pytest (incl. new sequentialization test + rewritten
    proportional-allocation test; simulators already supported multi-entry lanes),
    runtime_test PASS PoCL+NVIDIA, calibrated e2e correct on the NVIDIA megakernel.
    Traced diamond on PoCL: planned and measured panels now structurally agree; idle
    lane-time 42–50% → ~20% (rest is PoCL per-workgroup jitter, not scheduling).
    **Wall-clock:** diamond unchanged within PoCL noise (model gain 1.17× ≪ jitter);
    the lane-stealing shape (1 matmul + 7 cheap EW) improves 14.2 → 10.7 ms median
    (**1.33×**) with calibration on. NVIDIA at these sizes is launch-bound (no change,
    no regression). plot_schedule.py planned panel now reads the cost table the plugin
    actually used (path recorded in each trace line).

- ⚠️ **CONFIRMED #1 RISK 2026-07-15: cross-workgroup spin-barrier is UNRELIABLE on PoCL under
  iteration (LIVENESS axis — still open; poc/07 fixed only the visibility axis above).** The persistent-VLIW engine (vm2.cl) uses the poc/01 global barrier between schedule
  phases. On NVIDIA it is rock-solid (500 two-level + 300 chained-matmul back-to-back runs, zero
  hangs). On PoCL (CPU) it deadlocks NONDETERMINISTICALLY within ~30–50 iterations of ANY
  multi-level program (even `(a+b)*a`), at every lane count tried (24 down to 4). Root cause:
  PoCL maps workgroups onto a CPU thread pool and does not guarantee all N workgroups make
  concurrent progress, so a spin-barrier where WG-i waits on not-yet-scheduled WG-j hangs.
  AGGRAVATED tonight by the register-blocked MMA raising the shared megakernel's `__local` from
  2 KB→8 KB (declared for ALL programs), cutting PoCL co-residency headroom — the ceiling-1
  shared-resource tax, now a correctness problem on CPU. Single-shot executes still pass, so PoCL
  remains valid for CORRECTNESS spot-checks; numpy validators (vmreader) don't touch the barrier
  so pytest is unaffected. PERF/stress testing must use NVIDIA. **Fix options (next major item):**
  (a) Plan B — host-side kernel-launch-per-phase loop (host enforces the barrier at launch
  boundaries; the bytecode is engine-agnostic by design); (b) per-family / typed-lane kernel
  split so non-MMA programs carry small local footprint and stay co-resident; (c) PoCL-specific
  engine = host dispatch. NVIDIA/real-GPU path is unaffected and is the current perf target.

- ✅ **PIVOT 2026-07-14 (user-driven, M3): host-dispatch is the primary engine.** Each data
  instruction = one `clEnqueueNDRangeKernel` of a dedicated per-op kernel at full problem size
  (one WI per element; per-op local sizes), in-order queue, NO global barrier on the hot path.
  Control flow: host reads the cond scalar (~10µs) and selects the next range. Evidence that
  killed megakernel-as-primary:
  - poc/01's benchmark was accidentally RIGGED: the "separate launches" baseline used the VM's
    tiny persistent grid. Honest baseline (full-size launch): launches ≈ megakernel at 1M elems
    (1.04x), and both 3x faster at local=256 than the starved 188×64 config.
  - Co-residency cliff: 1128×256 persistent groups DEADLOCKED (spin > residency). The barrier
    needs an exact occupancy oracle per device/driver/kernel — fragility on every vendor.
  - Megakernel switch couples register pressure across ops and blocks cooperative kernels
    (tiled matmul, tree reduce).
  - Bytecode is engine-agnostic (deliberate) — this pivot changes ZERO format bytes.
  - 🅱️ Megakernel demoted to optional segment engine for long chains of tiny ops / tight scalar
    loops (validated, kept in tree; routing marked segments through it is a later optimization).
  - Note: the user's ORIGINAL brief said "a series of kernel dispatches, from a simple bytecode"
    — the megakernel detour is preserved below for the record.
- ✅ **SUPERSEDING DESIGN 2026-07-14 (user-proposed, agreed): tick-synchronous VLIW-style VM**
  — see docs/tile-isa.md, the spec of record. GPU as spatially-partitioned VLIW machine:
  persistent lane-interpreters (≈1–4/CU), schedule table (ticks × lanes) assigns DIFFERENT ops
  to DIFFERENT lanes in the same tick; tick boundary = validated barrier; control flow =
  uniform tick-range jumps (atomic cond reads). Independent ops run spatially parallel entirely
  on device — no host in the loop. Prior art: Mirage Persistent Kernel / Hazy megakernels.
  - ✅ **Three-layer split** (user + analysis): tensor bytecode stays the portable ISA;
    per-device schedule compile derives task descriptors (one per op, tiles NEVER materialized
    as instructions — cells reference tile RANGES) + schedule table. Tile-op vocabulary is the
    execution opset (EW_TILE/MMA_TILE/REDUCE_PARTIAL+COMBINE/GATHER_TILE/FUSED_TILE-reserved).
    Rationale: uniform tile costs make tick packing tractable; tile residency unlocks fusion;
    bytecode stays compact/device-neutral/symbolic-friendly (L1 dynamic shapes recompiles only
    the µs schedule layer).
  - ✅ **Cost model is MEASURED, not assumed** (user requirement): first-run µbenchmark per
    device → per-tile-op costs cached (device+driver key); LPT packing uses them.
  - ✅ **Instrumentation is a spec-level feature** (user requirement): logical-clock ranks
    (portable) + kernel-clock/event-profiling modes; bubble % reported under
    PJRT_OCL_VM_STATS=1; goal = prove execution units stay occupied, enable profile-guided
    re-packing.
  - Streamed-launch engine (above) remains the second engine behind the same bytecode;
    honest-benchmark referee decides per segment class. → `poc/04-vliw-vm`
  - ✅ **MODEL CORRECTION (user, 2026-07-14): per-lane bytecode streams, NOT global tick
    lockstep.** Each lane owns a linear instruction stream; sync is point-to-point via per-op
    completion counters (WAIT/SIGNAL entries, atomic polls). Lockstep ticks = degenerate case
    (WAIT-ALL each entry), kept only as debug/profiling mode. Rationale: imbalance absorbs
    program-wide instead of costing max-lane per tick; consumers pipeline behind producers;
    equals Mirage MPK's event model. Spec updated (docs/tile-isa.md).
    - ✅ Refinement (user): global sync EXISTS but is SCHEDULER-PLACED (BARRIER entries at
      dataflow joins); cost model shapes per-lane work so arrivals coincide → bubbles mostly
      absent by construction. Barrier arrival-rank instrumentation names the lane class to
      unload (validated in test E: NVIDIA → mma last; PoCL → ew last. Same graph!).
    - ✅ **Test E VALIDATED 2026-07-14 both platforms**: 4 lanes cooperating on 256³ matmul
      while 184 (resp. 20) lanes run 8 EW ops as many small entries; one global barrier;
      consumer phase. Correct + 0.58 ms NVIDIA / 13.3 ms PoCL.
  - ✅ **poc/04 (lockstep variant) VALIDATED 2026-07-14** (NVIDIA + PoCL): spatial co-scheduling in one tick,
    local-memory MMA tiles, cross-tick reduce — all correct; cost-aware packing 1.65x over
    naive on NVIDIA. ⚠️ Same policy LOST (0.81x) on PoCL because naive 1-tile/lane calibration
    is contaminated by ~20–30µs launch/barrier overhead (also why bubble% read >100). Fix
    specced: multi-K slope fit per tile-op per device. EW:MMA cost ratio 0.9 (NVIDIA) vs 5.6
    (PoCL) — user's measure-don't-assume requirement empirically confirmed.

### 1-old. Megakernel era (historical, still true for the segment engine)

- ✅ **Device-side megakernel VM** (persistent kernel, opcode switch) — user decision 2026-07-14.
  Motivation: minimal dispatch overhead.
  - ✅ **Strictly linear bytecode, no jumps/conditionals** — user decision 2026-07-14. StableHLO has
    no jump ops (verified against spec, see docs/stablehlo-notes.md); region ops (`while`/`if`/
    `case`/`reduce`/...) lower to one instruction referencing nested linear instruction lists,
    interpreted by the VM.
    - ❌ pc-manipulation/jumps in bytecode — rejected: user prefers stupid-linear execution;
      nothing in StableHLO needs it.
    - ✅ **Nested-list control flow VALIDATED 2026-07-14** (`poc/01` test4): `OP_WHILE` with
      cond/body sub-list refs + explicit frame stack in the interpreter; 2-deep nesting passes on
      PoCL + NVIDIA.
      - ⚠️ **Lesson**: the cond scalar MUST be read with `atomic_add(p,0)` — a plain load hit
        stale per-SM cache on NVIDIA, diverged the workgroups' loop decision, and deadlocked the
        barrier (PoCL was fine). Rule: uniform-control-flow values are always read atomically.
    - ✅ **stablehlo.while END-TO-END through the real plugin — 2026-07-15 (M4).** Lowering
      (`lowering._lower_while`) + scheduler (`scheduler._Scheduler` per-lane WHILE control entries
      with cond/body sub-streams placed after `root_len`) + a WHILE-aware python lane simulator
      (`vmreader._run_control`, mirrors vm2.cl's frame stack). Loop-carry model: N mutable carry
      buffers, init-copied from the operands; body computes fresh values then snapshot→commit
      copies them back (two levels with a barrier between ⇒ carry writes strictly after all body
      reads, so swap/passthrough bodies are safe despite carries not being SSA). Verified on
      NVIDIA: scalar mixed i32/f32 carry, fori_loop, multi-tile vector carry, multi-level body,
      zero-iteration, nested while, while-then-op — all bit-exact vs jax CPU, 40/40 deterministic.
      - ⚠️ **CROSS-LANE DATA RACE under iteration (extends the 1.2-atomics gap, lines below).**
        The barrier reliably publishes the *atomic* cond-flag read across workgroups, but a
        loop-carried DATA buffer written by lane A and read by lane B in a later phase races
        UNDER ITERATION on NVIDIA (regular global loads hit stale L1; the barrier's global-mem
        fence is workgroup-scoped). Measured: a fori whose scheduler split a scalar carry's
        copy-chain across 2 lanes gave 17/20/23/29 nondeterministically at ≥2 lanes, but 30/30
        correct at 1 lane; a same-lane-only while (manual op order) and a single-shot reduce
        (cross-lane but not iterated) are both 100%. Root cause is the kernel barrier's memory
        model, NOT the lowering/scheduler (both python validators, 1-lane device, and same-lane
        multi-lane device all pass). **Mitigation (M4, correctness-first):**
        `schedule_program` forces **n_lanes = 1** for any while-containing program, so every
        carry's producer/consumer share a lane (no cross-lane data movement). The multi-lane
        WHILE scheduler path stays exercised by the python simulator
        (`allow_multilane_while=True`), where cross-lane is exact. **Follow-up (M5):** harden the
        barrier (OpenCL 2.0 device-scope acquire/release, or an L1-bypassing cross-lane load) then
        re-enable multi-lane loop bodies — until then a loop with a heavy body is single-workgroup
        and slow (correct). Same root cause as the feature-detect item two bullets down.
  - ✅ **Cross-workgroup barrier — VALIDATED 2026-07-14** (`poc/01-device-vm`): Xiao&Feng-style
    arrival counter + phase flag with OpenCL **1.2** atomics passes correctness + 2000-instr
    dependency stress on both PoCL (24 grp) and NVIDIA (188 grp, ~1.1 µs/barrier). Megakernel vs
    separate launches on NVIDIA: **2.5x faster @1M-elem ops, 3.2x @4K** — the design pays off.
    Rules: never launch more groups than co-resident capacity (= CUs for now; PoCL would deadlock
    otherwise); 1.2-relaxed-atomics barrier is technically outside the 1.2 memory model —
    follow-up: feature-detect OpenCL 2.0 `atomic_load_explicit(memory_scope_device)` path.
  - 🔬 **Opcode dispatch** — no function pointers in OpenCL C → single big switch (works fine in
    poc/01); risk: compile time/register pressure as op library grows. Mitigation candidate:
    split VM by op family.
  - 🅱️ **Host-side dispatch loop** over the same bytecode (one clEnqueueNDRangeKernel per instr).
    Keep the bytecode dual-interpretable so this fallback stays cheap to activate.

## 2. StableHLO ingestion

- ❌ **Link MLIR + StableHLO C++ libs, built via CMake** — was the plan (user decision 2026-07-14),
  dropped 2026-07-14. Original trigger was a disk scare that turned out WRONG (I measured the
  root overlay, ~3 GB free; `/home/ubuntu/project` is a separate mount with ~445 GB — user
  corrected this). The pivot stands anyway on merits: python lowering is version-matched to JAX,
  no LLVM rebuild per JAX upgrade, hackable compile logic. C++ MLIR build (in `third_party/`
  inside the project mount) is a VIABLE fallback again, e.g. for a future C++ `vm` dialect.
  Prebuilt escape hatches checked and still dead:
  - ❌ Link jaxlib's bundled MLIR: `libjax_common.so` (334 MB, contains all of MLIR+StableHLO)
    exports only 27 dynamic symbols — Python module init wrappers; the MLIR C API is hidden.
    Verified with `nm -D` 2026-07-14. Not linkable.
  - ❌ LLVM release-tarball prebuilts + stablehlo source: stablehlo pins non-release LLVM commits;
    extracted tarballs alone (~10 GB) don't fit either.
- ✅ **Python-side lowering, out-of-process** (previously rejected, revived by the disk evidence —
  and it's arguably better): lowering is compile-time-only, so the C++ plugin spawns the venv
  Python (`sys.executable` passed via `register_plugin(..., options=...)` →
  `PJRT_Client_Create` create_options) as a subprocess during `PJRT_Client_Compile`, pipes the
  serialized VHLO artifact in, receives flat VMProgram bytecode out. Uses jaxlib's own StableHLO
  Python bindings ⇒ **version-matched to JAX by construction**, zero heavy C++ deps, lowering is
  plain debuggable Python. C++ side stays a pure executor. → `poc/03-python-lowering`
  - ❌ In-process CPython callback instead of subprocess — rejected: GIL re-entrancy from inside a
    PJRT C call is a hazard; subprocess is ~100s of ms per compile, acceptable.
  - 🔬 Custom MLIR `vm` dialect (from original plan) deferred; VMProgram is a plain binary format
    emitted by Python for now.
  - ❌ Hand-written textual-MLIR parser — fragile across JAX/MLIR versions, can't read
    bytecode/VHLO artifacts.
  - ✅ **VALIDATED 2026-07-14** (`poc/03-python-lowering`): full chain serialize → subprocess
    `lower_service.py` → VMProgram → numpy reference interpreter == `jax.jit` exactly (atol=0).
    Subprocess cost 0.14 s. Headline facts (detail: `poc/03-python-lowering/research.md`):
    - `PJRT_Client_Compile` receives `PJRT_Program{format:"mlir"}` whose code is a **VHLO
      portable artifact** (MLIR bytecode, producer `StableHLO_vX.Y.Z`); `compile_options` is a
      serialized `xla.CompileOptionsProto`. jax python passes the live module; jaxlib C++ does
      the serialization (`xla::Serialize` → `serializePortableArtifact`).
    - Version negotiation: plugin should advertise `stablehlo_current_version` (int64[3]) in
      `PJRT_Plugin_Attributes`; client targets min(plugin, client). Without it: 12-week window
      (1.13.7 on this jaxlib; current 1.17.0). `deserialize_portable_artifact` auto-upgrades.
    - ⚠️ `serialize_portable_artifact` MUTATES its input module to VHLO in place — clone first
      (bytecode roundtrip) or you corrupt jax's cached lowering in same-process tooling.
    - Artifact bytes embed python-traceback locations ⇒ not stable across call sites; any
      compile cache must key on semantics, not bytes.

## 3. Kernel strategy

- ✅ **Generic shape-agnostic kernel library** (strides/shapes as runtime args), compiled once per
  device at init, program binaries cached on disk. Start with a tiny op set, expand only when e2e
  works — user decision 2026-07-14.
  - 🔬 Kernel-table override mechanism for tuned per-vendor variants (M5), incl. specialized matmul.

## 3b. OpenCL C dialect for vm.cl (2026-07-15, first external-machine bug report)

- **Bug**: `clBuildProgram(prog, dev, "")` compiles **OpenCL C 1.2** (spec default), where
  `vmo_barrier`'s `atomic_work_item_fence` / `memory_order_*` / `memory_scope_device`
  (OpenCL C 2.0+) are *undeclared identifiers*. Strict compilers (Intel, user's laptop) reject
  vm.cl at plugin init; it only ever built here because PoCL and NVIDIA **non-conformantly expose
  the 2.0 atomics in their default dialect** (verified: forcing `-cl-std=CL1.2` on PoCL reproduces
  the exact 6-error report). No user-side workaround existed — the build ran before engine
  selection, so even host-dispatch CPU devices (which never execute the fences) died.
- Facts that shaped the fix (all measured on this machine, 2026-07-15):
  - On OpenCL 3.0 drivers `CL_DEVICE_OPENCL_C_VERSION` is **capped at "OpenCL C 1.2" by spec**;
    the real list is `CL_DEVICE_OPENCL_C_ALL_VERSIONS` (PoCL + NVIDIA report 3.0 only there).
  - ❌ In-source feature-macro guard (`__opencl_c_atomic_order_acq_rel` &&
    `__opencl_c_atomic_scope_device`): NVIDIA accepts the fence builtins under `-cl-std=CL3.0`
    but does **not define the macros** (`#error` probe) nor advertise the features in
    `CL_DEVICE_OPENCL_C_FEATURES` — the guard would silently compile the fences out and
    reintroduce the poc/07 cross-lane race on our primary GPU. Same under-advertising axis as
    poc/07 test E.
  - ❌ `-cl-std=CL2.0` can't be assumed: PoCL rejects it ("device doesn't support that version")
    despite supporting 3.0.
- ✅ **Probe cascade at init** (`runtime.cc`), most capable dialect first, first successful build
  wins: `-cl-std=CL3.0` (if in ALL_VERSIONS) → `-cl-std=CL2.0` (if supported) → `""` (lenient
  pre-3.0 drivers, old behavior) → `"" + -DVMO_NO_DEVICE_FENCE` (strict-1.2 last resort; compiles
  the fences out via macros in vm_common.cl). The winning variant sets
  `DeviceInfo::has_device_fence`; without it the runtime **forces host-dispatch** and
  `PJRT_OCL_ENGINE=mega` fails loudly (fence-less spin-barrier = poc/07 data race, never silent).
  Verified: 195/195 e2e on PoCL and NVIDIA (both pick CL3.0), NVIDIA `ENGINE=mega` still runs the
  megakernel, strict-CL1.2 simulation builds via the last-resort variant.
- **Rule**: never call `clBuildProgram` with empty options and 2.0+ features in the source —
  leniency of the dev machines masks it until the first strict compiler (Intel) sees the code.

## 4. PJRT layer

- ✅ **Hand-rolled PJRT C API — VALIDATED 2026-07-14** (`poc/02-pjrt-skeleton`): `jax.devices()`
  returns our OclDevice on both NVIDIA and PoCL with ~650 lines of C++, one vendored header,
  CMake+Ninja (~3 s build), zero XLA source dep. User's failure prediction did not materialize.
  ~30 of 138 API entries suffice for device enumeration. Incident log (full detail in
  `poc/02-pjrt-skeleton/NOTES.md`):
  - jaxlib dlsym's **`GetPjrtApi`** (lowercase "rt"), not `GetPjRtApi` as some docs write.
  - `PJRT_Error_ForEachPayload` must work from day one — stubbing it → infinite error recursion
    → core dump (framework calls it on every error).
  - `PJRT_Device_GetAttributes` returning UNIMPLEMENTED is a CHECK-crash (`LogFatalIfPjrtError`),
    not catchable; empty attributes are fine. Expect more CHECK-crash (not error) contracts in
    Compile/Execute/Event callbacks at M2 — implement those to spec, not as stubs.
  - Keep the trick: every stub returns UNIMPLEMENTED **carrying its own callback name** —
    makes each new jax version/feature self-diagnosing.
  - 🅱️ XLA C++ wrapper route (`pjrt_c_api_wrapper_impl.h`, full Bazel build) — retired unless the
    async Event contract proves intractable by hand.
  - ✅ **M2 e2e VALIDATED 2026-07-14**: `jax.jit((a+b)*a)` == numpy exactly on NVIDIA + PoCL via
    the full stack (compile → lowering subprocess → VMProgram → megakernel). Multi-output,
    chained calls, identity/output-aliasing, 2D all pass. New CHECK-crash contracts found at M2
    (both now implemented): `PJRT_LoadedExecutable_AddressableDeviceLogicalIds`,
    `PJRT_LoadedExecutable_GetDeviceAssignment` (wants a serialized xla.DeviceAssignmentProto —
    hand-encoded 9 protobuf bytes for the 1×1 case). `PJRT_Executable_OptimizedProgram` +
    `PJRT_Device_MemoryStats` + `PJRT_Client_TopologyDescription` errors are tolerated by jax.
    Events pre-signaled (fully synchronous v1) worked without incident — the feared async Event
    contract never materialized for single-device jit.
  - ⚠️ Once the .so exists at the default path, plugin discovery makes EVERY `import jax` in the
    venv use our backend (priority 500 > cpu 400): pure-lowering tests must pin
    `JAX_PLATFORMS=cpu` before importing jax; eager jax ops (even `jnp.arange`) compile through
    the plugin, so eager coverage == jit coverage.

## 4b. jax/PJRT version pin

- ✅ jax/jaxlib **0.10.2** ⇒ XLA pin via `third_party/xla/revision.bzl` at tag `jax-v0.10.2` ⇒
  XLA commit `5a9e73cbd92530cac2ac36f4736a774b2412afe2` ⇒ **PJRT C API 0.112** (vendored at
  `poc/02-pjrt-skeleton/vendor/pjrt_c_api.h`). Exact minor match ⇒ no ENABLE_PJRT_COMPATIBILITY
  needed. Recipe documented for future bumps.

## 5. Python packaging / discovery

- ✅ **Entry-points discovery** (`[project.entry-points.'jax_plugins']`) — recommended by openxla
  docs over bare `jax_plugins/` namespace dirs. `initialize()` calls
  `xla_bridge.register_plugin('opencl', priority=500, library_path=..., options=None)`.
  priority>400 makes it win under `JAX_PLATFORMS=''`; during dev prefer explicit `JAX_PLATFORMS=opencl`.
  - 🔬 jaxlib ↔ PJRT C API version matching is strict (no ABI guarantee yet): pin JAX and record the
    `PJRT_Api` major/minor we build against.

## 5b. Python package (M1, merged 2026-07-14)

- ✅ `python/pjrt_ocl` implements VMProgram v1 exactly (golden byte-layout test); 14/14 pytest.
  Options dict to C++: `python_exe`, `lower_service`; exit codes 0/2(unsupported)/3(internal).
- ⚠️ **No COPY opcode in v1** ⇒ returning an argument/constant lowers as output-map ALIASING of
  the producing buffer id; the executor must tolerate output regions == input/const regions.
- ⚠️ **FMA divergence**: XLA CPU contracts `a*b-c` under jit (no flag disables it; three tried) ⇒
  bit-exact comparisons vs jax.jit need integer-valued f32; real-valued data compares vs EAGER
  jax. Policy applies to all future e2e tests.
- 🧭 Splat constants currently expand into the const pool; FILL_F32 lowering is a follow-up.

## 5c. Dynamic memory north star (user, 2026-07-14)

- User anticipates a JAX-successor with fully dynamic device memory (realloc + data-dependent
  reshape). Direction: keep the door open, do NOT force the design. See docs/memory.md L0–L3
  spectrum; cheap door-keeping = indirectable operands + flat-arena discipline.

## 6. Naming

- ✅ **pjrt-ocl** (python package `pjrt_ocl`, JAX platform name `opencl`) — picked from user's
  shortlist (pjrt-ocl / pjrt-ocl-mk / ocl-ext-xla) 2026-07-14.

## 7. Backend selection

- ✅ **CPU-first development on PoCL**, then NVIDIA, then Intel/AMD — user decision 2026-07-14.
  Rationale: printf/debuggers/sanitizers work on a CPU OpenCL runtime.
- ✅ **Backend configurable**: `PJRT_OCL_DEVICE=<platform substring>[:<device index>]` env var,
  overridable via PJRT client-create options; default = first GPU, else first CPU.

## 8. Environment

- ✅ NVIDIA ICD registered manually 2026-07-14: `/etc/OpenCL/vendors/nvidia.icd` ←
  `libnvidia-opencl.so.1` (was missing; clinfo now lists the RTX PRO 6000 Blackwell).
- ✅ PoCL installed 2026-07-14 (`pocl-opencl-icd`): platform "Portable Computing Language",
  device cpu-haswell (AMD Ryzen 9 3900X).

## 9. First Intel Xe2 bring-up (2026-07-15, Lunar Lake host)

- Environment: Intel Core Ultra 9 288V (Lunar Lake) w/ builtin Arc 140V (**Xe2**, 8 Xe-cores,
  reports **64 compute units** = XVEs), inside Docker (needed `--device /dev/dri` passthrough +
  a Lunar-Lake-capable ICD: `intel-opencl-icd` **26.22** from `ppa:kobuk-team/intel-graphics`;
  24.04-archive 23.43 predates LNL and enumerates nothing).
- ✅ **Results**: `runtime_test` PASS; full pytest **198 passed / 1 skip** on Xe2 with EITHER
  engine — megakernel (`PJRT_OCL_VM_LANES=32`) or host-dispatch (`PJRT_OCL_ENGINE=host`).
  PoCL-on-LNL also 198/1. The `-cl-std` dialect probe (§3b) held up on the real Intel compiler.
- ❌ **Default lane sizing is wrong on Intel — megakernel deadlock out of the box.** JAX e2e
  fails `clFinish` = -5 (CL_OUT_OF_RESOURCES) at default `ngroups = 2×CU = 128`. Lane sweep at
  local=256: **32 lanes PASS, 33 lanes FAIL** — exactly the hardware residency: 8 Xe2-cores ×
  64 HW threads ÷ (256 items @ SIMD16 = 16 threads/group) = **32 co-resident groups**, i.e.
  **CU/2**, not 2×CU. Root cause: `CL_DEVICE_MAX_COMPUTE_UNITS` semantics differ per vendor —
  NVIDIA reports SMs (2×CU validated, poc/01/04), Intel reports **vector engines (XVEs)**, so
  2×CU oversubscribes 4× and the spin-barrier starves (§1's predicted "occupancy oracle"
  fragility, now measured on a second vendor).
- ✅ **FIX (2026-07-15): measured occupancy discovery, `poc/08-occupancy-discovery` → integrated.**
  Sorensen-Donaldson discovery protocol (gate/ticket/lock, 1.2 atomics on one buffer — safe on
  the strict-1.2 `VMO_NO_DEVICE_FENCE` build too): ticket holders spin until the gate closes
  (holds their residency slot), ticketless groups exit immediately ⇒ deadlock-free for ANY
  launch size. Runs at init as a probe mode INSIDE vm2 (`nlanes==0` sentinel, ~20 ms), because
  the answer is per-compiled-kernel: a lookalike probe kernel (8 KB SLM + reg pressure but
  SIMD32) discovered 64 on Xe2 while the real vm2 (SIMD16) discovers exactly **32** — the
  measured JAX boundary. `ngroups = min(discovered, 2×CU)`; the cap keeps NVIDIA at its
  validated sizing until discovery is re-validated there. `PJRT_OCL_VM_LANES` still overrides.
  Full suite green on Xe2 with no overrides after the fix.
  - 🔬 poc/08 side-finding: SLIM kernels over-discover (256 = whole launch) — Xe2 mid-thread
    preemption time-slices kernels that don't use barriers/SLM, so they don't need co-residency
    at all; barrier+SLM kernels (vm2) are non-preemptible and discovery = true residency. If a
    future kernel is preemptible, over-discovery is harmless (preemption keeps the spin-barrier
    live). Liveness at discovered count: PASS (1.9 µs/barrier on Xe2, 225 µs on PoCL);
    discovered+1 on Xe2: spins >60 s, host-killed — discovery is TIGHT.
## 10. Perf: while + matmul (2026-07-16)

Focus session on the two biggest gaps vs native CUDA (see docs/bench_plot.png): `while`
(was 28x at 16M elems) and `matmul` (6.8x). Two background research agents mined
HazyResearch/Megakernels (sync/scheduling) and the tensor-core GEMM refs
(ihavnoid/hgemmtest inline-PTX WMMA from OpenCL, CUTLASS m16n8k8 TF32 string).

### 10a. `while` — SOLVED (28x → ~4x at 16M, and the small-N floor halved)

Profiled the benchmark `fori_loop(0,32, v: v*1.5+1, x)`. Three independent costs, none of
them the barrier at large N:

1. **Scalar-const broadcasts materialized.** `v*1.5+1` lowered to `gather_strided`
   (broadcast 1.5 → full N-vector) + `mul` + `gather_strided`(1.0) + `add` — two full
   N-length const buffers written and read every iteration. FIX: **OP_AFFINE_F32**
   (`d = a*s + t`, s/t scalar immediates) + a lowering peephole that folds
   `mul(x, bcast_const)` / `add(x, bcast_const)` into it, **composes affine∘affine chains**
   (`(x*s1+t1)*s2+t2`), and DCEs the dead broadcasts (index-stable NOP substitution so
   WHILE cond/body ranges stay valid). `v*1.5+1` → ONE affine op, ZERO broadcast buffers.
2. **Redundant copy-back.** The while lowering snapshotted body returns into temps then
   copied temps→carries (2 full-length passes/iter, for swap/passthrough safety). FIX:
   **in-place carry update** — when a carry's new value is produced by a single
   elementwise (index-aligned) op that is the only body reader of that carry, retarget the
   producer to write the carry directly and drop both copies. Guarded off for bodies with
   nested WHILE/IF (a nested region's carry-init copy would otherwise be mistaken for the
   producer — caught by test_nested_while). Net: body of the benchmark = `c_x = c_x*1.5+1`
   IN PLACE, so the 64 MB carry stays **L2-resident** across all 32 iterations (Blackwell
   has 96 MB L2), matching CUDA's fused-loop traffic. 16M: 22.5 ms → **3.2 ms** (bit-exact
   vs JAX CPU).
3. **Barrier contention (small-N floor).** The spin-barrier used `atomic_add(&bar[1],0)` as
   a *read* — an atomic RMW forces every one of the (up to 376) spinning workgroups to take
   the phase cache line EXCLUSIVE, so it ping-pongs (~38 µs/barrier). FIX: coherent
   `atomic_load_explicit(..., relaxed, memory_scope_device)` — the line stays Shared, only
   invalidated once at the phase flip. 4K while: 2.47 → 1.8 ms. (The true small-N fix is
   barrier ELISION for lane-diagonal loops — each lane runs the whole loop with per-lane
   control, zero grid barriers — designed but not built; the megakernel research confirms
   it's the right model. Deferred.)

Encoding note: OP_AFFINE_F32 needs TWO f32 immediates but `imm` is the only free-form
serialized instr word (`dst/a/b` are range-checked, `aux` must be ≤ n_aux, `p3` is the
SELECT pred). Repurposed the 8th instr word (was a zero pad `pad1`) as `imm2` (the `t`
bits); parse allows it nonzero only for OP_AFFINE_F32. Device reads s/t from task p2/p3
(unvalidated for EW-affine). `mad(a,s,t)` matches JAX CPU's fma bit-for-bit.

These are GENERAL wins, not while-specific: scalar scale/bias folding helps any program
with `x*c`/`x+c`/affine chains (bias, normalization, scaling).

### 10b. `matmul` — megakernel register ceiling (open)

Baseline 17 TFLOP/s @ N=2048 vs cuBLAS 117 TFLOP/s (6.8x). Key facts established:
- cuBLAS at N=2048 gets only 117 TFLOP/s — well under TF32 tensor-core peak (~400+), so it
  is NOT tensor-core-bound at these sizes; Blackwell FP32 peak is ~125 TFLOP/s. So a
  well-tuned **portable** SGEMM could in principle approach cuBLAS here WITHOUT tensor cores.
- The kernel is **local-memory-bandwidth bound**: the 4×4 register microtile does only
  2 FMA per local load (global reuse is already high, so float4 *global* loads won't help).
  Raising arithmetic intensity needs a bigger register microtile.
- **8×8 (128×128 tile) HANGS the matmul at runtime** (not compile — runtime_test's EW/while
  pass at 128×128). 64 live accumulator registers in the SHARED megakernel almost certainly
  spill catastrophically (the megakernel's register budget is the max over ALL op paths).
  Rolling the K-loop didn't help. This is the fundamental tension the CLAUDE.md notes:
  aggressive matmul tiling is incompatible with one-kernel-does-everything.

SHIPPED (path a, partial): a standalone `mm2` SGEMM kernel + a pure-matmul fast path.
`runtime.cc` detects a program that is a single f32 TILE_MMA with no barrier/control
entries and, for LARGE matmul on GPU (M,N≥1024, K≥256; `PJRT_OCL_MM_KERNEL` forces),
launches `mm2` (one 256-thread workgroup per 128×64 tile, 8×4 register microtile,
double-buffered smem, transposed As for vectorized local loads) instead of the megakernel.
Standalone => independent register budget, so an 8×4 tile does not spill. N=2048:
17.1 → **21.1 TFLOP/s** (1.23×). Gated because below ~1024 the megakernel's 4×4/256-thread
MMA wins (more workgroups per unit work → better latency hiding). Correct on non-square /
non-tile-multiple / large shapes; 199 pytest + runtime_test green.

What the tuning sweep established (all standalone, N=2048/4096): tile/register configs
4×4-256t, 8×8-64t, 8×8-256t(128²), 8×4-256t all **plateau at ~17–23 TFLOP/s** — the 8×8's
64 accumulator registers cap occupancy at ~25% (2 workgroups/SM), and without
bank-conflict-free smem swizzling + register-level (not just smem) prefetch the kernel
can't approach the ~100 TFLOP/s a production FP32 SGEMM reaches. **cuBLAS hits 134 TFLOP/s
at N=4096 — ABOVE Blackwell's ~125 FP32 peak — so it is TENSOR-CORE (TF32) bound.** Hard
conclusion: **matmul parity REQUIRES TF32 tensor cores**; portable FP32 tops out ~1.3× off
even when perfect. Next: an NVIDIA-only TF32 tensor-core body inside `mm2` behind a build
guard (poc first, per hard rules), via inline PTX `wmma.load.*.shared`/`wmma.mma.sync`
(hgemmtest passes `__local` ptrs as `"l"` into `.shared` WMMA ops; CUTLASS
`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`). The `mm2` dispatch path is already
built to host it.

## 11. Scheduler: fuse lane-local elementwise chains (2026-07-16)

The scheduler split the dataflow into LEVELS (maximal antichains) with a global barrier
between every level, so a dependency chain paid a barrier per step even when it was pure
elementwise. But an elementwise dependency is **lane-local**: output element i reads only
input element i, so a lane that owns tile T produces everything tile T needs. Chaining such
ops on one lane per tile is strictly better than "parallel level + barrier + next level" —
same compute wall-clock (two independent equal ops on half the lanes each = the two ops
sequentially on all lanes), minus the barrier.

**Algorithm (not a hack — a graph coarsening).** A barrier is emitted only on a CROSS-LANE
edge: one where the producer or consumer is a *shaped* op (matmul/reduce/gather/broadcast/…,
whose output tiling differs), or the element count changes. Same-index elementwise
dependencies do NOT start a new phase. Within a phase, connected components of elementwise
ops (under data deps) become fused **chains**; each chain is a *unit* fed to the existing
chunk+LPT packer, which splits its tiles across lanes and emits the whole chain over each
tile range. Independent units still fan out across lanes in parallel; a shaped op is just a
singleton chain, so the matmul-∥-elementwise packing (docs §1 diamond) is unchanged.

**No engine change, no fence.** Consecutive entries of a chain run over the *same* tile range
on a lane, and each tile op's grid-stride loop is `for i=lo+lid; i<hi; i+=lsz` — so work-item
`lid` writes then re-reads the *same* elements across ops (thread-local program order). The
megakernel and host-dispatch both just run the extra entries; correctness needs neither a
cross-workgroup barrier nor a work-group fence.

Validator (vmreader) updated: a phase is no longer fully order-independent — entries within a
lane are ordered (the chain). It now runs each lane's entries in sequence and only permutes
LANE order (forward vs reverse) to assert no cross-lane write/read landed in one phase.

**Measured** (deep 7-op elementwise chain, `PJRT_OCL_FUSE=0` reverts to per-level barriers):
NVIDIA megakernel 1.06–1.19× (cheap in-kernel barriers); **PoCL host-dispatch 2.0–2.9×**
(each phase there is a kernel launch + clFinish, 7→1). Correct vs JAX CPU on both. 199 pytest
+ runtime_test green. Does NOT reduce memory traffic (ops still each read/write their buffers)
— that is elementwise *op* fusion (the affine folding, §10a), orthogonal to chaining.

## 12. CPU performance: why XLA CPU wins, and the fix (2026-07-16, poc/09)

- Context: first honest PoCL-vs-native-XLA-CPU bench (README, 2026-07-16) showed 2.6x (EW 16M)
  to ~90x (matmul 2048) to ~320x (matvec) deficits. poc/09 microbenches candidate kernel shapes
  on PoCL AND Xe2 against an 8-thread memcpy wall (84.7 GB/s).
- ❌ **Root cause 1 (EW): PoCL's work-group vectorizer only vectorizes the implicit WI loop.**
  Any explicit in-kernel loop around the body — including our tile loop, restructured variants
  with straight-line bodies, with or without guards — leaves the kernel SCALAR: 5 GB/s vs
  46 GB/s for explicit float8 (a4). Not fixable by loop restructuring; measured, not assumed.
- ❌ **Root cause 2 (matmul/matvec): the GPU MMA tile shape is pessimal on CPU.** __local
  staging is an extra memcpy and every WG barrier makes PoCL loop-split; barrier-free
  register-blocked float8 (1 WI/WG) is 4x faster standalone (60.9 vs 15.6 GFLOP/s @1024) and
  ~11x vs in-VM; GEMV via a row-dot kernel beats the MMA tile on BOTH devices (12.7 vs ~0.6
  GB/s PoCL; 73 vs 37 Xe2).
- ❌ **Root cause 3 (small ops): PoCL's launch floor is 17–52 µs** (pipelined, measured) vs
  XLA's ~12 µs full dispatch. Not ours to fix in-kernel; accepted.
- ✅ **No single EW pattern wins both device classes** (a4: CPU 46 / Xe2 62; current a1: CPU 5 /
  Xe2 104; a2 wins both but needs element-sized grids, incompatible with the lane/tile model).
  **CHOSEN: device-keyed build define `-DVMO_CPU_TILES` (set when `!is_gpu`) selects an
  explicit-float8 per-WI-contiguous EW tile body; GPUs keep the scalar coalesced loop.** This
  is the CLAUDE.md "vendor tuning behind the kernel-table" mechanism, realized as a build
  variant of the same source (precedent: the -cl-std probe, §3b).
- ✅ CPU-shaped SGEMM + GEMV kernels routed via the pure-matmul fast path (precedent: mm2,
  §10b). Iteration ladder recorded in poc/09 README: land b2 shape first (~11x), cache
  blocking/packing later (Eigen parity NOT the goal; ~8x off is acceptable for a debug backend
  that was ~90x off).
- ✅ **SHIPPED (2026-07-16, three iterations, each 199/1 green on PoCL AND Xe2):**
  1. `-DVMO_CPU_TILES` float8 EW bodies: add 16M f32 44.7 → **3.16 ms (63.8 GB/s)** — 14x, and
     3.7x FASTER than native XLA CPU (11.8 ms). Xe2 build bit-identical (104 GB/s unchanged).
     Cost-table cache key now includes kernel source + build opts (stale-cost bug otherwise).
  2. CPU-shaped `mm2` body (barrier-free 4x16 float8 register block, geometry 1 WI/WG) +
     **`gemv` kernel routed on BOTH device classes for N==1**: PoCL matmul 2048 3183 → **223 ms**
     (14x, 77 GFLOP/s; XLA/Eigen 618 — cache-blocking is the recorded next rung), PoCL matvec
     113 → **0.77 ms (147x)**; Xe2 matvec 0.456 → **0.253 ms** (1.8x, GPU win as predicted).
  3. float8 movers for contiguous rank-1 dyn_gather/dyn_scatter + vector reduce partials:
     dynamic_slice 16M 20.9 → **1.53 ms** (14x; XLA 5.7). reduce_sum 16M 1.15 ms (XLA 0.57 —
     read-only stream, ours still has tree/barrier overhead; acceptable, revisit if it matters).
- 🧭 Remaining known gaps, deliberately deferred: CPU matmul cache blocking (~8x to Eigen),
  reduce 2x, PoCL launch floor (~17-52 µs, PoCL-internal), i32/f16 EW tiles still scalar on CPU
  (extend vmo_ew_bin8 pattern when a workload cares).
