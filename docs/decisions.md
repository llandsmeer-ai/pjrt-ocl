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

### 10c. TF32 tensor cores for IN-PROGRAM matmul — SHIPPED (2026-07-16)

Front #3: make matmuls that appear *inside* a larger program (transformer QKV/out/FFN
projections + batched attention QKᵀ/AV) fast — they run in the megakernel's `vmo_mma_tile`,
NOT the standalone `mm2` fast path, so §9b's `mm_tc` never touched them. Two candidates:
(1) a guarded tensor-core body inside the megakernel; (2) selective per-phase dispatch of
matmul phases onto the standalone TC kernel.

**CHOSEN: (1) guarded TC `vmo_mma_tile` in a NVIDIA-only megakernel variant.** The vm2
program is rebuilt a SECOND time with `-DVMO_NV_PTX` (only on NVIDIA, only when the portable
build already got device fences); on success `vm_tc_kernel_` replaces `vm_kernel_` for
execution, else it stays null and the persistent engine transparently uses the portable
kernel (try-and-fallback, mirroring the dialect probe). Inline PTX therefore NEVER enters the
portable program — PoCL/AMD/Intel are untouched (runtime_test + e2e PASS on PoCL, portable
path). `PJRT_OCL_MEGA_TC=0` forces portable (A/B). The TC body computes the SAME 64×64 tile
(scheduler `MMA_T=64`, batch via `t.p3` preserved → batched attention gets tensor cores too)
with `wmma.mma.sync m16n16k8` tf32, reusing poc/08's driver workarounds (A.row/B.col,
`cvta.to.shared` on `__local` ptrs, broken `wmma.store.d.shared` → hand-mapped masked global
store). It keeps the SAME `As`/`Bs` local footprint (64×16 each) and a comparable register
count (`acc[2][8]+af[4]+bf[4]≈24` vs scalar `acc[4][4]+a[4]+b[4]`).

**The occupancy risk (the whole reason to fear approach 1) MEASURED AWAY.** WMMA fragments do
cost co-residency: the probe (poc/08 discovery) reports **raw residency 564 workgroups for the
portable vm2, 376 for the TF32 vm2** — a 33% drop. BUT the megakernel launch is already capped
at `2×CU = 376` lanes (validated NVIDIA sizing, §9), and 376 ≤ 564, so BOTH variants launch
the identical 376 lanes. The tax is fully absorbed by the existing cap → non-matmul ops are
NOT regressed. Confirmed directly, not just by lane count: a 12-op elementwise chain (0.739 ms)
and a 32-iter `while` (2.17 ms) are BIT-identical and TIME-identical between the two variants.
Note the tile sits exactly at the occupancy boundary (TF32 residency 376 == the cap): a bigger
register tile would push residency *below* 376 and then shrink every other op's lane count — so
64×64/`acc[2][8]` is the largest tile affordable here, which is also why in-program TF32 can't
reach the standalone `mm_tc`'s 128×128 intensity.

**Measured (RTX PRO 6000 Blackwell, `PJRT_OCL_MEGA_TC=0` = portable baseline):**
- In-program matmul (chained, stays in megakernel): N=512 3.8→**4.6** TFLOP/s (1.2×),
  N=1024 11.7→**17.7** (1.5×), N=2048 17.1→**27.3** (1.6×). Smem-bandwidth bound (BK=16,
  single-buffered), so ~2× under the standalone `mm_tc` — the occupancy-preserving tile trades
  intensity for not taxing other ops.
- Batched attention shape (G=32, 128×64×128): 2.1→**2.3** TFLOP/s (~1.1×) — these matmuls are
  tiny (67 MFLOP each) and latency/overhead-bound, so tensor cores barely engage.
- Transformer `--config base`: portable 14.52 → **13.70 ms** (1.06×), vs CUDA 0.458 ms
  (gap 31.7× → **30.0×**). Correct: mean −0.0128 (=CUDA), std 1.1539 vs 1.1544 (TF32 ~1e-2 rel).

**Honest conclusion.** TF32-in-megakernel is a clean, SAFE, always-correct win with zero
portability cost and zero non-matmul regression — shipped ON by default on NVIDIA. But the
transformer at base is **overhead/latency-bound, not matmul-bound**: its matmuls are small
(M=512, attention K=64) where even portable matmul is far from compute-bound, so faster matmul
only buys 1.06×. The remaining 30× gap to CUDA is dominated by per-phase barrier/launch and the
many small elementwise/reduce/softmax/layernorm phases + per-Execute H2D/D2H — a different
front (barrier elision for lane-diagonal loops, §10a; per-Execute overhead), not matmul.

**REJECTED: (2) selective per-phase dispatch.** Its only advantage over (1) is giving matmul an
independent register budget for a 128×128 tile — but (1) proved it needs no such rescue
(occupancy untouched). Against it: the transformer has ~48 matmul phases/iter, each would need
a `clFinish` barrier (breaking the single-launch megakernel), and the standalone kernel is not
batch-aware. Since in-program matmuls at these sizes are already overhead-bound (batched
2 TFLOP/s), pulling them into separate launches adds more overhead than the higher intensity
saves. Only worth revisiting if approach (1) had hurt occupancy, which it did not.

### 10d. TF32 megakernel tile micro-tuning campaign (2026-07-16, transformer `large`)

Front: the transformer `large` gap to native CUDA is 9.6× and 96% of its FLOPs are matmul
(measured), so it IS matmul-bound — the one lever left (§14b/transformer-perf memory). Attacked
the in-megakernel TF32 `vmo_mma_tile` (§10c) measure-first. **Baselines** (RTX PRO 6000 Blackwell):
in-megakernel TF32 matmul (`PJRT_OCL_MM_KERNEL=0` keeps a pure matmul in the megakernel) N=2048
18.0 / N=4096 16.1 TFLOP/s; standalone FP32 `mm2` 21.6 / 25.1; cuBLAS 72.9 / 146.6; transformer
`large` 35.08 ms / 9.55 TFLOP/s, `base` 9.82 ms / 2.13 TFLOP/s.

**Root cause established (the ceiling is real).** The 64×64 tile has arithmetic intensity ≈
`64·64·K·2 / ((64·K+K·64)·4)` = **16 FLOP/byte** (reuse == tile width 64), far below Blackwell's
~83 FLOP/byte balance. Two consequences, both measured: (a) at large K where B exceeds L2 (square
N=K≥4096, B=64 MiB) the tile is **global-bandwidth bound** — N=4096 (16.1) is SLOWER than N=2048
(18.0) despite more parallelism; (b) the only intensity fix is a bigger output tile, which the
megakernel's co-residency cap forbids (a bigger register/acc tile drops residency below the 2·CU=376
lanes the cross-workgroup barrier needs — §10c, re-confirmed here). So the standalone `mm2` beats the
in-megakernel TF32 at N≥2048 (25 vs 16) purely on its larger 128×64 tile + double buffering, NOT
tensor cores (it's FP32).

**What was tried (register/occupancy-neutral knobs only — the tile can't grow):**
- **Smem leading-dim padding — SHIPPED (+4.5% on `large`).** The wmma m16n16k8 A/B fragment loads
  read 16 rows at stride `MMA_BK`==16, colliding on the same 32 smem banks (8-way conflict). Pad
  the TF32 staging leading dim to 20 (`gcd(20,32)=4` → 8 distinct bank offsets). Costs 25% more
  staging smem (still ~16 KiB/wg, nowhere near the limiter), zero accumulator registers. Measured:
  in-megakernel N=2048 18.0→19.6, N=4096 16.1→16.9; transformer `large` 9550→9984 GFLOP/s (+4.5%),
  `base` 2133→2174 (+1.9%). TF32-only (`VMO_NV_PTX`); portable path byte-identical.
- **Group-M L2 swizzle — SHIPPED (+7.6% on huge squares; WASH on the transformer).** Remap
  tile→(tr,tc) into GROUP_M=8-tall column strips so co-resident workgroups reuse the same B panel
  from L2 (CUTLASS/Triton "grouped" order). Pure bit-exact index remap. Measured: in-megakernel
  N=4096 16.9→18.1 (+7.6%), N≤2048 unchanged. **Wash on the transformer** — its FFN/projection
  B-panels (≤16 MiB) fit L2, so those matmuls are tile-compute bound not BW bound. Kept anyway: free,
  bit-exact, and the megakernel is the only path for view-folded / in-program large matmuls that the
  `mm2` fast path (pure single-matmul programs, contiguous operands) can't take.
- **Double-buffered K-loop staging — WASH, reverted.** Two smem panels, prefetch next K-block while
  computing (one barrier/iter). N=2048 18.0→17.9, N=4096 16.1→16.4 — no move. The tile is not
  global-latency bound (padding/swizzle addressing bank-conflict throughput and L2 reuse are what
  matter), so hiding global latency buys nothing. Added smem for no gain → dropped.
- **BK 16→32 — REGRESSION, reverted.** Deeper K-block (fewer barriers/mma) N=2048 18.0→13.6,
  N=4096 16.1→14.0. More staging smem drops occupancy; the tile is not barrier-bound. Dropped.

**Direction 2 (route hot matmuls to the standalone kernel) re-evaluated and still REJECTED for the
transformer.** The enabler would be host-dispatch (per-phase `vm2_seg` launches, where a matmul
phase could launch `mm2` instead). Measured NVIDIA host-dispatch baseline (`PJRT_OCL_ENGINE=host`):
`base` 9.7→14.3 ms, `large` 33.5→46.0 ms — the per-phase launch overhead (+12.5 ms on `large`)
*exceeds* the entire matmul time the megakernel spends, so even a 2× matmul kernel nets a wash. The
only routing that could win is a **hybrid**: keep the single-launch megakernel for non-matmul phases
and split out only the big projection/FFN matmuls into separate `mm2`(-TF32) launches. That needs a
standalone TF32 128×128 kernel (independent register budget → the intensity the megakernel can't
have) AND splitting the persistent megakernel into per-segment relaunches at matmul boundaries
(arena state persists across launches, so feasible, but ~30 relaunches/iter and mm2 is not
batch-aware). This is the genuine remaining lever for `large` (potential ~1.5–1.8× if the segment
relaunch overhead stays ~1 ms) but is a large architectural change deferred as future work.

**Net shipped:** transformer `large` 35.08→33.51 ms, **9.55→10.0 TFLOP/s (+4.7%)**, gap to native
CUDA (91.9 TFLOP/s) 9.63×→9.19×; `base` 9.82→9.70 ms, 2.13→2.16 TFLOP/s, gap 21.9×→21.6×. All 5
`--check` configs PASS (TF32 atol 5e-2); portable `MEGA_TC=0` f32-exact (max_abs 2.2e-6); PoCL
portable f32-exact (2.4e-7); 240 pytest pass / 1 skipped. Honest bottom line: the in-megakernel TF32
tile is at its architectural ceiling (~16–20 TFLOP/s, intensity-capped by the co-residency-locked
64×64 tile); micro-tuning bought a clean +4.7% but matmul PARITY needs the hybrid split above.

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
- ✅ **Low-N + matmul follow-up (2026-07-17, poc/10 + phase batching):**
  - Low-N root cause MEASURED (PJRT_OCL_PHASES): every host-dispatch phase paid a blocking
    seg_tab write + clFinish (~66 µs on PoCL); a small dynamic_slice is 3 phases, a while
    iteration ~3. FIX: 256-slot seg_tab ring + staged phases flushed as ONE non-blocking write
    + k kernel enqueues per drain group; drains only at while-cond reads (implicit), ring wrap,
    program end. A/B at 262K: gather −33%, while −36%, single-phase ops unchanged. Remaining
    while floor = the per-iteration cond read; the structural fix is §10a's barrier elision
    (designed, still not built) or scheduler-side cond+body phase fusion.
  - CPU SGEMM cache-blocking ladder (poc/10, each step verified): packed B panels 1.6x,
    6x16 register block 1.45x, KC=512 sweeps 1.27x → in-VM 2048: 268 → **110 ms
    (156 GFLOP/s, 2.6x off Eigen; 88x at the start of the CPU work)**. Default CPU matmul;
    `PJRT_OCL_MM_CPU=reg` keeps the register kernel (per-hardware choice + ragged-N fallback).
    Stop point recorded: packed A / prefetch / per-core-type tiles not worth it for a debug
    backend.

## 13. General principle: access-map–driven fusion (2026-07-16, transformer workload)

Driving a realistic transformer forward pass (`tools/bench_transformer.py`: layernorm, MHA,
GELU-FFN, residuals; random weights) exposed the real gaps. The specific fixes all reduce to
ONE principle, now the guiding design rule for the compiler/runtime.

**Principle.** Model every producer→consumer edge by its *access map* — the function from a
consumer output index `i` to the producer indices it reads. If that map is a static function
of `i` (identity → elementwise; `i//seg`/`0` → broadcast; affine strided → transpose/slice/
reshape), the producer need **not be materialized** and **no barrier** crosses the edge — the
consumer inlines the read. Materialization + a barrier are required **only** at genuine
many-to-one / data-dependent edges: reductions, contractions (matmul), dynamic gathers. Fusion
= compose access maps along edges up to the nearest such boundary; the fused region's *leaves*
(program inputs + reduction/matmul/gather outputs) are the only things in memory. This is the
classic loop-fusion / polyhedral view, and it unifies every ad-hoc fold we had: scalar-affine
(§10a) = broadcast with seg=whole; chain fusion (§11) = identity map; and now shape-op folding.

**Mechanism (shipped): viewed operands.** An elementwise op reads `src[view_index(i)]` through a
strided descriptor (the `{rank,out_dims,in_strides,src_off}` map gather already uses). Lowering
pass `_fuse_views` folds an OP_GATHER_STRIDED (broadcast/transpose/slice/reshape/reverse) into
its consuming viewable f32 EW operands (view aux-offset in imm/imm2 → task p2/p3) and NOP's it;
a gather feeding a matmul/reduce/other-gather stays materialized. Kernel: `vmo_view_idx` + a
scalar viewed path in `vmo_ew_tile_f32`. Both validators read via `rt.viewed`. No fence: a
viewed read is a strided load from an already-materialized (prior-phase) buffer. 210 pytest +
runtime_test green; transformer bit-close to JAX CPU. General for broadcast/transpose-heavy
code; marginal on the *latency-bound* transformer (whose real cost was the reduces, below).

**Also shipped alongside (transformer bring-up):**
- Segmented reduction (OP_REDUCE_SEG) for innermost-suffix partial reductions (softmax/
  layernorm reduce the last axis); non-suffix still needs a transpose first.
- dot_general → batched/broadcast matmul: lhs leading free dims flatten into M (`x@W`); equal
  leading batch dims G give G contiguous per-batch matmuls (attention QKᵀ/AV), batch in p3.
- **Reduce parallelism fix (§ the big transformer win): workgroup-per-segment collaborative
  reduce.** Thread-per-output starved layernorm's n_out=512 reduce to ONE workgroup; now one
  segment/tile reduced by the whole workgroup via a local tree → all lanes busy. layernorm 2x,
  softmax 3.8x, base transformer 14.5→8.5 ms (gap to CUDA 32x→19x).

Remaining front: in-program matmul (attention/FFN) still runs the megakernel SGEMM at ~5
TFLOP/s vs cuBLAS TF32 ~46 — being attacked by bringing the proven inline-PTX tensor-core
kernel (poc/11-tensor-core-mma, per docs/matmul-tensorcore-brief.md; renumbered — poc/08 is occupancy discovery) back to the lane-bytecode path (guarded, portable fallback).

## 14. Transformer optimization campaign — profiling & remaining fronts (2026-07-16)

Drove a realistic GPT-style forward pass (`tools/bench_transformer.py`, base = 4×128×512, 8
heads, 6 layers, batch=4) OpenCL/NVIDIA vs native JAX CUDA. **14.5 ms → 7.6 ms (gap 32× →
17×)**, all bit-close to JAX CPU / matching JAX CUDA's TF32. What each front bought:

- **Segmented reductions + batched/broadcast dot_general** — enabling ops so it runs at all.
- **Front 1 — access-map fusion (§13, viewed operands)**: general; marginal here (latency-bound).
- **Front 2 — workgroup-per-segment reduce**: layernorm 2×, softmax 3.8×; **14.5 → 8.5 ms**.
- **Front 3 — TF32 tensor cores in the megakernel (§10c)**: in-program matmul 1.5–1.6× but only
  **8.5 → 7.6 ms** — because the matmuls are small (M=512, attention K=64) and latency-bound.

**Profiled breakdown (base, PJRT_OCL_PROFILE + schedule inspection):**
- 100% kernel time (in_copy 0.36 ms, out_copy 0.01 ms — H2D/D2H are NOT the issue).
- **209 barriers / phases** (~34/layer); lane-count sweep 32→376 is FLAT, so barriers (whose cost
  scales with lanes) are NOT dominant (~1.7 µs each, measured earlier ≈ 0.35 ms total).
- Task mix: **210 EW, 48 MMA, 66 GATHER, 36 RSEG**. Matmul ≈ 2.5 ms (35%, TF32 saved 0.85 ms of
  it — near its ceiling); **non-matmul ≈ 5 ms (65%)**.

**The remaining gap is NOT one lever — it's distributed, and every piece is a large change:**
- **Transposes into matmul (8 of the 66 gathers/layer → dot):** the attention head reshapes
  (`(B,T,H,hd)→(B,H,T,hd)`) materialize full tensors + are phases. The fix is the access-map
  principle applied to matmul: **strided/batched matmul operands** so the dot reads the
  pre-transpose buffer via per-dim strides (batch may decompose into multiple strided sub-dims).
  Highest single value (~1 ms) but a substantial mma-kernel + dot-lowering change.
  ✅ **SHIPPED 2026-07-16 (§14a below): correct + portable, but a WALL-CLOCK WASH here.**
- **Gather chains (gather→gather→EW):** the inner gather can't fold because the outer already
  viewed the operand — needs strided-view *composition* (compose two access maps into one).
  Low value here (broadcasts are tiny) — deferred.
- **Small / batched matmul efficiency** (attention's 32× tiny 128×64×128): cuBLAS batched-GEMM
  territory; hard.
- **Flash-attention-style fusion** (QKᵀ→softmax→AV in fewer kernels): the biggest conceptual win,
  the biggest effort.

Conclusion: the general mechanisms (access-map fusion, collaborative reduce, TF32) are in and
correct; closing the last ~17× is a set of large, mostly-independent efforts, not a single fix.

### 14a. Shape-op → matmul operand fold (strided/batched matmul reads) — SHIPPED 2026-07-16

Front "transposes into matmul": extend the §13 access-map fold (which handled only elementwise
operands, `_fuse_views`) to **matmul operands**. A dot operand `A[g,m,k]` (`B[g,k,n]`) that was
produced by a transpose/reshape/broadcast (an `OP_GATHER_STRIDED` pure index map) now reads the
**pre-transpose SOURCE** in place instead of materializing the transposed tensor + a barrier phase.

**Mechanism (the clean insight): reuse the gather descriptor + the contiguous flat index.** The
dot treats its operand as a contiguous `[G,M,K]` tensor — element `(g,m,k)` at flat index
`g*M*K+m*K+k` — and the gather output IS that operand row-major. So the fold needs **no new index
math**: pass the contiguous `[G,M,K]`/`[G,K,N]` flat index through the gather's OWN descriptor
(`vmo_view_idx`, the same `{rank,out_dims,in_strides,src_off}` map). This is fully general — any
number of batch/M/K axes, and the attention batch `g=(b,h)` decomposing into two strided sub-dims
falls out for free (it's just the leading axes of `out_dims`). No new descriptor format.
- **Encoding:** task_t widened `32→40 B` with **p4/p5 = operand a/b VIEW aux-offset (+1; 0 =
  contiguous)** (all of p0–p3 = M,N,K,G were taken). Lowering carries them on `Instr.aview/bview`
  (non-serialized, like `reads_hint`) → scheduler → task p4/p5; a 2-word aux header at `Instr.aux`
  mirrors them so the tensor-interpreter validator (runs on re-parsed bytecode) recovers the fold.
- **Kernel:** both `vmo_mma_tile` variants (portable scalar 4×4 AND the `-DVMO_NV_PTX` TF32
  tensor-core body) branch on av/bv in the As/Bs staging load; **av==0 keeps the exact contiguous
  fast path** (no regression to the common case, no PoCL/AMD/Intel changes — the branch is uniform).
- **Lowering pass** `_fuse_matmul_views` (mirrors `_fuse_views`): a gather folds iff **every** reader
  is a dot reading it on a not-yet-viewed slot; retarget each dot operand to the gather source +
  descriptor, NOP the gather (DCE). `PJRT_OCL_MM_VIEWFOLD=0` disables (A/B + revert lever).
- ⚠️ **BUG FOUND + FIXED (regression-tested): unwritten view source.** When the SAME buffer feeds
  a dot operand directly AND (via a second gather) the other operand as a view (self-attention
  `q @ q.T`), folding the shared gather away leaves the view reading an **unwritten** buffer.
  Symptom: portable result rel-err ≈ 1.0 (garbage) on large random values — INVISIBLE to the small
  integer-valued unit tests (they happened to still match). Fix: a gather is not foldable into a dot
  if that dot already references the gather's output as a **viewed** operand source (it must stay
  materialized). Caught only by an fp comparison on random-magnitude data → added that as a test.

**Measured (base transformer, this machine's baseline is ~9.9 ms, not §14's 7.6 ms — clock/driver
drift; A/B on the SAME build via the flag):**
- **Eliminated 24 gather phases/model** (66→42 GATHER tasks, 4/layer: split(q), k-transpose,
  split(v), out-merge-transpose), **18 matmuls/model now fold** their transpose reads (~96 fewer
  per-lane barrier entries). No tensor materialized for those transposes.
- **NVIDIA wall-clock: WASH** — 9.90 ms fold-off vs 9.90 ms fold-on. **PoCL: wash** (small 283 vs
  290 ms). The strided staging load adds a per-load rank-wise div/mod that ≈ cancels the saved
  phase cost, and §14 already established the base transformer is **latency/overhead-bound, not
  matmul-bound** (its matmuls are tiny: M=512, attn K=64). This is the honest outcome, not a win.
- **Correct + portable:** portable megakernel (NVIDIA `MEGA_TC=0`) and PoCL host-dispatch both
  **f32-exact vs JAX CPU (max |err| 2.2e-6)**; TF32 megakernel matches JAX CUDA (max |err| 4.6e-3,
  TF32 noise). 215 pytest (+5 fold tests incl. the self-attn regression) + runtime_test PASS on
  NVIDIA **and** PoCL; both mma variants + all engines exercised.

**Kept ON by default** (it's the general access-map mechanism and does remove real
materialization + phases — a memory/phase win that a *compute*-bound or larger workload would feel;
harmless where latency-bound). Behind `PJRT_OCL_MM_VIEWFOLD=0` per the "revert if it ever regresses"
rule. Gather→gather→EW *composition* (compose two access maps) is still the deferred sibling (§14).

### 14b. The CUDA gap vs. model size — we're overhead-bound, not compute-bound (2026-07-16)

Measured ours (TF32 megakernel, NVIDIA) vs. native JAX CUDA on the same GPU, forward pass:

| config       | ours (ms) | CUDA (ms) | gap       | ours GFLOP/s |
|--------------|-----------|-----------|-----------|--------------|
| tiny         | 1.58      | 0.13      | 11.9×     | 37           |
| small        | 4.75      | 0.21      | 22.7×     | 198          |
| base         | 9.82      | 0.44      | 22.5×     | 2,132        |
| large_l1     | 5.77      | 0.57      | 10.2×     | 9,683        |
| **large** (6L) | **35.07** | **3.67** | **9.6×** | **9,553**  |

`large` = the full compute-bound config (D=1024, ff=4096, 16 heads, 6 layers; `large_l1` is one
such layer). As the work becomes matmul-dominated the picture inverts: our throughput jumps to
**9.5 TFLOP/s** and the gap **more than halves** (22× → 9.6×) and *holds* at full depth. So base's
22× is overhead/small-op-bound (many tiny barrier phases, §14 profiling), NOT a fundamental matmul
deficit — on compute-bound work we are within ~10× at ~9.5 TFLOP/s, and that residual is
cuBLAS-vs-our-tiling (the in-megakernel TF32 path runs at ~10% of native; the tuned standalone
`mm2` TF32 kernel is faster but only fires for pure-matmul programs, §10c/§10b). The honest answer
to "comparable range of performance": **yes on compute-bound layers (~10×, holds end-to-end at 6
layers), no on tiny/overhead-bound ones.** (Full `large` measured after arena reuse §16 unblocked
it — it was a hard LoweringError at 2 layers before.)

## 15. Fixed-trip while: OP_FOR + bytecode unroll (2026-07-16, poc/12)

**Observation**: essentially every `stablehlo.while` JAX emits is a *counted loop*
(`lax.scan`/`fori_loop`: carry k init'd to a constant, cond `arg_k < const`, body returns
`arg_k + const_step`). Data-dependent whiles are rare in practice. When the trip count is known
at compile time, the cond sub-list — and, critically, every *runtime read* of the cond — is
unnecessary; only data dependencies between iterations need synchronization.

**What was built** (`_detect_fixed_trip` in lowering.py; `PJRT_OCL_WHILE=while|for|unroll|auto`):
- **OP_FOR (op 53) / ENT_FOR (0xFFFFFFFB) / TASK_FOR**: body sub-list + trip count in the entry
  (`wait_flag`); the VM frame's `phase` word counts remaining iterations. Persistent engine:
  1 global barrier/iteration instead of 2 + cond phases + per-lane atomic cond read. Host-dispatch
  engine: **no blocking cond read at all** — the whole loop streams into the enqueue ring (the
  per-iteration cond read was the last remaining sync after §11's phase batching).
- **Unroll**: body inlined `trip` times, pure SSA (no carries, no copies), the counter bound to a
  per-iteration const-pool scalar so its add-chain DCEs and cross-iteration fusion applies —
  a fori of `x*1.01+0.5` over 10 steps collapses to ONE affine instruction via `_compose_affines`.
- **auto** (default): unroll iff `trip <= PJRT_OCL_UNROLL_TRIPS` (64) AND
  `trip × est. body result bytes <= PJRT_OCL_UNROLL_ARENA_MB` (256 MB); else OP_FOR.

**Measured** (poc/12 bench, best-of-5; fori-ew = `x = x*a+b` vector a/b; scan-rnn =
`c = c*0.9+xs[t]` stacking ys):
- **NVIDIA (persistent VM)**: FOR = **3.2–3.5×** over WHILE on fori-ew (e.g. 4096×T512:
  27.9 → 7.9 ms), **1.5×** on scan-rnn. Unroll doubles that again where it fits
  (4096×T8: 0.52 → 0.09 ms = **matches XLA CPU exactly**; scan 1M×T8 1.10 ms **beats** XLA's 1.87).
- **PoCL (host-dispatch)**: FOR = 1.1–2.7× on fori-ew (4096×T8: 1.9 → 0.71 ms); unroll up to
  **21×** over WHILE at 4096 (T128: 29.8 → 1.4 ms) and ~2× at 1M×T8 scan (275 → 127 ms, after
  this session's passthrough fix).
- Scan at LARGE n×T is bound by the dynamic_update_slice identity copy (full ys buffer
  re-materialized every iteration — 4096×T512 ≈ 4 GB of traffic dwarfing loop overhead in every
  mode). Next lever for scan: in-place DUS into the loop carry — SHIPPED, see §15a.

**Traps hit** (fixes in this branch):
- Unrolling past ~2 GiB of arena silently misaddresses: buffer offsets are u32 AND bit 31 is
  VMO_IO_BIT — a 512-trip 1M-elem forced unroll returned `inf`. Now a clean LoweringError; the
  bump allocator has no SSA liveness reuse (the M1 "reuse" line item remains unimplemented —
  implementing it would widen unroll's applicable range considerably).
- Outputs are I/O ports: a result buffer nothing writes (trip-0 unroll, passthrough) reaches
  PJRT as garbage. The arena-based validators can't see it — only real-plugin e2e caught it.
- Pre-existing, scan-blocking: `_fuse_views` folded the DUS identity gather into downstream
  readers, orphaning the scatter (DCE'd it → ys returned all zeros; there were NO scan tests).
  Gathers now fold only if their dst has exactly one writer and their src is never written later
  (carries are multi-write). Viewed OP_COPY also dropped its view descriptor in
  `_copy_to_task`/numpy interp.
- Passthrough carries (scan's xs) paid 2 full-length snapshot copies per iteration for nothing —
  now skipped (PoCL scan 1M×T8: 440 → 275 ms before the loop even changes mode).
- A worktree branches from **origin/main**, not local main: the missing local redseg barrier fix
  made PoCL assert `region_entry_barrier != NULL` at plugin init and perfectly impersonated a
  new-kernel-control-flow bug. Merge local main into worktrees before debugging PoCL builds.

**Decision**: `auto` is the default (unroll small, OP_FOR the rest, plain WHILE only for genuine
data-dependent conds). Detection is deliberately narrow (LT/signed, positive const step) —
widen only when a real program shows a different counted shape.

### 15a. In-place dynamic_update_slice into the loop carry — SHIPPED 2026-07-17

The §15 "next lever". DUS is pure (`ys' = operand with slice replaced`), lowered as an
identity-gather copy of the WHOLE operand into a fresh buffer + a scatter of the update row —
so every scan iteration paid O(T·n) traffic for an O(n) write, O(T²·n) per scan.

**Mechanism** (lowering-only; no new opcodes, no kernel/runtime changes): in the while/FOR body
commit phase, when carry k's return value is produced by exactly that pair, the gather's source
is carry k's own buffer (verified structurally: row-major contiguous aux, zero offset, full
length), the DUS out_buf is returned in exactly one slot and never read in the body, and the
carry is read by NOTHING but the identity gather and written by nothing else — then: NOP the
gather, retarget the scatter's dst to the carry buffer, skip both snapshot copies. The pure
semantics collapse to a mutation because no body instr can observe the carry mid-update.

**Measured** (bench poc/12, `results_*_inplace_dus.csv`, identical sig checksums): NVIDIA FOR
4096×T512 42.2→20.4 ms (2.1×), 1M×T8 1.79→1.11 ms and 1M×T32 4.84 ms — both now TIED with XLA
CPU. PoCL FOR 4096×T512 1735→205 ms (8.4×), 1M×T32 3582→237 ms (15×). WHILE mode gains equally.
Unroll does NOT get the fold (unrolled iterations are SSA, each writes a fresh ys) and now
LOSES to FOR on large scans (PoCL 4096×T512: 940 vs 205 ms); the auto arena gate already
steers those to FOR since the estimate counts the full (T,n) DUS result per iteration.

**Trap: the scheduler's no-WAR assumption.** `_depends` modeled only RAW+WAW ("the program is
SSA" — python/NOTES.md A2). Carries are NOT SSA, and it never bit because the ys temp-copy
always forced a phase break between the scatter and the carry copy-backs. Removing that copy
let the in-place scatter's runtime-index read (the counter carry, via reads_hint) share a
barrier phase with the counter copy-back — a genuine cross-lane race, caught immediately by
the schedule simulator's lane-order-independence check (great validator). Fix: WAR edges in
`_depends`. They never fire in a program's SSA bulk (verified 0 phase-count change on a
transformer-ish block; fori/transformer benches unchanged) — only at carry commit points.

**Remaining scan gap vs XLA CPU** (4096×T128: 5.2 vs 1.3 ms) is per-iteration phase overhead
(dynamic_slice + scatter each barrier per row), no longer bulk traffic. That's barrier-elision
territory (§12 / megakernel-survey), not copy elimination.

## 16. Arena is a bump allocator — no liveness reuse (found 2026-07-16, transformer `large`)

**Discovery**: added a compute-bound `large` transformer config (8×256×1024, 16 heads, ff 4096,
6 layers) to test whether the CUDA gap closes when matmul dominates (base is small-op/overhead-
bound, §14). It **crashes** — but not at runtime: lowering raises
`arena 2174157440 bytes exceeds the 31-bit offset space` at just **L=2**. (Bisected: L=1 lowers
& runs correct vs JAX-CPU; L≥2 overflows.)

**Root cause**: `_Ctx.new_buffer` (lowering.py) is a pure **bump allocator** —
`offset = self._arena; self._arena += aligned_size`. Buffer offsets are assigned once at creation
and never reused, so the arena grows with the **sum of every intermediate ever emitted**, not the
**peak live set**. A 2-layer large transformer emits ~236 instrs whose temporaries (attention
scores 33.6 MB, ffn hidden 33.6 MB, plus every EW temp) accumulate to 2.17 GB, past the u32
offset cap (2^31; bit 31 is the I/O-port flag). This is the M1 "SSA liveness for reuse" item —
deferred and never done (the only reuse today is the narrow in-place while-carry + viewfold).

**Not** a resource limit (device max-alloc 23.7 GB, biggest single tensor 33.6 MB) and **not**
the megakernel/barrier (individual large matmuls, softmax, layernorm all run fine; L=1 runs).

**SHIPPED 2026-07-16 (`lowering._reuse_arena`, runs in `lower_module` after
`_compose_affines`/`_fuse_matmul_views`/`_fuse_views`/`_dce_nops`, before the 2^31 cap backstop).**
Buffer IDs are UNCHANGED — only `Buffer.arena_byte_offset` moves; everything downstream keys on IDs
(scheduler patches offsets from the buffer table, runtime/validators read the table). No C++ change.

- **KEY CORRECTNESS INSIGHT — liveness is measured in scheduler PHASE time, NOT program-instruction
  order.** The scheduler runs independent ops in PARALLEL across lanes and inserts a global barrier
  only BETWEEN phases (`_build_levels`/`_phases`). It assumes SSA (each buffer written once) and by
  design adds **no WAR edge** (`_depends` omits WAR). Aliasing two buffer IDs onto one offset
  introduces exactly a WAR hazard the scheduler can't see — so instruction-order liveness would be
  *silently wrong*: an independent producer/consumer pair that lands in the SAME phase runs
  concurrently on different lanes, and the recycled slot's write races the still-live read. The fix
  is to alias only when a **barrier is guaranteed** between the last use of one buffer and the first
  def of the other, i.e. their PHASE intervals are disjoint. The pass recomputes the phase partition
  from the SAME instrs + `PJRT_OCL_FUSE` flag the real scheduler uses (offsets don't affect it, so it
  matches the schedule that will execute) by instantiating a throwaway `_Scheduler` and calling
  `_build_levels(range(main_len))`. First cut used instruction index — caught immediately by
  reasoning about `_cross_lane_dep`; phase time is the corrected model.
- **Algorithm**: per-buffer live interval `[lo,hi]` in phase time (a phase = one entry of
  `_build_levels`; each WHILE/FOR is its own "while" phase). Then offline greedy placement: biggest
  buffer first, lowest 64B-aligned offset whose `[off,off+size)` misses every already-placed buffer
  with an *overlapping* phase interval (inclusive overlap ⇒ two buffers sharing a phase never share a
  slot). O(n²) over a few hundred–thousand buffers — negligible.
- **Regions**: a WHILE/FOR's ENTIRE sub-list (every iteration, nested regions included — expanded
  transitively via the instr's cond/body ranges) and its carries collapse to the region op's single
  phase. So nothing a region touches is reused *within or across* the region. Conservative but safe;
  while/for arenas are tiny anyway. Carry init-copies (root, before the region) + result-aliases
  (root, after) naturally extend the carry interval across the whole region span.
- **Pins**: inputs `lo=0` (non-port inputs are bulk-copied into the arena BEFORE phase 0, so a
  reused slot could otherwise be clobbered by the initial copy-in — this pin is load-bearing);
  outputs `hi=end` (D2H after the program); consts `[0,end]` (uploaded once at load). Zero-copy I/O
  PORTS (bit 31, assigned by the runtime for the first 8 in-XOR-out buffers) ignore the arena offset
  entirely, so pinning + not-relocating them is automatic — but note only 8 ports exist, so the
  `large` transformer's ~53 remaining weight tensors ARE non-port arena inputs (all live from phase
  0), which is the arena's floor.
- **Views**: a folded gather source (§13/§14a) is read by its viewer through the operand's `a`/`b`
  field after the fold, so `_reads_of` already counts it as a read of the SOURCE — its interval
  extends to its last viewer with no special-casing. Verified by the `q @ q.T` viewfold test.
- **Before/after arena (PJRT_OCL_ARENA_DEBUG=1, this machine):** tiny 8.4→2.3 MiB (3.6×),
  base **715.8→105.0 MiB (6.8×)**, `large` (6 layers) **6204→584 MiB (10.6×)** — was a hard
  LoweringError at 2.17 GB @ L=2; now the full 6-layer `large` fits well under the 2 GiB cap.
- **A bug the tests caught**: the golden byte-layout test (`test_golden_layout_jax_lowered_add`)
  asserted `off == i*64` — a bump-allocator artifact. Reuse assigns offsets by interval (the output,
  with the longest span, is placed first), so buffer 0 no longer sits at offset 0. The buffer-ID
  fields (`ADD dst=2 a=0 b=1`) are unchanged and still correct; relaxed the assertion to "offsets are
  a permutation of {0,64,128}, 64B-aligned, in range". This is exactly the right failure — it proved
  offsets moved while IDs stayed stable.
- **Verification matrix (all PASS):** 239 pytest (+5 new `tests/test_arena_reuse.py`: offset reuse,
  peak-vs-sum bound, while-region safety, viewfold-source liveness, offset-in-range — each checked by
  the dual vmreader validators) + 1 skip; runtime_test PoCL+NVIDIA. Transformer `--check` vs JAX-CPU:
  NVIDIA TF32 tiny/small/base/large_l1/**large** all PASS (large max_abs 1.3e-2 = TF32 noise);
  NVIDIA portable megakernel `MEGA_TC=0` base/large **f32-exact** (max_abs 1.2e-5/2.2e-6 — the
  strongest no-corruption signal: an early free gives rel-err ≈ 1.0, §14a); NVIDIA `ENGINE=host`
  small/base f32-exact; PoCL host-dispatch tiny/base f32-exact (2e-7/2e-6). `large` timing: **35.0
  ms/iter (9.6 TFLOP/s)** OpenCL-NVIDIA vs 3.5 ms native CUDA (~10×) — it runs and is correct.
- **Kept conservative** (correct-but-larger beats corruption): whole-region collapse (no reuse
  inside a while/for body); inclusive phase-overlap (a producer/consumer handoff within one phase
  doesn't share a slot); dead (DCE'd, never-referenced) buffers parked at offset 0. None of these
  matter for the `large` arena (weights dominate its floor). A `PJRT_OCL_ARENA_DEBUG=1` stderr line
  reports bump-vs-reuse sizes (env-gated, zero-cost otherwise) — kept as a permanent diagnostic.

## 17. Matmul launch geometry must key on `is_gpu`, not `host_dispatch` (found 2026-07-16)

**Found while profiling `large`** (forced `PJRT_OCL_ENGINE=host` on the NVIDIA GPU to get a
per-phase breakdown): a standalone large matmul crashed with `mm2_pack launch failed`, then —
after a first fix — returned silently WRONG results (max_abs 169).

**Root cause**: `LaunchMatmul` chose its launch geometry from `rt_->host_dispatch()`:
- packed CPU-SGEMM path (pack B panels + 6×16 `mm2p`), and
- register CPU path (`lsz=1, gsz=(M+3)/4`),

both intended for CPU devices, vs. the GPU tiled path (`lsz=256`, tiles×256). But
`host_dispatch_ = !is_gpu || !has_device_fence` (runtime.cc): host-dispatch is the EW-engine
choice, and it is ON for **fence-less GPUs** too. So a GPU without a device fence — or any GPU
forced onto the host engine — launched the mm2 kernel with CPU geometry, which the kernel does not
implement correctly on a GPU. Two failure modes: (1) `mm2_pack`/`mm2p` kernels are only compiled
for non-GPU devices, so the pack path launched a **null** kernel (`launch failed`); (2) the
register path launched but computed garbage (wrong thread→output mapping for GPU).

**Fix**: matmul geometry now keys on `is_gpu()` (added an accessor). GPU devices always use the
GPU tiled geometry regardless of the EW engine; only genuine CPU devices take packed/register.
The packed-scratch alloc is likewise gated on `!is_gpu() && mm_pack_kernel()`. Matmul dispatch is
independent of the EW engine (`mm_ok_ ? LaunchMatmul : …`), so this is safe. **Verified**: GPU
`ENGINE=host` large matmul now max_abs 2e-4 (was 169 / crash); GPU-normal + CPU paths unchanged;
234→235 pytest (added `test_e2e_matmul_host_dispatch`, which forces the host engine and would have
caught this on any GPU CI). This was latent for real fence-less-GPU vendors — exactly the AMD/Intel
portability targets — so it is a genuine correctness fix, not just a debug-path curiosity.

## 18. PoCL barrier-placement portability rule (2026-07-17, merge fallout)

- ❌ The merged collaborative segmented reduce (§14, front 2) crashed PoCL 5.0 at LAZY kernel compile —
  `pocl::Kernel::createParallelRegionBefore: Assertion 'region_entry_barrier != NULL'` — killing
  runtime_test and the e2e subprocess tests (main-process pytest stayed green because the
  crashing kernel was never launched there; NVIDIA/Intel compile the same source fine, and
  upstream's PoCL evidently tolerates it).
- 🔬 Bisected by stubbing tile bodies: the trigger is a **barrier() as the LAST statement of a
  switch case inside vmo_exec_tiles' tile loop** (i.e. immediately before the loop backedge).
  Removed — provably safe here (after the tree's final barrier only lid 0 reads As[0]; every
  tile op re-barriers before reading shared local slots). Early `return`s on paths that precede
  barriers (even workgroup-uniform, spec-legal ones) were restructured to if/else at the same
  time as defense in depth.
- ✅ **Kernel-library rule going forward: in any function inlined into the tile dispatch,
  (a) no `return` on a path that precedes a barrier, (b) no barrier as the final statement
  before the dispatch loop's backedge.** Validate on PoCL (the strictest region-former) before
  merging barrier-bearing kernels; a laptop-green NVIDIA/Intel run does not cover this.

## 19. Fusion pattern → singular fused op (methodology, 2026-07-17)

**The general principle** (established by profiling the transformer `base`, §14b/§18): when an op
sequence's intermediate results are **immediately reduced and broadcast back** — a
reduce → broadcast → elementwise → reduce → broadcast chain — each step is a separate
**cross-workgroup phase** with a full global-memory round-trip. At small tensor sizes every phase
is latency-bound (~28–30 µs floor: the barrier + memory latency can't be hidden), so a 7-phase
layernorm costs ~0.21 ms while cuBLAS/XLA fuse it into ~1 kernel at ~7 µs (**~30× gap**). This —
NOT matmul — is the dominant cost on realistic (base-scale) transformer workloads: our small
matmuls are already competitive/faster than CUDA; the loss is entirely in layernorm/softmax/gelu
running at ~1–4% of memory bandwidth (component profile in `docs/` / session notes).

**The fix pattern**: RECOGNIZE the fixed idiom and lower it to a SINGLE fused megakernel op that
does the whole computation **in local memory with one global read + one global write** — the
workgroup-per-segment collaborative pattern already used by `vmo_redseg_tile` (§14 front 2). A
`seg`-wide row is staged into local once; all reduces (max/sum/sumsq) run as local tree-reduces;
the normalize is applied and written back — zero intermediate global buffers, one phase instead of
five to seven.

**How to add a new fused op** (the reusable recipe — apply next time a workload shows a
reduce+broadcast idiom eating phases, e.g. RMSNorm, logsumexp, log_softmax, GroupNorm, attention's
scale+softmax):
1. **Identify the idiom** and confirm it reduces over the **innermost (suffix) axis** — that's what
   the segment model (`OP_REDUCE_SEG`/`TILE_RED_SEG`) already tiles as workgroup-per-segment.
2. **Add a tensor opcode** (`OP_*`) + **tile-op** (`TILE_*_SEG`); `imm`/`imm2` carry seg size + any
   scalar param (eps). Params that are per-channel vectors (layernorm's `*g+b`) stay as separate
   EW — they fuse cheaply via §11/§13; keep the fused op to the phase-heavy reduce core.
3. **Kernel**: clone `vmo_redseg_tile`'s structure — stage segment to `__local`, tree-reduce,
   compute, write once. MUST follow the §18 PoCL rules (no `return` before a barrier; no barrier as
   the last statement before the tile-loop backedge; `valid` guard for over-assigned tiles).
4. **Recognize + rewrite**: a lowering pass detects the idiom and emits the single op. Prefer a
   post-lowering peephole on OUR VM-instr stream (robust to StableHLO/jaxlib variation — everything
   funnels through `OP_REDUCE_SEG` + viewed EW) over matching raw StableHLO; GATE it hard and fall
   back to the decomposed path on any mismatch (never wrong, only sometimes-unfused).
5. **Wire** scheduler `n_tiles` (= n_out segments), numpy interp + schedule-sim validators, and add
   dual-validator tests. **Verify**: phase-count drop, component-ms drop, transformer `--check`
   still exact on all devices, full-model ms win — keep only if it moves the needle (§14a rule).

**Expected payoff**: layernorm ~7→~2 phases, softmax ~5→~1; on `base` these two ops are ~3.8 ms of
9.7 ms today, so the ceiling is large. `gelu` is pure-EW (no reduce) and should already chain-fuse
(§11) — if it doesn't, that's a chain-fusion gap, not a new fused op.

### 19a. SHIPPED 2026-07-17: OP_SOFTMAX + OP_LAYERNORM (the first two fused norms)

Implemented both per the recipe above. **Kept ON by default; `PJRT_OCL_FUSE_NORM=0` reverts.**

- **Opcodes/tile-ops**: `OP_SOFTMAX`(54)/`OP_LAYERNORM`(55) in lowering.py;
  `TILE_SOFTMAX_SEG`(11)/`TILE_LAYERNORM_SEG`(12) in scheduler.py; `TOP_*`/`kTop*` (kMaxTileOp→12)
  in vm_common.cl/runtime.h. `imm`=seg, `n`=n_out (like OP_REDUCE_SEG); layernorm `imm2`=eps f32 bits.
- **Recognizer** `_fuse_norm` (lowering.py, runs after `_dce_nops`, before `_reuse_arena`; a second
  `_dce_nops` cleans the dead intermediates). Post-lowering peephole on OUR instr stream, anchored on
  `OP_REDUCE_SEG`: **softmax** = MAX-redseg → `sub(x,·)` → `exp` → SUM-redseg → `div(exp,·)`;
  **layernorm** = SUM-redseg → `div(·,seg)`=mean → `sub(x,mu)` → `mul(sq)` → SUM-redseg →
  `div(·,seg)`=var → `affine(var,1,eps)` → `pow(·,-0.5)` → `mul(x-mu,·)`. A `bcast_src` helper
  walks OP_GATHER_STRIDED producers so it matches whether the broadcast is a leftover gather or a
  folded view. **Gated hard** on every op kind + producer→consumer linkage + `seg<=1024` (local
  staging) + the affine scale==1 / pow exp==-0.5 / divisor==seg consts; any mismatch → decomposed
  path untouched. Rewrites the FINAL op in place to the fused op reading X; the rest DCE. **The
  trailing per-channel `*g+b` is left as separate EW/gather** (it chain-fuses cheaply; the win is the
  reduce core). Only 2 kinds needed (softmax MAX+SUM, layernorm SUM+SUM). Idiom variation actually
  seen: softmax keeps a materialized reshape-gather between the reduce and the broadcast-view (the
  `keepdims` (…,1) reshape) that `_fuse_views` can't fold because the EW slot is already viewed —
  `bcast_src` handles it; layernorm's broadcasts are all folded views (no leftover gather).
- **Kernels** `vmo_softmax_seg`/`vmo_layernorm_seg` (reduce.cl), cloned from `vmo_redseg_tile`:
  stage the seg row into `__local As` once, tree-reduce in `Bs` (softmax: max then sum; layernorm:
  sum+sumsq one pass, **var = E[x²]−E[x]²**), write once. eps = `as_float(t.p2)`, `rsqrt(var+eps)`.
- **PoCL bug found + fixed (the whole debugging cost of this task).** A **divergent-trip-count**
  grid-stride loop (`for(j=lid;j<seg;j+=lsz)`, lanes doing different iteration counts) that
  **follows a barrier and does a GLOBAL store** is miscompiled by **PoCL 5.0's work-item-loop /
  parallel-region former** → intermittent **heap corruption**: crash (`free(): invalid pointer` /
  `POclReleaseMemObject refcount>0`) when the store target is an I/O port, silent wrong values when
  it's the arena — **non-deterministic, ~50% of runs** (5/5 decomposed stable, ~half of fused
  crash). NVIDIA (both engines) and the dual Python validators are all correct — it is purely
  PoCL-CPU codegen. Bisected: staged-copy+barrier+**multi-lane** port store crashes; +**lid-0-only**
  store is stable (this is why `vmo_redseg_tile`, which stores from lid 0, never hit it); a
  **uniform-trip** multi-lane store (round the bound up to `ceil(seg/lsz)*lsz`, guard with `j<seg`)
  is stable AND keeps the store parallel. Fix = `SEG_UNIFORM(seg,lsz)` on **every** post-barrier
  seg-loop (verified 16/16 stable). This is a new, sharper instance of the §18 PoCL barrier rule:
  **post-barrier grid-stride loops must have a work-item-UNIFORM trip count.**

**Before/after (this machine; A/B via `PJRT_OCL_FUSE_NORM` on the SAME build):**

| metric                              | fused OFF | fused ON | note |
|-------------------------------------|-----------|----------|------|
| barriers: layernorm (4,128,512)     | 7         | **2**    | 2 = fused op + trailing `*g+b` |
| barriers: softmax (4,8,128,128)     | 5         | **0**    | single op, no cross-wg barrier |
| standalone layernorm (4,128,512) ms | 0.238     | **0.102**| 2.3×; CUDA 0.091 (→1.1× gap, was ~30×) |
| standalone softmax (4,8,128,128) ms | 0.158     | **0.043**| 3.7×; **faster** than CUDA's 0.058 |
| **base transformer ms/iter (TF32)** | 9.73      | **7.34** | GFLOP/s 2151→2852 |
| base gap vs native CUDA (0.433 ms)  | 22.5×     | **17.0×**| |

**Correctness (all PASS):** 262 pytest + 1 skip (+10 `tests/test_ops_fused_norm.py`: fires, dual
validators, `FUSE_NORM=0`/`seg>1024` fallbacks) + runtime_test PoCL & NVIDIA. transformer `--check`:
NVIDIA TF32 tiny/small/base/large_l1/large; NVIDIA `MEGA_TC=0` base/large **f32-exact** (2.2e-6 /
1.3e-5 — layernorm's one-pass var, NOT TF32 noise); PoCL Portable tiny/base **f32-exact** (5e-7 /
2.4e-6), stable across repeated runs.

**Surprised by**: how large the standalone win is (softmax now *beats* native CUDA on this shape,
layernorm ~1.1× — both were ~30× off) and that the base end-to-end still improves 1.33× despite the
matmuls being untouched. **Kept conservative**: recognizer gates on the exact TF32-era jaxlib 0.10.2
idiom; `*g+b` left separate; `seg<=1024`. gelu was NOT touched — it is pure-EW and already
chain-fuses (§11), consistent with the §19 note; no new fused op needed there.

## 20. dynamic_slice start scalars: aux byte offsets must be LOADER-patched (found 2026-07-17)

**Found while** reworking the per-op benchmark (§21): `lax.dynamic_slice(x, (k,), ...)` with a
runtime `k` passed as a *program input* silently sliced at offset 0 on the real device (both
engines, PoCL and NVIDIA), and a 16-link chained version segfaulted PoCL. All 240 pytest
validators passed throughout.

**Root cause — two independent invalidations of the same design.** The dynslice handler recorded
each start scalar's `arena_byte_offset` into the aux pool at lowering time
(`idx_byteoff[rank]`, read by `vmo_dyn_base` on device). That value is doomed twice:
1. `_reuse_arena` (§16) reassigns EVERY buffer's arena offset after handlers run — the docstring
   premise "byte offset is fixed at allocation time" died when §16 landed. Any arena-resident
   index scalar could end up read from its pre-reuse offset.
2. A start scalar that is a program input may be assigned an I/O PORT at load time
   (`runtime.cc` `assign_port`, zero-copy `cl_mem` — never in the arena at all). The recorded
   arena offset then points at memory nothing ever wrote (→ base 0 or garbage → OOB segfault
   under PoCL).

**Why validators never caught it**: the numpy validators address start scalars by BUFFER ID
(`idx_bufid[rank]`, carried in aux for exactly that purpose) — only the real device consumed
`idx_byteoff`. This is the same blind spot as the §15 "outputs are I/O ports" trap: anything
arena-offset-shaped is invisible to arena-based validators; only real-plugin e2e sees it.

**Fix (chosen): patch aux at LOAD time, single source of truth.** The loader already resolves
buffer id → arena-offset-or-port-handle for task dst/a/b (`elem_off`); it now also walks dyn
gather/scatter tasks and rewrites `aux[idx_byteoff[d]] = elem_off(aux[idx_bufid[d]])` before
uploading the aux buffer. The kernels read the scalars through `AP()`, which already resolves
bit-31 port handles, so ports work with no kernel change. The Python handler writes placeholder
0s. Rejected alternatives: (a) lowering-side post-pass after `_reuse_arena` — fixes staleness
but cannot know port assignment without duplicating `assign_port` in Python (silent-corruption
coupling); (b) copying ported scalars into the arena via an extra 1-element instruction —
correct but pollutes every dynamic_slice with instruction overhead.

**Rule going forward**: an aux word that names a buffer LOCATION must be patched by the loader
from a buffer id; lowering-time offsets are only valid for things `_reuse_arena` doesn't move
(shapes, strides, trip counts). Verified: previously-failing repros + full pytest + PoCL/NVIDIA
device runs of chained dynamic_slice.

**Unrelated observation recorded while validating** (not a bug): on NVIDIA, f32 matmul runs on
the tf32 tensor-core WMMA path (`-DVMO_NV_PTX`, §14) — a single 64x64 matmul shows median rel
error ~8e-4 vs f64 (tf32 mantissa = 10 bits, 2^-11 ≈ 5e-4), where PoCL shows ~1e-7 (true f32
FMA). Same trade cuBLAS/XLA make by default on Ampere+; worth remembering when comparing
against references with tight tolerances.

## 21. Per-op benchmark: in-program op chaining to kill dispatch noise (2026-07-17)

`tools/plot_bench.py` timed one op per `jit` call; at small N the measurement was dominated by
per-call noise (python dispatch, PJRT execute overhead, launch latency — µs-scale, same order
as the kernels), giving jagged curves and run-to-run swings.

**Rework**: every benchmarked function now applies its op CHAIN=16 times inside ONE jitted
program as a data-dependent chain (link i consumes link i-1's output; matmul B-matrices scaled
by 1/sqrt(n) so values stay finite), and the reported time is call_time/CHAIN, min-of-7-rounds
with iteration counts auto-calibrated so every timed round lasts ≥50 ms (timer resolution).
Host-side per-call overhead is amortized 16x; both backends execute one program containing 16
real op instances — for ours that's 16 VM instructions (the megakernel's actual regime), for
XLA 16 kernels/thunks in one executable.

**The load-bearing detail is `stablehlo.optimization_barrier` between links**: without it XLA
fuses/CSEs a repeated elementwise chain into far fewer kernels and the CUDA side reads ~16x
faster than reality. Our lowering now supports the op as pure buffer aliasing (zero
instructions — we have no cross-op optimizer to fence), so it is free on both sides.

**Trap 1: a barrier stops CSE/fusion but NOT dead-code elimination.** First gather version
threaded only the (unchanged) offset through the barrier; each link's slice output fed nothing,
so BOTH backends DCE'd 15 of the 16 slices and the panel read a physically impossible
2.7-4.3 µs/op at N=16M (>10 TB/s). Sanity-check every chained panel against bandwidth
arithmetic. Fix: the next offset must GENUINELY depend on the previous slice's data —
`k += (y[0] * z).astype(int32)` where `z` is an opaque zero (`optimization_barrier(0.0)`
hoisted out of the loop), which no simplifier can fold.

**Trap 2: XLA looks THROUGH the consumer.** With the data dependency alone, XLA rewrote
`y[0]` as a 1-element slice of `x` (slice-of-dynamic-slice simplification) — the k-chain then
ran on tiny kernels and the big slices were dead again (CUDA still ~3 µs/op flat at 16M). Fix:
pass `y` through `optimization_barrier` BEFORE indexing it; the barrier forces `y` materialized
and simplifications cannot look through it. After both fixes, CUDA lands at ~12 µs/op at 16M —
which IS physically sane: `x` is 64 MB and fully L2-resident (128 MB L2 on Blackwell), so the
~5.4 TB/s effective is L2, not HBM, bandwidth. Ours ~30-50 µs/op (per-instruction floor ~5
instrs/link, then bandwidth). Corollary: a data-dependent chain makes per-op time honest only
if the consumed data is (a) genuinely needed and (b) barrier-shielded from producer fusion.

Measured with the rework (NVIDIA, 2026-07-17): run-to-run deviation of ours-column medians
0.1-0.8% per panel (max 5%); the old one-op-per-call method also carried a ~2x BIAS at small N
(29-39 µs/call where the op itself is ~15 µs — the rest was python/PJRT dispatch, now /16).

Chaining is also what surfaced the §20 dynslice bug — repetition-within-program is a better
correctness probe than one-shot calls; keep using it.

## 22. Per-tile execution was latency-bound, not dispatch-bound (2026-07-17, chained-bench fallout)

The §21 noise-reduced bench showed severe flat gaps vs CUDA: EW add/mul ~15 µs/op for ANY
N in 16K..2M (CUDA flat ~3.3 µs), gather ~30 µs, matvec 27-46x, while-loop 2.6x above 512K.
Diagnosis (probe: per-op time scales with per-thread loop trip count, not N): the megakernel's
per-INSTRUCTION cost was fine (~3-5 µs at 1 tile — near CUDA's launch floor), but one EW tile =
one workgroup running a scalar stride-256 loop → 64 dependent global-memory round trips per
thread (~230 ns each ≈ 15 µs, 13 GB/s per lane), and TILE_SIZE=16384 caps an op's parallelism
at N/16K workgroups — a 2M-element op used 128 of 376 lanes; mid-size ops used a handful.
Bytecode dispatch itself was NOT the problem — don't optimize the interpreter loop for this.

Fixes (all shipped, all portable core-OpenCL):
1. **f32 EW vector fast path** (ew.cl): float4 lanes + 2x manual unroll (8 independent wide
   round trips instead of 64 serial scalar ones) for bin/un/affine/fill when all operand
   pointers are 16B-aligned; scalar tail covers the last tile. GPU-only (#else of
   VMO_CPU_TILES); the CPU float8 chunk path is untouched. add 16K..262K: 15.1 → 3.1 µs
   (CUDA 3.3); while-loop 1M: 466 → 141 µs — now BEATS CUDA (181, launch-bound) at every N.
2. **Device-tuned EW tile size**: EW_TS is now a -D at program build chosen in runtime.cc
   (GPU 4096, CPU 16384) and advertised to the scheduler via PJRT_OCL_EW_TS (plugin.cc env,
   scheduler.TILE_SIZE reads it) — kernel and host tiling MUST agree; CalBuilder takes it as a
   parameter. 4x lane parallelism at equal work; PoCL keeps 16384 (~300 µs/tile host overhead
   makes more tiles pure loss there).
3. **dynamic_slice/update contiguous GPU fast path** (dynslice.cl): the generic body pays a
   ~20-cycle serial div/mod chain PER ELEMENT even for rank-1 stride-1 slices; added the
   4-wide 2x-unrolled copy twin of the existing CPU float8 path (vloadn tolerates the
   runtime-odd base). Static gather.cl got the same rank-1 fast path. gather bench 29 → 12-16 µs;
   the remaining 3x vs CUDA is the per-link scalar-index chain (tiny ops + barrier phases) —
   that is survey R1/R2 (scoreboard/fusion) territory, not tile code.
4. **GEMV routing** (ops/dot.py _dot_to_task): dot_general with N==1, G==1, no folded views
   now emits TILE_RED_SEG in dot mode (p3=1, vector operand in t.b; out[o] = dot(row o, x))
   instead of TILE_MMA — the 64x64 MMA tile wasted 63/64 of its work at N=1 and serialized
   the K loop. One row per tile, whole-workgroup coalesced loads (float4 + dot() on GPU),
   M-way parallel. matvec 1024: 101 → 8.2 µs (12x; CUDA 3.7). CPU keeps a FLAT single-loop
   accumulate body: the branchier vectorized CFG re-triggered the §18 PoCL 5.0
   region_entry_barrier assert — dot mode folds into the existing loop via ternary there.

Verified: 252/252 pytest on NVIDIA AND PoCL; chained bench + transformer rerun (see README).
Residual gaps after this: matmul tile ceiling (survey R3), small-op/barrier-chain overhead
(survey R1/R2). EW at 2M sits ~2x off CUDA's L2-resident chain speed (7.7 vs 3.5 µs) — a
deeper unroll may close some; diminishing returns vs R2 fusion.

## 23. FUTURE DIRECTION — register-resident map-region fusion (the base 13× lever)

**The gap that remains** (§14b/§19 profiling): `base` runs **107 serial barrier-phases** for 192 VM
ops; ~5.8 ms, 13.4× off native. Per-op kernels are competitive/faster than CUDA *standalone* — but
the whole is far behind because CUDA fuses a layer into a handful of pipelined on-chip kernels while
we run ~100 discrete phases, each with a cross-workgroup barrier AND a global-memory round-trip.

**Key finding (2026-07-19):** our current "fusion" removes the *barrier*, not the *round-trips*.
Every EW tile-op still does `AP(t.a)` → compute → `AP(t.dst)` — global in, global out. §11 chain
fusion just lets a chain `[tid0,tid1,…]` run barrier-free on one lane; each op STILL materializes its
full intermediate to the arena. A 6-op GELU chain = 6 global writes + 6 reads. We eliminated the
sync, not the DRAM traffic — which is exactly why per-op wins don't compound.

**The lever (user idea, 2026-07-19): make bytecode ops pass intermediates in REGISTERS and work on
much smaller sets.** A general "fused map-region" tile-op:
- Lowering/scheduler finds maximal **map-regions** — contiguous runs of ops whose output tile depends
  only on the input tile at a static position (elementwise, broadcast-reads, transpose/reshape views).
  Boundaries = genuinely cross-lane ops (reduce, matmul, gather/scatter, dynamic index).
- Each region → ONE tile-op run as a **per-tile register pipeline**: load the region's inputs into
  registers/local ONCE, interpret the region's op sub-list on that scratch (a straight-line
  mini-opcode loop, like `while` w/o branch), store outputs ONCE. Intermediates never leave registers.
- K ops + K round-trips + K−1 barriers → **1 load + 1 store, 1 phase**. Most of the 107 base phases
  are exactly these map-ops. This is XLA's kLoop/kInput fusion, done inside our VM, and it GENERALIZES
  the hand-fusions we already ship (§11 barrier, §13 views, §19 norm) into one mechanism.

**Constraints (honest):** only map-ops fuse into the pipeline (reduce/matmul/gather stay boundaries,
though a reduce can fuse its *surrounding* EW as §19 does); register pressure bounds tile-size ×
chain-length (tile size becomes a tuning knob — spills kill it); the VM gains a "region" opcode
whose operand is a straight-line sub-list interpreted over a local tile buffer (a small fusing
tile-executor in the megakernel). Divergent/data-dependent-index ops can't join the pipeline.

**Status: not started — this is the documented next major architecture project** for the base
(overhead/memory-bound) regime, distinct from the matmul intensity-cap lever (§10c, large regime).
Measurement-first when taken up: rank the 107 phases by cost, fuse the fattest map-regions first.

## 19b. LOGGED FOLLOW-UP — OP_GELU as a dedicated fused opcode (implement later)

`gelu` is pure-elementwise (`0.5*x*(1+tanh(0.7978*(x+0.044715*x³)))`, ~6 map-ops) — no reduce, so it
does NOT need the §19 collaborative-reduce treatment; §11 chain fusion already runs it barrier-free.
BUT (§23 finding) those 6 ops still each round-trip their intermediate through global memory, so gelu
measured ~2.3× off CUDA at base sizes. Two ways to fix, both LOGGED for later:
- **Dedicated `OP_GELU`** (§19-style): recognize the gelu idiom → one opcode that computes the whole
  tanh-approx in registers per element, one global read + one write. Trivial once the recognizer
  pattern is written; guaranteed win; but one-off.
- **Falls out of the general map-region fusion (§23 / Idea A)** for free — gelu is exactly a pure-map
  region. If the region-op lands, a dedicated OP_GELU is redundant.

**Decision: don't hand-write OP_GELU yet** — let the general region-op (§23, being implemented) subsume
it; only add a dedicated OP_GELU if the general mechanism stalls or gelu needs to ship sooner.
**UPDATE (§24, then SHIPPED §26):** the poc/14 gate INVERTED this — at base the general op wins only
1.3× and carries an occupancy-regression risk, so the dedicated OP_GELU was the right near-term lever.
Shipped 2026-07-19; see §26.

## 24. Fused map-region tile-op — DESIGN + PoC gate → SPECCED, NOT BUILT (2026-07-19, poc/14)

**Outcome up front:** the PoC gate (poc/14) says GO-but-size-dependent, and the megakernel
integration analysis says the *general interpreted* region-op is the wrong near-term lever — ship
dedicated fused opcodes (OP_GELU) instead, and defer the general op to the R3 VM-split. Design +
evidence below; "Integration findings" at the end is the decision.

Specs §23 / ideas-for-v2 Idea A: a **fused map-region tile-op** that keeps a run of
pure-map ops' intermediates ON-CHIP (one global load per region input + one store per output)
instead of round-tripping every EW intermediate through the arena. This is the memory-traffic
lever §11 chain fusion left on the table (it killed the barrier, not the round-trips).

### PoC gate (poc/14-map-region) — GO, size-dependent

Hand-emitted the GELU tail (tanh approx, **9 pure-map micro-ops reusing x 4× — a real DAG**) as a
single region interpreted over on-chip scratch, vs the faithful "today" path (9 **vectorized**
float4 EW passes in ONE launch = §11 chain, round-tripping each plane through global). Same
micro-op program, numerically identical (maxerr 2.95e-7, pure f32). Fair delta =
float4-vectorized fused region vs vectorized 1-launch chain:

| n (f32)        | regime            | NVIDIA fused speedup | note |
|----------------|-------------------|----------------------|------|
| 262 144 (1 MiB)| overhead-bound    | ~1.0× (wash)         | too small; dispatch-bound |
| 1 048 576 (4 MiB) = **base FFN GELU** | **L2-resident** | **1.3×** | 4 MiB fits the 128 MB L2 |
| 16 777 216 (64 MiB) | HBM-bound    | **3.1–3.5×**         | round-trips now hit HBM |
| 67 108 864 (256 MiB)| HBM-bound    | **3.2×**             | reg variant ≈ hardcoded gelu ceiling |

PoCL CPU @1 MiB: **2.85×** (round-trips bite even at base size — cache pressure).

**Four findings that shape the design:**
1. **GO, but the base-size win is modest (1.3×) because base is L2-resident on this GPU; the
   dramatic 3×+ is HBM-bound.** The microbench also UNDERSTATES the real win — it has zero VM
   per-instruction dispatch overhead, whereas the real megakernel pays bytecode-dispatch +
   grid-stride setup for EACH of the 9 EW ops (§22: ~3–5 µs/instr); collapsing 9→1 banks that too.
   So the end-to-end base-transformer gain will be real but single-digit-% on the EW component;
   the big payoff is larger/HBM-bound workloads (`large`, big batch) and the phase/instr-count drop.
   Per §14a: keep only if it moves the needle end-to-end — MEASURE on the real model before ON-by-default.
2. **Vectorization is load-bearing.** The *scalar* interpreter is a wash even vs the multi-launch
   chain (dispatch-bound). The region tile-op's inner loop MUST be float4-vectorized (float4 scratch
   slots, 4 elems/iter) — that alone turns the wash into the win.
3. **Local-staging is the portable default (step 1).** Switch-addressed registers (step 2) slightly
   edge local on NVIDIA at large N but LOSE on PoCL CPU (switch dispatch is costly there); `__local`
   wins or ties everywhere and is simpler. Registers are a per-device tuning, not needed to bank it.
4. **Interpreter ceiling is adequate** — reg variant reaches hardcoded-gelu speed at 256 MiB. The
   bottleneck is memory traffic, not interpretation. No JIT / device-compile needed.

### Design (as specced; single-output v1)

**Opcode.** `OP_MAP_REGION` (56) in lowering; `TILE_MAP_REGION` (13) in scheduler/kernel. The
region is variable-length ⇒ its descriptor lives in the **aux pool** (Instr.aux word offset); the
32-B Instr carries `dst` (region output buf), `n` (element count → n_tiles), `aux` (descriptor off).
Input buffer ids ride in `Instr.reads_hint` (non-serialized) so the scheduler's dep analysis sees
them without parsing aux (precedent: dynamic_slice start scalars, §20).

**Aux descriptor layout** (u32 words):
```
[0]=n_inputs  [1]=n_micro  [2]=n_slots
n_inputs × { buf_id, view_aux_off(+1; 0=direct), dst_slot }   # prologue loads (view = §13 access-map)
n_micro  × { kind, dst_slot, a_slot, b_slot, s_bits, t_bits } # straight-line SSA sub-block
[last]   = out_slot                                            # epilogue store → Instr.dst
```
Operands INSIDE the region reference **scratch slots**, not arena ids (the §23 reason a per-opcode
"chain flag" can't work — real regions are DAGs, GELU reuses x 4× / binary ops need 2 live inputs).
`kind` = the vmo_ew micro-op (MUL/ADD/SUB/DIV/…/AFFINE a*s+t/TANH/EXP/…). n_slots bounds region
width; gate `n_slots ≤ 8` (register-file bound, PoC-validated) + n_inputs small.

**Kernel** `vmo_map_region` (new ops/region.cl): float4-vectorized grid-stride tile loop; per
float4 index — load each region input into its slot (`R[slot][lid]`, applying the §13 view address
for broadcast/transpose leaves), interpret the micro sub-list over `__local float4 R[NSLOTS][WG]`,
store `R[out_slot]` to `Instr.dst`. **No cross-workgroup barrier** (pure map — element i independent),
so §18/§19a PoCL barrier rules DON'T bite here (no barrier at all); works on host-dispatch engine
unchanged. On-chip scratch = NSLOTS×WG×16 B (8×256×16 = 32 KB local) — sits within occupancy for the
spin-barrier (§10c): measure residency, shrink WG/NSLOTS if it drops below the barrier's need.

**Recognizer** `_fuse_region` (lowering, after `_fuse_views`/`_fuse_norm`, before `_reuse_arena`):
find maximal connected runs of viewable-f32-EW instrs (the §11 chain set) with a SINGLE
externally-live output; encode the micro-program (topo order = SSA order), map leaves→input slots
(carrying any folded view aux-off), NOP the members, emit one OP_MAP_REGION. **Gated HARD** (all-f32,
n_slots≤8, single output, no cmp/select/convert/fill/dynamic index); any mismatch → decomposed path
untouched. `PJRT_OCL_FUSE_REGION=0` reverts. Multi-output regions (residual forks) = a future step.

**Validators.** vmreader numpy interp (execute the micro sub-list over ndarrays) + the schedule-sim,
per the §19 dual-validator net; add region tests (fires / FUSE_REGION=0 fallback / n_slots>8 fallback).

### Integration findings — STOP on the general op, ship dedicated fused ops instead (2026-07-19)

Designing the megakernel integration surfaced two things the standalone PoC (full occupancy, no VM
dispatch) hides, and they flip the near-term decision. **The general interpreted region-op is NOT
built; hard gate stays a design.** Reasons, measured:

1. **At base size the interpreted region is only 1.3×, but a DEDICATED fused op is ~4×.** The PoC's
   `gelu_hard` (hardcoded, fully inlined, float4 — i.e. a §19-style dedicated `OP_GELU`) hit
   **0.0036 ms @base vs the chain's 0.0149 (4.1×)**, where the *interpreted* region managed only
   1.3× (its per-op dispatch + runtime-slot indexing eats the round-trip saving at L2-resident
   sizes). So for the base regime, **dedicated fused opcodes beat the general interpreter** — which
   INVERTS §19b's "let the general region-op subsume OP_GELU": at base sizes the interpreter's
   overhead is the very thing that makes the general op lose to the one-off. The general op's value
   is *generality* + the *HBM-bound 3×*, not base-size speed.
2. **Megakernel integration has an occupancy/register-pressure cost that regresses ALL ops.** A
   float4 local-staging region scratch (8 slots × 256 lanes × 16 B = 32 KB SLM) blows the local
   budget — and §10c already measured the TF32 megakernel sitting *exactly at* the 376-lane
   residency boundary (MMA_ASZ/BSZ ≈ 10 KB each), so ANY added SLM pushes residency below the cap
   and shrinks every other op's lane count. The register-switch variant (region_reg4, zero SLM,
   fastest on NVIDIA at large N) sidesteps SLM but adds ~32 vector registers to the *shared*
   translation unit → raises the whole megakernel's register count → same occupancy hit by a
   different door (CLAUDE.md's "watch register pressure as the op library grows; split the VM by op
   family"). Either way the monolithic-megakernel integration is **entangled with the R3 "split VM
   by op family" architecture** (megakernel-survey) — a much bigger, riskier change than the
   standalone PoC implies. Local staging is only free of this on the **host-dispatch (CPU) engine**
   (no spin-barrier, no co-residency) — where the PoC already shows local > registers (2.85×).

**Decision (this session): do NOT ship the general interpreted region-op now.** Per §14a
("don't build the general mechanism on a false premise") and CLAUDE.md's measure-don't-assume /
correctness-first rules, the PoC gate did its job — it revealed that at the *target* (base) regime
the general op wins only 1.3× AND carries a broad-occupancy-regression risk, while the cheaper,
lower-risk lever (dedicated fused opcodes, §19/§19b recipe) wins ~4× at base with near-zero risk.

**Recommended sequence instead:**
- **Near-term (low-risk, base-regime): dedicated fused opcodes for the hot map-idioms** — `OP_GELU`
  first (§19b, exactly the PoC's `gelu_hard` as one EW-unary case + a recognizer), then any other
  measured idiom. One extra dispatch case each, no SLM, negligible register cost, ~4× on its
  component at base. This is the §19 methodology, now PoC-justified over the general op for base.
- **Later (the general interpreted region-op): pursue together with R3 (split the megakernel by op
  family)** so the region interpreter gets its own kernel + register/SLM budget without taxing the
  EW/MMA path, and target it at the **HBM-bound / `large`** regime where the PoC shows 3×+. On the
  **CPU host-dispatch engine** it can use local staging and land sooner (no co-residency bind).

poc/14 + this section are the decision basis; the recognizer/region-op/VM-loop remain specced-but-
unbuilt above for when R3 makes the integration cheap.
## 25. Async / prefetched DRAM loads to hide tile latency — EXPLORED (2026-07-19, poc/13)

§22 diagnosed per-tile execution as latency-bound (a tile = one workgroup grid-striding a loop of
dependent global round-trips). The textbook fix is async/prefetched loads: issue tile N+1's loads
while computing tile N. poc/13 tests this measure-first on the two representative loop shapes, three
variants each, on NVIDIA + PoCL. All variants bit-exact (`maxerr=0`). Verdict: **a verified
near-negative — register double-buffering is a small real lever, `async_work_group_copy` is a
portability trap; do NOT add async to the VM.**

**Loops:** (A) streaming EW `d=a*s+t` (§22 headline, no reuse); (B) matmul K-loop global→local stage
(64×64 tile, BK=16, 256 lanes, 4×4 microtile — mirrors the shipped `vmo_mma_tile`, 8 KB As/Bs).
**Variants:** (a) baseline direct loads; (b) manual double-buffer = prefetch next tile's globals into
**registers** while computing current; (c) `async_work_group_copy` into `__local` + `wait_group_events`.
Persistent-grid faithful (grid = 2·CU, grid-stride). EW GB/s are L2-resident (buffers ≤64 MB reused
across reps) → read as a *relative* ranking, not HBM bandwidth.

**NVIDIA RTX PRO 6000 Blackwell (188 CU, grid 376):**
- EW (GB/s): scalar → **reg-DB** → async. 256K 203→**256**→184; 1M 799→**925**→699;
  4M 2061→**2544**→1570; 16M 3212→**4641**→2321. reg-DB is **1.25–1.45×**; async is a **~30 % loss**.
- MMA (GFLOP/s), single / reg-DB / async: 512³ 1601/**1831**/249; 1024³ 3676/**3874**/896;
  2048³ 4954/**5226**/1197. reg-DB is a consistent **5–14 %**; **async is 4–15× SLOWER**.
- Occupancy (poc/08 handshake): 8 KB and 16 KB `__local` BOTH → 752 co-resident groups (≫ 376 cap).
  Local footprint isn't the binding constraint; and the winning reg-DB uses **no extra local** (regs
  are the 2nd buffer) — only ~8 extra registers.

**PoCL (Ryzen 3900X CPU, 24 CU, cap 48):** async is the *opposite* — fastest for EW (lowers to
`memcpy`: 1M 10.5→30.9 GB/s, 4M 14.1→26.5), a wash/slight-loss for MMA (10–20 % slower). Occupancy
24 groups at both footprints (512 KB local).

**What worked / no-op'd / regressed:**
- `async_work_group_copy`: **device-polar.** NVIDIA's OpenCL runtime has no async DMA engine — it
  emulates the copy serially, so the chained per-row/col staging (TM+TN=128 calls) collapses matmul to
  0.25 TFLOP/s. PoCL lowers it to `memcpy` and wins. A core-path feature that is 15× faster on one
  device and 15× slower on another is **unusable portably** (violates the "no vendor-poison in core"
  rule, like the §9 `2×CU` heuristic and §18 barrier placement).
- Register double-buffering: the real lever. For EW it IS **already shipped** — the §22 float4+2×-unroll
  fast path is exactly register-level prefetch (8 in-flight loads); nothing more to do for streaming.
  For the MMA K-loop a reg-prefetch double-buffer is a genuine but modest 5–14 %.
- `prefetch()` builtin: not separately benchmarked — it's a hint the NVIDIA ICD ignores and PoCL treats
  as a no-op; the reg-DB variant already realizes the same "loads in flight early" effect explicitly.

**Occupancy/§10c note:** doubling `__local` 8→16 KB did NOT drop co-residency below the 376 cap here,
so a *local*-based double buffer would be affordable on this GPU — but the better design (reg-DB) needs
no local at all. The caveat that matters at integration: §10c's TF32 `vmo_mma_tile` already sits AT the
376 boundary, so adding rA/rB registers there could tip it — must re-measure on the real kernel.

**Recommendation (for later integration, not now):**
1. **Do not add `async_work_group_copy` / async staging to the VM.** Verified regression on NVIDIA.
2. EW async lever is **already banked** (§22 float4 path). No action.
3. The MMA K-loop reg-prefetch double-buffer (5–14 %, no local cost) is the only new candidate. Fold it
   into `vmo_mma_tile` **only on the portable non-TF32 path**, and only after re-checking register
   pressure against the §10c 376-lane boundary. It is not a headline: §14b shows base is overhead-bound,
   not matmul-bound, so 5–14 % on matmul ≈ ~1.05× on compute-bound `large`, negligible on `base`.
4. It **composes with the §23 region-op**: "async-load the region's inputs" is the same reg-prefetch
   idea (load region inputs into registers once, compute the sub-list, store once) — so the register
   double-buffer belongs *inside* the region-op work, not as a standalone async mechanism.

**Verified:** poc/13 builds (`make`) and runs clean on NVIDIA + PoCL; all 3×(4 EW + 5 MMA) variants
bit-exact vs host/baseline (`maxerr=0`). Data: `poc/13-async-prefetch/results_{nvidia,pocl}.csv`.

## 26. SHIPPED 2026-07-19: OP_GELU — dedicated fused elementwise opcode (§19b/§24)

Implemented the §24 near-term recommendation (dedicated fused opcode over the general
interpreted region-op). **Kept ON by default; `PJRT_OCL_FUSE_GELU=0` reverts.** GELU is
**pure elementwise** (no reduce/segment), so unlike softmax/layernorm (§19a) this is NOT a new
segmented tile-op family — it is one extra **EW-unary subop** riding the existing TILE_EW float4
fast path (one global read + one write per element, whole tanh-approx computed in registers).

- **Opcode/subop**: `OP_GELU`(56) in lowering.py → `to_task` maps it to `TILE_EW` with
  `SUB_GELU`(41, appended after SUB_AFFINE) in vm_common.cl. Kernel: `vmo_gelu1/8/4` +
  `case SUB_GELU` in `vmo_ew_un/un8/un4` and `vmo_ew_is_un()` range-extended (ops/ew.cl). Numpy
  `_gelu_np` (validators) matches the kernel's `VMO_GELU_BODY` exactly. No new SLM, no barriers,
  negligible register cost — sidesteps the §24-finding-2 occupancy risk that killed the *general*
  region-op for the base regime.
- **Recognizer** `_fuse_gelu` (lowering.py, runs right after `_fuse_norm`, before `_reuse_arena`;
  the following `_dce_nops` drops the dead chain). Post-lowering peephole on OUR instr stream,
  anchored on the final `MUL`. **Two spellings reach us with an identical backbone**
  (x²=x·x, x³=x²·x, `0.044715·x³`, `x+·`, `0.7978845608··`, `tanh`) and differ ONLY in the
  `0.5·x·(1+tanh)` tail factoring: `jax.nn.gelu` → `x·(0.5·tanh+0.5)`; the transformer's manual
  `0.5*x*(1+tanh(...))` → `(0.5·x)·(1·tanh+1)`. **First cut matched only the jax.nn form and MISSED
  the transformer** (the real target) — generalized the tail to `(s_x·X)·(s_t·(tanh+1))` gated on
  the tanh-factor bias==scale AND `s_x·s_t == 0.5`, matching any algebraically-equivalent factoring
  while the backbone constants (0.044715 / 0.7978845608) stay exact-checked and X stays reused
  throughout. **Gated hard**: any op-kind / constant / linkage / reused-X mismatch → decomposed
  chain untouched (never wrong, only sometimes-unfused). Lookalikes (wrong consts, non-reused arg)
  correctly do NOT fire. The exact **erf** variant (`approximate=False`) can't even VHLO-serialize
  in jaxlib 0.10.2 — nothing to match; documented follow-up.

**Surprised by: fused OP_GELU is MORE accurate than the decomposed chain, not just faster.** At true
f32 (`PJRT_OCL_MEGA_TC=0`) base is `max_abs 2.4e-6` fused vs `1.2e-3` decomposed — the single-expression
kernel matches XLA-CPU's in-register/FMA gelu, whereas our decomposed path round-trips each
intermediate (exactly the §23 "we removed the barrier, not the round-trips" point, now visible in
*accuracy* too).

**Before/after (this machine; A/B via `PJRT_OCL_FUSE_GELU` on the SAME build, NVIDIA):**

| metric                                       | fused OFF | fused ON | note |
|----------------------------------------------|-----------|----------|------|
| standalone gelu (4,128,2048)=4 MiB, L2-res   | 0.0428 ms | **0.0278**| 1.54×; **beats** CUDA 0.0349 |
| standalone gelu 64 MiB (HBM-bound)           | 0.749 ms  | **0.121** | **6.2×**; CUDA 0.089 (gap 8.5×→1.37×) |
| base transformer ms/iter (TF32)              | 5.80      | **5.33**  | 1.09×; GFLOP/s 3610→3930 |
| base gap vs native CUDA (0.4385 ms)          | 13.2×     | **12.2×** | |
| large transformer ms/iter (TF32)             | 28.24     | **26.83** | 1.05× |

The base end-to-end win (1.09×) is single-digit-% as §24 predicted (base FFN gelu is L2-resident →
memory win is modest; the big 6.2× is HBM-bound), but it *does* move the base needle (§14a) AND
collapses 8 VM instrs→1 per layer. Consistent with §24: dedicated opcode banks the base regime
cheaply; the general region-op (deferred to R3) is for the HBM-bound/`large` regime.

**Correctness (all PASS):** 275 pytest + 1 skip (was 262+1; +13 `tests/test_ops_gelu.py`: both
spellings fire, lookalikes don't, dual validators agree, `FUSE_GELU=0` fallback). runtime_test
NVIDIA + PoCL PASS. transformer `--check`: NVIDIA TF32 tiny/small/base/large; NVIDIA
`PJRT_OCL_MEGA_TC=0` base **2.4e-6** / large **1.2e-5** f32-exact; PoCL Portable tiny/base
**2.3e-6** f32-exact.

**Pre-existing issue surfaced (NOT gelu-caused), two of them:**
1. **runtime_test was stale.** Its two EW/while tests hardcoded a 16384-element tile size; since
   §22 gave GPUs `EW_TS=4096`, a *fresh* build computes only 4×4096 of 65536 elements → deterministic
   `48956 = 49152 − 196` bad on NVIDIA (PoCL's 16384 masked it; main's binary was pre-§22 stale).
   Fixed: both tests now size `N = LANES · rt->ew_ts()` (one tile/lane at the device's real tile
   size). PASS on both devices. The transformer `--check` passing throughout proved the *kernel* was
   always fine — only the test's tiling assumption was wrong.
2. **PoCL fused-norm intermittent (§19a) still flickers.** PoCL base occasionally reads `~1.3e-3`
   instead of `~2.4e-6` (~1 in 6 runs), reproduced on **pristine main** (no OP_GELU) → it lives in
   the fused softmax/layernorm barrier path, not gelu (gelu is barrier-free and stable). Logged for a
   future §19a follow-up; unrelated to this change.

## 27. Register-resident map-region fusion IS free inside the one megakernel — GO (2026-07-20, poc/15)

**Verdict up front: GO.** §24 finding-2 ("the general interpreted region-op is entangled with the
R3 split-VM; any added register/SLM pressure regresses ALL ops because the TF32 megakernel sits
*exactly at* the 376-lane boundary") was **too pessimistic — it reasoned about occupancy instead of
measuring it.** Measured: a register-resident `TOP_MAP_REGION` case added to the real `vm2` costs
**+0 registers to the whole-kernel max, +0 SLM, and holds co-residency at 376 (the §10c floor)**,
while computing GELU f32-exact. Region fusion can live in the single megakernel with NO split, NO
relaunch. (poc/15-region-budget; NVIDIA RTX PRO 6000 Blackwell, 188 SM, sm_120.)

### The instrument (deterministic, unlike the spin-probe)

The occupancy discovery probe (`vmo_discover`) is **bimodal 376/752** run-to-run (backfill
over-counts co-residency), so it can't resolve a fine register→occupancy curve. Instead, poc/15's
`regprobe` builds the real `kVmClSource` with **`-cl-nv-verbose`** and reads ptxas's exact per-kernel
**register / smem / stack / spill**. Two build-flag knobs on the real megakernel (never emitted by
lowering): `-DVMO_PROBE_REGS=N` (a switch case holding N per-thread float accumulators
simultaneously live, scoped in-case) and `-DVMO_REGION_POC` (the real interpreted region case). A new
`PJRT_OCL_EXTRA_BUILD` env injects either flag into the runtime's kernel build so the *actual*
residency + GELU correctness can be measured end-to-end through `runtime_test`.

### The occupancy model, anchored to hardware and validated both ends

188 SM × 65536 regs/SM; 256 threads/WG (8 warps). WG/SM = floor(65536 / (roundup(R)·256)), capped at
the ~4-WG/SM hardware ceiling. Cliffs: **R ≤ 128 → 2 WG/SM = 376; R ≥ 129 → 1 WG/SM = 188.** Measured:

| build (vm2)                 | regs | smem  | stack | WG/SM | co-resident (spin-probe) |
|-----------------------------|------|-------|-------|-------|--------------------------|
| baseline portable           | 92   | 8196  | 240   | 2     | 376 / 752 (bimodal)      |
| baseline TF32 (`VMO_NV_PTX`)| 94   | 10244 | 240   | 2     | 376 / 752 (bimodal)      |
| **region PoC portable**     | **88** | 8196 | 320  | 2     | **376 / 752 (bimodal)**  |
| **region PoC TF32**         | **88** | 10244| 320  | 2     | **376 / 752 (bimodal)**  |
| +64 switch-regs (op 99)     | 95   | —     | —     | 2     | (95 < 128 ⇒ 2 WG/SM)     |
| +80 switch-regs             | 182  | —     | —     | 1     | 188                      |
| +96 switch-regs             | 197  | —     | —     | 1     | **188 (always)**         |

The spin-probe reads 376 *or* 752 for every in-budget build (≤128 regs) and never 188; the
over-budget 197-reg build reads **188, always** — the unambiguous occupancy discriminator. So the
region case (88 regs) is in the SAME occupancy class as the baseline (92–94 regs) and can never fall
to the 188 that a >128-reg kernel forces; the launch caps at `min(2·CU=376, measured)` = 376 either
way. The deterministic regcount is the truth; the 752 blips are the discovery protocol counting
backfilled groups. The cliff itself is sharp and reproducible: 197-reg build → 188 (×2). So
**baseline has 34–36 registers of headroom before the 128-reg cliff** — the TF32 megakernel at 94
regs is in the *middle* of the 2-WG/SM shelf (81–128), NOT at its edge. §24's "exactly at the
boundary" conflated "at the 376 lane cap" with "at the register cliff"; they are 34 registers apart.

### Reconciling 752 (poc/13) vs 376 (§10c)

Not a contradiction — different kernels, different binding resource. poc/13's EW/MMA microkernels use
~8–24 regs and hit the **4-WG/SM hardware ceiling (752)**; the 92–94-reg megakernel is
**register-limited to 2 WG/SM (376)**. The 752→376 gap is *entirely* the megakernel's register
pressure. §10c's earlier "portable 564" (3 WG/SM ⇒ R ≤ 80) was before the op library grew
(GELU/softmax_seg/layernorm_seg pushed portable to 92 regs ⇒ 2 WG/SM); both variants now sit at 376.

### The three hypotheses — all confirmed

1. **Registers are MAX-not-SUM across mutually-exclusive switch cases.** N = 8, 16, 24, 32, 48, 64
   in-case float accumulators ALL held vm2 at ~95–96 regs (+3–4 over the 92 baseline) — the region
   case reuses the matmul case's physical registers (disjoint live ranges). Only at N=80 (182 regs)
   does the case become the new max and blow past 128. So a switch-addressed-register region using
   **≤ ~64 float (16 float4 slots)** costs ~0. §24's "adds ~32 vector registers to the whole
   megakernel" is false: 32 in-case float regs cost +0 to the kernel max.
2. **Local (SLM) is shared and need not grow.** `vm_main.cl` declares ONE `As`/`Bs` scratch and
   passes it to every family; the region case declares NO `__local` → smem stays 8196 (portable) /
   10244 (TF32), byte-identical to baseline. Hypothesis 2 confirmed structurally + by measurement.
3. **poc/14's "hit" was an impl choice, not an architectural limit.** §24 finding-2's 32 KB SLM =
   `8 slots × 256 lanes × 16 B` — i.e. a per-**workgroup** `__local float4 R[NSLOTS][WG]` staging
   tile. Using per-**thread** slots (`float4 R[NSLOTS]` private) removes it: dynamic slot indexing
   lands the array in per-thread LOCAL/stack memory (+80 bytes stack frame, 0 spill), which consumes
   neither the register file nor SLM, so occupancy is untouched (88 regs, 376 lanes). The
   switch-addressed-register alternative (slots stay in registers, faster) also fits — 16 float4
   slots is inside the 34–64-reg headroom. Either structuring is occupancy-free; §24 assumed the
   worst of both (SLM tile AND additive registers) and never measured the megakernel.

### The megakernel-native region-op design (fits the fixed budget)

Per-lane budget that keeps ≥ 2 WG/SM = 376 on this part: **≤128 registers/thread** and the shared
~10 KB `As`/`Bs` SLM (well under the 48 KB/WG SLM limit at 2 WG/SM). The region op:
- **Per-thread slots, never a per-workgroup SLM tile.** `float4 R[NSLOTS]` scoped in the switch case;
  intermediates never leave the lane. Reuse the shared `As`/`Bs` only as an explicit spill target if
  a region ever needs it — the default touches zero SLM.
- **float4-vectorized single grid-strided tile loop, no cross-workgroup barrier** (pure map: element
  i independent) — so §18/§19a PoCL barrier rules don't bite and it runs on both engines unchanged.
- **Inputs ride the task's own dst/a/b handles** (loader-resolved, no aux-handle patching, §20); the
  aux descriptor carries only the straight-line micro-program (`{kind,dst,a,b,s,t}` over slots) +
  slot map. Real DAGs / input reuse work (GELU reuses slot0 = x 4×).
- **Recognizer-splitting for over-budget regions.** A region wider than NSLOTS (or a chain that would
  push the case past the register headroom) is split by the recognizer into budget-sized on-chip
  sub-regions, materializing one boundary tensor between them — still ONE kernel, no relaunch
  (register-tiling, done in the bytecode). Gate `n_slots ≤ 8` (design A) / `≤ 16` (register variant).

### Evidence / correctness

`regprobe` builds clean on NVIDIA; register table above. GELU region (8 pure-map micro-ops over 2
float4 slots, slot0 reused 4×) through `runtime_test`: **PASS, maxerr 5.96e-08** (f32-exact) on BOTH
the TF32 and portable (`MEGA_TC=0`) builds; residency in the same 376/752 class as baseline on the
TF32 (binding) build. **PoCL host-dispatch (CPU) runs the identical region case unchanged — GELU
PASS, maxerr 5.96e-08** (no barrier, no vendor code → portable across both engines).
Product code untouched unless `-DVMO_REGION_POC` is set (the case is `#ifdef`-guarded; op 13 reserved
in the enum + parser; `PJRT_OCL_EXTRA_BUILD` env is inert unless set). Not yet wired into lowering /
the scheduler recognizer — that is the follow-up now that the occupancy blocker is measured away.

**Caveat (out of scope here): performance.** This settles the *occupancy* GO/NO-GO only. §24's
finding-1 (at L2-resident base sizes a dedicated fused opcode beats the interpreter ~4× vs 1.3×)
stands — the general region-op's payoff is the HBM-bound/`large` regime and the phase/instr-count
collapse, not base-size speed. The value of §27 is that the general op no longer has to wait for the
R3 split-VM: it can ship in the one megakernel whenever its end-to-end perf is shown to move the
needle (measure-first, §14a). GO on feasibility; perf gating unchanged.

## 28. SHIPPED 2026-07-20: OP_MAP_REGION — register-resident map-region fusion, real (§23/§27)

The culmination of the §23/§24/§27 arc: the proven §27 PoC is now a real
recognizer-driven op. A maximal run of pure-map f32 EW ops collapses into ONE
`OP_MAP_REGION` (tensor opcode 57) → `TILE_MAP_REGION` (tile op 13) whose
intermediates never leave the lane — one global load per input, interpret the
straight-line micro-program over per-thread `float4 R[8]` slots, one store. The
`#ifdef VMO_REGION_POC` guard is REMOVED: the case ships in the default megakernel
and is correct there. `PJRT_OCL_FUSE_REGION=0` reverts the recognizer.

### What was built
- **Kernel** (`ops/region.cl`, un-gated): interpreter over an arbitrary map
  micro-program (`{kind,dst,a,b,s,t}` over slots); micro-op ALU extended to the
  EW set (add/mul/sub/div/max/min/neg/exp/log/sqrt/rsqrt/tanh/abs/affine), builtins
  byte-matching `ops/ew.cl` so a region is numerically identical to the chain it
  replaces. float4-vectorized grid-stride tile loop + scalar tail, no `__local`,
  no cross-workgroup barrier. Inputs ride the task's own dst/a/b handles (≤2
  inputs, no aux-handle patching — §20 avoided per §27).
- **Recognizer** `lowering._fuse_region` (after the other fusion passes, so
  OP_SOFTMAX/LAYERNORM/GELU are single ops = region boundaries): finds
  **within-phase** connected runs of eligible map ops (phase computed exactly like
  `_reuse_arena`, so a dependency that threads the whole program — the residual
  stream, whose links sit in different phases across the attention/FFN/norm
  boundaries — is NOT wrongly grouped), does linear-scan slot allocation, encodes
  the micro-program into aux, emits `OP_MAP_REGION`. **Over-budget / long chains
  SPLIT** into budget-sized single-output on-chip sub-regions (still one kernel;
  the boundary tensor is a real arena buffer read by the next sub-region).
- **Gated HARD**: co-scheduled in one phase, single externally-live output, ≤2
  inputs per (sub)region, ≤8 slots, all-f32, no viewed operands, splits that
  yield only singletons rejected (no op-count win). Any mismatch → decomposed
  chain untouched. `PJRT_OCL_REGION_SLOTS=<n>` lowers the budget (forces the
  split in tests).
- **Scheduler**: `OP_MAP_REGION` → one `TILE_MAP_REGION` task, n_tiles like EW.
- **Validators** (`ops/region.py` + `tests/test_ops_region.py`, 14 tests): numpy
  micro-program interp (validator a) + schedule-sim (validator b), matching the
  kernel; fires/doesn't-fire cases + the over-budget split.

### Occupancy re-verified — the region case is FREE in the default build
`regprobe` on the shipped kernel (region case now always compiled): **vm2 = 88
registers** (portable, 8196 smem) / **88** (TF32, 10244 smem), 320-byte stack, 0
spill — identical to §27's gated-PoC measurement and LOWER than the pre-region
baseline (92/94). 88 < 128 ⇒ 2 WG/SM ⇒ the §10c 376-lane floor holds. So the
whole point of §27 is confirmed end-to-end: register-resident region fusion lives
in the ONE megakernel, no VM split, no relaunch, +0 registers to the kernel max.

### Honest measurement (§14a) — moves the needle on its target class; transformer flat
- **Transformer (base + large): fires on 0 regions; phases 107 → 107; base/large
  ms identical REGION on/off (5.48 / 26.84 ms TF32).** Measured reason: the
  dedicated OP_GELU/SOFTMAX/LAYERNORM already absorbed every ≤2-input map chain,
  so the ONLY remaining within-phase multi-op map chains are the 12 layernorm
  affines `x·γ+β` — **3 inputs** (x + broadcast γ + broadcast β), which exceed the
  2-input design; the residual-stream adds are single ops per phase (boundaries
  between them). So on THIS model the general op is inert — no win, **no
  regression** (a valid §14a outcome).
- **Its target class (≤2-input map chains): a large, monotone win.** A 24-op
  1-input chain (`t=tanh(t·0.5+a)`×8) through the real plugin, NVIDIA, REGION on
  vs off: 256K **1.25×**, 1M **1.32×**, 4M **1.79×**, 16M (64 MiB, HBM-bound)
  **3.82×** (0.590 vs 2.254 ms) — exactly §24's "1.3× at L2, 3×+ at HBM"
  prediction, now with the memory-traffic saving realised (24 arena round-trips →
  1 load + 1 store; §23's "we removed the barrier, not the round-trips" finally
  closed). Correct to **1.79e-7** vs numpy on BOTH NVIDIA and PoCL; region GELU
  through `runtime_test` **5.96e-8** on both engines.

### Decision: KEEP, default ON
Correct everywhere (289 pytest + 1 skip; TF32 tiny/small/base/large; MEGA_TC=0
base/large f32-exact 2.4e-6/1.2e-5; PoCL tiny/base 4.8e-7/2.3e-6; runtime_test
both devices), occupancy-free, and a 1.25–3.82× win on the map-chain class with
zero transformer regression. Defaulting ON matches the §26 gelu precedent.

### Next step to make it fire on the transformer (the 3-input `x·γ+β` idiom)
Support **>2 inputs + broadcast-view loads**: carry the 3rd/4th input handle in a
free task field (p2/p3, loader-patched like dst/a/b — a small localized change, no
aux-handle patching) and add strided/broadcast load addressing (§13 access-maps)
so γ/β broadcasts read in place. That captures the 12 layernorm affines
(107 → ~95 phases + 12 removed intermediate round-trips) and any `scale·x+bias`
per-channel idiom. Deferred: measured here as the concrete lever, low-risk but
out of this session's 2-input scope.

## 29. Decomposing the base 5.0 ms in-kernel time — it is MATMUL, not EW/reduce (2026-07-20)

**Question.** base transformer is ~12× slower than native CUDA (mega 5.44 ms /
kernel 5.01 ms vs CUDA 0.44 ms); 93% is in-kernel. Decompose the 5 ms and prove
where it goes. Hypothesis on the table: "strictly-serial globally-barriered VLIW
interpreter, no overlap → each phase is a latency-bound small-data pass and the
grid barrier serializes them, so utilization is few-%."

**Verdict up front: the hypothesis is HALF right, and the wrong half rewrites the
roadmap.** The *mechanism* is exactly as described — 107 serial global-barrier
phases, each straggler-bound, mean lane utilization **31%** (lanes idle 69% of
every phase). But the premise that the latency-bound phases are elementwise/
reduce/norm is **now false**: the §19/§26/§28 fusion campaign already collapsed
all non-matmul compute to **0.4 ms (8%)**. The 5 ms is **MATMUL** (84%). We spent
the last campaigns fusing the cheap 8% while matmul — the expensive 84% — sat
unattacked. R2 (EW fusion) is essentially done and is no longer a lever; the
levers are R1 (overlap the many small independent matmuls to fill the grid) and
R3 (a bigger matmul tile for the per-tile intensity cap).

### Instrumentation (behind flags; product build byte-identical, 289 pytest + runtime_test green)
Two hooks, both inert unless a `-D` is injected via `PJRT_OCL_EXTRA_BUILD` (§27's
mechanism) and/or an env is set. Default kernel/behaviour unchanged (base --check
PASS max_abs 4.6e-3; runtime_test PASS).
- **Op-class stubbing** `-DVMO_STUB_MASK=<bits>` (`vm_main.cl` `vmo_exec_tiles`):
  bit *k* set ⇒ tile-op *k* returns immediately (skips its tile work; the entry/
  phase/barrier structure is untouched). `full − stub_X` = X's wall contribution
  *as the phase straggler* (only the straggler bounds a global-barriered phase, so
  this is the correct wall attribution). Correctness intentionally void; timing only.
- **Per-phase device timestamps** `-DVMO_PHASE_TS` + env `PJRT_OCL_PHASE_TS`
  (`vm_common.cl` `vmo_now_ns` reads NV `%globaltimer` — a *GPU-global* ns counter,
  so arrival times are comparable across workgroups, unlike per-SM `clock64`).
  Every lane writes its arrival ns into the existing `stats` buffer at each barrier;
  host (`runtime.cc` ExecuteDevice) reads it back → per phase: `wall = release[b] −
  release[b−1]` (release = last-lane arrival), `skew = max−min arrival`,
  `idle = release − mean arrival` (avg per-lane wait). Only the VMO_NV_PTX (TF32)
  build has globaltimer — which is the default base path on NVIDIA.

### The 5.0 ms labelled budget (base, NVIDIA RTX PRO 6000 Blackwell, TF32 megakernel)
Op-class stub subtraction (kernel= from PJRT_OCL_PROFILE, mean of last iters):

| stub                     | kernel (ms) | ⇒ class wall (ms) | % of 5.01 |
|--------------------------|-------------|-------------------|-----------|
| full                     | 5.01        | —                 | —         |
| MMA (op 1)               | 0.82        | **matmul 4.19**   | **84%**   |
| EW (op 0, incl gelu/affine) | 4.80     | EW 0.21           | 4%        |
| SOFTMAX_SEG (11)         | 4.90        | softmax 0.11      | 2%        |
| GATHER (2,7)             | 4.96        | gather 0.05       | 1%        |
| LAYERNORM_SEG (12)       | 4.98        | layernorm 0.03    | 0.6%      |
| RED_SEG (10)             | 5.00        | reduce ~0.01      | ~0%       |
| MMA-only (~1, all else stubbed) | 4.61 | matmul isolated 4.61 | — |
| all compute stubbed floor ≈ stub_MMA − Σnon-mma | ≈0.41 | **barrier/interp/empty-phase 0.41** | 8% |

Labelled 5.0 ms ≈ **matmul 4.2 / EW 0.21 / softmax 0.11 / gather 0.05 / layernorm
0.03 / reduce 0.01 / barrier+interpreter+empty-phase 0.41**. (Matmul's straggler
attribution 4.19 ms and its isolated 4.61 ms bracket it: when non-matmul ops share
a matmul phase their work hides under the matmul straggler, so isolated > marginal.)

### Per-phase timestamps — the serialization/imbalance is real, and it IS the matmul
`PJRT_OCL_PHASE_TS`, base: **107 phases, sum_wall 4.96 ms** (validates the
instrument against the 5.01 ms profile), **skew ≈ wall in EVERY phase** (the
first-arriving lane reaches the barrier with ~no work while the last-arriving lane
defines the phase — every phase is straggler-bound), **sum_idle 3.44 ms, mean lane
utilization 0.306** (an average lane is busy only 31% of each phase; 69% is spent
waiting at barriers). Wall is concentrated, not spread:

| phase wall bucket | # phases | Σ wall (µs) | what |
|-------------------|----------|-------------|------|
| ≥300 µs           | 5        | 1884 (38%)  | FFN matmuls (1/layer; 6th is 296 µs) |
| 100–300 µs        | 9        | 1256 (25%)  | projection / attention matmuls |
| 30–100 µs         | 16       | 1044 (21%)  | smaller matmuls + fused norms |
| 10–30 µs          | 41       | 593 (12%)   | tiny ops at the latency floor |
| <10 µs            | 36       | 195 (4%)    | barrier + trivial-op floor |

Top 14 phases (all matmul-heavy) = 3.1 ms (63%). The 107 global barriers' own cost
is small (~1.7 µs each ≈ 0.18 ms; consistent with §14's flat lane-count sweep) —
the tax is not the barrier primitive, it is **stragglers idling 375 lanes while a
few finish a small matmul**.

### Per-phase utilization — the small matmuls are latency/occupancy-bound (proved by size)
The 64×64 MMA tile means an M×N matmul spawns `⌈M/64⌉·⌈N/64⌉` tiles = busy lanes
(of 376). base shapes: QKV/out projections (512×512×512) = 64 tiles = **17%**
occupancy; attention per-head (128×64×128, ×32 batch) = 2–4 tiles/head; FFN
(512×512→2048) = 256 tiles = 68%. Mostly far under 376 → most lanes idle → the 31%
mean. Confirmed by sweeping size on the SAME code:

| config    | phases | mean lane util | matmul share | matmul TFLOP/s | CUDA TFLOP/s | gap |
|-----------|--------|----------------|--------------|----------------|--------------|-----|
| base      | 107    | **0.31**       | 84%          | **4.7**        | 47.5         | 10× |
| large_l1  | 17     | **0.72**       | 92%          | **14.1**       | 97.9         | 7×  |

As phases grow (D=1024/F=4096), lane utilization jumps 31→72%, matmul throughput
4.7→14 TFLOP/s (approaching our own §10d in-megakernel tile ceiling ~18–20), and
the CUDA gap shrinks 10→7× — the textbook signature of latency/occupancy-bound
small ops, now *attributed to the matmul phases specifically*.

### Splitting the 10× matmul gap into its two buckets
base matmul 4.7 TFLOP/s vs cuBLAS-in-CUDA 47.5:
1. **Occupancy/latency (~3.8×):** 4.7 → our own ~18 TFLOP/s tile ceiling (§10d) is
   the price of small matmuls filling 31% of the grid + being individually tiny.
   Dominant at base; nearly gone at large_l1 (14 of 18).
2. **Per-tile intensity cap (~2.5–7×):** the 64×64 single-buffered co-residency-
   locked tile tops out ~18–20 TFLOP/s (§10d: intensity 16 FLOP/byte, a bigger
   register tile drops residency below the 376-lane barrier floor). This is the
   *whole* residual gap on large (compute-bound), a minor one at base.

So base's 10× is **mostly bucket 1 (occupancy)**, large's 7× is **mostly bucket 2
(tile intensity)**. Different levers for different regimes.

### Ranked causes (of the base 5.0 ms) and the fix each maps to
1. **Matmul under-occupancy — ~2.4 ms recoverable, megakernel-native (R1).** Small
   independent matmuls run in serial global-barriered phases at 31% lane occupancy.
   In one layer the 3 QKV projections are mutually independent, the 32 attention
   heads are independent, FFN tiles are independent — today each is its own barrier
   phase. **R1 (per-tile dependency scoreboard replacing the global barrier)** lets
   independent matmul phases co-occupy the 376 lanes instead of serializing. Est: if
   base matmul reached large_l1's 72% util, matmul 4.2 → ~1.8 ms, kernel → ~2.6 ms
   (**~1.9×**, gap 10→~5×). This is the single highest-leverage megakernel-native
   change and it directly consumes the 3.44 ms of measured lane-idle.
2. **Matmul per-tile intensity cap — megakernel-native (R3).** The 64×64 tile's ~18
   TFLOP/s ceiling caps even a *fully-occupied* matmul at ~7× off cuBLAS. Fix: R3's
   compile-time VM split (a `vm2_heavy` variant with a 128×128 double-buffered
   register tile, independent register budget, no relaunch). Biggest lever for the
   `large`/compute-bound regime; secondary at base (occupancy dominates there).
3. **Barrier/phase serialization floor — ~0.4 ms + a 0.79 ms tail of 77 tiny
   phases.** 107 serial global barriers, each straggler-bound. R1 subsumes this
   (a scoreboard removes the barriers and lets the tail overlap the fat phases).

**NOT a cause any more: non-matmul compute (0.4 ms, 8%).** EW/softmax/gather/
layernorm/reduce are already fused to near-nothing (§19/§26/§28). R2 (EW micro-
program fusion) is effectively complete for this workload — further EW/norm work
would chase 8% and is explicitly deprioritised. This corrects §14's "65% non-matmul"
finding, which predated the fusion campaign.

### Single highest-leverage megakernel-native fix
**R1 — dependency scoreboard replacing the global barrier**, so the many small
*independent* matmuls (3× QKV, 32 attention heads, independent FFN tiles) overlap
and fill the lane grid instead of serializing at 31% occupancy. It attacks the
dominant bucket (occupancy, ~2.4 ms of lane-idle at base) and dissolves the 107-
barrier serialization in one structural change; the WAIT/SIGNAL fields are already
reserved in the bytecode (`vm_main.cl`), the scheduler already owns the producer/
consumer graph, and the barrier fix already proved device-scope coherent spinning
works. R3 (bigger matmul tile) is the follow-on for the compute-bound `large` end.

**Reproduce:** `PJRT_OCL_EXTRA_BUILD="-DVMO_PHASE_TS" PJRT_OCL_PHASE_TS=1 python
tools/bench_transformer.py --config base` (per-phase wall/skew/idle + summary);
`PJRT_OCL_EXTRA_BUILD="-DVMO_STUB_MASK=0x2" PJRT_OCL_PROFILE=1 …` (stub op 1 = MMA;
bit k = tile-op k). Instrumentation is `#ifdef`/env-gated — the shipped build is
unchanged.

## 30. R1 — per-tile dependency scoreboard replacing the global barrier (DESIGN, 2026-07-20)

**Goal (from §29).** The 107 serial global barriers serialize INDEPENDENT small
matmuls (3 QKV, 32 heads, FFN tiles) that each fill only 17–31 % of the 376 lanes;
mean lane util 0.31. Replace the grid-wide barrier with point-to-point WAIT/SIGNAL
so independent work overlaps and fills the grid. Est ~1.9× (kernel 5.0→2.6 ms).

**Correction to the §29 premise.** §29 said "the WAIT/SIGNAL fields are already
reserved in the bytecode". They are NOT free: `Entry.wait_flag/wait_count/
signal_flag` were repurposed by §15 (OP_FOR) / WHILE to carry cond/body ranges and
trip counts (scheduler.py `_emit_while`, vm_main.cl ENT_WHILE/ENT_FOR). So a
scoreboard needs its OWN per-entry sync fields (grow the entry struct) or a
side-table — the compute-tile entries never use those three fields, only
WHILE/FOR/IF control entries do, so a compute-tile can borrow them, but that
overlaps semantically with control flow and is fragile. Design uses NEW fields.

### The primitive (proven, §10c/poc07)
Producer: `VMO_FENCE_DEV_REL()` then `atomic_inc(flag)` (release) — the SAME
device-scope release the spin-barrier uses; poc07 test E proved plain cross-lane
data writes become visible after a device-scope acquire on the consumer. Consumer:
spin `while(atomic_load_acquire(flag) < threshold)` then `VMO_FENCE_DEV_ACQ()`.
Point-to-point instead of all-to-all. Same #1-risk primitive, same portability
envelope (NVIDIA honours memory_scope_device; PoCL native; strict-1.2 → host
engine, scoreboard OFF).

### The two coupled changes
1. **Sync**: each producer TASK gets a flag = its arrival counter; each of its
   tiles `atomic_inc`s the flag on completion. A consumer tile WAITs until every
   producer task's flag == that task's n_tiles. wait_count = #producer tasks; a
   side "wait list" of (flag_idx, threshold) pairs indexed by (wait_off, wait_n).
2. **Lane assignment MUST change** (the non-obvious half). Today `_pack_units`
   resets `loads=[0]*n_lanes` per phase, so every phase packs onto lanes 0..T-1 —
   3 independent QKV matmuls all land on the SAME low lanes. Removing the barrier
   alone gives NO overlap: a lane runs its linear stream in order, and the same
   lanes are reused. Fix: a GLOBAL list-scheduler over the whole tile-DAG that
   spreads independent ready tiles across ALL lanes (carry `loads` across phases /
   assign independent tasks to disjoint lane ranges) so they co-occupy the grid.
   A lane's stream stays linear and topologically valid; a WAIT only fires on a
   true cross-lane producer.

### Correctness / deadlock
Acyclic DAG incl. WAR carries (§16) and region ops — verified by the existing
`_depends` (RAW+WAW+WAR). Region ops (while/if/for) stay grid-barrier sync points
(a scoreboard across a data-dependent loop back-edge is future work); the
scoreboard overlaps only the straight-line compute BETWEEN region ops, which is
where the 107 barriers / matmul phases live. Exact flag thresholds = producer
n_tiles; a consumer never waits on a flag no tile signals (every producer task
emits all its tiles). Liveness (co-residency) is UNCHANGED risk vs the spin-barrier
— point-to-point waiting still needs producers co-resident; NVIDIA 376 co-resident
(occupancy-capped), PoCL needs lanes ≤ cores (already true / host engine).

### Fallback
`PJRT_OCL_SCOREBOARD=0` (default until proven on all vendors) → today's global-
barrier path, byte-identical. Scoreboard is opt-in, NVIDIA-first.

### Gate (poc/16-scoreboard, hard, measure both devices)
K independent matmul-like tasks + 1 dependent. MEASURE (a) correctness race-free
under a stress loop (acquire/release orders the writes), (b) independent tasks
actually OVERLAP — per-WG %globaltimer timestamps show lane utilization rising
toward full grid vs the serial-barrier baseline, (c) real speedup on this pattern.
NVIDIA AND PoCL. **If overlap does not materialize or correctness can't be made
race-free, STOP and report — do not rewire the scheduler on a false premise (§14a).**

### GATE RESULT — mechanism PASSES, premise FALSE, STOPPED (2026-07-20, poc/16)
**The scoreboard mechanism works and is race-free on both devices** (poc/16):
- NVIDIA RTX PRO 6000: independent tiles OVERLAP — lane util 0.115→0.362 (3.2×),
  wall 2.3× (K=16: util 0.056→0.326 = 5.8×, wall 4.0×); **0/94000 wrong dependent
  reads** over 2000 iters (device-scope acquire/release orders the non-atomic
  producer writes point-to-point, poc/07 test E). No hangs.
- PoCL (Ryzen 3900X): race-free (0/9000 over 1500 iters), wall 15× (the CPU
  spin-barrier is catastrophic — §10c — so barrier-free wins big). No hangs.

**BUT the base transformer — R1's target — has ZERO independent phase-level work,
so a scoreboard cannot help it.** Measured on the real lowered schedule
(`poc/16-scoreboard/analyze_schedule.py`, `critical_path.py`):
- §29's premise is FALSE as stated. The scheduler ALREADY fuses the 3 QKV
  projections into ONE 3-matmul phase (`_phases` detects their mutual
  independence) and attention heads are ALREADY batched into one MMA task
  (`p3=batch`). Those were never "each its own barrier phase" — that intra-phase
  parallelism is already extracted (occupancy 0.51 for the fused QKV phase).
- Everything else is a strict RAW dataflow chain: QKᵀ→softmax→AV→out-proj→
  residual→LN→FFN1→gelu→FFN2→residual→next layer.
- **Overlap ceiling = total_phase_cost / critical_path, using the OPTIMISTIC
  RAW-only edge set (minimal constraints = max possible overlap): 1.000× on
  tiny/small/base/large_l1/large — 0.0 % of any config is overlappable.** The
  program IS its own critical path.

**Why the 31 % util then, and what actually fixes it.** Each matmul is
individually SMALL (64–256 tiles < 376 lanes) AND the matmuls are serially
DEPENDENT — the low occupancy is small-per-op + chain, not independent work
artificially serialized. A scoreboard only overlaps INDEPENDENT phases, of which
there are none. The lever that raises util is **bigger matmul tiles / bigger
effective matmuls (R3)** — large_l1's 0.72 comes from D=1024/F=4096 giving 4× more
tiles per matmul (fills the grid within ONE phase), NOT from overlap. §29
conflated "fill the grid" (R3, size) with "overlap independent matmuls" (R1), and
R1's independence does not exist in this workload. §29's own bucket-2 already
pinned the residual `large` gap on the per-tile intensity cap (R3).

**DECISION (§14a): STOP — do NOT rewire the production scheduler/VM for R1.** The
mechanism is proven and kept as ready infrastructure (poc/16 + this design) for a
workload that DOES have independent branches — a *parallel* transformer block
(GPT-J/PaLM: attention ∥ FFN off the same LN), mixture-of-experts, ensembles, or
independent batched models. For the current serial-block transformer it buys 0×;
integrating the #1-risk primitive (races, deadlocks, co-residency) for no measured
gain, off by default, is exactly the §14a "don't ship" case. Production code
UNTOUCHED (only docs + poc/16 added) → 289/1 pytest baseline preserved by
construction. Next real matmul lever: **R3** (compile-time `vm2_heavy` variant with
a 128×128 double-buffered register tile, §29/§10d).

**Reproduce:** `cd poc/16-scoreboard && make && PJRT_OCL_DEVICE=NVIDIA ./poc16`
(mechanism gate); `CFG=base .venv/bin/python poc/16-scoreboard/critical_path.py`
(overlap ceiling on the real schedule).

## 31. Megakernel-native matmul package (R1 + go-to-188 + P3) — MEASURED, all three a WASH, kept behind flags (2026-07-20)

**Goal.** Attack the base 84%/large matmul gap (§29/§10d) with three coupled,
staged, individually-verified changes, measured end-to-end, default OFF: **R1**
(per-tile scoreboard replacing the grid barrier), **go-to-188** (128×128 in-
megakernel TF32 tile, relaunched at 1 WG/SM via the existing occupancy
discovery), **P3** (double-buffered K-loop). Hardware: RTX PRO 6000 Blackwell
(188 SM), TF32 megakernel; CUDA ref = JAX-on-cuda (cuBLAS).

### What shipped as opt-in infrastructure (default byte-identical)
- **128×128 tile**: `-DVMO_MEGA_BIGTILE` (env `PJRT_OCL_MEGA_BIGTILE=1`) widens
  the WMMA `vmo_mma_tile` from 64×64 (TC_RF=1,TC_TNW=2) to 128×128
  (TC_RF=2,TC_TNW=4, 64 f32 accumulators/thread), generalising the warp→fragment
  map and the m16n16k8 D-fragment masked store. Scheduler `MMA_T` is env-driven
  (`PJRT_OCL_MMA_T`, runtime advertises 128 only when the big-tile TF32 program
  actually builds); cost-cal MMA geometry + cache key track the edge.
- **P3 pipeline**: `-DVMO_MMA_PIPE` (env `PJRT_OCL_MEGA_PIPE=1`) double-buffers
  the K-loop staging (two smem panels, prefetch next K-block while the tensor
  cores consume the current — the mm2 pattern, register/smem prefetch, NOT
  async_work_group_copy §25). Independent of tile size.
- Both only enter the NVIDIA `-DVMO_NV_PTX` program; portable/PoCL untouched.

### go-to-188 works mechanically — and is a WASH-to-REGRESSION
Occupancy discovery INDEPENDENTLY measures the drop (not just computed):
`PJRT_OCL_INFO` reports `lanes=188 measured-residency=188` with the big tile
(vs 376 default) — the 64-accumulator tile crosses the 128-reg cliff (§27), all
188 co-resident, barrier stays safe. Correct at TF32 (base --check max_abs
4.6e-3 PASS, large_l1 4.2e-3 PASS; non-tile-multiple 256×320×192 rel 9e-4). But:

| in-megakernel matmul (PJRT_OCL_MM_KERNEL=0) | 64-tile | 128-tile | +P3(64) | +P3(128) | cuBLAS |
|---------------------------------------------|---------|----------|---------|----------|--------|
| N=2048 TFLOP/s                              | **19.1**| 14.1     | 18.6    | 14.1     | 116.5  |
| N=4096 TFLOP/s                              | **18.3**| 13.9     | 18.4    | 14.1     | 133.3  |

The 128×128 tile is **strictly slower** in-megakernel: halving occupancy
(376→188) removes more latency-hiding than the doubled per-tile intensity buys,
because the large matmul was never occupancy-starved (it already fills the grid)
and the WMMA tile is **smem-bandwidth bound, not intensity/occupancy bound**
(§10d, re-confirmed). P3 is a **complete wash** on top (18.6/18.4 ≈ 19.1/18.3):
double-buffering hides *global-load* latency, which was not the bottleneck —
exactly §10d's earlier double-buffer-is-a-wash finding, now re-derived at 188.

### End-to-end transformer (iters 30; CUDA ref cuBLAS)
| config    | default 64-tile | big-tile 188 | pipe-only | big+pipe | CUDA   |
|-----------|-----------------|--------------|-----------|----------|--------|
| base ms   | **5.37** (12.4×)| 10.10 (23×)  | 5.49      | 10.16    | 0.434  |
| base util | **0.311**       | 0.184        | —         | —        | —      |
| large_l1 ms| **4.39** (7.9×)| 5.15 (9.2×)  | —         | 5.11     | 0.557  |
| large_l1 util| **0.718**    | 0.720        | —         | —        | —      |

The big tile makes **base ~1.9× WORSE** (its 512×512 / 128×64-attention matmuls
are small: at 128×128/188 lanes they cover FEWER tiles over FEWER lanes → util
0.311→0.184, the occupancy-bound regime §29 predicted). large_l1 is 1.17× worse
and its util is unchanged (0.72) — the bigger tile did not raise grid-fill because
large already fills it; it only removed occupancy. Pipe-only is a wash (5.49 vs
5.37). Every gate green: default 289/1 pytest, runtime_test PASS, --check
tiny/small/base/large_l1 PASS (TF32), portable MEGA_TC=0 f32-exact (2.4e-6), PoCL
f32-exact + flags correctly no-op (host engine, portable path).

### R1 scoreboard — NOT rewired into production (proven 0×, §30 + §14a)
§30 already proved the scoreboard mechanism is race-free (poc/16) but buys **0×
on this serial-block transformer** (critical path / total = 1.000× on
tiny…large): there is no independent phase-level work to overlap, so removing the
grid barrier cannot recover the 3.36 ms of measured lane-idle — an early-finishing
lane's NEXT op is RAW-dependent on the straggler it "skipped", so a point-to-point
WAIT stalls it identically. The barrier PRIMITIVE costs only ~0.18 ms (§29); the
tax is inherent imbalance, which a scoreboard does not touch without independent
work. R1's stated §31 job ("be the sync the 188-lane big-tile runs under") is
**moot**: the big-tile regime it supports is itself a measured wash above, and the
188-lane barrier is already co-resident-safe (§27) with no scoreboard. Rewiring
the production scheduler/VM (global list-scheduler + new per-entry sync fields +
device-scope spin + WAR/region deadlock surface) for a proven-0× result, off by
default, is the textbook §14a "don't ship". Kept as poc/16 + this design for a
workload that HAS independent branches (parallel/GPT-J block, MoE, ensembles).

### DECISION (§14a): keep all three behind flags, default OFF; do NOT change the default
The in-megakernel TF32 tile is at its architectural ceiling (~18–20 TFLOP/s,
smem-BW-bound at the co-residency-locked size); neither a bigger output tile
(net loss: −occupancy > +intensity) nor one-level K-loop double-buffering
(hides the wrong latency) moves it, and R1 has no independent work to overlap.
**cuBLAS parity (116–133 TFLOP/s) needs a genuinely different engine** — a
multi-stage register-blocked WMMA pipeline with warp specialisation / cp.async
software pipelining (cuBLAS-class), which a single output-tile widen + 1-deep
prefetch do not approximate. The flags are retained as measured A/B infrastructure
and this honest negative result; the shipped default is unchanged.

**Reproduce:** `PJRT_OCL_MEGA_BIGTILE=1 PJRT_OCL_INFO=1 … bench_transformer.py
--config base` (lanes=188); `PJRT_OCL_MEGA_BIGTILE=1 PJRT_OCL_MM_KERNEL=0
MMN=2048 mmbench` (isolated matmul); `PJRT_OCL_MEGA_PIPE=1` (P3 A/B). All default
OFF; unset flags ⇒ byte-identical 64-tile/376-lane path.

## 32. Decode workload measured — still 12×, and it's the SAME wall (2026-07-20, tools/bench_decode.py)

Tested the regime megakernels are built for — batch-1 autoregressive DECODE (one token + KV cache,
all matmuls become memory-bound matVECs). Hypothesis (§31): our launch-elimination + fusion thesis
should WIN here where cuBLAS's compute tuning is irrelevant. **Measured: it does NOT.**

| config | ours µs/tok | ours GB/s | CUDA µs/tok | CUDA GB/s | gap |
|--------|-------------|-----------|-------------|-----------|-----|
| small  | 1523 | 9   | 344  | 40  | 4.4× |
| base   | 5702 | **14 (0.8% of peak)** | 464  | 176 | 12.3× |
| large  | 8283 | 39  | 1036 | 316 | 8.0× |

**But it's NOT the matvec kernel.** A *standalone* `(1,K)@(K,N)` matvec is only **1.9–2.9× off cuBLAS**
(ours 52–115 GB/s vs 98–328) — our GEMV path is fine. The full decode model is 14 GB/s because it's
**~107 barriered phases each doing a tiny matvec** — the per-phase overhead (barrier + fixed-376-lane
grid hugely underutilized on ~D outputs) dominates the tiny per-token work. base decode = ~1.4ms of
actual weight reads + ~4.3ms of phase overhead.

**This is the SAME wall as prefill (§29/§31), amplified.** Too many phases, too much per-phase
overhead relative to the work. Decode makes it worse because per-phase work is tiny (matvec vs matmul).

**How the literature wins decode (survey §1.1):** Hazy fuses the WHOLE forward pass into **~7 fused
instruction types** (RMSNorm+QKV-matvec+RoPE = ONE instruction, activation read once, weights
streamed) + no global barriers. **~7 phases, not 107.** That's DEEP fusion that folds matmul TOGETHER
with its surrounding norm/activation into one instruction body — hand-written fused CUDA. Our EW/norm
fusion (§19/§24) doesn't touch the matmul boundaries, so every matmul is still its own phase.

**Conclusion:** both regimes hit the same structural limit — a generic interpreted VM with one
op(-group) per phase and a barrier between can't reach the literature's ~7-instruction deep fusion,
which requires fusing matmul WITH its epilogue/prologue into single register-resident instructions.
That's the real megakernel superpower, it's hand-written in the literature, and it's hard/non-portable
to do generically. Our per-op kernels are competitive (~2×); the phase count × per-phase overhead is
the gap, in BOTH regimes. R2c (matmul epilogue fusion, decisions.md R-list) is the frontier that
would start to close it — fold the norm/residual/activation into the matmul's store.

## 33. R2c — matmul-inclusive deep fusion (epilogue), the frontier of §29/§31/§32 (2026-07-20)

**Goal (from §32).** Both regimes hit the SAME wall: 108 barriered phases, each a
tiny latency-bound pass. §19/§26/§28 fused all *non-matmul* compute to ~8% but never
touched the matmul boundary, so every matmul + its surrounding norm/activation/residual
is still ≥2 separate phases. The literature (Hazy, survey §1.1) runs ~7 fused
instructions because it folds matmul TOGETHER with its epilogue into one register-resident
instruction body. R2c does that generically: a matmul computes its output TILE in
registers/accumulators before the store; run a per-element micro-program on that tile
BEFORE storing → fold `matmul → {scale, +bias, gelu/relu, +residual}` from ≥2 phases into ONE.

### Reuse the §27/§28 substrate
`ops/region.cl`'s `vmo_region_micro` already interprets a straight-line map micro-program
over a per-thread value (the map-region ALU: add/mul/sub/div/max/min/neg/exp/log/sqrt/
rsqrt/tanh/abs/affine, byte-matching `ops/ew.cl`). Factor it into a SHARED function
(moved to `vm_common.cl`, concatenated first, so both `ops/mma.cl`'s epilogue and
`ops/region.cl` call it; SUB_GELU added so the FFN activation is expressible) and call it
from `vmo_mma_tile`'s store site. No new occupancy: the accumulator tile is already in
registers; the epilogue micro-ops reuse it (§27 max-not-sum; the epilogue adds only a
short scalar loop, no new live array).

### Encoding
- **task_t / VmTask / TASK_STRUCT grow by two u32**: `p6` = epilogue descriptor aux
  word-offset (+1; 0 = no epilogue), `p7` = the epilogue's second-input buffer handle
  (residual/bias), loader-patched to a byte offset exactly like dst/a/b (only when p6≠0).
  40B → 48B task; the C++ reads `tasks[en.task]` as a raw struct so all three layouts
  move together.
- **Descriptor** (int words at aux[p6-1]): `[n_micro]` then n_micro × `{kind, src, s_bits,
  t_bits}`. `kind` is a SUB_* op; `src` = 0 (unary on the accumulator), 1 (binary reading
  p7 per-element = residual: `p7[g*M*N + gr*N + gc]`), 2 (binary reading p7 per-column =
  bias: `p7[gc]`). Applied per stored accumulator element at logical (g,gr,gc):
  `v = vmo_region_micro(kind, v, y, s, t)` where y=v (unary) or the p7 read (binary).
- **In-memory Instr fields** `epi` (aux off +1) / `epi_res` (residual buf id), NOT
  serialized — mirrors how aview/bview ride to `to_task` and become task p4/p5. `to_task`
  sets p6=ins.epi, p7=ins.epi_res. The descriptor itself lives in the (serialized) aux
  pool; the kernel finds it through p6. The residual RAW dependency rides in
  `Instr.reads_hint` (OP_DOT READS = {a,b} | reads_hint) so the scheduler/arena-reuse
  order the DOT after the residual is produced.

### Recognizer `_fuse_mma_epilogue` (gated `PJRT_OCL_FUSE_MMA_EPI`, default ON if it lands)
Runs after `_fuse_norm/_fuse_gelu/_fuse_region` (so GELU/scale are already single ops =
epilogue candidates) and before `_reuse_arena`. For each `OP_DOT` that will use TILE_MMA
(skip the N==1 gemv route), greedily walk the SINGLE consumer of the matmul output while it
is a pure-map epilogue op with matching element count (=G*M*N) and not a program output:
  - `OP_GELU` → micro (SUB_GELU, src=0)
  - `OP_AFFINE_F32` (unary, a==cur) → micro (SUB_AFFINE, src=0, s=imm, t=imm2)  [QKᵀ scale]
  - `OP_ADD_F32` (cur + external `res` of size G*M*N) → micro (SUB_ADD, src=1), p7=res  [residual]
Stop at the first non-eligible consumer, a multi-consumer output, a second binary (only one
p7), or a size/view mismatch. Collect the micro chain, retarget DOT.dst to the last folded
op's dst, NOP the folded ops, set DOT.epi / reads_hint. GATED HARD; any mismatch leaves the
decomposed chain untouched. `PJRT_OCL_FUSE_MMA_EPI=0` reverts.

### Prologue (stage 4, phase 2) — scoped, not built this session
Fold the post-reduce LN normalize affine `(x-µ)·rsqrt(v+eps)·g+b` into the following
matmul's LOAD so the normalized activation never materializes. LN's reduce stays a phase;
its normalize fuses into the QKV/FFN1 matmul stage. `PJRT_OCL_FUSE_MMA_PRO`. Deferred: the
epilogue is the higher-leverage half (it removes the residual/activation phases that §29's
tail-of-77-tiny-phases is made of); the LN normalize is already inside the single
TILE_LAYERNORM_SEG op, so a prologue only saves the ONE materialized normalized tensor per
LN, and needs strided/broadcast load addressing for g/b.

### Flash-attention (QKᵀ→softmax→AV) — bigger separate follow-on, scoped only
The softmax REDUCE between QKᵀ and AV makes this a specialized fused instruction (online
softmax / running max+sum), NOT a simple store-epilogue. Out of scope here; noted as the
next matmul-inclusive lever after the epilogue/prologue land.

### Measurements — mechanism REAL + CORRECT, phases collapse 22%, e2e a SMALL win (§14a)
Built (task_t 40→48B, shared micro-interpreter, recognizer, C++ loader) and measured
on RTX PRO 6000 Blackwell (TF32 megakernel; CUDA ref = JAX-on-cuda/cuBLAS).

**Headline — phase count (compute barriers), EPI ON vs OFF:**

| config       | phases OFF | phases ON | Δ         |
|--------------|-----------|-----------|-----------|
| base prefill | 108       | **84**    | −24 (−22%)|
| base decode  | 108       | **84**    | −24 (−22%)|
| large_l1     | 18        | **14**    | −4        |

Fires on all 4 epilogue patterns/layer: QKᵀ→×scale (affine), out-proj→+residual,
FFN1→gelu, FFN2→+residual. The remaining 24 base EW ops are the LN affine `x·γ+β`
(prologue target, not epilogue) + the reshape gathers (not map-fusible).

**Prefill (ms/iter, iters 40):**

| config    | EPI ON | EPI OFF | Δ      | CUDA  | gap ON |
|-----------|--------|---------|--------|-------|--------|
| base      | 5.41   | 5.49    | −1.5%  | 0.436 | 12.4×  |
| large_l1  | 4.465  | 4.471   | wash   | 0.563 | 7.9×   |

**Decode (µs/token):**

| config | EPI ON | EPI OFF | Δ      | CUDA  | gap ON |
|--------|--------|---------|--------|-------|--------|
| base   | 3835   | 3864    | −0.8%  | 464   | 8.3×   |
| large  | 7918   | 7919    | wash   | 1025  | 7.7×   |

**Correctness (all PASS):** 289 pytest + 1 skip; runtime_test NVIDIA+PoCL; TF32
tiny/small/base/large_l1 (4.1e-4 / 1.6e-3 / 4.6e-3 / 4.2e-3); **NVIDIA MEGA_TC=0
scalar-f32 base 2.38e-6** and **PoCL f32 base 2.26e-6** (f32-EXACT — proves the
epilogue math on BOTH the WMMA and the portable scalar mma paths). Occupancy
UNCHANGED: `PJRT_OCL_INFO` lanes=376 measured-residency=376 (the scalar epilogue
loop reuses the accumulator registers, no 128-reg cliff — §27 max-not-sum holds).

### Verdict (§14a): KEEP, default ON — but it does NOT close the gap; here is why
The mechanism is correct everywhere, occupancy-free, collapses 22% of phases, and
is a small consistent win at base (−1.5% prefill / −0.8% decode) with zero
regression at large — the §26/§28 precedent (correct + base needle + no regression
⇒ default ON). `PJRT_OCL_FUSE_MMA_EPI=0` reverts byte-identically.

BUT the honest finding is that a store-epilogue **cannot** reach the literature's
107→7 collapse, and the e2e numbers show why: §29 already pinned base at **matmul
84% / EW+softmax+norm 8% / barrier-floor 8%**, and the phases an epilogue folds are
exactly the *cheap 8%* (residual/activation/scale), not the matmul. So the ceiling
of this lever is ~8%, and we banked ~1.5% of it (the rest is offset because the
folded residual read now counts under the matmul straggler instead of a free
lane-idle phase). Decode is the same: the removed phases are the tiny EW ones; the
memory-bound matVEC weight-read phases (the real decode cost, §32) are untouched.

**What WOULD close it (scoped, not built):** fusing the MATMUL phases *together* —
(a) **flash-attention** QKᵀ→softmax→AV as one instruction (the softmax reduce makes
it a specialized online-softmax op, not a store-epilogue), and (b) **QKV / prologue**
fusion that folds the pre-matmul LN normalize into the matmul load. Those attack the
84%/matvec phases the epilogue leaves alone. The epilogue is the necessary substrate
(shared micro-interpreter, task p6/p7, recognizer) they build on, but on this
serial-block transformer it is a small win, not the frontier's payoff. Prologue
(stage 4) deliberately NOT built this session: it targets the remaining 24 *even
cheaper* LN-affine EW phases, so its e2e ceiling is below the epilogue's — not worth
the strided-load risk until a flash-attention-class lever lands (§14a).

## 34. Flash-attention (QKᵀ→softmax→AV as ONE online-softmax op) — BUILT, MEASURED, REVERTED (default OFF) — 2026-07-21

**Goal (the §33 "what WOULD close it" lever a).** Fuse the batched per-head
attention block `DOT(QKᵀ)·scale → softmax(-1) → DOT(AV)` into ONE instruction
via **online softmax** (flash recurrence: running max `m`, denom `l`, output
accumulator `acc[hd]`, rescale each K/V tile by `exp(m_old−m_new)`), so the
(T×C) score matrix and the (1×C) decode score row NEVER materialize. This
attacks the two MATMUL phases the §33 store-epilogue leaves alone.

### What was built (all correct, all portable)
- **`OP_FLASH_ATTN`(58)/`TILE_FLASH_ATTN`(14)** + a NEW self-contained kernel
  `kernels/ops/attention.cl` (`vmo_flash_attn`): one workgroup per (head, query
  row), streams the C keys in tiles of `lsz`, local tree-reduces for the per-tile
  max/sum, `acc[hd]` in `__local` rescaled per tile. Reuses the shared As/Bs MMA
  panels (no new occupancy) and obeys §18/§19a PoCL rules (uniform-trip key loop;
  cleanup barrier at the loop TOP, never before the backedge; no return before a
  barrier). SCALAR only — did NOT touch `mma.cl`'s tensor-core tile (a parallel
  agent owns it); per the brief, WMMA-inside-attention was out of scope.
- **Read-through-view (correct-by-construction).** Q/K/V are read through the
  SAME strided view descriptors (qv/kv/vv) the decomposed DOT1/DOT2 used, with
  the matmul's own flat-index formula — so the fused op reads byte-identical
  inputs for ANY folded transpose/reshape (decode: kv only; prefill: qv,kv,vv all
  fold). No shape assumption; a wrong layout is impossible because the addressing
  is shared. Task: a=Q, b=K, p0=V (loader-patched), p1=H, p2=T, p3=descriptor
  aux-offset `[H,T,C,hd,scale,causal,qv,kv,vv]`. Loader change: the §33 epilogue
  p7-patch was gated on `kTopMma` so flash may use p6/p7 freely (byte-identical:
  only MMA ever set p6).
- **Recognizer `_fuse_attention`** (lowering, post-`_fuse_mma_epilogue`): anchors
  on `OP_SOFTMAX`, walks BACKWARD through an optional identity-reshape gather +
  scale-affine to DOT1 (scale recovered from its §33 epilogue OR the affine), and
  FORWARD to the single-consumer DOT2. Hard-gated on every shape relation +
  single-consumer linkage + hd≤256; any mismatch → decomposed path untouched.
- Dual validators (tensor interp + schedule sim), `tests/test_ops_flash.py`
  (12 tests: fires on decode/prefill/full-MHA idioms, both validators vs jax,
  disabled/oversized/default-off gates).

### Correctness — PROVEN exact on both engines (the online rescale is right)
289→**301 pytest** (+12) + 1 skip. runtime_test PASS NVIDIA+PoCL. **NVIDIA
MEGA_TC=0 scalar-f32 base FLASH ON = 2.15e-6** (vs decomposed 2.38e-6 — the
online-softmax kernel is f32-EXACT, the reassociation is numerically clean).
PoCL flash-on stable across repeats (finite, matches decode stats — no §19a
heap-corruption; the barrier discipline holds). Isolated decode C=2048 dev-vs-ref
2.7e-5. Occupancy UNCHANGED: lanes=376 (the scalar case reuses As/Bs + a few
scalar registers; no cliff).

### Phase count DROPS as designed — but wall-clock REGRESSES (the §33 wall, again)
Base (6 layers), FLASH ON vs OFF: **prefill/decode 84 → 72 compute phases**
(−12 = 3→1 per attention × 6 layers). The mechanism does exactly what it claims.

But it is a **measured wall-clock REGRESSION everywhere it fires**, worsening
with sequence length (RTX PRO 6000, TF32 megakernel, A/B via `PJRT_OCL_FLASH`):

| workload (where flash fires)         | flash ON | flash OFF | ON/OFF |
|--------------------------------------|----------|-----------|--------|
| prefill base   (T=128)  ms/iter      | 12.20    | 5.41      | **2.3× slower** |
| prefill base_s512 (T=512) ms/iter    | 9.56     | 1.85      | **5.2× slower** |
| prefill base_s1k  (T=1024) ms/iter   | 33.84    | 2.94      | **11.5× slower** |
| decode base    (C=256)  µs/token     | 4318     | 3807      | **1.13× slower** |

**Why (two independent killers):**
1. **Prefill replaces two TF32 tensor-core matmuls with a scalar kernel.** QKᵀ
   and AV run on WMMA in the decomposed path; the scalar online-softmax kernel
   cannot compete, and the gap grows with T (2.3×→11×). The 12 phases saved are
   dwarfed by losing the tensor cores — §29 already pinned base at *matmul 84%*,
   and this makes the matmul SLOWER, not faster.
2. **Decode (T=1) runs only H workgroups** (=8 for base). One WG per (head,query)
   → 8 of 376 lanes busy; the decomposed gemv path parallelizes over the C
   dimension (more tiles). And §32 already showed decode is **weight-HBM-bandwidth
   bound**, not attention-bound — even a perfect attention op barely moves decode
   e2e.

### Extra finding: it can't even fire at long context yet
`_fuse_attention` anchors on `OP_SOFTMAX`, which `_fuse_norm` only emits for
**seg ≤ 1024** (its local-staging cap, §19a). So for C>1024 (base_c2k/c4k,
base_s2k) softmax stays decomposed and flash never fires — the "wash" numbers
first seen there were both-decomposed (dev-vs-dev diff 0.00). The target
long-context regime is exactly where it's blocked. Firing it there needs a
second recognizer anchored on the decomposed `REDUCE_SEG` softmax; not built,
because the increasing-regression trend (2.3×→11× as T grows 128→1024) makes the
outcome certain — a longer C only does MORE serial scalar work.

### Verdict (§14a): REVERT — default OFF, kept behind `PJRT_OCL_FLASH=1`
Correct, portable, occupancy-free, collapses 14% of phases — but a wall-clock
regression wherever it fires and blocked at the long context it targets. Per the
§14a rule (keep only if it moves the needle) and the brief's PoC gate ("if it
doesn't win even at long context, STOP and report"): **default OFF**, byte-
identical revert. The op/kernel/recognizer are KEPT as the correct substrate for
the version that *could* win — needs (a) **tensor-core** QKᵀ/AV inside the
attention body (WMMA, the §33 substrate doesn't cover this), and (b) **split-KV /
partial+reduction** (survey §1.1) so decode's low (H·T) occupancy fills the GPU.
Both are large, hardware-specific builds; on this serial-block f32 transformer a
scalar single-pass flash is the wrong tool. `PJRT_OCL_FLASH=1` re-enables it for
that future work. NOT merged — reported for review.
## 35. NVIDIA cp.async multi-stage tensor-core matmul — cp.async is DEAD on this ICD; standalone tile clears the PoC gate but does NOT transfer to the megakernel (2026-07-21, poc/17)

**Goal.** Close the matmul gap (in-megakernel TF32 ~18-20 TFLOP/s vs cuBLAS
116-133, §31/§10d) with the one thing that requires leaving portable OpenCL: a
cuBLAS-class WMMA tile — a **multi-stage register-blocked pipeline with
`cp.async`** (async global→shared copy via inline PTX) so weight loads overlap
tensor-core compute and the smem-bandwidth bottleneck (§31) is relieved.
NVIDIA-only behind `VMO_NV_PTX`; portable path untouched. PoC-gated first
(poc/17-nv-mma). Hardware: RTX PRO 6000 Blackwell (sm_120), driver 595.71.05.

### RESULT 1 (the blocker): cp.async does not EXECUTE on this OpenCL ICD
`poc/17/probe.c` — the driver **emits correct PTX** (dumped binary:
`.target sm_120`, `.version 9.2`, literal `cp.async.cg.shared.global [%r1],[%rd2],16;`
+ `cp.async.commit_group; cp.async.wait_group 0;`), ptxas accepts it, the kernel
runs — **but the async copy never delivers data.** Every completion form is WRONG
(1024/1024 mismatch): `.cg`+`wait_group 0`, `.ca`, `+fence.proxy.async.shared::cta`,
`wait_group`+200k-iter spin (rules out a wait-only bug), and
`cp.async.mbarrier.arrive`+`mbarrier.try_wait`. The CONTROL — a **synchronous
`st.shared` through the IDENTICAL `cvta.to.shared`-derived shared address** — is
CORRECT (0 mismatch). So the shared-address mapping is fine and **cp.async itself
is the broken primitive**: the NVIDIA *OpenCL* runtime does not wire up the
Ampere+ async-copy unit (a CUDA-only path here). **The deep software pipeline
cuBLAS-class GEMM needs cannot be built through OpenCL→PTX on this driver** — the
§14a "the path can't express it, and here is the measurement" outcome.

### RESULT 2: the WMMA ceiling WITHOUT cp.async (synchronous staging), standalone
`poc/17/bench17` — standalone tf32 m16n16k8 tile (NO megakernel barrier, so free
of the co-residency cap), best-of-7, warmed clocks:

| tile | 2048³ TF/s | 4096³ TF/s |
|------|-----------|-----------|
| 64×64  1-buf (== in-megakernel tile shape) | 27.2 | 29.6 |
| 128×64 2-buf | 37.7 | 46.2 |
| **128×128 BK16 2-buf (the knee)** | **47.5** | **55.2** |
| 128×128 BK32 2-buf | 25.1 | 30.7 |
| 256×128 (any buf) | ~33 | ~38 |
| cuBLAS | 116.5 | 133.3 |

**The 128×128 synchronous double-buffered tile CLEARS the ≥40-60 gate at
47.5/55.2 TF/s** (2.5-3× the in-megakernel tile). But the win is (a) tile
**intensity** (128×128 register accumulator), not async staging — synchronous
double-buffer (ILP overlap of `ld.global` with the tensor cores) is all that is
available and it already helps (128×64: 1-buf 35.6 → 2-buf 46.2 at 4096); (b)
**still ~2.4× under cuBLAS** (no 3-4 stage latency-hidden pipeline without
cp.async; tf32 m16n16k8 is a smaller MMA than cuBLAS's; no smem swizzle beyond
the §10d LDS pad).

### RESULT 3: the standalone ceiling does NOT transfer into the megakernel
`poc/17/mmbench.py` on the real plugin (`PJRT_OCL_MM_KERNEL=0`, matmul stays in
the VM), reproduces §31 exactly:

| in-megakernel matmul | 2048³ | 4096³ |
|----------------------|-------|-------|
| 64-tile (shipped default) | 19.3 | 17.4 |
| 128×128 + pipe (`MEGA_BIGTILE`+`MEGA_PIPE`, §31) | 14.2 | 14.0 |

The **identical 128×128 tile is 55 TF/s standalone but 14 in-megakernel.** The
gap is entirely the megakernel structure: the cross-workgroup spin-barrier forces
all lanes co-resident (the 64-accumulator tile crosses the 128-reg cliff → 188
lanes, no oversubscription to hide latency, §27) and the whole-VM register budget
is a max over every op path. A dedicated kernel pays none of this. cp.async was
the sanctioned lever to relieve that latency/smem-BW bottleneck in-megakernel —
and it does not work here.

### DECISION (§14a): do NOT integrate, do NOT change the default; keep it honest
- cp.async — the mechanism the whole thesis required — is **non-functional** on
  this OpenCL ICD (RESULT 1).
- Wiring the 128×128 tile into the megakernel is an **already-measured
  regression** (§31, re-confirmed RESULT 3): 14 vs 19 TF/s and base e2e ~1.9×
  worse, because its speed is a dedicated-kernel property the co-residency barrier
  destroys.
- The genuine remaining lever is the **hybrid split** (route the big
  projection/FFN matmuls to a dedicated 128×128 TF32 kernel — poc/17's 55-TF/s
  tile — while the persistent megakernel keeps every non-matmul phase). §10d
  measured plain host-dispatch as overhead-bound (+12.5 ms/large); the hybrid
  needs per-segment megakernel relaunch (arena persists → feasible) and a
  batch-aware standalone kernel. Large architectural change, deferred — but poc/17
  now supplies the standalone kernel it would dispatch to and its measured ceiling.

**Portability + gates confirmed (nothing shipped touched — poc/17 is standalone):**
default pytest **289 passed / 1 skipped**; `runtime_test` PASS; NVIDIA TF32
`--check` tiny/small/base/large **PASS** (max_abs 1.3e-4…3.0e-3, TF32 tol);
NVIDIA portable `MEGA_TC=0` f32-exact (max_abs 2.4e-6); PoCL portable f32-exact
(7.2e-7). Default e2e unchanged: base 5.41 ms (gap 12.2× vs CUDA 0.44), large
27.3 ms (gap 7.4× vs CUDA 3.67). **NOT merged — reported for review.**

**Reproduce:** `cd poc/17-nv-mma && make && ./probe` (cp.async verdict) `&&
./bench17` (standalone ceiling); `JAX_PLATFORMS=opencl PJRT_OCL_MM_KERNEL=0
python mmbench.py 4096` (in-megakernel; add `PJRT_OCL_MEGA_BIGTILE=1
PJRT_OCL_MEGA_PIPE=1 PJRT_OCL_MMA_T=128` for the big tile).

## 36. Standalone TF32 WMMA tile pushed to the sync ceiling + hybrid dispatch — the hybrid WINS 1.45x on large (compute-bound), loses on base (2026-07-21, poc/17 + mm_tc)

Follow-on to §35. Two parts: (1) "optimize the standalone tile to hell"; (2)
route the transformer's big matmuls to it (hybrid) and measure e2e.

### Part 1 — the sync-only WMMA ceiling is ~57 TF/s; every classic GEMM lever is a WASH
`poc/17/bench17` (parametrized: TM/TN, BK, NBUF, warp grid WM×WN, VEC4 float4
staging, PAD, PIPE fragment-pipeline), best-of-7, warmed, RTX PRO 6000 Blackwell:

| config | 2048³ TF/s | 4096³ TF/s |
|--------|-----------|-----------|
| 128×128 W4×2 BK16 NBUF2 (§35 base, scalar uncoalesced-B) | 47.4 | 56.2 |
| + float4 coalesced staging (VEC4) | 43.8 | 57.0 |
| + fragment-pipeline (PIPE) | 43.9 | 57.7 |
| 128×256 / 256×128 (512–1024 t, ≤64 acc/thr) | 36–41 | 47–54 |
| BK32, PAD8, more warps | all ≤ base | all ≤ base |
| cuBLAS (ref) | 116.5 | 133.3 |

**Nothing beat the §35 tile by more than noise (~±3%).** The B global load WAS
stride-N uncoalesced; fixing it (coalesced/float4) moved 4096 by 56.2→57.0 —
noise, because the tile is **not** global/L2-BW bound. The lane sweep is decisive:
188 lanes (1 WG/SM)=33, 376 (2 WG/SM)=56, ≥564 no gain. We are **pinned at 2
WG/SM = 16 warps/SM** — a 128×128 f32 accumulator is 16384 regs, so 256 threads
already spend 64 regs/thread on accumulators; the register file (65536/SM) caps
residency at 2 blocks and there is no oversubscription to hide latency. That is
exactly the hole cp.async multistage fills — and cp.async is DEAD here (§35). So
**~57 TF/s @4096 / ~47 @2048 is the honest sync ceiling, ~2.3× under cuBLAS**;
the gap is latency-hiding the register file cannot buy and PTX cannot async its
way around on this ICD. Bigger tiles are impossible (a 256×256 f32 accumulator =
65536 regs = the whole SM file).

### Part 2 — hybrid dispatch: standalone mm_tc for the big matmul phases, VM for the rest
Ported the §35 tile into the plugin as **`mm_tc`** (vm_main.cl, `#ifdef
VMO_NV_PTX`, same arena/IO-port ABI as mm2). It replaces the scalar SGEMM mm2 on
the GPU/TF32 pure-matmul fast path: **standalone matmul 20.9/24.4 → 43.5/53.0
TF/s @2048/4096** (mmbench, 2.2×), TF32-exact incl. ragged M/N/K (max_rel 7e-4).

`PJRT_OCL_MM_HYBRID=1` then interleaves mm_tc into a full program. The single
persistent spin-barrier megakernel **cannot be split mid-flight**, and its base
is 8ms cheaper than host-dispatch (27.7 vs 35.4 ms/large) — so the hybrid runs on
the **host-dispatch engine** (already segments per-phase, enqueues back-to-back
on the in-order queue with NO clFinish) and, for any phase that is EXACTLY big
TF32 matmul tasks, flushes the pending VM phases then enqueues mm_tc per matmul
(patched dst/a/b handles — the raw prog_.tasks carry buffer ids; arena+id is
misaligned → context loss, the -36 trap). The scheduler packs the independent
Q/K/V projections into one phase, so a routable phase carries several matmuls.

**The FFN/out matmuls are epilogue-fused (§33 R2c, p6≠0) and mm_tc has no
epilogue path** — so MM_HYBRID also disables `_fuse_mma_epilogue` in lowering
(one switch), leaving the big FFN matmuls plain-routable and running GELU/residual
as their own cheap VM phases. That is what unlocks the win:

| large (6-layer, B8 T256 D1024 F4096) | ms/iter | vs default |
|--------------------------------------|---------|-----------|
| spin-barrier default (§35 baseline) | 27.7 | 1.00× |
| host-dispatch base (engine cost) | 35.4 | 0.78× |
| host-dispatch + noEpi base | 40.1 | — |
| **HYBRID (host + mm_tc, routes 30 matmuls/iter)** | **19.1** | **1.45×** |
| CUDA (ref) | 3.69 | — |

`large` is ~92% big-matmul FLOP; routing them to ~45 TF/s (from ~17 in-VM) saves
~21 ms, and even after eating the host-dispatch engine's +8/+12 ms overhead the
net is **27.7 → 19.1 ms, gap to CUDA 7.5× → 5.2×.** `large_l1` 4.53→3.42 (1.32×).
Correct on all (max_abs 1.2e-2, TF32 tol). **base REGRESSES 5.4→7.3 ms**: its
matmuls are M=512 (16–64 tiles, mm_tc starves the SMs) and it is overhead-bound,
so the host-dispatch tax is unrecovered — the §29/§14b small-op wall, unmoved.

### DECISION (§14a): keep, OFF by default, opt-in for compute-bound large configs
- **mm_tc pure-matmul fast path (mm2→mm_tc on GPU/TF32): kept ON** — strictly
  better (2.2×), TF32-exact, NVIDIA-only (portable mm2 untouched).
- **Hybrid: opt-in `PJRT_OCL_MM_HYBRID=1`** — a clean 1.3–1.45× on compute-bound
  large transformers, a loss on small/overhead-bound ones. Correct either way.
  It rides the host-dispatch engine (the megakernel can't be segmented without
  core-VM pc-range surgery); a spin-barrier-segmented hybrid would start from the
  27.7 ms base instead of 35.4 and land ~16 ms — the biggest remaining lever, and
  the reason this is not yet default.

**Gates:** default pytest **301 / 1 skip**; NVIDIA TF32 `--check`
tiny/small/base/large **PASS** (max_abs 4e-4…1.3e-2); NVIDIA `MEGA_TC=0`
f32-exact (2.4e-6); PoCL f32-exact (7e-7); default large unchanged (27.8 ms).
Hybrid `--check` large **PASS** (1.2e-2). **NOT merged — reported for review.**

**Reproduce:** `cd poc/17-nv-mma && make && ./bench17` (Part 1 sweep);
`JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=NVIDIA PJRT_OCL_MM_KERNEL=1 python
mmbench.py 4096` (mm_tc fast path); `PJRT_OCL_MM_HYBRID=1 python
tools/bench_transformer.py --config large` (hybrid e2e; `--check` for
correctness; `PJRT_OCL_MM_HYBRID_DBG=1 --iters 1` to log routed matmuls).
## 37. Diverse workload testbench: coverage + perf survey vs native CUDA (2026-07-21)

**What/why.** Built `tools/bench_suite/` — 18 seeded, f32, jitted workloads spanning AI
(MLP, CNN, LSTM, GRU, transformer, attention, layernorm, batchnorm, embedding+softmax) and
scientific/physics (2D heat stencil, N-body, RK4 Lorenz, logistic map, Monte-Carlo π, FFT,
spring-mass chain, Hodgkin-Huxley neuron, a **real brax** `inverted_pendulum` reset+step) — to
measure how far the transformer-tuned backend generalizes, and to let real failures rank the M3
op-coverage backlog. `run_suite.py` runs each workload through OUR OpenCL VM and native JAX CUDA
in separate subprocesses (backend is process-global), catches lowering failures, extracts the
missing op, compares outputs, and emits `docs/workload-coverage.md`. This is a survey — no
plugin/product code changed.

**Result: 11/18 run on our backend (61%), all numerically correct vs CUDA** (allclose
atol/rtol 2e-2; the "max rel" blowups are near-zero-reference artifacts, not real error).

**Perf spread (passers, ours/CUDA gap):** median **~7x**, range **0.81x–18.7x**.
- *Near/beating CUDA (≤~3x):* layernorm **0.81x** (the §19 fused-reduce idiom — the one place we
  win), logistic_map **1.03x** and other `while`/`scan` loops (per-iteration barrier overhead
  dominates BOTH sides so the ratio collapses), attention 2.4x, spring_mass 3.2x.
- *Far (≥~7x):* transformer **18.7x**, LSTM/GRU 7–8x, heat2d 7.9x, rk4 7.3x, hh_neuron 11.9x —
  matmul-heavy or long small-op/scan chains, exactly where XLA fuses elementwise runs and calls
  cuBLAS/TF32 while we pay per-instruction barriers + un-fused arena traffic. Consistent with the
  §14/§36 transformer picture; the survey shows it holds across workload classes.

**Coverage gaps, ranked by workloads unlocked (the M3 test-driven priority list):**
1. **partial-axis reduce** (2: batchnorm, nbody) — we support only full or innermost-suffix
   reductions (§ reduce). A reduce over axis 0 / a middle axis is rejected at lowering. Cheapest
   high-value win: a permuting gather before REDUCE, or a strided REDUCE tile op. Also the single
   most common "almost worked" idiom (mean/var/sum over a non-last axis).
2. **stablehlo.convolution** (1 here: cnn; would also unlock any real CNN / brax-vision) — no
   conv in the library at all.
3. **stablehlo.gather** (1 here: embedding_softmax; ALSO required by brax's step HLO) —
   data-dependent indexing; only `dynamic_slice` exists today.
4. **stablehlo.shift_right_logical** (1: monte_carlo) — threefry RNG lowers to bit-shifts we
   lack, so *any* `jax.random` workload fails. Small op, unlocks the whole RNG/Monte-Carlo class.
5. **complex dtype** (1: fft) — FFT emits `complex<f32>`; our arena has no complex storage. FFT
   op itself also absent. Large lift; low priority.
6. **platform allowlist** (1: brax_step) — NOT a lowering gap: brax/mujoco **reject our custom
   PJRT platform at env-construction** (`Unsupported device: OclDevice … platform "NVIDIA CUDA"`),
   before any op dispatches. So brax can't target us without a host-side patch; even if it could,
   its step needs gather + scatter + case + atan (confirmed from the CUDA stablehlo dump). A
   hand-written spring-mass analogue (`spring_mass`, PASS 3.2x) represents the physics class.

**Install/compat findings.** `brax` 0.14.2 installs cleanly under jax 0.10.2 and runs a real env
on CUDA. `jaxley` 0.13.0 installs but hits a **version wall at runtime** under jax 0.10.2:
`jnp.clip(a_max=…)` (removed kwarg) → `TypeError` inside `jaxley.solver_gate.save_exp`. Recorded,
not fought; the `hh_neuron` hand-written Hodgkin-Huxley analogue (PASS 11.9x) covers that class.

**Honest bottom line.** The backend **generalizes in coverage** — every f32 elementwise / plain
matmul / suffix-reduce / scan-or-while program that dodges the six gaps above simply runs and is
correct, across AI and scientific workloads. But its **tuning generalizes narrowly**: only the
layernorm/softmax fused-reduce idiom it was optimized for lands at CUDA parity; everything
matmul- or loop-heavy sits 7–20x behind. The top ROI item by breadth is **partial-axis reduce**
(unlocks 2 immediately, is the most common near-miss), then **gather** (2 real workloads counting
brax) and **threefry shift ops** (unlocks the entire RNG class from one small op).

**Reproduce:** `. ./env.sh && .venv/bin/python tools/bench_suite/run_suite.py --md
docs/workload-coverage.md` (add `--only <names>` for a subset). Verified: full 18-workload run
reproduces 11 PASS / 7 FAIL with stable gaps across two runs.

## 38. SHIPPED: general data-dependent gather (stablehlo.gather → OP_GATHER_INDEX) — unlocks embedding_softmax (2026-07-21)

**What/why.** Closes coverage gap #3 (§37). Before this, the only gather we had was
`OP_GATHER_STRIDED` — a *compile-time-affine view* (broadcast/transpose/slice/reverse). Real
gathers (embedding lookup `emb[ids]`, `take`, brax's step) need each output element's operand
base offset to come from a **runtime start_indices tensor**. New op `OP_GATHER_INDEX` /
`kTopGatherIndex` (tile op 15) does exactly that, fully general over stablehlo's gather
`dimension_numbers`.

**Design.** The whole gather reduces to a flat per-output-element affine form the kernel
evaluates (all row-major, one output element per grid-stride step):
`op_off(i) = Σ_e coord_e(i)·op_stride[e] + Σ_k clamp(S_k, 0, dim−slice)·idx_op_stride[k]`,
`S_k = start_indices[Σ_e coord_e(i)·si_stride[e] + k·si_vec_stride]`, then `dst[i]=operand[op_off]`.
The two stride arrays are disjoint (op_stride≠0 only on OFFSET output dims, si_stride≠0 only on
BATCH output dims), so both sums accumulate in one decode loop with no per-dim branch. The lowering
(`ops/gather_index.py`) computes op_stride/si_stride/idx_op_stride/clamp_max/si_vec_stride from the
dnums (offset_dims↔non-collapsed operand dims; batch output dims↔start_indices batch dims;
start_index_map↔operand dims). Whole-element copy ⇒ dtype-agnostic (esz mover, i32/i64 indices).
Reuses the dynamic_slice **loader-patch** trick: the start_indices location can't be known at
lowering time (arena reuse moves offsets; inputs live in I/O ports), so aux carries the buffer id
and the C++ loader patches the byte-offset/port-handle word at load (`elem_off`); the dep on the
indices buffer rides in `reads_hint`.

**Verified.** New `tests/test_ops_gather.py` (6 cases: 1D embedding, embedding+softmax,
2D index batch, i64 indices, two-component (row,col) scalar gather, clip/OOB clamping) — both
numpy validators PASS. Full pytest **307 passed / 1 skipped** (was 301). E2e through the real
plugin: raw `emb[ids]` is **bit-exact** vs numpy on NVIDIA (TF32 + MEGA_TC=0 f32-exact) and PoCL
CPU; `embedding_softmax` end-to-end allclose vs jax-CPU (maxabs 5e-6 TF32, 2e-8 exact, 1.5e-8
PoCL). `bench_suite --only embedding_softmax`: now **PASS, ours 0.078 ms vs CUDA 0.072 ms =
1.08x**, correct=close — coverage now **12/18**. (The 1.08x is a small-op program; the gather is a
copy-bound view, so it rides at CUDA parity like the other reduce/loop-bound passers.)
