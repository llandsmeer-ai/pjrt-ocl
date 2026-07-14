# Decision log (tree)

Institutional memory of the project. Every design exploration gets a node: **TRIED** (with the
actual error/measurement), **FAILED/OK**, **CHOSEN** and why. Update in the same session as the
exploration. Nested bullets = sub-decisions opened by a parent choice.

Legend: ✅ chosen · ❌ tried & rejected (keep the evidence!) · 🔬 open, needs PoC · 🅱️ fallback kept viable

## 1. Execution model

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
