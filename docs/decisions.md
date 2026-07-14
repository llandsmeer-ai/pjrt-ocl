# Decision log (tree)

Institutional memory of the project. Every design exploration gets a node: **TRIED** (with the
actual error/measurement), **FAILED/OK**, **CHOSEN** and why. Update in the same session as the
exploration. Nested bullets = sub-decisions opened by a parent choice.

Legend: ✅ chosen · ❌ tried & rejected (keep the evidence!) · 🔬 open, needs PoC · 🅱️ fallback kept viable

## 1. Execution model

- ✅ **Device-side megakernel VM** (persistent kernel, opcode switch) — user decision 2026-07-14.
  Motivation: minimal dispatch overhead.
  - ✅ **Strictly linear bytecode, no jumps/conditionals** — user decision 2026-07-14. StableHLO has
    no jump ops (verified against spec, see docs/stablehlo-notes.md); region ops (`while`/`if`/
    `case`/`reduce`/...) lower to one instruction referencing nested linear instruction lists,
    interpreted by the VM.
    - ❌ pc-manipulation/jumps in bytecode — rejected: user prefers stupid-linear execution;
      nothing in StableHLO needs it.
  - 🔬 **Cross-workgroup barrier** — #1 project risk. Candidate: persistent threads + atomic
    arrival counter with acq_rel/seq_cst atomics on global memory; only safe with all workgroups
    co-resident (occupancy-derived launch size). → `poc/01-device-vm`
  - 🔬 **Opcode dispatch** — no function pointers in OpenCL C → single big switch; risk: compile
    time/register pressure as op library grows. Mitigation candidate: split VM by op family.
  - 🅱️ **Host-side dispatch loop** over the same bytecode (one clEnqueueNDRangeKernel per instr).
    Keep the bytecode dual-interpretable so this fallback stays cheap to activate.

## 2. StableHLO ingestion

- ✅ **Link MLIR + StableHLO C++ libs, built once via CMake** (outside repo, e.g. ~/third_party) —
  user decision 2026-07-14. Robust for MLIR bytecode + VHLO portable artifacts; gives us MLIR infra
  for a future custom `vm` dialect.
  - 🔬 Pin LLVM + stablehlo commits (record here once built); match stablehlo version to the JAX
    release we pin.
  - ❌ Hand-written textual-MLIR parser — rejected without PoC: fragile across JAX/MLIR versions,
    can't read bytecode/VHLO artifacts.
  - ❌ Python-side lowering shim — rejected: non-standard plumbing through PJRT compile.

## 3. Kernel strategy

- ✅ **Generic shape-agnostic kernel library** (strides/shapes as runtime args), compiled once per
  device at init, program binaries cached on disk. Start with a tiny op set, expand only when e2e
  works — user decision 2026-07-14.
  - 🔬 Kernel-table override mechanism for tuned per-vendor variants (M5), incl. specialized matmul.

## 4. PJRT layer

- 🔬 **Hand-rolled PJRT C API** (vendored `pjrt_c_api.h`, no XLA source dep, CMake-only) — try this
  FIRST (user preference), but user predicts it may fail; record every crash/unimplemented-callback
  incident here before switching. → `poc/02-pjrt-skeleton`
  - 🅱️ **XLA C++ wrapper route**: inherit `xla::PjRtClient` (cf. `tfrt_cpu_pjrt_client.h`), then
    `pjrt::CreateWrapperClient` + `pjrt::CreatePjrtApi` via `pjrt_c_api_wrapper_impl.h`
    (reference: `pjrt_c_api_cpu.cc`). Cost: full XLA Bazel build. See docs/pjrt-integration.md.
  - Mandatory C API surface per openxla docs: `GetPjRtApi`, `PJRT_Client_Create`;
    optional: `PJRT_Plugin_Initialize`, `PJRT_Plugin_Attributes`, `PJRT_TopologyDescription_Create`.

## 5. Python packaging / discovery

- ✅ **Entry-points discovery** (`[project.entry-points.'jax_plugins']`) — recommended by openxla
  docs over bare `jax_plugins/` namespace dirs. `initialize()` calls
  `xla_bridge.register_plugin('opencl', priority=500, library_path=..., options=None)`.
  priority>400 makes it win under `JAX_PLATFORMS=''`; during dev prefer explicit `JAX_PLATFORMS=opencl`.
  - 🔬 jaxlib ↔ PJRT C API version matching is strict (no ABI guarantee yet): pin JAX and record the
    `PJRT_Api` major/minor we build against.

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
