# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

**pjrt-ocl**: a pip-installable PJRT plugin (python package `pjrt_ocl`) that lets JAX (and
eventually PyTorch/XLA) execute on **any OpenCL-capable hardware**. Primary targets: **Intel Xe2,
AMD MIxxx (CDNA), NVIDIA GPUs**. The OpenCL platform/device is **configurable** (env var
`PJRT_OCL_DEVICE=<platform substring>[:<device index>]` + PJRT client-create options; default:
first GPU, else first CPU). Development order: **CPU first via PoCL** (easier debugging, printf,
sanitizers), then the local NVIDIA GPU, then other vendors. Do not publish to PyPI yet — the goal
for now is "works end-to-end locally".

Core design: programs are **not** compiled per-dispatch. At PJRT-compile time we lower StableHLO
into a flat **bytecode** (a list of instructions referencing a fixed kernel/opcode table). At
execute time a **device-side megakernel VM** (persistent OpenCL kernel, à la ThunderKittens)
interprets that bytecode: a big opcode switch, cross-workgroup barriers between instructions.
The OpenCL C kernel library is generic (shape-agnostic, strides as runtime args), compiled once
per device at plugin init and cached as program binaries — the OpenCL compiler is never invoked
on the hot path.

## Architecture (the pipeline)

```
jax.jit(f)                                  [user]
  → StableHLO (serialized MLIR)             [JAX/XLA does this]
  → PJRT_Client_Compile                     [our plugin: hand-rolled PJRT C API impl]
      → spawn venv-python lowering subprocess (pjrt_ocl.lowering)
        — uses jaxlib's OWN StableHLO python bindings ⇒ version-matched to JAX for free
        — python exe path arrives via register_plugin options → PJRT_Client_Create
      → lowering (in Python): parse VHLO artifact, buffer assignment, instruction selection
      → VMProgram { const pool, buffer plan, instr list }   ← the "bytecode", plain binary format
      → C++ plugin only ever sees VMProgram bytes; it is a pure executor
  → PJRT_LoadedExecutable_Execute
      → upload instr list once; enqueue vm.cl megakernel
      → device VM: for(pc..){ switch(op){...}; global_barrier(); }
```

Key properties of the execution model:
- **Strictly linear bytecode — no jumps/branches, ever.** StableHLO itself has no jump ops (all
  control flow is structured regions: `while`, `if`, `case`, and region-carrying ops like
  `reduce`/`sort` — see docs/stablehlo-notes.md). Region ops lower to a single instruction whose
  operands reference **nested instruction lists**; the VM interprets a `while` instruction by
  alternately running the cond sub-list and body sub-list. Sub-lists are themselves linear.
- The VM launches with a **fixed grid sized to device residency** (persistent threads); every
  instruction internally uses grid-stride loops over its logical iteration space.
- Cross-workgroup barrier between instructions is the hardest portability risk (see docs/decisions.md).
  Fallback plan B (keep viable at all times): the same bytecode interpreted by a host-side loop of
  `clEnqueueNDRangeKernel` calls. Design the bytecode format to be interpretable both ways.

## How to work (session & agent organization)

- One session = one coherent work block (a milestone or a PoC). Cross-session state lives ONLY in
  git + CLAUDE.md + `docs/` — assume the next session remembers nothing else. Start each session
  by reading `docs/decisions.md` and `git log --oneline -15`.
- Keep the current riskiest/most iterative item **in the main session** (e.g. poc/01 barrier
  work); delegate self-contained, verifiable chunks to **background agents in git worktrees**
  (e.g. poc/02 PJRT boilerplate, later per-op-family coverage work in M3). Long dumb jobs
  (LLVM build) run as background shell tasks in parallel.
- Delegated agents must return: what they tried/failed (for `docs/decisions.md` — the main
  session merges these entries) and how their result was verified. Unverified agent work is
  treated as not done.
- Commit small and often; never end a session with undocumented design findings.

## Hard rules

- **PoC-first**: every risky mechanism gets a minimal standalone proof-of-concept under `poc/NN-name/`
  before being integrated. Never integrate unproven mechanisms into the main tree.
- **Decision log**: `docs/decisions.md` is a decision tree. Every design exploration gets an entry:
  what was tried, what failed (with the actual error/measurement), what was chosen and why.
  Update it in the same session as the exploration — this is the project's institutional memory.
- **Portability discipline**: core VM and kernel library use OpenCL 3.0-core features only
  (target the 1.2-ish common subset; feature-detect at init). No vendor extensions in the core
  path — vendor-specific tuning goes behind the kernel-table override mechanism. Never assume fp64.
- `docs/` holds distilled references (PJRT API notes, StableHLO op semantics, OpenCL memory-model
  notes). When you burn >15 min figuring out an external API fact, write it down there.

## Environment facts (verified 2026-07-14)

- Two OpenCL platforms available (`clinfo -l`): NVIDIA CUDA (RTX PRO 6000 Blackwell Max-Q) and
  PoCL (AMD Ryzen 9 3900X CPU, `pocl-opencl-icd`). **Develop and test on PoCL first** — printf,
  host debuggers and sanitizers work there — then validate on NVIDIA. The NVIDIA ICD had to be
  registered manually: `/etc/OpenCL/vendors/nvidia.icd` containing `libnvidia-opencl.so.1`.
- `sudo` available without password. Python 3.12.3, CMake, gcc/g++/clang, ninja NOT yet installed.
- `opencl-headers` + `ocl-icd-opencl-dev` + `clinfo` + `ninja-build` installed.
- **jax 0.10.2 / jaxlib 0.10.2** in `.venv/` (project venv; use `.venv/bin/python` for everything
  Python). Record the matching PJRT C API version in `docs/decisions.md` once known.
- **Only ~5 GB free disk** (host-shared overlay; not cleanable). This killed the build-LLVM plan —
  see docs/decisions.md §2. Do NOT start large source builds; lowering uses jaxlib's bundled
  StableHLO python bindings instead.

## Planned repo layout

```
pjrt_plugin/           C++ plugin: PJRT C API impl + OpenCL runtime (pure VMProgram executor)
  pjrt/                hand-rolled PJRT C API surface (client/device/buffer/executable)
  runtime/             OpenCL context/queue/allocator, binary cache, VM launcher,
                       lowering-subprocess plumbing
  kernels/             vm.cl megakernel + generic op library (OpenCL C)
python/pjrt_ocl/       python package: jax_plugins entry point, packaging, AND
  lowering/            stablehlo → VMProgram lowering (runs as compile-time subprocess)
poc/                   numbered standalone proof-of-concepts (each with its own README)
tests/                 pytest, comparing against JAX CPU backend
docs/                  decisions.md (decision tree), reference notes
```

## Commands

(These are the contract; keep them working as the code appears.)

- Configure/build plugin: `cmake -S . -B build -G Ninja && cmake --build build`
- Build+run a PoC: `cmake --build build --target poc-NN && ./build/poc/NN-name/poc-NN`
- C++ unit tests: `ctest --test-dir build`
- Python e2e tests: `pytest tests/` (needs `pip install -e python/` once packaging exists)
- Smoke test JAX sees us: `JAX_PLATFORMS=opencl python -c "import jax; print(jax.devices())"`
- Select OpenCL backend: `PJRT_OCL_DEVICE="Portable"` (PoCL CPU) / `PJRT_OCL_DEVICE="NVIDIA"` —
  platform-name substring, optional `:<device index>`
- Inspect what JAX will hand us: `python tools/dump_stablehlo.py "<jax expr>"` (write this tool early;
  `jax.jit(f).lower(args).compiler_ir('stablehlo')` is the API)
- OpenCL sanity: `clinfo -l`

## Milestones

Work through these in order; each has an explicit exit criterion. Details/status live in
`docs/roadmap.md` once created.

- **M0 – PoCs for the three risky mechanisms (parallelizable, standalone):**
  - `poc/01-device-vm`: pure OpenCL persistent megakernel interpreting a hand-written instruction
    list (add/mul on buffers). Must prove: opcode switch dispatch, grid-stride execution,
    **cross-workgroup barrier via atomics** (acquire/release on global mem), residency-limited
    launch sizing. Bring it up on PoCL first, then validate on NVIDIA. Measure per-instruction
    overhead vs separate kernel launches. This PoC decides whether device-VM survives or we fall
    back to plan B.
  - `poc/02-pjrt-skeleton`: minimal .so implementing just enough PJRT C API (from a vendored
    `pjrt_c_api.h`) that `jax.devices()` lists our device. This tests the "hand-rolled C API vs
    XLA C++ wrappers" question — expect friction; record every unimplemented-callback crash in
    the decision log before considering the XLA-wrapper route.
  - `poc/03-python-lowering`: pure-Python PoC: take `jax.jit(f).lower(...)` → serialized VHLO
    artifact bytes → deserialize with `jaxlib.mlir` bindings → walk stablehlo ops → print
    ops/types/shapes and emit a strawman VMProgram. Also prove subprocess plumbing: read artifact
    from stdin, write VMProgram to stdout.
- **M1 – Bytecode + lowering:** define `VMProgram` serialization (opcode enum, operand/buffer
  refs, launch geometry); implement stablehlo→bytecode for elementwise f32 ops + constants;
  static buffer plan (arena + offsets, SSA liveness for reuse).
- **M2 – End-to-end:** `jax.jit(lambda a,b: a+b)` produces correct results on the GPU through the
  real plugin. Buffer H2D/D2H, `PJRT_Client_Compile` → `Execute` wired to the VM. Exit criterion:
  pytest comparing against CPU backend passes.
- **M3 – Op coverage, test-driven:** as soon as M2-level "1+1" works, adopt a real corpus of JAX
  programs and let its failures drive which ops/features to implement next (test-driven design).
  Candidates (evaluate, record choice in docs/decisions.md): a curated subset of upstream JAX's
  own test suite run against our backend, StableHLO's interpreter conformance/testdata, and a
  generated corpus (hypothesis-based random jax.numpy programs checked against the CPU backend).
  Maintain a coverage scoreboard (ops passing / total) in `tests/` and grow it monotonically.
  Typical op order: broadcast(-in-dim), reshape/transpose (strided views where possible),
  reductions, dot_general (naive tiled matmul first), select, compare, convert.
- **M4 – Control flow:** `while`/`if`/`case` as instructions referencing nested instruction
  lists; scalar condition read on device between sub-list runs. No jumps — linear lists all the
  way down. This is where the device-VM design must prove its worth.
- **M5 – Hardening & perf:** dtype matrix (f16/bf16 where supported, int32/64, bool), buffer
  donation, binary cache keyed by (device, driver), per-op profiling hooks, kernel-table overrides
  for tuned matmul. Then other vendors: **AMD hardware will be provided by the user in time** —
  keep the codebase ready for it (no NVIDIA-isms); Intel Xe2 via CI or remote access.

## Technical notes / gotchas

- **PJRT C API**: single self-contained header in openxla/xla (`xla/pjrt/c/pjrt_c_api.h`) — vendor
  it with the commit hash recorded. JAX discovers plugins via a `jax_plugins` entry point in
  `pyproject.toml` (preferred) whose module exposes `initialize()` calling
  `xla_bridge.register_plugin('opencl', priority=500, library_path=...)`. Mind
  `PJRT_Api.pjrt_api_version` — jaxlib refuses mismatched versions (no ABI stability yet).
  Distilled integration notes: `docs/pjrt-integration.md`
  (source: https://openxla.org/xla/pjrt/pjrt_integration).
- **Global barrier in OpenCL**: there is no portable grid-wide barrier primitive. The known-viable
  pattern is persistent threads + atomic arrival counter + seq_cst/acq_rel atomics on `global`
  memory — but it is only safe when all workgroups are co-resident, so launch size must come from
  occupancy queries, not problem size. Validate on each vendor; this is the project's #1 risk.
- **OpenCL C has no function pointers** — the VM's dispatch is a switch over opcodes in one
  translation unit. Watch compile time / register pressure as the op library grows; mitigation is
  splitting into multiple VM kernels by op family (each still handles instruction *ranges*).
- **StableHLO versioning**: JAX serializes portable "VHLO" artifacts; use stablehlo's
  deserialization API rather than parsing raw MLIR from a mismatched version.
- **JAX defaults to f32** (x64 disabled) — convenient, since fp64 is absent on Intel Xe2 consumer
  parts. Don't gate anything on fp64.
