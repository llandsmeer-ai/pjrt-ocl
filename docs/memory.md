# Memory management design

Three problems with three owners. (Status markers: ✅ current, 🔜 planned milestone, 🧭 open.)

## 1. Intra-program temporaries — Python lowering, compile time

- One flat f32 arena per executable; every tensor = (offset, size) slice; instructions carry
  element offsets. Rationale: OpenCL ≤1.2 cannot store buffer pointers in buffers (no portable
  SVM), so offsets-into-one-allocation is forced anyway — and it makes the memory plan a pure
  compile-time artifact.
- StableHLO is SSA ⇒ exact static lifetimes via use-def chains (this is where MLIR earns its
  keep). Linear bytecode ⇒ liveness = last textual use. 🔜 M1: linear-scan slot allocation with a
  free list ⇒ **arena = peak live set**, not sum of values. (MLIR's one-shot bufferization passes
  solve this in C++; for linear programs a ~50-line Python linear-scan is equivalent.)
- `while` carried values: StableHLO while is functional and type-invariant across iterations ⇒
  lowering pins iter-arg/result pairs to the SAME slot (loop runs in place, no per-iteration
  copies); those slots are excluded from reuse for the loop's duration.

## 2. Inter-execution tensors (PJRT buffers) — C++ plugin, runtime

- ✅ M2 bring-up: `PJRT_Buffer` = host memory; H2D into arena per execute, D2H out. Correct,
  simple, slow; also doubles host RAM per resident array (32 GB machine — don't linger here).
- 🔜 M3: device-resident buffers: each `PJRT_Buffer` owns a `cl_mem`; `BufferFromHostBuffer` does
  H2D once; Execute does device→device copies (`clEnqueueCopyBuffer`) buffer↔arena-slot. jax
  chains never cross PCIe.
- 🔜 M3: donation: donated input's slot aliased to an output slot in the IO map (skip one d2d
  copy, invalidate the input buffer). Format-wise just an IO-map property.
- 🧭 deferred until profiled: shared device pool where PJRT buffers are suballocations and arena
  IO slots alias them (zero-copy execute) — entangles donation/aliasing; only if d2d copies show
  up in profiles.

## 3. Recycling — C++ plugin, runtime

- 🔜 M3: size-bucketed free list for `cl_mem` on `PJRT_Buffer_Destroy` (jax churns allocations;
  drivers are slow at it).
- Arenas live as long as their executable and jax caches executables ⇒ many live arenas is the
  main VRAM pressure point. 🧭 If it bites: arenas become per-execution leases from a shared pool
  (cheap for us: offsets are arena-relative; instruction buffer re-upload is tiny).

## Dynamic shapes — how they fit (2026-07-14 discussion)

Key property: the VM decouples compute from shape. Fixed persistent-kernel launch + grid-stride
loops ⇒ element counts are just data; barrier structure is shape-independent. Only the ARENA PLAN
needs shapes ⇒ dynamism = "when is the memory plan evaluated?".

- **L0 (✅ implicit)**: jax.jit recompiles per shape; data-dependent shapes are forbidden in jit.
  Static per-executable arena is exactly right. Nothing to build.
- **L1 (🔜 design VMProgram v2 for it)**: shape-polymorphic bytecode for symbolic-dim StableHLO
  (`jax.export`). The slot-reuse plan's STRUCTURE is shape-independent; only sizes vary, as
  products of dim vars. Lower once with symbolic sizes (linear-expression encoding + dim-var
  table in v2); at execute: read dims from inputs, evaluate sizes (~µs), lay out arena, patch
  n/offsets, re-upload instr buffer, cache per shape-tuple. A "recompile" in µs instead of the
  ~150 ms lowering subprocess.
- **L2 (🧭)**: bounded dynamism within one execution (XLA model, `stablehlo.dynamic_*`,
  `PJRT_Buffer_DynamicDimensionIndices`): allocate at bounds, actual size lives in a scalar arena
  slot, instructions read `n` from arena (flag bit / twin opcodes — grid-stride doesn't care;
  size scalars read atomically like while-conds). Real lowering work, no architectural change.
- **L3 (🧭 research note only)**: unbounded data-dependent allocation. Fights the pre-planned
  arena; a device-side bump allocator over a heap region is *expressible* (offsets are integers)
  but dealloc/compaction is ugly. Nothing in jax needs it.

Rule of thumb: recompile-per-shape through M3; make format v2 L1-ready (sizes as expressions);
never attempt shape-varying while carried values (StableHLO forbids them anyway).
