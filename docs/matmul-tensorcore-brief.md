# Task brief: TF32 tensor-core matmul (agent handoff)

**Mission:** close the `dot_general` (matmul) performance gap to native CUDA/cuBLAS on NVIDIA
by adding a **TF32 tensor-core** SGEMM path, reachable from OpenCL via **inline PTX WMMA**,
behind the existing `mm2` pure-matmul dispatch and gated to NVIDIA. Everything else in the
plugin must be untouched and must still build/run on non-NVIDIA devices (PoCL, AMD, Intel).

Read `docs/decisions.md` §9 (esp. §9b) and this file before starting. Follow the repo's
**hard rules** in `CLAUDE.md` — in particular **PoC-first**: prove the mechanism in
`poc/08-tensor-core-mma/` before integrating anything into the main tree.

---

## Why this is the only path to parity (already established — don't re-litigate)

- Baseline matmul is ~17 TFLOP/s; a standalone FP32 SGEMM (`mm2`, already shipped) reaches
  ~21 TFLOP/s and every FP32 tile/register config **plateaus at ~20 TFLOP/s** in this
  session's sweep (occupancy/register-limited; see §9b).
- **cuBLAS hits 134 TFLOP/s at N=4096 — ABOVE Blackwell's ~125 FP32 peak — so it is
  TF32-tensor-core bound.** No portable FP32 kernel can reach it. TF32 tensor cores are the
  only route to <2× (ideally ~parity). Note JAX's CUDA backend itself uses TF32 for f32
  matmul by default, so a TF32 result is the correct apples-to-apples comparison.

## What already exists (build ON this — do not rebuild it)

The **pure-matmul fast path is already wired**; you're replacing the kernel body, not the
plumbing. In `pjrt_plugin/src/runtime.cc`:
- `LoadedProgram::Load` detects a program that is a single f32 `TILE_MMA` task with no
  barrier/control entries and sets `mm_ok_` (+ `mm_M_/N_/K_`, and the offset/port-patched
  `mm_dst_/mm_a_/mm_b_` handles). Currently gated to large GPU matmul (M,N≥1024).
- `ExecuteDevice` routes `mm_ok_` programs to `LoadedProgram::LaunchMatmul`, which sets args
  and `clEnqueueNDRangeKernel`s the `mm2` kernel (one workgroup per output tile).
- `mm2` kernel lives in `pjrt_plugin/kernels/vm_main.cl` (search `__kernel void mm2`): a
  portable FP32 128×64/8×4 double-buffered SGEMM. **Keep it as the fallback.**
- `PJRT_OCL_MM_KERNEL=0/1` env forces the fast path off/on.

Your job: add a **TF32 tensor-core kernel** selected when the device is NVIDIA and the PTX
compiles; fall back to the portable `mm2` otherwise.

---

## Plan (phased; each phase has a gate)

### Phase 0 — PoC (poc/08-tensor-core-mma/, standalone, NO plugin changes)
Write a minimal C + OpenCL host program that:
1. picks the NVIDIA platform/device (substring "NVIDIA"),
2. compiles a kernel containing inline-PTX tensor-core ops (start with the **proven f16
   WMMA** form below to de-risk the *mechanism*, then the **TF32** form),
3. runs a single/few tiles and checks the result against a CPU reference.

**Gate 0:** the PTX **compiles under NVIDIA's OpenCL** and one 16×N tile is numerically
correct. This is the #1 risk — if inline PTX won't compile from OpenCL here, stop and report.
(The f16 WMMA form is copied verbatim from a repo that does exactly this — see refs — so it
should compile; TF32 is the goal.)

Build the PoC like the others: a small `Makefile`/`CMakeLists` linking `-lOpenCL`. A
`poc/NN` CMake target convention exists (`cmake --build build --target poc-NN`); mirror an
existing `poc/*/` for the pattern. Keep a `README.md` recording what compiled, what didn't
(with the exact driver error), and the measured TFLOP/s.

### Phase 1 — full TF32 tensor-core SGEMM (still standalone in poc/08)
Grow the PoC into a real GEMM: block tile in shared memory, warp-level `wmma.mma` accumulation
loop, epilogue store. Tune for throughput (target: clearly beat the 21 TFLOP/s FP32 mm2;
stretch: approach cuBLAS's ~100–134 TFLOP/s). Techniques from the refs: shared-mem staging
of A/B tiles, `wmma.load.*.shared` fragments, K-loop over `wmma.mma`, double-buffered smem,
optional bank-conflict swizzle. Measure at N=1024/2048/4096, square + non-square + edge
(non-multiple-of-tile) shapes.

**Gate 1:** a standalone kernel that is correct (vs CPU, TF32 tolerance ~1e-2 rel) and
materially faster than the FP32 `mm2` (>40 TFLOP/s is a good bar; higher is better).

### Phase 2 — integrate behind a device/compile guard
- Put the tensor-core kernel in its **own OpenCL program**, compiled in a **separate
  `clCreateProgramWithSource`/`clBuildProgram`** that is attempted **only on NVIDIA**
  (detect via device-name substring and/or `cl_nv_device_attribute_query`). If that build
  **fails**, log and leave the tensor-core kernel unavailable — the runtime must **fall back
  to the portable `mm2`**. Do NOT put inline PTX in the shared megakernel program (`vm2`
  etc.): that program must keep compiling on PoCL/AMD/Intel.
  - Rationale: the runtime already probes build variants for the megakernel (see the
    `-cl-std` / `VMO_NO_DEVICE_FENCE` probe in `runtime.cc` `Create`); mirror that
    try-and-fallback discipline for the tensor-core program.
- Add a `cl_kernel mm_tc_kernel_` (tensor-core) alongside `mm_kernel_` (portable). In
  `LaunchMatmul`, use the tensor-core kernel when available, else the portable one. Handle
  its own grid geometry (its tile size differs from 128×64).
- The kernel receives the same VM buffer handles (`arena`, `iop[]` ports, `dst/a/b` handles)
  and resolves them with the existing `VMO_BASE`/`AP` macros (see `vm_common.cl`) — so I/O
  ports keep working. `M/N/K` come from `mm_M_/N_/K_`.

**Gate 2:** on NVIDIA, large matmul uses the tensor-core kernel and is faster; on PoCL the
plugin still builds and `runtime_test` + the while/matmul e2e still pass (the tensor-core
program simply isn't created there).

### Phase 3 — precision, tests, docs, benchmark
- TF32 drops ~13 mantissa bits, so tolerances must loosen for the tensor-core path. The
  e2e/matmul tests currently compare against JAX (CPU is f32; but JAX **CUDA** uses TF32).
  Pick tolerances that pass against JAX CUDA-equivalent precision (rel ~1e-2) and document
  the choice. Do NOT loosen tolerances for the non-tensor-core paths.
- Update `docs/decisions.md` §9b (TRIED/OK with real measurements), `README.md` Performance
  section, and regenerate `docs/bench_plot.png/.csv` via `tools/plot_bench.py`.
- Consider a `PJRT_OCL_MM_TC=0/1` override and revisiting the size gate (tensor cores may win
  at smaller N than the FP32 mm2 does).

**Gate 3:** 199+ pytest pass, `runtime_test` PASS on NVIDIA **and** PoCL, benchmark refreshed,
decision log updated.

---

## Exact inline-PTX references (verbatim; verified against real sources)

### OpenCL inline-PTX syntax (NVIDIA)
`asm volatile("ptx;" : outputs : inputs : clobbers);` — `%0,%1,…` are operands numbered
across the output then input lists; `%%` emits a literal `%` (for special regs). Constraints
that compile under NVIDIA's OpenCL: **`"=r"`/`"r"`** = 32-bit int reg, **`"=f"`/`"f"`** =
32-bit float reg, **`"l"`** = 64-bit int/pointer reg. Read the true warp lane with
`asm("mov.u32 %0, %%laneid;" : "=r"(laneid));` — it is NOT `get_local_id(0)%32`.

### f16 WMMA from OpenCL (proven — de-risk the mechanism with this first)
From `ihavnoid/hgemmtest` (`hgemm.cl`), real OpenCL compiled by NVIDIA's driver. A `__local`
pointer passed with the **`"l"`** constraint into a **`.shared`**-qualified WMMA op *just
works* — no `cvta` dance:
```c
asm("{\n"
    ".reg .b32 a0,a1,a2,a3,a4,a5,a6,a7;\n"
    ".reg .b32 b0,b1,b2,b3,b4,b5,b6,b7;\n"
    "wmma.load.a.sync.aligned.m16n16k16.shared.col.f16 {a0,a1,a2,a3,a4,a5,a6,a7}, [%4], %6;\n"
    "wmma.load.b.sync.aligned.m16n16k16.shared.row.f16 {b0,b1,b2,b3,b4,b5,b6,b7}, [%5], %7;\n"
    "wmma.mma.sync.aligned.col.row.m16n16k16.f16.f16 "
    "  {%0,%1,%2,%3}, {a0,a1,a2,a3,a4,a5,a6,a7}, {b0,b1,b2,b3,b4,b5,b6,b7}, {%8,%9,%10,%11};\n"
    "}" : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
        : "l"(a_local_ptr), "l"(b_local_ptr), "r"(a_stride), "r"(b_stride),
          "r"(c0),"r"(c1),"r"(c2),"r"(c3));
// store:
asm("wmma.store.d.sync.aligned.col.m16n16k16.f16 [%4], {%0,%1,%2,%3}, %5;"
    : : "r"(d0),"r"(d1),"r"(d2),"r"(d3), "l"(dst_global_ptr), "r"(N));
```
Accumulators carried as `int` (b32) packing 2× f16. Local A stored column-major, B row-major;
`barrier(CLK_LOCAL_MEM_FENCE)` between the gmem→smem copy and the mma loop.

### TF32 mma.sync (the goal — from NVIDIA CUTLASS `arch/mma_sm80.h`)
```c
asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(D[0]),"=f"(D[1]),"=f"(D[2]),"=f"(D[3])
    : "r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),   // A: 4 tf32 regs (pass f32 bits as uint32)
      "r"(B[0]),"r"(B[1]),                        // B: 2 tf32 regs
      "f"(C[0]),"f"(C[1]),"f"(C[2]),"f"(C[3]));  // C/D: 4 f32
```
- TF32 is bit-identical to f32 with the low 13 mantissa bits ignored: pass f32 bits straight
  in as `uint32_t`, or round first with PTX `cvt.rna.tf32.f32`. Layout: A row-major (`.row`),
  B col-major (`.col`).
- Per-thread fragment↔element map (m16n8k8; groupID=`lane>>2`, tg=`lane&3`):
  - **A [16×8], 4 regs/thread:** rows `{lane/4, lane/4+8}`, cols `{lane%4, lane%4+4}`.
  - **B [8×8], 2 regs/thread:** rows `{lane%4, lane%4+4}`, col `lane/4`.
  - **C/D [16×8], 4 f32/thread:** c0,c1 at row `lane/4`, cols `(lane%4)*2, (lane%4)*2+1`;
    c2,c3 at row `lane/4+8`, same cols.
- There is also a **`wmma` TF32** form (`wmma.mma.sync.aligned.row.col.m16n16k8.f32.f32` with
  `wmma.load.*.tf32`) whose fragment layout is compiler-managed (opaque) like the f16 WMMA —
  often easier to get right than hand-managing the `mma.sync` per-thread layout. **Try the
  `wmma` TF32 path first** (mirrors the proven f16 WMMA structure); drop to raw `mma.sync`
  only if needed for perf.
- `ldmatrix` alternative: needs a **32-bit `.shared`** address in an `"r"` operand
  (`cvta.to.shared.u64` then narrow to u32, or bit-cast a `__local` ptr to `uint`). Prefer
  `wmma.load.*.shared` (takes `__local`+`"l"` directly) to avoid this.

### OpenCL-vs-CUDA PTX gotchas
1. `.shared`-qualified WMMA loads take a `__local` pointer as `"l"` directly — no
   `cvta.to.shared`. This is the clean OpenCL path.
2. `ldmatrix` has no address-space-qualified variant → needs a 32-bit shared address.
3. Warp/lane identity ≠ OpenCL linear id — read `%laneid`.
4. All of this is NVIDIA-only; keep it in a **separate program built only on NVIDIA** with
   fallback. PoCL/AMD reject inline PTX outright.

Sources: hgemmtest `hgemm.cl`; `sschaetz/nvidia-opencl-examples/.../inlinePTX.cl`;
`alexarmbr/matmul-playground`; NVIDIA CUTLASS `include/cutlass/arch/mma_sm80.h`; PTX ISA
§9.7.15. (Fuller notes in `docs/decisions.md` §9b.)

---

## Environment / build / test (READ THIS — a stale-.so trap cost hours last time)

- **Always `. ./env.sh` first** (pins caches off the full root overlay).
- **Build the plugin with `cmake --build pjrt_plugin/build`.** The `.so` that tests and
  ad-hoc `JAX_PLATFORMS=opencl` scripts load is **`pjrt_plugin/build/libpjrt_ocl.so`**
  (hardcoded in `tests/test_e2e.py`, and `python/pjrt_ocl/__init__.py`'s default). There is a
  *second* CMake config at top-level `build/` (`cmake -S . -B build` → `build/pjrt_plugin/...`);
  building that one while tests load the other means **your kernel edits silently no-op**
  (an unknown EW subop becomes a no-op → wrong-but-fast, no error). Symptom seen last time.
  The NVIDIA driver recompiles kernels by source hash (no plugin-side binary cache), so a real
  rebuild always takes effect once the right `.so` is loaded.
- Correctness oracle: compare device output against **JAX CPU** (`JAX_PLATFORMS=cpu`) — it
  matches the device's `mad`/fma bit-for-bit for non-tensor paths. For the TF32 path expect
  ~1e-2 rel divergence (that's TF32, not a bug).
- Quick device run: `JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=NVIDIA .venv/bin/python -c "..."`.
  PoCL: `PJRT_OCL_DEVICE="Portable"` (first run after a cache clear is a slow LLVM compile —
  give it minutes, it is NOT a hang).
- Tests: `.venv/bin/python -m pytest tests/ -q`; C++: `./pjrt_plugin/build/runtime_test`
  (set `PJRT_OCL_DEVICE`). Bench: `.venv/bin/python tools/plot_bench.py --out docs/bench_plot.png`.
- Measure TFLOP/s: `2*N^3 / seconds`. cuBLAS reference on this box: ~112 TFLOP/s @2048,
  ~134 @4096. Current portable mm2: ~21 @2048.

## Working discipline
- Work in the provided **git worktree** (isolated); do NOT touch `main`. Commit small and
  often on your branch. Never leave undocumented findings — update `poc/08/README.md` and
  `docs/decisions.md` §9b as you go.
- Unverified work is treated as not done: every result needs a correctness check + a measured
  number. If a phase gate fails, stop and report the exact error/measurement rather than
  pressing on.
- Deliverable back to the main session: what compiled/failed (with driver errors), the
  measured TFLOP/s curve vs cuBLAS, and — if integrated — confirmation that PoCL still builds
  and all tests pass.
