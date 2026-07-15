# poc/06 notes — for the decision tree

Optimizing the VM's `MMA_TILE` from 4.3 → 26.2 TFLOPS (portable) on the RTX PRO
6000. Peak (clpeak FP32) = 105.9 TFLOPS. Constraint: tile function, 256 threads
fixed, one workgroup = one output tile, callable in a loop (no grid/global-sync
assumptions). Every number best-of-5, GPU boost clocks pre-warmed.

## What worked (the progression)

1. **Register blocking** was the single biggest win: 4.3 → 19–25 TFLOPS. Each
   thread computes an RM×RN microtile in registers (64 FMA / 16 local loads at
   8×8), vs the naive 1 output/thread (1 FMA / 2 local loads). 128×128 tile with
   8×8 microtiles (16×16 threads) beat 64×64/4×4 (25 vs 19 TFLOPS @4096).
2. **Non-transposed A local staging** (`As[m][k]`) is the key layout choice: it
   keeps BOTH the global load and the local store contiguous → coalesced DRAM
   reads AND conflict-free `vstore4`. The obvious "transpose A into local"
   (`As[k][m]`) forces a scatter store; with a 128-stride that is a 32-way local
   bank conflict — it cratered float4 loads to 5.7 TFLOPS before the layout fix.
3. **Deeper K panel** (BK 8 → 16) gave a small, free win (24.6 → 26.2) — fewer
   barriers per K, more compute between syncs. BK32 adds only ~0.6 more.
4. **GPU clock warmup matters for measurement:** the first ~1 s of kernels runs
   at a low boost state. Cold vs warm changed the *same* naive kernel from 2.1
   to 4.3 TFLOPS. `bench.c` now warms up before timing; without it the
   sequential sweep is also contaminated by thermal drift (later configs slower).

## What did NOT work (negative results — as valuable)

- **float4 vectorized staging (VECW=4): net-negative-to-marginal.** After the
  non-transposed layout fix it stopped regressing, but scalar loads consistently
  *matched or beat* it (BK16 scalar 26.2 vs BK16 vec4 ~24; BK32 vec4 only 26.8).
  The staging is not the bottleneck once loads are coalesced. Kept only as an
  NVIDIA-tuned option, not the core path — also because it crashes PoCL (below).
- **Double buffering (prefetch next K-panel): no gain.** ~0% over single-buffer
  at the same config. NVIDIA already hides DRAM latency with 2 co-resident
  workgroups/SM; the extra 16–32 KB local just costs occupancy. Textbook win on
  bandwidth-starved GPUs, dead weight here.
- **Strided / interleaved microtile mapping (thread owns cols tx, tx+16, …):**
  meant to remove the 4-way B-read bank conflict and coalesce C writes. It made
  128×128 *worse* — 26 → 15.6 TFLOPS (isolated, repeatable). Conclusion: LDS
  bank conflicts are NOT the bottleneck here; the contiguous mapping's better
  register reuse / store pattern dominates. Reverted.
- **More lanes for occupancy:** 2/CU (376 lanes) is optimal. 3–5/CU gave ≤1%
  and often nothing — the 128×128 tile (80 regs/thread, 16–32 KB local) is
  occupancy-bound at ~2 workgroups/SM, so oversubscription only adds contention.
- **Bigger tiles to cut DRAM traffic:** 128×256 and 256×128 (16×16 microtiles,
  128 regs/thread) spill — 128×256 fell to 13.9, 256×128 to 23.8 TFLOPS. With
  256 threads fixed you cannot grow the tile without growing per-thread
  registers past the spill point. This is the structural ceiling of the
  "256 threads, one workgroup per tile" contract for fp32 SIMT.

## PoCL portability findings (correctness backend)

- **Scalar tiles up to BK16 are correct on PoCL** (512³ exact) — steps 1–4.
  Perf on the Ryzen CPU is ~40 GFLOP/s and irrelevant, as specced.
- **BK32 scalar is MISCOMPILED by PoCL** (wrong result + absurd 446 "GFLOP/s" on
  a CPU) while correct on NVIDIA. Local mem (32 KB) is under PoCL's 48 KB limit,
  so it is a PoCL codegen bug on the 32-deep `#pragma unroll`ed barrier loop, not
  overflow. ⇒ cap the portable tile at BK16.
- **Any VECW=4 config crashes PoCL's compiler:**
  `Kernel.cc:129 Assertion 'region_entry_barrier != NULL'`. PoCL's work-group
  barrier-region former chokes on the vectorized staging loop preceding a
  barrier. An early `return` in `stage_panel`'s fast path made it worse (two
  inlined predecessors for the caller's barrier); rewriting to single-exit
  if/else did not fix the vec4 case. ⇒ float4 staging is NVIDIA-only, correctly
  living behind the per-vendor override.

## Decision

- **Portable core `MMA_TILE`: 128×128, 8×8 microtile, BK16, scalar, single-
  buffer.** 26.2 TFLOPS (24.7% peak), 6.1× over naive, correct on NVIDIA + PoCL.
  16 KB local, ~80 regs/thread.
- **NVIDIA override: +float4 staging, BK32** → 26.8 (27.1 @752 lanes). Marginal,
  non-portable; belongs in the kernel-table override, not the core path.
- Target was ≥30 (stretch 50); we reach ~26. Honest ceiling for straightforward
  fp32 SIMT SGEMM under the fixed-256-thread tile contract on this OpenCL driver.

## What to try next (to push past 30)

1. **Register-vectorized inner loop** — hold `acc`/`a`/`b` as float4 and issue
   `fma` on float4 so the compiler packs FFMA pairs; the biggest untried lever,
   and independent of the staging vectorization that didn't help.
2. **`cl_khr_subgroups` broadcast** — feed A (or B) operands via
   `sub_group_broadcast` instead of local memory, cutting LDS traffic; NVIDIA
   exposes subgroups in recent drivers (feature-detect). Marked in tile-isa.md
   ceiling assessment as the SIMT escape hatch short of tensor cores.
3. **2-level (warp) tiling** — a warp-tile between the block-tile and the
   thread-tile to raise arithmetic intensity without growing the block tile past
   the spill point; the standard CUTLASS structure, reachable in OpenCL C.
4. **`-cl-mad-enable` / `-cl-fast-relaxed-math`** build flags — cheap to test for
   FFMA contraction (not yet tried; correctness impact must be checked).
5. Accept the ceiling and record it: tile-isa.md already notes CLBlast-class
   40–70% of SIMT peak needs SASS-level tuning we can't reach from OpenCL on
   NVIDIA; 25% from a clean, portable, VM-integrable tile is a reasonable stop.
