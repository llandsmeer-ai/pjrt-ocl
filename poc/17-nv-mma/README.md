# poc/17 — NVIDIA cp.async / multi-stage tensor-core matmul (§35)

**Goal.** Attack the matmul gap (in-megakernel ~18-20 TFLOP/s vs cuBLAS 116-133)
with the one thing that requires leaving portable OpenCL: a cuBLAS-class WMMA
tile — a **multi-stage register-blocked pipeline with `cp.async`** (async
global→shared copy) so weight loads overlap tensor-core compute and the
smem-bandwidth bottleneck (§31) is relieved. Strictly NVIDIA-only, behind the
existing `VMO_NV_PTX` gate; portable path untouched. **PoC gate first.**

## Files
- `probe.c` / `probe.cl` — **cp.async feasibility gate.** Does the NVIDIA
  OpenCL ICD actually *execute* inline-PTX `cp.async`? `make run-probe`.
- `bench17.c` / `mma17.cl` — **standalone tensor-core matmul ceiling.** tf32
  m16n16k8 WMMA tile, sweeps tile size / synchronous multi-buffering / lanes at
  2048³ and 4096³. `make run-bench`.
- `mmbench.py` — in-megakernel matmul TFLOP/s via the real plugin
  (`JAX_PLATFORMS=opencl PJRT_OCL_MM_KERNEL=0`), 64-tile vs `PJRT_OCL_MEGA_BIGTILE`.

## RESULT 1 (the blocker): cp.async is NON-FUNCTIONAL on this ICD

NVIDIA RTX PRO 6000 Blackwell (sm_120), driver 595.71.05, OpenCL 3.0 CUDA.
The driver **emits correct PTX** — the dumped program binary shows
`.target sm_120`, `.version 9.2`, and literal `cp.async.cg.shared.global … ;`
+ `cp.async.commit_group; cp.async.wait_group 0;` — ptxas accepts it, the kernel
runs. **But the copy never delivers data.** Every completion form tried is WRONG
(1024/1024 mismatch), while the CONTROL — a synchronous `st.shared` through the
*identical* `cvta.to.shared`-derived address — is CORRECT (0 mismatch), so the
shared-address mapping is fine and **cp.async itself is the broken primitive**:

| form | result |
|------|--------|
| `st.shared` (sync control, same address) | **CORRECT** |
| `cp.async.cg` + commit/`wait_group 0` | WRONG |
| `cp.async.ca` | WRONG |
| + `fence.proxy.async.shared::cta` | WRONG |
| `wait_group` + 200k-iter spin (rules out a wait-only bug) | WRONG |
| `cp.async.mbarrier.arrive` + `mbarrier.try_wait` | WRONG |

Conclusion: the NVIDIA **OpenCL** runtime does not wire up the Ampere+ async-copy
unit (a CUDA-only feature path here). **The deep software pipeline that
cuBLAS-class GEMM needs cannot be built through the OpenCL→PTX path on this
driver.** This is the §14a "measured ceiling — the path can't express it"
outcome.

## RESULT 2: the WMMA ceiling *without* cp.async (synchronous staging)

`bench17` (standalone, no megakernel barrier, so the tile is free of the
co-residency cap). Best-of-7, tf32, warmed clocks:

| tile | 2048³ TF/s | 4096³ TF/s |
|------|-----------|-----------|
| 64×64  BK16 1-buf (== in-megakernel tile shape) | 27.2 | 29.6 |
| 128×64 BK16 2-buf | 37.7 | 46.2 |
| **128×128 BK16 2-buf (the knee)** | **47.5** | **55.2** |
| 128×128 BK32 2-buf | 25.1 | 30.7 |
| 256×128 BK16 (any buf) | ~33 | ~38 |
| in-megakernel 64-tile (mmbench, shipped default) | 19.3 | 17.4 |
| in-megakernel 128×128+pipe (`MEGA_BIGTILE`, §31) | 14.2 | 14.0 |
| cuBLAS | 116.5 | 133.3 |

The **128×128 synchronous double-buffered tile clears the ≥40-60 gate at
47.5/55.2 TF/s** (2.5-3× the in-megakernel tile) — but note the win is (a) tile
**intensity** (128×128 register accumulator), not async staging, and (b) only
realisable in a **dedicated** kernel. It is still **~2.4× under cuBLAS**: without
cp.async there is no 3-4 stage global-latency-hidden pipeline, the tf32 m16n16k8
MMA is smaller than cuBLAS's, and there is no smem swizzle beyond LDS padding.

## RESULT 3: the ceiling does NOT transfer into the megakernel

The identical 128×128 tile is **55 TF/s standalone but 14 TF/s in-megakernel**
(mmbench, reproduces §31). The gap is entirely the megakernel's structure: the
cross-workgroup spin-barrier forces all lanes **co-resident** (the 64-accumulator
tile crosses the 128-reg cliff → 188 lanes, no oversubscription to hide latency)
and the whole-VM register budget is a max over every op. A dedicated kernel pays
none of this. cp.async was the sanctioned lever to relieve that latency/smem-BW
bottleneck — and it does not work here.

## DECISION
**Do NOT integrate / do NOT change the default.** cp.async (the mechanism the
thesis required) is unavailable, and the big-tile megakernel path is an
already-measured regression (§31). The genuine remaining lever is the **hybrid
split** (route big matmul phases to a dedicated 128×128 TF32 kernel — this PoC's
55-TF/s tile — while the megakernel keeps everything else), which §10d measured
as host-dispatch-overhead-bound and deferred as a large architectural change.
Full detail: docs/decisions.md §35.

## §36 update — sync ceiling swept, hybrid BUILT and it wins on large

`bench17` is now parametrized (TM/TN, BK, NBUF, warp grid WM×WN, VEC4 float4
staging, PAD, PIPE fragment-pipeline). Full sweep verdict: **every classic GEMM
lever is a wash within ±3%** — the tile is register-file-capped at 2 WG/SM (lane
sweep: 188→33, 376→57, ≥564 flat), not BW-bound, so ~57 TF/s @4096 / ~47 @2048
is the honest sync ceiling (~2.3× under cuBLAS; only cp.async — dead — could buy
the missing latency hiding).

The hybrid IS built (no longer deferred): the tile ships in the plugin as
`mm_tc` (pjrt_plugin/kernels/vm_main.cl), replacing scalar mm2 on the GPU/TF32
pure-matmul fast path (24→53 TF/s), and `PJRT_OCL_MM_HYBRID=1` routes a full
program's big TF32 matmul phases to it on the host-dispatch engine. **large
transformer 27.7→19.1 ms (1.45×), gap to CUDA 7.5×→5.2×**; base regresses
(overhead-bound, M=512 too small). Opt-in, not merged. Detail: decisions.md §36.

## §38 update — the tf32 57 ceiling is BROKEN by fp16/bf16 WMMA (~92 TF/s)

`mma17_hp.cl` + `bench17hp.c` (`make bench17hp && ./bench17hp`). The §35/§36
"~57 TF/s is THE sync ceiling" was measured **only on tf32 m16n16k8**. Switching
the MMA inputs to **fp16/bf16 (m16n16k16, 2× tensor rate)** and using a
**thinner accumulator-per-thread** tile (256×128 W8×4 = 1024 threads → 32 acc
regs/thr vs 64 → 3–4 WG/SM instead of 2) reaches:

| config | 2048³ | 4096³ | precision |
|--------|-------|-------|-----------|
| tf32 128×128 (§36 ceiling) | 47 | 57 | 10-bit mant |
| **f16 256×128 W8×4** | **72** | **92** | 10-bit mant (= tf32), max_abs 3e-3 |
| bf16 256×128 W8×4 | 71 | 91 | 7-bit mant, max_abs 2.7e-2 |
| cuBLAS tf32 | 116 | 133 | — |

**1.6× over the tf32 ceiling at tf32-equivalent accuracy** (fp16 and tf32 share a
10-bit mantissa; fp16 only has a smaller exponent range). §36's "register-file-
capped, latency the RF can't buy" was a tf32-specific artifact: fp16 halves
fragment reg pressure *and* doubles compute headroom, so the wide/thin tile's
extra occupancy finally converts to latency hiding. Gotchas found:
`wmma.load.shared.{f16,bf16}` **faults (-36) unless smem is 16-byte aligned**
(`__attribute__((aligned(16)))`); f16 A/B = 8 .b32 regs, bf16 = 4. cp.async
**independently re-confirmed dead** (`make run-probe`). PoC-only; not integrated
(needs an f16 input path + range guards). Full detail: docs/decisions.md §38.
