# poc/06 — fast SGEMM tile function

Optimized OpenCL SGEMM **tile function** to replace poc/04's naive `mma_tile`.
It is a tile function for the persistent-lane VM, **not** a free-standing GEMM:
one workgroup (256 threads, fixed) computes one `TMxTN` output tile of
`C[M,N] = A[M,K] @ B[K,N]` (all row-major dense in one float arena, addressed
by element offsets), callable in a loop from an interpreter — see `exec_tiles`
in `poc/04/vliw.cl`. No global sync, no grid assumptions, no workgroup-size
changes. M/N/K need not divide the tile edges (edge-guarded scalar path).

The tile function is `mma_tile_fast` in `mma.cl`, parametrized by `-D` options
(`TM TN BK VECW DB`) and compiled once per config. `bench.c` drives it
persistent-style (lanes = 2 x CU count, each lane a contiguous tile range),
best-of-5 at M=N=K=2048 and 4096, and verifies correctness against a host
reference at 512^3 with integer-valued floats (exact compare).

## Build / run

```
. ../../env.sh                 # MUST: pins caches off the full root overlay
make
timeout 120 ./bench                         # NVIDIA (default)
OCL_PLATFORM=Portable WARMUP=0 VM_LANES=24 timeout 120 ./bench   # PoCL (correctness)
```

env: `OCL_PLATFORM` (default `NVIDIA`), `VM_LANES` (default 2 x CU),
`WARMUP` (default 1; set 0 for PoCL), `ONLY=<label substring>` (filter steps).

## Progression (NVIDIA RTX PRO 6000 Blackwell Max-Q, 188 CU, 376 lanes, best-of-5)

Peak = **105.9 TFLOPS** (clpeak FP32). GFLOP/s below; TFLOPS = /1000.

| # | step | 2048 GF/s | 4096 GF/s | % peak (4096) | 512 correct | PoCL |
|---|------|----------:|----------:|--------------:|:-----------:|:----:|
| 1 | naive 16×16 (poc/04 `mma_tile`) |  4338 |  4311 |  4.1% | yes | correct |
| 2 | register-blocked 64×64, 4×4 µtile, BK16 | 17311 | 19072 | 18.0% | yes | correct |
| 3 | 128×128, 8×8 µtile, BK8 | 18335 | 24601 | 23.2% | yes | correct |
| 4 | **128×128, 8×8 µtile, BK16 (portable champion)** | 19475 | **26162** | **24.7%** | yes | **correct** |
| 5 | 128×128, BK32, float4 staging (NVIDIA-only) | 19627 | **26843** | **25.3%** | yes | compiler crash |

Lane sweep (warmed) on step 5: 26.9 (376) → 27.1 (752/940) TFLOPS — a further
~1%; 2/CU is essentially optimal (the 128×128 tile is occupancy-bound).

## Final numbers

- **Portable champion (step 4):** 128×128, BK16, scalar loads, single-buffered.
  **26.2 TFLOPS @4096 = 24.7% of the 105.9 TFLOPS peak, a 6.1× speedup over the
  4.3 TFLOPS naive baseline.** Correct on both NVIDIA and PoCL.
- **NVIDIA-tuned override (step 5):** float4 staging + BK32 → **26.8 TFLOPS
  (27.1 at 752 lanes) = 25.3% of peak.** Marginal (+0.7 TFLOPS) and it crashes
  PoCL's compiler, so it belongs behind the per-vendor kernel-table override,
  not in the portable core path (matches CLAUDE.md's portability rule).

Short of the ≥30 TFLOPS target. At 26 TFLOPS the kernel is ~44% of DRAM
bandwidth (measured traffic) — partly memory-bound, and larger tiles that would
cut traffic spill registers under the fixed 256-thread / persistent-lane
constraint. See `NOTES.md` for the failed experiments and what to try next.

## Final tile footprint (portable champion: TM=TN=128, BK=16, VECW=1)

- **Local memory:** `BK*(TM+TN) = 16*256 = 4096 floats = 16 KB` per workgroup
  (single-buffered). `As` is `TM×BK` (un-transposed, `As[m*BK+k]`), `Bs` is
  `BK×TN` (`Bs[k*TN+n]`). Double-buffering (`-DDB=1`) doubles this to 32 KB.
  The NVIDIA override (BK32) uses 32 KB — still within PoCL's 48 KB, but PoCL
  miscompiles the deeper unrolled barrier loop (see NOTES).
- **Registers:** `RM*RN = 8*8 = 64` fp32 accumulators + `RM+RN = 16` operand
  regs ≈ 80 fp registers/thread (plus loop/index). This is the occupancy limiter
  (~2 workgroups/SM).
- **VM integration:** `mma_tile_fast(arena, aoff, boff, coff, M, N, K, tile,
  As, Bs, full)` — one call per output tile; `tile` is the flat tile index
  (`tr = tile / tiles_n`, `tc = tile % tiles_n`), exactly the poc/04 range-cell
  contract. `full` selects the aligned/interior fast path (VECW==4 only);
  pass 0 to force the always-correct edge-guarded scalar path.
