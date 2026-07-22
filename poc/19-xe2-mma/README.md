# poc/19 — Xe2 f32 SGEMM tile-geometry sweep

**Question.** The shipped GPU `mm2` kernel (the standalone SGEMM the
host-dispatch engine uses for a pure `a@b`) runs at ~1.17 TFLOP/s at 2048³ on
Intel Arc 140V (Xe2) — roughly a quarter of the part's f32 peak, and only par
with an 8-core CPU. Is its tile geometry (128×64 output tile, 256 threads, 8×4
register microtile, `MM2_BK=16`) the right shape for Xe2, or was it tuned for
NVIDIA?

**Method.** `mma19.cl` is the shipped `mm2` kernel with `TM/TN/TD/BK` lifted to
build `-D` options (same double-buffered `__local` staging, same
register-blocked microtile). `bench19.c` builds one program per config, times
`C=A@B` (min of 8 runs, kernel-only, after 3 warmups) and spot-checks 24 output
cells against a CPU reference. No plugin, no python.

```bash
cc -O2 -o bench19 bench19.c -lOpenCL -lm
./bench19 2048 Intel        # [N] [platform-substr]
```

## Result (GFLOP/s, Intel Arc 140V, 64 CUs, 128 KB SLM)

| config (TM×TN td bk → RM×RN) | N=512 | N=1024 | N=2048 |
|---|---|---|---|
| **128×64 td16 bk16 (8×4)** — *shipped baseline* | 675 | 1079 | 1171 |
| **128×64 td16 bk8 (8×4)** — *chosen* | **795** | **1564** | 1522 |
| 128×64 td16 bk4 (8×4) | 690 | 1654 | 1595 |
| 64×64 td16 bk8 (4×4) | 633 | 1502 | 1647 |
| 64×64 td16 bk4 (4×4) | 597 | 1556 | 1658 |
| 128×128 td16 bk8 (8×8) | 428 | 1608 | **1866** |
| 128×128 td16 bk4 (8×8) | 399 | 1497 | 1776 |
| 64×128 td16 bk8 (4×8) | 437 | 1340 | 1353 |
| 128×64 td16 bk32 (8×4) | 266 | 568 | 472 |
| 64×64 td8 bk16 (8×8) — *64 threads/WG* | 470 | 561 | 486 |
| 256×64 td16 bk16 (16×4) | 631 | 718 | 874 |

Every config is numerically `ok` (max|·| ≤ 2.5e-5 vs the f64 CPU reference).

## Findings

1. **`MM2_BK` was the bug, not the register block.** Dropping the staged
   K-block 16→8 is worth **~1.3–1.45×** at 1024–2048 with the tile shape
   otherwise unchanged. Double-buffered staging allocates `2*BK*(TM+TN)` floats;
   at bk16 that is 24 KB per workgroup, at bk8 only 12 KB — so twice as many
   workgroups stay co-resident on the 128 KB SLM and the driver can latency-hide
   global loads. `bk32` (48 KB) is catastrophic (−60%), confirming the
   occupancy story rather than an instruction-scheduling one.
2. **It is NOT register spilling.** The initial hypothesis was that 8×4 = 32
   accumulators over-pressured Intel's GRF. False: the 8×8 (64-accumulator)
   128×128 tile is the *fastest* config at 2048. Arithmetic intensity wins.
3. **Thread count matters more than register-block size.** `td8` (64
   threads/WG, 8×8 block) is the worst non-degenerate config at every size —
   256 threads/WG is required to fill the XVEs.
4. **The best large-N tile is the worst small-N tile.** 128×128 bk8 leads at
   2048 (1866) but collapses at 512 (428): only (512/128)² = 16 output tiles,
   far too few to fill 64 CUs. Since one fixed geometry serves *every*
   host-dispatch matmul, the balanced **128×64 bk8** is the robust pick — best
   at 512, near-best at 1024/2048, never a regression.

## Shipped

`MM2_BK 16 → 8` in `pjrt_plugin/kernels/vm_main.cl` (docs/decisions.md §45).
End-to-end through the plugin on Xe2, single-call `a@b`: 2048³ 16.05→11.75 ms
(1.37×), 1536³ 7.91→4.71 ms (1.68×), 1024³ 3.32→2.37 ms (1.40×). Full
`pytest tests/` green on Xe2 (406 passed, 1 skip).

## Size-adaptive geometry — MEASURED, and NOT worth it

The obvious follow-up was to compile both 128×64 bk8 and 128×128 bk8 and pick at
launch on M/N, since 128×128 leads by 1.23× on square 2048. Measured on the
THREE matmul shapes a real transformer actually issues (`--config large_l1`),
the win does not hold up:

| shape (M×N×K) | 128×64 bk8 | 128×128 bk8 | |
|---|---|---|---|
| 2048×1024×1024 (QKV/out proj) | 1635 | 1785 | 1.09× |
| 2048×4096×1024 (FFN up) | 1533 | 1732 | 1.13× |
| 2048×1024×4096 (FFN down) | **1699** | 1599 | **0.94× — regresses** |

Average ~1.05×, and one of the three shapes gets *worse*. Against that: it needs
either a second full program build at init or a refactor of the kernel body to be
macro-instantiable (blocked today by its nested `#define MM2_STAGE`). **Not
shipped** — the cost and regression risk exceed a ~5% average.

## Not done (future)

- **XMX / DPAS**: Arc 140V has Intel's matrix engine, reachable via
  `cl_intel_subgroup_matrix_multiply_accumulate`. It is bf16/f16/int8 only, so
  an f32 matmul would have to round operands the way the NVIDIA path already
  does for TF32 (§35–41). Largest remaining Xe2 matmul lever; a vendor
  extension, so it belongs behind the kernel-table override, not the core path.
