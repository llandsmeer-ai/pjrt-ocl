# poc/09 — why XLA CPU beats us on PoCL (and what actually fixes it)

**Question:** the PoCL backend is 2.6x (elementwise) to ~90x (matmul) to ~320x
(matvec) slower than JAX's native XLA CPU backend on the same 8 cores. Which
mechanism, per op class, and what kernel shapes fix it?

**Method:** standalone microbenches of candidate kernel patterns on PoCL *and*
Xe2 (same binary, `PJRT_OCL_DEVICE=` selects), against an 8-thread `memcpy`
machine baseline. `make && ./poc09`.

## Results (LNL Core Ultra 9 288V, 8 cores / Arc 140V)

Machine wall: memcpy 53.6 GB/s (1 thread), **84.7 GB/s (8 threads)**.
For scale: XLA CPU's `a+b` at 16M ≈ 12 GB/s effective.

### [A] elementwise `d = a+b`, 16M f32 (2 reads + 1 write)

| pattern | PoCL | Xe2 |
|---|---|---|
| a1 **current VM tile loop** (per-WI stride-`lsz`) | 5.1 GB/s | **104 GB/s** |
| a1b/a1c restructured w/ straight-line body (± guard) | 4.5–4.9 | 105 |
| a2 classic 1-elem-per-WI | 38.9 | **112.6** |
| a3 contiguous chunk/WI, scalar | 9.6 | 51.7 |
| a4 **contiguous chunk/WI, float8** | **46.1** | 61.6 |
| a5 1 WI/WG × 8 WGs, float8 | 46.3 | 6.3 |
| a6 WI-coalesced float8 (a1 vector-widened) | 20.2 | 79.1 |

Findings:
- **PoCL's work-group vectorizer only fires on the implicit WI loop** (a2). ANY
  explicit in-kernel loop around the body defeats it — restructuring (a1b/a1c)
  does not help; this is not about guards or uniformity. With the vectorizer
  dead, a1 runs scalar with 8 threads → 5 GB/s.
- **Explicit vector types bypass the problem** (a4/a5: the body IS an AVX op),
  but per-WI-contiguous chunks de-coalesce GPU access (Xe2 104→62) and 1-WI/WG
  starves the GPU entirely (6 GB/s). **No single pattern wins both devices.**
- a2 wins both but needs element-sized grids — incompatible with the fixed-lane
  tile/entry execution model (a lane interprets many tiles per launch).

### [B] SGEMM 1024³ / [C] GEMV 2048² / [D] launch floor

| | PoCL | Xe2 |
|---|---|---|
| b1 the VM's MMA tile shape (local staging + WG barriers) | 15.6 GFLOP/s | 1502 GFLOP/s |
| b2 CPU-shaped: 1 WI/WG, 4×16 register block, float8, no barriers | **60.9** | 98 |
| c1 GEMV 1 row/WI scalar | 6.0 GB/s | 62 GB/s |
| c2 GEMV 1 row/WI float8 | **12.7 GB/s** | **73 GB/s** |
| d1 empty-kernel launch (amortized, pipelined) | 17 µs | ~1 µs |
| d2 add-4K launch | 52 µs | ~1 µs |

- The MMA tile's __local staging + barriers are pure overhead on CPU (PoCL
  loop-splits at each barrier); a barrier-free register-blocked kernel is ~4x
  faster standalone and ~11x vs the in-VM measurement (5.4 GFLOP/s at 2048).
  Still ~8x off XLA/Eigen (~475 GFLOP/s) — cache blocking/packing is the next
  rung, NOT required for the first iteration.
- GEMV: a dedicated row-dot kernel wins on **both** devices (the MMA tile
  wastes 63/64 of its work on a width-1 RHS — README already flags this on
  NVIDIA).
- The ~17–50 µs PoCL launch floor explains the small-op gap vs XLA's ~12 µs
  dispatch; it is PoCL-internal (thread wakeup), not ours. Not actionable
  in-kernel; deprioritized.

## Decision (see docs/decisions.md §11)

1. **EW tiles get an explicit-float8, per-WI-contiguous body variant selected
   at program build time by a device-keyed define (`-DVMO_CPU_TILES` when
   `!is_gpu`)** — a4's 46 GB/s on CPU (9x current), zero change for GPUs.
   Same source file, both bodies next to each other.
2. **CPU-shaped SGEMM** (`b2` shape) routed through the existing pure-matmul
   fast path (`mm2` precedent) for non-GPU devices; iterate on blocking after.
3. **GEMV kernel for both device classes**, routed when a matmul RHS/LHS has
   width 1.
4. PoCL launch floor: documented, no action.
