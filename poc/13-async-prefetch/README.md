# poc/13 — async / prefetched DRAM loads: do they hide our tile latency?

§22 established that per-tile execution is **latency-bound** (a megakernel tile
= one workgroup running a grid-stride loop of dependent global round-trips).
Classic fix from the literature: **async / prefetched loads** — issue tile N+1's
loads while computing tile N. This PoC asks, measure-first: does it actually help
*our* loops, on *our* hardware, portably?

Two representative loop shapes (§22 candidates), each built THREE ways:

- **LOOP A — streaming elementwise** (`d = a*s + t`): the §22 headline. Pure
  streaming, **no data reuse**.
- **LOOP B — matmul K-loop** global→local stage: 64×64 tile, BK=16, 256 lanes,
  4×4 microtile — mirrors the shipped portable `vmo_mma_tile` (8 KB As/Bs). Real
  data reuse, the classic double-buffering case.

Variants: **(a)** baseline direct loads · **(b)** manual double-buffer (prefetch
next tile's globals into **registers** while computing current) · **(c)**
`async_work_group_copy` into `__local` + `wait_group_events`.

Persistent-grid faithful: fixed grid = 2·CU groups (the megakernel cap),
grid-stride over tiles. Also runs the poc/08 occupancy handshake at 8 KB vs
16 KB `__local` to price the §10c co-residency cost.

## Run

```
make
PJRT_OCL_DEVICE=NVIDIA   ./poc13 > results_nvidia.csv
PJRT_OCL_DEVICE=Portable ./poc13 > results_pocl.csv
```

All variants are bit-checked (`maxerr` column; 0.0 everywhere here). EW GB/s are
**L2-resident** (buffers reused across best-of-N reps, ≤64 MB) — treat as a
*relative* ranking of the three variants, not absolute HBM bandwidth.

## Results (2026-07-19, best-of-N wall)

### NVIDIA RTX PRO 6000 Blackwell (188 CU, grid 376)

**LOOP A — streaming EW (GB/s, higher better):**

| n | (a) scalar | (b) reg-DB | (c) async |
|---|---|---|---|
| 256K | 203 | **256** | 184 |
| 1M | 799 | **925** | 699 |
| 4M | 2061 | **2544** | 1570 |
| 16M | 3212 | **4641** | 2321 |

**LOOP B — matmul K-loop (GFLOP/s, higher better):**

| M×N×K | (a) single | (b) reg-DB | (c) async |
|---|---|---|---|
| 512³ | 1601 | **1831** (1.14×) | 249 |
| 512×2048×512 | 3602 | **3827** (1.06×) | 892 |
| 1024³ | 3676 | **3874** (1.05×) | 896 |
| 512×512×2048 | 1671 | **1892** (1.13×) | 251 |
| 2048³ | 4954 | **5226** (1.05×) | 1197 |

Occupancy: 8 KB and 16 KB `__local` both → **752 co-resident groups** (≫ 376
cap) — local footprint is not the binding constraint here.

### PoCL (AMD Ryzen 9 3900X CPU, 24 CU)

**EW (GB/s):** async is *fastest* on CPU (memcpy path): e.g. 1M 10.5 → 30.9;
4M 14.1 → 26.5. reg-DB helps modestly; scalar slowest.
**MMA (GFLOP/s):** all ~11–12; double-buffer ≈ wash, **async 10–20 % slower**.
Occupancy: 24 groups at 8 KB and 16 KB (512 KB local available; cap 48).

## Verdict (full write-up: docs/decisions.md §25)

1. **`async_work_group_copy` is a hard NO on NVIDIA.** 4–15× *slower* for matmul
   staging, ~30 % slower for EW. NVIDIA's OpenCL runtime has no DMA path — it
   emulates the copy serially. On PoCL it lowers to `memcpy` and is fine/faster.
   Fully device-dependent → unusable in the portable core path.
2. **Register double-buffering is the real (small) lever.** EW: reg-prefetch
   (= the *already-shipped* §22 float4+2× unroll) is the winner at 1.25–1.45×;
   there is nothing more to do for streaming. MMA K-loop: a reg-prefetch
   double-buffer is a **consistent but modest 5–14 %**, with **no local cost**
   (registers are the second buffer) and only ~8 extra registers.
3. **Recommendation:** don't add async to the VM. The EW win is already banked.
   The MMA double-buffer is worth folding into `vmo_mma_tile` *only* on the
   portable (non-TF32) path and only after re-checking register pressure against
   the §10c 376-lane occupancy boundary — 5–14 % on matmul that §14b shows is
   not the base bottleneck. It composes with the §23 region-op (async-load a
   region's inputs is the same reg-prefetch idea) but is not itself a headline.
