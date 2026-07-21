# poc/18 — CPU SGEMM: 2D grid + unrolled accumulators (PoCL)

**Question:** poc/10 left the packed CPU SGEMM (`mm2p`) at ~156 GFLOP/s @2048,
~2.6x off Eigen, and the poc/10 README stopped there. Two structural things
were still on the table: (1) the 6x16 accumulator block was an `a0[6]/a1[6]`
*array* — does PoCL/LLVM keep it in registers? (2) one workgroup owned a full
6-row stripe of the *entire* N, so at small M only a handful of workgroups run
— do the 24 threads even fill?

**Findings (this harness, `v6` = the winner; ratios are the datum):**

| step | @512 | @1024 | @2048 |
|---|---|---|---|
| v3 poc/10 (array accs, 1D grid, KC=512) | 178 | 230 | 298 |
| v4 = v3 + **fully-unrolled named accumulators** | 244 | 266 | 374 |
| v6 = v4 + **2D grid (PC=2 panel-groups)** | 342 | **503** | **555** |

- **Named accumulators (v4, +25% @2048).** With `float8 a0[6],a1[6]` PoCL's
  LLVM leaves the arrays in memory across the k-loop (SROA fails), so every FMA
  round-trips through the stack. Unrolling into 12 named `c00..c15` keeps them
  in ymm. This alone is the single biggest lever.
- **2D grid (v6, the big one @≤1024).** One WI/WG maps to one PoCL host thread;
  a 1D grid of `ceil(M/6)` groups means at M=1024 only 171 workgroups exist and
  the 24 threads are starved + imbalanced. Splitting N into `PC`-panel groups
  (PC=2 → 32 cols) multiplies the workgroup count by `N/32`, filling all cores
  and keeping each WG's B slice L1/L2-hot. @1024 230→503 (2.2x). **PC=1..2 best;
  PC≥8 collapses** (too few groups again / bad WG size).
- **Fast-math (`-cl-fast-relaxed-math`) is a WASH** — the FMA is already
  explicit via `mad()`, so nothing to gain, and we keep f32-exactness (CPU has
  no TF32; the megakernel must match XLA-CPU bit-closely).
- **KC sweeps barely matter now**: KC=1024 vs single-sweep are within noise for
  K≤2048 (the 6x32 C tile reloaded between sweeps is tiny). KC blocking only
  pays for very large K (keeps the packed-B panel L2-resident).
- **Power-of-2 residual**: @2048 the standalone kernel hits ~555, but the SAME
  kernel *in the VM arena* dips to ~370 — classic power-of-2 leading-dimension
  cache-set aliasing (A stride K·4 and C stride N·4 both = 8 KiB → same L1 set),
  worsened by the arena's A/C relative offsets. 1536/1792 (non-power-of-2) run
  557–602. Fixing needs packed-A / dim padding; deferred (poc/10 stop point).

**Shipped** (`vm_main.cl` `mm2p`, `runtime.cc` LaunchMatmul 2D geometry): the
default CPU packed SGEMM is now 2D-gridded with unrolled accumulators. In-VM
single-matmul sweep: 62/219/230/301 → 112/447/600/370 GFLOP/s (256/512/1024/
2048). See decisions.md §12a.

Build+run: `make && PJRT_OCL_DEVICE=Portable ./poc18` (env `PC=`, `KC=`).
