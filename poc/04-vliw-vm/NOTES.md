# poc/04 notes — for the decision tree

## Model evolution within this PoC (record of the misunderstanding)

1. I first built a GLOBAL (ticks × lanes) lockstep table — barrier every tick. Works (tests
   A–D) but pays max-lane per tick and needs per-tick balancing.
2. User corrected: **per-lane bytecode streams** — N streams, one megakernel, each block
   interprets its own stream at its own pace; unequal stream lengths are the point (4 lanes =
   one fat cooperative matmul entry; other lanes = MANY small vector entries).
3. Global synchronization DOES exist but is **scheduler-placed** (BARRIER entries at dataflow
   joins); the cost model shapes per-lane work so lanes arrive together → bubbles mostly absent
   by construction, and only possible at barriers.
4. Lockstep = degenerate case (barrier after every entry); kept as debug/deterministic-replay
   mode. WAIT/SIGNAL per-op counters implemented too — finer-grained option when the scheduler
   wants joins without stopping every lane.

## Validated

- Spatial co-scheduling works: one tick, ADD on lanes [0,94), MUL on [94,188), both correct.
  The tick model + poc/01 barrier compose exactly as specced.
- Cooperative tile ops inside persistent interpreters work: 16×16 local-memory MMA tiles and
  local-tree REDUCE_PARTIAL — things the flat megakernel could not express.
- Range cells ({task, tile_lo, tile_hi}) keep the table compact; no per-tile instructions.
- Instrumented mode (one launch per tick + CL event profiling) gives exact tick durations.

## The calibration lesson (critical path)

- Naive calibration (time a tick with 1 tile/lane) is CONTAMINATED by launch + barrier
  overhead (~20–30 µs), which swamps a ~10 µs tile. Consequences observed:
  - bubble metric reports >100% "busy" (per-tile costs overestimated),
  - on PoCL the cost-aware packer made things WORSE (0.81x) — bad inputs, bad schedule.
- Fix (next iteration): calibrate with K tiles per cell for K in {1, 8, 64}; per-tile cost =
  slope of the fit (subtracts fixed overhead); calibrate per tile-op AND per operand scale
  (memory-bound EW cost is bytes-driven).
- Cross-device ratios differ hugely (EW:MMA = 0.9 NVIDIA vs 5.6 PoCL) — cost tables must be
  per-device, cached, and versioned by driver (as specced).

## Other observations

- MMA_T=16 with 256-thread lanes wastes nothing (16×16 = 256 exactly); larger tiles
  (32×32 with 2×2 per thread) are the obvious next step for arithmetic intensity.
- The naive packer's lane-split heuristic (fixed mm_lanes/ew_lanes) is a placeholder; the real
  scheduler should be LPT bin-packing per dataflow level with calibrated costs (spec).
- PoCL executes this fine at 24 lanes — the design degrades gracefully to CPUs.
- Not yet in the PoC: control flow as tick-range jumps (mechanism already validated in poc/01;
  same atomic-cond-read + uniform-jump pattern applies to the tick loop), GATHER_TILE,
  FUSED_TILE, per-tile dependence/pipelining (Mirage-style events).

## MMA gap analysis (2026-07-14, BENCH_MMA mode)

- clpeak FP32 peak: 105.9 TFLOPS. Our MMA tiles: 2.95 TFLOPS @188 lanes, 4.29 @376 lanes
  (2048³, best of 5) → ~4% of peak, ~25x from cuBLAS-class SGEMM.
- Dominant cause: 16×16 tiles ⇒ DRAM-bound (measured ~1.07 TB/s traffic at 4% compute);
  then no register blocking (1 FMA / 2 local loads). Fix is internal to MMA_TILE (M5):
  128×128 tiles + 8×8 register blocks + float4 loads + double buffering ⇒ 30–80 TFLOPS typical.
- Vendor ceilings from OpenCL: NVIDIA tensor cores UNREACHABLE (SIMT fp32 only);
  Intel exposes cl_intel_subgroup_matrix_multiply_accumulate; AMD MFMA has no OpenCL path.
