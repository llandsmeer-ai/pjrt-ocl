# Tile ISA & schedule model (v1 spec)

Agreed 2026-07-14 (user + design discussion; see decisions.md #1). Three layers:

```
StableHLO ──lower──▶ tensor bytecode (VMProgram)      portable, compact, symbolic-friendly
                        │ per-device schedule compile (µs — rerun per shape tuple, L1)
                        ▼
                    task descriptors + schedule table  device-specific artifact
                        │ consumed by
                        ▼
                    VLIW engine: persistent lane-interpreters, tick-synchronous
```

The tensor bytecode remains the ISA of record; the streamed-launch engine consumes it directly.
The tile layers are an optional, per-segment execution strategy.

## Tile-op vocabulary (execution opset, fixed & small)

| tile op | unit of work (one cell slot, one workgroup) |
|---|---|
| NOP | idle this tick |
| EW_TILE(subop) | elementwise chunk of TS elements (TS device-tuned, e.g. 16384); subop = add/mul/sub/div/unary/cmp/select/fill/iota… |
| MMA_TILE | one TMxTN output tile of a matmul, local-memory staged, K-loop inside |
| REDUCE_PARTIAL | one input chunk → one partial accumulator slot |
| REDUCE_COMBINE | fold partial slots → final (next tick after partials) |
| GATHER_TILE | strided-gather chunk (broadcast/transpose/slice/reverse) |
| FUSED_TILE(template) | reserved: producer-consumer chains on a resident tile |

Standardized tile shapes are what make cell durations predictable → dense ticks.

## Task descriptors

One per tensor-op instance, emitted by schedule compilation (NOT per tile):
`{ tile_op, dst/a/b buffer ids, params (dims, subop, K, …), n_tiles }`.
A lane derives tile coordinates from (descriptor, tile index) by divmod — tiles are never
materialized as instructions. Tile sizes are chosen per device at schedule-compile time;
the tensor bytecode stays device-neutral.

## Schedule table

`cell[tick][lane] = { task_id | NOP, tile_lo, tile_hi }` — a lane executes tiles
[tile_lo, tile_hi) of the task, serially, within the tick. Dependences are compiled into tick
order: all tiles of a producer land in ticks strictly before any consumer tile (per-tile
dependence/pipelining is a marked future refinement — Mirage-style events instead of ticks).
Tick boundary = the validated inter-workgroup barrier (poc/01). Lanes = persistent workgroups,
1–4 per CU, counts in the validated co-residency regime (≤ ~3× CUs, 256 threads).

Control flow: tick-range jumps. All lanes read the cond scalar atomically after a tick barrier
(poc/01 rule) and uniformly continue at the while/if sub-range. Nested, linear, no jumps —
unchanged semantics, now over tick ranges.

## Cost model & calibration (hardware-dependent by design)

Per-tile-op costs vary per device ⇒ **measured, not assumed**:
- On first client creation per device (and cached to disk keyed by device+driver), run a
  µbenchmark schedule: N ticks of each tile-op class → µs per EW_TILE, MMA_TILE, GATHER_TILE,
  REDUCE_*. Seconds of one-time work.
- The scheduler packs cells so each tick's max-lane cost ≈ mean (LPT bin-packing within a
  dataflow level, then greedy level merge). Cost model is a lookup of calibrated numbers.
- Recalibrate when device/driver hash changes; manual override env `PJRT_OCL_CALIBRATE=1`.

## Instrumentation (optional, for bubble analysis)

Goal: measure per-tick occupancy so idle lanes are visible, not assumed away.
- **Logical-clock mode** (portable): after finishing its cell (before the barrier), lane 0 of
  each group records `atomic_inc(&order)` into `inst[tick][lane]`. Completion-rank spread per
  tick ≈ imbalance; NOP/busy counts give occupancy. Near-zero overhead, always available.
- **Wall-clock mode**: `cl_khr_kernel_clock` where exposed (feature-detect); NVIDIA fallback via
  inline-PTX `%globaltimer` if we ever need it. Otherwise: debug engine runs one launch per tick
  with CL event profiling (exact tick durations, no per-lane detail).
- Reporting: `bubble % = 1 - (Σ cell_cost / (n_lanes × Σ tick_max))`, printed per execute under
  `PJRT_OCL_VM_STATS=1`; feeds back into calibration (profile-guided re-packing, later).

## Physics notes (recorded so we don't oversell)

- Spatially co-scheduled memory-BW-bound ops share DRAM bandwidth: parallel ≈ serial for
  BW-saturated ops. Wins live in small/latency-bound ops, imbalanced graphs, scalar chains.
- Register pressure: interpreter carries the fattest tile-op. Mitigation if it bites: typed
  lanes (per-family interpreters on static SM subsets).

## Validation path

`poc/04-vliw-vm`: (a) two independent EW tasks co-scheduled in one tick on disjoint lanes —
correct; (b) matmul as MMA tiles (local memory) — correct vs host; (c) reduce as
partials+combine across ticks — correct; (d) calibration µbench + cost-aware vs naive packing —
measured bubble % improves; (e) perf vs serial megakernel and streamed launches on a wide graph.
