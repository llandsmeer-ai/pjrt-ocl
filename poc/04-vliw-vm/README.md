# poc/04 — tick-synchronous VLIW VM (spec: docs/tile-isa.md)

**Status: architecture VALIDATED on NVIDIA + PoCL. Calibration methodology needs work
(see NOTES.md) — it is the critical-path item for scheduling quality.**

Persistent lane-interpreters (1 workgroup/CU × 256 threads) execute a schedule table
(ticks × lanes → {task, tile range}); different lanes run different ops in the same tick;
tick boundary = poc/01 barrier. Fully on-device execution: one kernel launch per program.

## Build & run

```
make && ./vliw                 # NVIDIA (default)
OCL_PLATFORM=Portable ./vliw   # PoCL
```

## Results (2026-07-14)

| test | NVIDIA (188 lanes) | PoCL (24 lanes) |
|---|---|---|
| A: 2 independent EW ops co-scheduled in ONE tick, disjoint lanes | PASS | PASS |
| B: matmul 128³ as 64 local-memory MMA tiles | PASS | PASS |
| C: reduce 1M = partial tiles tick → combine tick | PASS | PASS |
| D: calibrated EW:MMA tile-cost ratio | 0.9 | **5.6** |
| D: cost-aware vs naive packing (wide graph: 512³ matmul + 8×1M adds) | **1.65x faster** | 0.81x (slower!) |

Headline: the same schedule policy wins 1.65x on one device and loses on another —
**hardware-measured cost models are not optional** (user called this; data agrees).
Bubble metric of the lockstep engine reported >100% busy: calibration contamination (NOTES).

## Test E — the model of record: per-lane streams (user-corrected design)

`vliw_async` kernel: each lane interprets ITS OWN instruction stream (unequal lengths);
global sync only at scheduler-placed BARRIER entries; optional WAIT/SIGNAL per-op counters.
Scenario: lanes 0–3 cooperate on a 256³ matmul (1 fat entry each) while lanes 4+ run 8
elementwise ops as many small entries; one global barrier; consumer phase (Z = C + C).

| | NVIDIA | PoCL |
|---|---|---|
| E correctness | PASS (0.58 ms) | PASS (13.3 ms) |
| last barrier arrival | **mma lane** (matmul = critical path) | **ew lane** (EW dominates on CPU) |

The barrier arrival-rank instrumentation directly names the lane class to unload —
the cost-model feedback signal, per device, for free.
