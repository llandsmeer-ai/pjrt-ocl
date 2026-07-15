# poc/05 — typed-lanes cross-kernel co-residency (docs/tile-isa.md ceiling-1)

**Status: VALIDATED on NVIDIA + PoCL, with a hard occupancy ceiling on both — see NOTES.md
for the verdict.**

Question: can two DIFFERENT kernels, launched concurrently on separate in-order queues of the
same device, stay co-resident and synchronize through atomic flags in a shared buffer? This
decides whether "typed lanes" (per-op-family kernels with separate register budgets, ceiling-1
in docs/tile-isa.md's "Ceiling assessment") are viable, or whether the megakernel-VM's one
fat interpreter is mandatory.

## Design

`kernels.cl` has two independent `__kernel` functions:
- `mma_ish` (GA groups x 256 threads): heavier dummy compute (64 FMA-ish iters/round).
- `ew_ish` (GB groups x 256 threads): lighter dummy compute (8 iters/round).

Each round: dummy compute -> SIGNAL own counter (`atomic_add`, lane 0) -> WAIT (atomic spin,
lane 0) until the partner kernel's counter has reached the round's threshold -> group barrier so
every thread sees the partner's writes. This is exactly poc/04's `entry_t` WAIT/SIGNAL idiom,
applied ACROSS two kernel launches instead of within one. `flags[0]`/`flags[1]` are two `uint`
counters in one small shared buffer; thresholds are `(r+1)*GB` / `(r+1)*GA` so the scheme works
for GA != GB (each flag is only ever incremented once per group per round by its owner kernel).

Host (`main.c`) launches `mma_ish` on queue A and `ew_ish` on queue B (two separate in-order
`cl_command_queue`s on the same context/device), `clFlush`es both, then polls
`clGetEventInfo(..., CL_EVENT_COMMAND_EXECUTION_STATUS, ...)` non-blockingly every 2ms against a
`WATCHDOG_S` (default 10s) wall-clock budget. On trip: print `DEADLOCK` and `_exit(2)` — no
further CL calls, so a genuinely stuck spinning kernel cannot hang process teardown. Timing for
successful runs uses `CL_QUEUE_PROFILING_ENABLE` device-clock timestamps (the host poll loop's
2ms granularity is far too coarse for these sub-millisecond kernels — see NOTES.md). Correctness:
after completion, both flag counters must equal exactly `GA*rounds`/`GB*rounds`, and every
work-item's final scratch write must show the full round count (catches early-exit/divergence).

## Build & run

```
. ../../env.sh          # mandatory: root overlay is full, pins caches onto the project mount
make
./poc <platform_substr> <GA> <GB> [rounds=1000]      # ONE config per process
timeout 60 ./poc NVIDIA 94 94 1000                    # NVIDIA
timeout 60 ./poc Portable 12 12 1000                  # PoCL
```

Every invocation MUST be wrapped in `timeout 60` — the internal watchdog is a graceful
detector, `timeout` is the hard backstop.

## Results (2026-07-15, RTX PRO 6000 Blackwell Max-Q [188 CUs] / PoCL Ryzen 9 3900X [24 CUs])

Timing = device-clock span (earliest kernel START to latest kernel END across both queues),
`CL_QUEUE_PROFILING_ENABLE`. "x CUs" = `(GA+GB)/device_max_compute_units`.

### NVIDIA — required matrix + oversubscription ladder

| GA | GB | x CUs | rounds | status | ns/round | rounds/s |
|---|---|---|---|---|---|---|
| 94 | 94 | 1.0x | 1000 | OK | 637.5 | 1,568,696 |
| 188 | 188 | 2.0x | 1000 | OK | 966.3 | 1,034,905 |
| 47 | 141 | 1.0x (asym) | 1000 | OK | 636.8 | 1,570,273 |
| 282 | 282 | 3.0x | 1000 | OK | 1,352.7 | 739,260 |
| 376 | 376 | 4.0x | 1000 | OK | 1,787.2 | 559,545 |
| 564 | 564 | 6.0x | 1000 | OK | 2,759.2 | 362,428 |
| 611 | 611 | 6.5x | 500 | **DEADLOCK** | (watchdog, 10s) | — |
| 658 | 658 | 7.0x | 300 | **DEADLOCK** | (watchdog, 10s) | — |
| 752 | 752 | 8.0x | 300 | **DEADLOCK** | (watchdog, 10s) | — |

Ceiling bisected between 6.0x (works) and 6.5x (deadlocks) = between 1536 and 1664 threads/SM
for this (256-thread, ~1KB local mem, trivial-register) kernel pair — consistent with a common
`CL_DEVICE_MAX_WORK_GROUP_SIZE`-class hardware occupancy cap of 1536 threads/SM. **This is a
property of these specific tiny kernels, not a universal constant** — see NOTES.md.

### PoCL — required matrix + oversubscription

| GA | GB | x CUs | rounds | status | ns/round | rounds/s |
|---|---|---|---|---|---|---|
| 12 | 12 | 1.0x | 1000 | OK | 127,173 | 7,863 |
| 6 | 18 | 1.0x (asym) | 1000 | OK | 114,493 | 8,734 |
| 24 | 24 | 2.0x | 300 | **DEADLOCK** | (watchdog, 10s) | — |
| 36 | 36 | 3.0x | 300 | **DEADLOCK** | (watchdog, 10s) | — |
| 13 | 12 | 1.04x (25 total) | 100 | **DEADLOCK** | (watchdog, 10s) | — |
| 12 | 13 | 1.04x (25 total) | 1000 | **DEADLOCK** | (watchdog, 10s) | — |

PoCL's ceiling is a **hard wall at exactly `GA+GB <= device CU count`** (24) — even one group
over (25 total, either kernel) deadlocks. Matches poc/01's NOTES.md finding for the
single-kernel case exactly: PoCL's thread pool has one worker per CU with no time-slicing of
spinning groups.

### Comparison to poc/01's single-kernel barrier cost (~1.1 µs/instr on NVIDIA, ~58 µs/instr PoCL)

At the baseline 1x-CU-count config (which is the realistic operating point — docs/tile-isa.md
specs "1-4 [lanes] per CU"), one typed-lanes round-trip SIGNAL+WAIT costs:
- NVIDIA: 637 ns — **~0.58x** the cost of poc/01's intra-kernel global barrier (1.1 µs). Cheaper,
  not more expensive: this mechanism only synchronizes two shared counters (not an N-way
  rendezvous of every group), so it is structurally lighter than a full barrier.
- PoCL: 127 µs — **~2.2x** poc/01's single-barrier cost (58 µs). Plausible: two independent
  cross-thread spin-waits plus OS thread-scheduling overhead on a 24-core CPU.

See NOTES.md for the full verdict and the group-count rule for docs/tile-isa.md ceiling-1.
