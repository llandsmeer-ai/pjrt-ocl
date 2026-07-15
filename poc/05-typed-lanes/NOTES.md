# poc/05 notes — verdict for docs/tile-isa.md ceiling-1

## Verdict

**Typed lanes (per-op-family kernels, concurrent launches synced via shared atomic flags) ARE
viable on both NVIDIA and PoCL — mechanism validated, correct, and cheaper per sync than the
existing intra-kernel barrier — but ONLY under an occupancy-derived group-count rule, not a
fixed one. The rule differs sharply by device class:**

- **NVIDIA**: co-residency of two different kernels' workgroups holds up to a real hardware
  occupancy ceiling — measured at **~6x the CU count** (bisected between 6.0x=OK and
  6.5x=DEADLOCK, i.e. between 1536 and 1664 threads/SM) for this PoC's specific kernels (256
  threads/group, ~1KB local memory, trivial register footprint). Below that ceiling, sync cost
  degrades *gracefully* with oversubscription (637 ns/round at 1x -> 2759 ns/round at 6x) —
  no cliff, no correctness risk, just more rounds queued behind fewer physically-resident SM
  slots. Above the ceiling: **hard deadlock**, caught cleanly by the watchdog every time.
  - This 6x number is an upper bound specific to tiny kernels, not a constant to hardcode. A
    real "mma-ish" typed lane (the actual MMA_TILE kernel, register/local-mem heavy per
    poc/04's `mma_tile`) will have a LOWER occupancy ceiling than this PoC's dummy compute.
    **The real rule (matches the CLAUDE.md rule already written for the single-VM case): launch
    geometry for typed lanes must come from a per-kernel occupancy query
    (`clGetKernelWorkGroupInfo(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE)` /
    device max-threads-per-CU vs. the kernel's actual register/local-mem footprint), summed
    across ALL concurrently-launched typed kernels, not derived from problem size.** A
    conservative fallback (sum of groups across all typed kernels <= 1x CU count) is always
    safe and costs nothing extra vs. the current single-VM design's launch sizing.
- **PoCL**: co-residency holds up to and including **exactly `sum(GA, GB, ...) == CU count`**
  (24) — one group over on EITHER kernel (25 total) deadlocks, unconditionally, regardless of
  the GA/GB split (tested `13,12` and `12,13`, both 25 total, both deadlocked; `12,12` and
  `6,18`, both 24 total, both passed). This is the exact same rule poc/01 already found for the
  single-kernel VM (`Do not launch more groups than CUs on PoCL`) — typed lanes do not make the
  PoCL constraint worse, but they DO mean the existing CU budget must be split BETWEEN the typed
  kernels rather than given entirely to one interpreter. E.g. if the current single-VM design
  uses `nlanes = CUs` on PoCL, switching to two typed kernels of GA+GB=CUs halves the lanes
  available to either op family relative to today — a real (if bounded) capacity cost on CPU
  that NVIDIA does not pay (NVIDIA has ~6x headroom to spend on splitting instead of a hard 1x
  wall).

**Practical recommendation for the scheduler**: typed lanes are safe to build IF the
per-device launch-sizing step (already planned — "occupancy queries, not problem size", per
CLAUDE.md) is extended to size the SUM of all concurrently-resident typed-kernel launches
against the device's real concurrent-block capacity, calibrated per kernel (like poc/04's
tile-cost calibration, but for occupancy instead of throughput). On PoCL specifically, budget
conservatively at `sum(lanes) <= CUs`; on NVIDIA, an occupancy query per kernel (not a fixed
multiplier) determines how much headroom exists above 1x.

## What was tried / what happened

- First design pass used the exact poc/04 WAIT/SIGNAL `entry_t` idiom (atomic_add signal +
  atomic-poll spin wait + group barrier for cross-thread visibility), just split across two
  `__kernel` functions on two `cl_command_queue`s instead of two entries in one lane's stream.
  Worked correctly on the first attempt on BOTH platforms, no memory-model surprises beyond what
  poc/01/04 already found (atomics for anything steering control flow / synchronization).
- **Methodology bug found and fixed**: the host watchdog poll loop uses a 2ms `nanosleep`
  interval (fine for liveness detection). Using that same host wall-clock delta as the
  *performance* measurement was wrong — for these sub-millisecond kernels (94,94 x 1000 rounds
  finishes in ~0.6ms real time), the 2ms poll granularity dominated and gave numbers that did not
  scale linearly with round count (measured 4148 ns/round at 500 rounds vs 2059 ns/round at 1000
  rounds for the SAME config — a dead giveaway of quantization, not real behavior). **Fix**:
  `CL_QUEUE_PROFILING_ENABLE` on both queues; report `min(START_a,START_b)` to
  `max(END_a,END_b)` from `clGetEventProfilingInfo`, which scales correctly with round count
  (376,376: 500 rounds -> 0.908ms, 1000 rounds -> 1.787ms, ~2x as expected) and is independent of
  host poll timing. Host wall-clock is now used ONLY for the watchdog's liveness budget, never
  for the reported rounds/sec.
- Bisected the NVIDIA oversubscription ceiling by binary search once 3x (282,282) passed cleanly
  and 8x (752,752) reliably deadlocked: 4x OK, 6x OK, 6.5x DEADLOCK, 7x DEADLOCK. All deadlocks
  were caught by the 10s watchdog every time (well under the 60s outer `timeout`); no case ever
  needed the outer timeout as the actual backstop — `_exit(2)` on trip was instant and clean, no
  hang, no corrupted terminal state, no stuck process observed in any of ~15 runs.

## Driver quirks observed

1. **Must `clFlush()` BOTH queues after enqueueing both kernels, before polling/waiting on
   either.** An in-order `cl_command_queue`'s `clEnqueueNDRangeKernel` only guarantees the
   command is queued, not that it reached the device. Waiting on queue A's event first (e.g. via
   `clFinish(qa)`) without having flushed queue B risks the driver never having submitted B's
   kernel at all, producing a **false deadlock that is a host-side queuing bug, not a
   co-residency failure**. Flushing both immediately after both enqueues avoids this ambiguity
   entirely and is cheap.
2. **NVIDIA event status reports `CL_SUBMITTED` (not `CL_RUNNING`) for the entire watchdog
   window on deadlocked (oversubscribed) runs**, even though the kernel is provably executing
   on-device (real wall time elapses, atomics are being spun on — confirmed by the fact that
   sub-ceiling configs with less work finish faster, proportionally). Do not use the
   `CL_EVENT_COMMAND_EXECUTION_STATUS` transition to `CL_RUNNING` as a liveness heartbeat on
   NVIDIA — it is not a reliable signal here. Wall-clock elapsed time against the watchdog budget
   is the only dependable deadlock signal.
   - Contrast: PoCL DOES report `CL_RUNNING` (status 1) throughout its deadlocked runs — the two
     platforms' event-status semantics diverge under oversubscription in an observable way,
     another data point for "do not assume portable meaning of intermediate CL event states."
3. **PoCL's co-residency wall is exact and CU-count-derived, not a soft degradation** — unlike
   NVIDIA's graceful-then-cliff pattern, PoCL goes straight from "works" (24 total groups) to
   "deadlocks after the full 10s watchdog with zero progress" (25 total groups). No intermediate
   slowdown regime to observe or tune around; it's a hard admission-control boundary.
4. Kernel-launch/build overhead (PoCL kernel cache compile on first invocation) is easily
   mistaken for the phenomenon under test if `env.sh` isn't sourced first — first-ever run in a
   fresh `POCL_CACHE_DIR` was ~100x slower than steady-state (unrelated to co-residency; it's
   PoCL JIT-compiling and caching the `.cl` source to native code). Not a bug, just a reminder
   that `. env.sh` (mandatory anyway, for the disk-full constraint) also stabilizes timing runs.

## Not covered by this PoC (future work if typed lanes get built for real)

- More than 2 concurrent typed kernels (the real design may want 3+ op families: MMA/EW/REDUCE/
  GATHER lanes simultaneously) — the pairwise flag scheme generalizes (N counters, N-1 wait
  conditions per kernel) but was not tested here.
- Occupancy query API (`clGetKernelWorkGroupInfo`) was not used to DERIVE the ceiling
  automatically — the 6x NVIDIA number here is empirical/bisected, not queried. Wiring the
  occupancy query up and cross-checking it against this PoC's measured ceiling is the natural
  next step before typed lanes get integrated.
- Register-heavy kernels (the actual MMA_TILE) will have a materially lower ceiling than this
  PoC's near-register-free dummy compute; re-run this same matrix with poc/04's real `mma_tile`
  body substituted in before trusting a specific multiplier for the real scheduler.
