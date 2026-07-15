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

## Schedule artifact: PER-LANE INSTRUCTION STREAMS (model of record, user-corrected 2026-07-14)

The scheduler emits **one linear bytecode stream per lane**. One megakernel launch with N
workgroups; lane i interprets stream i at its own pace — **no global synchronization**.

Stream entry: `{ task_id | NOP, tile_lo, tile_hi, wait_flag, wait_count, signal_flag }`
- If `wait_flag != NONE`: atomic-poll `flags[wait_flag] >= wait_count` before executing
  (the poc/01 atomic-read pattern; a waiting lane spins — co-residency required, as always).
- Execute tiles [tile_lo, tile_hi) of the task serially.
- If `signal_flag != NONE`: `atomic_add(&flags[signal_flag], tiles_done)`.

Dependencies are per-op completion counters: each tensor op gets a flag; producers' lanes add
their tile counts; a consumer entry waits for `flags[op] == n_tiles(op)`. Consumers start the
moment their dependency is met — lanes pipeline past each other, imbalance absorbs across the
program instead of costing max-lane per step. Per-TILE (finer than per-op) dependence counters
are a marked refinement of the same mechanism.

"Ticks" survive only as a scheduling heuristic (a mental level-structure for the packer) and as
an optional lockstep debug mode (WAIT-ALL after every entry ≡ poc/04's validated tick barrier) —
useful for deterministic replay and per-step profiling.

Control flow: each lane's stream contains the same structured WHILE/IF entries (cond scalar read
atomically; all lanes take identical control decisions — uniformity rule from poc/01); bodies
are per-lane sub-streams. A control region acts as a natural all-lanes sync point (cond producer
signals; all lanes wait it).

Lanes = persistent workgroups, 1–4 per CU, 256 threads, validated co-residency regime.

## v1.1: LOCAL-memory tile slots for fusion (user, 2026-07-15; register premise CORRECTED)

Original framing was "explicit data moves keep working memory in registers." **That does not
work and is dropped**: GPU registers are not addressable, so a slot file indexed by a runtime
slot id spills to local/private memory regardless. LOAD/STORE never reaches registers.

What survives — the FUSION benefit, restated on the correct (local-memory) premise: arena =
global; each lane owns a **local-memory tile-slot file** (~5 slots × 64×64 f32). `LOAD_TILE`/
`STORE_TILE` move arena↔slot; compute tile ops may address slots (operand/dest arena-or-slot
flag, reserved in the v0 entry encoding).
- **Fusion by scheduling**: a consumer scheduled on the SAME lane right after its producer reads
  the slot from LOCAL memory instead of round-tripping through the arena — elementwise chains do
  1 global load + 1 global store total; matmul epilogue (bias/act) fuses onto the accumulator
  tile in local memory. No compile-at-dispatch.
- Scheduler additions: slot liveness = per-lane linear-scan allocation over the local-mem slot
  file; **lane-affinity constraint** (slots lane-private; cross-lane consumers go via arena).

### Register pressure — MEASURED, not a slot-file problem (2026-07-15)

Question raised: could a "global register file reused across opcodes by casting" cut the
megakernel's register footprint? **No — the compiler already does it, measured on NVIDIA
(PTX virtual-register counts, same compiler ⇒ comparable):**
- EW-only kernel: 4 f32 vregs. MMA-only (8×8 reg block): 595. Switch over BOTH: 598 ≈ MMA, NOT
  4+595. ⇒ the allocator reuses registers across mutually-exclusive cases automatically
  (max-over-cases, not sum). A manual union/register-file adds nothing and risks forcing
  spills (dynamically-indexed union arrays can't stay in registers).
- BUT the combined kernel costs 598 regs on EVERY lane, incl. ones running a cheap add, because
  occupancy = kernel PEAK pressure and MMA's 64 accumulators are all simultaneously live. No
  reuse trick shrinks a peak that's live at once. ⇒ ceiling-1 is intrinsic; the ONLY escape is
  the fat op in a separate kernel (typed lanes, poc/05-validated). Repo: probe in job tmp.
- Rollout unchanged: Phase 1 = arena-mode ops only (slot fields reserved); slot-fusion + typed
  lanes land in Phase 3.

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

## Ceiling assessment (2026-07-14, "does anything block cuBLAS-class perf?")

- Architecture: NO blocker — persistent-lane tile ranges ≡ CUTLASS stream-K structure;
  interpreter overhead amortizes to ~0.
- Ceiling 1 (engineering): one binary per launch ⇒ fattest tile-op's registers/local tax all
  lanes. Mitigation: typed lanes as CONCURRENT kernel launches syncing via shared atomic flags.
  ✅ **poc/05 VALIDATED 2026-07-15**: two different kernels on separate in-order queues DO stay
  co-resident and sync via shared atomic flags. NVIDIA: works to ~6x CU count (dummy kernels),
  cross-kernel handshake CHEAPER than intra-kernel barrier (0.58×). PoCL: hard wall at
  sum(groups) ≤ CUs. RULE: launch geometry = per-kernel occupancy query summed over concurrent
  typed kernels, NOT a fixed multiplier (real MMA kernel's register use lowers its ceiling).
  Quirks: clFlush all queues before waiting (else host-stall looks like deadlock); NVIDIA event
  status stays CL_SUBMITTED during deadlock (not a liveness signal) — use a host watchdog.
- Ceiling 2 (accept): no SASS/PTX access from OpenCL on NVIDIA ⇒ CLBlast-class 40–70% of SIMT
  peak is the realistic target (cuBLAS ~85–90%).
  📈 **poc/06 progress 2026-07-15**: naive 16×16 (4.3 TFLOPS) → register-blocked 128×128 BK16
  = **26.2 TFLOPS / 24.7% peak, portable** (6.1×). Biggest lever: 8×8 register µtiles +
  non-transposed A staging. 16 KB local, ~80 reg/thread (occupancy limiter, 2 wg/SM optimal).
  Negatives: float4 staging net-neutral & crashes PoCL compiler (→ vendor override only);
  double-buffering 0% (NVIDIA hides latency at 2 resident wg); interleaved µtile map REGRESSED.
  Need warm GPU clocks to measure (2.1 vs 4.3 cold/warm). Next for 30+: float4 register
  accumulators, subgroup broadcast, -cl-mad-enable. This is the MMA_TILE the VM adopts in Phase 3.
- Ceiling 3 (fundamental): matrix units unreachable from OpenCL on NVIDIA (tensor cores) and
  AMD (MFMA); Intel exposes cl_intel_subgroup_matrix_multiply_accumulate. SIMT fp32 rate is our
  matmul ceiling ⇒ ~4–8x behind tensor-core TF32/BF16 on matmul-heavy ML on NVIDIA. Per-vendor
  escape hatches = marked future branches.
