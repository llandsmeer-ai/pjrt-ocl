# poc/12 — fixed-trip while: OP_FOR construct vs bytecode unroll vs OP_WHILE

Nearly every `stablehlo.while` JAX emits is a **counted loop** (`lax.scan`,
`fori_loop`): carry k starts at a constant, cond is exactly `arg_k < const`,
body returns `arg_k + const_step`. Nobody hand-writes data-dependent whiles in
practice. If the trip count is known at compile time, the cond sub-list never
needs to execute and — critically — nothing has to be *read back* to decide
whether to continue:

| path       | per-iteration cost (persistent GPU engine)         | (host-dispatch CPU engine) |
|------------|-----------------------------------------------------|-----------------------------|
| OP_WHILE   | cond phases + 2 global barriers + per-lane atomic cond read | blocking host read of the cond flag (queue drain) |
| OP_FOR     | 1 global barrier                                    | nothing — the whole loop streams into the enqueue ring |
| unroll     | root-list phases (cross-iteration fusion applies)   | same, and phases batch across iterations |

The unroll path additionally binds the counter to a per-iteration const-pool
scalar, so the counter add-chain is DCE'd and `_compose_affines` can collapse
an affine recurrence across iterations (a 10-step `x*a+b` scalar recurrence
becomes ONE instruction).

## Run

```
.venv/bin/python poc/12-fixed-trip/bench.py --device Portable   # PoCL
.venv/bin/python poc/12-fixed-trip/bench.py --device NVIDIA
```

Workloads (all f32, checked bit-comparable across modes via the `sig` column):
- `fori-ew`: `x = x*a + b` with **vector** a, b (not scalar-foldable — the
  affine-composition collapse is deliberately off the table) for T steps.
- `scan-rnn`: `c = c*0.9 + xs[t]`, stacking `ys` — dynamic_slice +
  dynamic_update_slice per step, the canonical scan shape.

`PJRT_OCL_WHILE=while|for|unroll` pins the path per subprocess; `xla-cpu` rows
are the JAX CPU backend on the same machine.

## Results (2026-07-16; best-of-5 wall ms per execute)

Full data: `results_pocl.csv` / `results_nvidia.csv`. Highlights (ms):

**NVIDIA (persistent VM engine)** — FOR is 3.2–3.5x over WHILE on fori-ew,
unroll ~2x more and reaches XLA-CPU parity at small sizes:

| workload | n×T | while | for | unroll | xla-cpu |
|---|---|---|---|---|---|
| fori-ew | 4096×8   | 0.52 | 0.16 | **0.09** | 0.09 |
| fori-ew | 4096×512 | 27.9 | 7.9  | **4.1**  | 2.9  |
| fori-ew | 1M×512   | 53.6 | **28.6** | (arena guard) | 2.9* |
| scan-rnn | 4096×8  | 1.09 | 0.70 | **0.42** | 0.39 |
| scan-rnn | 1M×8    | 2.24 | 1.79 | **1.10** | 1.87 |

**PoCL (host-dispatch engine)** — FOR removes the blocking per-iteration cond
read; unroll additionally batches phases across iterations (up to 21x):

| workload | n×T | while | for | unroll | xla-cpu |
|---|---|---|---|---|---|
| fori-ew | 4096×8   | 1.91 | 0.71 | **0.24** | 0.12 |
| fori-ew | 4096×128 | 29.8 | 21.0 | **1.4**  | 0.82 |
| fori-ew | 1M×8     | 20.3 | 19.2 | **10.8** | 10.2 |
| scan-rnn | 4096×8  | 8.9  | 6.9  | **4.4**  | 0.16 |
| scan-rnn | 1M×8    | 274.6 | 270.2 | **126.8** | 0.16* |

\* XLA CPU rows marked * are after XLA-side loop optimizations our comparison
can't disable (and CPU-contended rows fluctuate); the primary comparison is
while/for/unroll on the same backend.

## In-place DUS follow-up (2026-07-17)

Scan at large n×T was bound by the dynamic_update_slice identity copy (the
full ys buffer re-materialized per iteration), not loop mechanics. SHIPPED:
when the DUS operand is the loop carry itself and nothing else in the body
reads it, the lowering NOPs the identity copy and scatters the update row
straight into the carry buffer — per-iteration ys traffic drops from O(T·n)
to O(n). Re-run of `scan-rnn` (full data: `results_*_inplace_dus.csv`;
identical `sig` checksums):

| device | n×T | for (before) | for (after) | xla-cpu |
|---|---|---|---|---|
| NVIDIA | 4096×512 | 42.2 | **20.4** | 4.8 |
| NVIDIA | 1M×8     | 1.79 | **1.11** | 1.09 (tied) |
| NVIDIA | 1M×32    | —    | **4.84** | 4.84 (tied) |
| PoCL   | 4096×512 | 1734.7 | **205.5** | * |
| PoCL   | 1M×8     | 270.2  | **60.6**  | * |
| PoCL   | 1M×32    | 3582.0 | **237.4** | * |

WHILE mode gains too (the fold is mode-independent). Unroll does NOT get the
fold (unrolled iterations are pure SSA writing fresh buffers), so unroll now
LOSES to FOR on large scans (PoCL 4096×512: 940 vs 205 ms) — the auto-mode
arena estimate already steers those to FOR because the per-iteration DUS
result counts the full (T,n) ys size. The remaining gap to XLA CPU at
mid-size scans is the per-row dynamic_slice/scatter phase overhead, no longer
bulk memory traffic.

Scheduler prerequisite: in-place carry commits broke the "program is SSA, WAR
edges unneeded" assumption — the in-place scatter's runtime-index read of the
counter carry could share a barrier phase with the counter copy-back (caught
by the schedule simulator's order-independence check). `_depends` now adds
WAR edges; they never fire in the SSA bulk (0 extra phases on a
transformer-ish block), only at carry commit points (python/NOTES.md A2).

## Decision

`PJRT_OCL_WHILE=auto` (default): unroll when `trip <= PJRT_OCL_UNROLL_TRIPS`
(64) and `trip x estimated body bytes <= PJRT_OCL_UNROLL_ARENA_MB` (256 — the
arena is a bump allocator; >=2 GiB overflows the u32/bit-31 offset space, now
a clean compile error), else OP_FOR. Plain OP_WHILE only for genuinely
data-dependent conds. Full write-up: docs/decisions.md §15.
