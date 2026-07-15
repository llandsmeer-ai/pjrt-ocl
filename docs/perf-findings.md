# Per-op performance findings (2026-07-15, Phase 3 dir. 2)

Harness: `tools/bench_ops.py` + `tools/bench_ops.sh`. NVIDIA RTX PRO 6000
(~1.7 TB/s HBM, 105.9 TFLOPS f32 clpeak). Measured through the real plugin,
device-resident buffers. PoCL excluded — its spin-barrier is unreliable under
iteration (decisions.md #1). Reference = JAX's own CPU backend (no CUDA jaxlib
here, so "JAX GPU" is not available as a baseline).

## Q1: do ops parallelize with execution units (lanes)?  YES — all of them.

Throughput vs lane count (2048² ops; matmul 1024³):

| op | 1 lane | 8 | 47 | 188 | 376 | scales to |
|---|---|---|---|---|---|---|
| add (GB/s) | 15 | 91 | 227 | 221 | 221 | plateaus ~47 lanes |
| broadcast (GB/s) | 4 | 25 | 71 | 78 | 76 | plateaus ~47 |
| reduce (GB/s) | 5 | 37 | 129 | 204 | 235 | scales to 376 |
| matmul (GFLOP/s) | 75 | 584 | 2716 | 6200 | 7443 | scales to 376 |

Every op speeds up monotonically with lanes until it saturates — the VLIW
engine genuinely parallelizes work across execution units. Memory-bound ops
(add/broadcast) saturate early (~47 lanes); compute/reduction-heavy ops
(matmul, reduce) keep scaling to full occupancy. Default 2×CU=376 is right for
the latter and merely harmless-plateau for the former.

## Q2: vs JAX CPU backend — we win on all ops.

| op | JAX CPU | ours (best) | speedup |
|---|---|---|---|
| add 2048² | 11.7 GB/s | 227 GB/s | 19× |
| broadcast 2048² | 22 GB/s | 78 GB/s | 3.5× |
| reduce 2048² | 134 GB/s | 235 GB/s | 1.75× |
| matmul 1024³ | 1069 GF/s | 7443 GF/s | 7× |

## Perf bugs / opportunities (evidence-based)

1. **Per-execute floor = 43 µs** (tiny op, lane-independent). Healthy for a GPU
   dispatch (XLA ~10–30 µs); Python/PJRT + kernel launch + device copies +
   clFinish. Not the bottleneck at useful sizes. No action needed now.

2. **Memory-bound ops hit only ~13–26 % of HBM peak** (add 227, reduce 235,
   broadcast 78 GB/s vs 1.7 TB/s). Leading cause: **the arena input/output
   copies**. `ExecuteDevice` device→device-copies each input into the monolithic
   arena and each output back out, so a memory-bound op moves ~2× its useful
   bytes. Evidence: `neg` (1 input) and `add` (2 inputs) take nearly the same
   time despite add's larger total traffic → time tracks fixed structure +
   copies, not useful bytes. **Fix (priority next perf item): zero-copy buffer
   binding** — pass input/output cl_mems as kernel args and have the loader mark
   buffer ids as input-slot / output-slot / arena, so the VM reads inputs and
   writes outputs directly (no arena round-trip). Architectural (vm2 ABI +
   loader), deferred from tonight to avoid destabilizing the working engine.

3. **broadcast is the weakest (78 GB/s)** — the strided GATHER does scalar
   indexed loads. Coalescing / special-casing stride-0 broadcast (a pure
   fill-from-vector) would help; lower priority than #2.

4. **matmul 7.4 TFLOPS end-to-end** at 1024³ vs 17 pure-compute (poc/06) — the
   gap is dispatch + the single-task-spread-thin effect (decisions: lone big op
   suits the streamed-launch engine, not the persistent VLIW). Climbs with size.

## Bottom line

Both questions answered: we parallelize across execution units on every op, and
we beat the JAX CPU backend everywhere (1.75–19×). The one clear perf bug is the
arena-copy memory tax on bandwidth-bound ops — fix is zero-copy buffer binding,
scoped as the next perf task.
