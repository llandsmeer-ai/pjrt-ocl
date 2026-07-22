# poc/16 — per-tile dependency scoreboard vs the grid barrier (R1, §29/§30)

Gate for R1: replace the megakernel's grid-wide spin barrier with point-to-point
WAIT/SIGNAL flags so INDEPENDENT work overlaps and fills the lane grid instead of
serializing at 17–31 % occupancy (§29).

## Build & run
```
. ./env.sh
cd poc/16-scoreboard && make
PJRT_OCL_DEVICE=NVIDIA  ./poc16      # gate on GPU (util via %globaltimer)
PJRT_OCL_DEVICE=Portable ./poc16     # gate on PoCL CPU (wall + correctness)
# knobs: G lanes, K indep tasks, P tiles/task, WORK inner-loop, ITERS, LSZ
```

`main.c` — K independent "matmul-like" tasks + 1 dependent, run two ways in one
persistent kernel: MODE 0 (barrier, today's serial phases at P/G occupancy) and
MODE 1 (scoreboard: producers spread across ALL lanes, each `atomic_inc`s its task
flag on completion; the dependent tiles WAIT until every flag reaches P, using the
same device-scope acquire/release as `vmo_barrier`). Measures correctness (stress
loop counting stale/wrong dependent reads), overlap (per-WG timestamps → lane
util), and wall speedup.

## Result — MECHANISM PASSES, but the target workload has no independent work

**The scoreboard mechanism works and is race-free on both devices:**

| device | correctness (stress) | lane util (bar→sb) | wall speedup |
|--------|----------------------|--------------------|--------------|
| NVIDIA RTX PRO 6000 | 0/94000 wrong (2000 iters) | 0.115 → 0.362 (3.2×) | 2.3× |
| NVIDIA, K=16        | 0/6900 wrong        | 0.056 → 0.326 (5.8×) | 4.0× |
| PoCL (Ryzen 3900X)  | 0/9000 wrong (1500 iters) | n/a (no gtimer)  | 15× |

Device-scope acquire/release orders the non-atomic producer writes for the
consumer on a different workgroup (poc/07 test E), point-to-point — proven
race-free over ~100k checks. Overlap materializes: independent tiles co-occupy the
grid, util rises toward full, wall drops. No hangs.

**BUT — the base transformer (the R1 target) has 0 % independent phase-level work.**
`analyze_schedule.py` + `critical_path.py` on the real lowered schedule:
- The scheduler ALREADY fuses the 3 QKV projections into ONE 3-matmul phase, and
  attention heads are ALREADY batched into one MMA task. §29's premise that these
  are "each its own barrier phase" is false — that parallelism is already extracted.
- Everything else (QKᵀ→softmax→AV→out-proj→residual→LN→FFN1→gelu→FFN2→residual→
  next layer) is a strict RAW dataflow chain.
- Overlap ceiling (total_phase_cost / critical_path), even with the OPTIMISTIC
  RAW-only edge set: **1.000× on tiny/small/base/large_l1/large — 0 % overlappable.**

A scoreboard only overlaps INDEPENDENT phases; the transformer has none. The low
31 % lane util is because each matmul is individually SMALL (64–256 tiles < 376
lanes) AND the matmuls are serially dependent — not because independent matmuls are
artificially serialized. You cannot fill the grid by overlapping a dependency
chain. The lever that DOES raise util is bigger matmul tiles / bigger effective
matmuls (R3): large_l1's 72 % comes from D=1024/F=4096 giving 4× more tiles per
matmul, not from overlap.

## Verdict (§14a)
**STOP — do not rewire the production scheduler/VM for R1.** The mechanism is
proven and kept here as ready infrastructure for a workload that HAS independent
branches (parallel transformer block à la GPT-J/PaLM, MoE, ensembles, independent
batched models). For the current serial-block transformer, R1 buys nothing; the
remaining matmul gap is R3 (per-tile intensity / bigger tile), which §29's own
bucket-2 already identified for the compute-bound end. See docs/decisions.md §30.
