# poc/01 — device-side bytecode VM (persistent megakernel)

**Status: VALIDATED on PoCL (CPU) and NVIDIA — the megakernel design survives.**

Proves the core execution model: a persistent OpenCL kernel interprets a strictly linear
instruction list (single f32 arena, offsets as buffer refs, grid-stride per instruction,
cross-workgroup barrier between instructions).

## Build & run

```
make
./vm                              # PoCL (default)
OCL_PLATFORM=NVIDIA ./vm          # NVIDIA
VM_BENCH_N=4096 VM_BENCH_K=2000 ./vm   # small-op overhead focus
```

Env knobs: `OCL_PLATFORM`, `VM_GROUPS`, `VM_LOCAL`, `VM_STRESS`, `VM_BENCH_N`, `VM_BENCH_K`.

## Tests

1. **correctness**: 5-instruction program (iota/fill/mul/sub/revadd), checked exactly on host.
2. **barrier stress**: chains of `REVADD` ping-pongs — every instruction has a cross-workgroup
   data dependency on the previous one; a single missed/broken barrier corrupts the result.
3. **overhead**: K adds executed inside the VM vs K separate `clEnqueueNDRangeKernel` launches.
4. **nested while**: 2-deep `OP_WHILE` (nested instruction lists + frame stack, no jumps);
   3×4 loop doubling a 64k vector with scalar loop counters, checked exactly.

## Results (2026-07-14)

| target | correctness | stress | barrier cost | vm vs launches |
|---|---|---|---|---|
| PoCL, Ryzen 3900X (24 groups × 64) | PASS | PASS (200 instr) | ~58 µs/instr | 0.93x @1M elems |
| NVIDIA RTX PRO 6000 (188 groups × 64) | PASS | PASS (2000 instr) | ~1.1 µs/instr | **2.54x @1M, 3.19x @4K** |

See NOTES.md for caveats (co-residency, memory-model status of the 1.2-style barrier).
