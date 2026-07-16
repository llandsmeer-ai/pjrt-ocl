# poc/10 — CPU SGEMM cache-blocking ladder (PoCL)

**Question:** poc/09's barrier-free register kernel left CPU matmul ~8x off
XLA/Eigen. How much does classic cache blocking recover through OpenCL?

**Ladder** (each verified against a scalar reference on non-square,
non-multiple shapes; pack cost included in all timings):

| variant | N=1024 | N=2048 |
|---|---|---|
| v0 poc/09 b2 (4×16 regs, stride-N B reads) | 37.1 GFLOP/s | 33.0 |
| v1 + **packed B** (16-col panels, sequential k-loop) | 52.3 | 52.6 (**1.6x**) |
| v2 + **6×16 register block** (12 f8 accs + 2 B + 1 A = 15/16 ymm) | 49.5 | 76.3 (**1.45x**) |
| v3 + **KC=512 blocking** (C accumulated across sweeps) | **62.0** | **96.6 (1.27x)** |

(In-run ratios are the datum; absolute numbers drift with laptop thermals.)

Findings:
- The stride-N·4B B read was the dominant cost (v1's 1.6x from a pure layout
  change; the pack pre-pass is O(K·N) and parallel — noise next to O(M·N·K)).
- Wider register block only pays at 2048 (at 1024, B panels are L2-resident
  either way); KC blocking pays at both.
- Ceiling check: ~97 GFLOP/s ≈ 11% of the 288V's paper FMA peak vs Eigen's
  ~70%. The rest needs packed A, alignment + prefetch tuning, per-core-type
  micro-tiles (LNL P vs E cores), deeper unrolling — steep effort for a debug
  backend; **stop here** (decisions.md #11).

**Decision:** integrate v3 as the default CPU matmul (`mm2p` + `mm2_pack`
pre-pass, KC sweeps enqueued back-to-back on the in-order queue); keep the
poc/09 register kernel selectable via `PJRT_OCL_MM_CPU=reg` for hardware that
prefers it (and as the fallback when N % 16 != 0).
