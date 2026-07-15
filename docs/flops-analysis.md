# What's limiting GPU FLOPs? (deep dive, 2026-07-15)

NVIDIA RTX PRO 6000 Blackwell. clpeak FP32 (FMA-counted) = **105.9 TFLOPS** — this
is the SIMT fp32 ceiling and the number every % below is against.

## Measured (all real, this session)

VM matmul through the plugin (376 lanes, device-resident buffers, best-of-5):

| M=N=K | time | TFLOP/s | % of 106 |
|---|---|---|---|
| 512 | 0.11 ms | 2.5 | 2% |
| 1024 | 0.29 ms | 7.4 | 7% |
| 2048 | 1.13 ms | 15.3 | 14% |
| 4096 | 7.69 ms | **17.9** | 17% |

Standalone SGEMM kernel (poc/06, no VM), 2048 / 4096:

| tile config | 2048 | 4096 |
|---|---|---|
| naive 16×16 (old) | 4.3 | 4.3 |
| **64×64 4×4 BK16 (what the VM uses)** | 17.3 | **19.0** |
| 128×128 8×8 BK8 | 18.3 | 24.6 |
| 128×128 BK16 (portable champion) | 19.4 | **26.1** |
| 128×128 BK32 float4 (NV-only) | 19.6 | 26.8 |

## The key finding: the bytecode / VM is NOT the bottleneck

At 4096³ the VM hits **17.9 TFLOPS vs the dedicated 64×64 kernel's 19.0** — a ~6% gap
(arena copies + 43 µs dispatch floor + megakernel switch). **The VM reaches the kernel
ceiling at scale.** Changing the bytecode format cannot raise this — the bytecode already
lowers a matmul to exactly the tile-op the standalone kernel runs. The earlier "2 TFLOPS
at 4096" figure was a bad measurement (cold clocks / no warmup) and is retracted.

So the question "what limits FLOPs" is really "what limits the SGEMM *kernel*", plus one
architectural tax and a small-size regime. Four gaps, largest first:

### 1. SGEMM kernel efficiency ceiling — 26 TFLOPS = 25% of SIMT peak (the big one)

Even the best standalone kernel gets 24.7% of the 106 TFLOPS SIMT peak. A well-tuned
portable OpenCL SGEMM (CLBlast class) reaches 40–70%, so there is ~2× left in *kernel*
work, none of it bytecode-related:
- **Occupancy-bound**: 128×128/8×8 uses ~80 registers/thread ⇒ ~2 workgroups/SM. More
  occupancy needs a smaller per-thread register footprint (narrower microtile) — which
  trades against arithmetic intensity. This GPU's sweet spot is 128×128 (poc/06 sweep).
- **No float4 register accumulators / vectorized inner loop** yet (float4 *staging* was
  net-neutral once loads coalesced; float4 *accumulate* is the untried lever).
- **No subgroup/warp cooperation** (`cl_khr_subgroups` broadcast to cut LDS traffic).
- **OpenCL compiler ≠ ptxas**: no SASS scheduling / register-tuning control on NVIDIA.
  This alone caps us below cuBLAS's ~85–90% of SIMT peak.

### 2. Architectural tax — the shared megakernel forces the smaller tile (~1.4×)

The VM uses **64×64 (19 TFLOPS)** not the **128×128 champion (26 TFLOPS)** — a 37% loss —
because vm2.cl is ONE megakernel for all ops. Its occupancy is set by the *fattest* op:
128×128's 64 accumulators (~80 regs) + 16 KB local would crush occupancy for *every* op
(elementwise, gather, …) and already broke PoCL co-residency at 8 KB. Measured: adding the
MMA case took the kernel from 377→545 f32 vregs. This is **ceiling-1** and it IS
architectural — the fix is **typed lanes / a dedicated GEMM kernel** (poc/05 validated that
separate concurrent kernels co-reside and sync). The bytecode already supports this (it is
engine-agnostic); no format change needed, just a second kernel the scheduler routes DOT
tiles to. Worth ~1.4× on matmul with zero cost to other ops.

### 3. Fundamental hardware/API ceiling — tensor cores are unreachable from OpenCL

106 TFLOPS is the **SIMT fp32** peak. cuBLAS gets its 5–10× more from **tensor cores**
(tf32/bf16/fp8), which **OpenCL cannot address on NVIDIA** — no extension exposes them.
So a CUDA/tensor-core backend will always beat us by ~5–10× on matmul-heavy bf16 ML,
*by construction*, no matter how good our kernel is. Escape hatches are per-vendor:
Intel exposes `cl_intel_subgroup_matrix_multiply_accumulate`; a future path is Vulkan
cooperative-matrix interop. On AMD, MFMA is also OpenCL-invisible. This is the hard wall.

### 4. Small-matmul regime — dispatch floor + occupancy ramp (2.5–7.4 TFLOPS ≤1024)

The 43 µs per-execute floor and the persistent-lane occupancy ramp mean 512–1024 matmuls
run at 2–7 TFLOPS. Real ML has many smallish/skinny matmuls, so this matters. Levers
(scheduler-level, not bytecode-format):
- **Stream-K tiling**: when M×N tiles < SM count, split K across SMs for occupancy.
- **Right-sized dedicated launch** for lone matmuls (the streamed-launch engine) instead of
  376 persistent lanes.

## Answers to the specific questions

- **Is it architectural?** Partly — exactly one architectural lever (§2, the megakernel
  tile-size tax, ~1.4×). The rest is kernel tuning (§1, ~2×) and a hard hardware/API wall
  (§3, the 5–10× tensor-core gap).
- **Should we change the bytecode to facilitate the bottlenecks?** **No.** The bytecode
  reaches the kernel ceiling at scale and is engine-agnostic by design. The one bytecode/
  scheduler-adjacent win is **epilogue fusion** (matmul→bias→activation kept in registers via
  the reserved FUSED_TILE / slot-file ABI) — it doesn't raise peak GEMM FLOPs but removes the
  write-C-then-reread traffic on real transformer layers, so it raises *effective* throughput.
  And **stream-K scheduling** helps §4. Neither needs a format change — the slot fields are
  already reserved in the entry encoding.
- **The bytecode itself?** Not the problem.

## Priority order to raise FLOPs

1. **Typed lanes / dedicated GEMM kernel** (§2): unlock 128×128, ~1.4×, zero cost elsewhere.
   poc/05 already proved the mechanism. Highest value-per-effort.
2. **Kernel tuning inside that GEMM** (§1): float4 accumulators, subgroups, `-cl-mad-enable`
   → toward CLBlast-class 40–70% (~2×). Bounded by the OpenCL-not-ptxas ceiling.
3. **Stream-K + right-sized launch** (§4): fixes the small/skinny-matmul regime.
4. **Epilogue fusion** (bytecode-adjacent): effective throughput on real layers.
5. **Accept §3** on NVIDIA; pursue Intel subgroup-matrix / Vulkan coop-matrix per vendor.
