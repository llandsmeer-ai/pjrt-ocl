# poc/15 — register-budget probe for in-megakernel map-region fusion (§27)

**Question (from §24 finding-2):** can a register-resident fused map-region op live
INSIDE the one megakernel without regressing co-residency below the §10c spin-barrier
floor (2×CU = 376 lanes on the 188-SM RTX PRO 6000 Blackwell)? §24 assumed NO ("needs
R3 / split the VM"). This PoC measures it.

## Instrument

`regprobe.c` builds the REAL `vm2` megakernel (`kVmClSource` from the dev build's
`vm_cl_source.h`) with `-cl-nv-verbose`, so the NVIDIA driver prints the exact
per-kernel **register count / smem / stack / spill** from ptxas. This is deterministic,
unlike the occupancy spin-probe (`vmo_discover`), which is bimodal 376/752 due to
backfill over-counting.

Two knobs, both build-flags on the real megakernel (never emitted by lowering):

- `-DVMO_PROBE_REGS=N` — a switch case (op 99) holding N per-thread float accumulators
  SIMULTANEOUSLY live, scoped in-case. Sweeps register pressure to find the occupancy
  cliff and test max-vs-sum across mutually-exclusive switch cases.
- `-DVMO_REGION_POC` — the real `TOP_MAP_REGION` interpreted region case
  (`kernels/ops/region.cl`): per-thread float4 slots, straight-line micro-op sub-list,
  one global load per input + one store, no cross-workgroup barrier.

Real residency + GELU correctness go through `runtime_test` with
`PJRT_OCL_EXTRA_BUILD=-DVMO_REGION_POC` (injects the flag into the runtime's kernel
build) and `REGION_POC=1` (runs test C).

## Results (RTX PRO 6000 Blackwell, 188 SM, 65536 regs/SM, sm_120)

| build                      | vm2 regs | smem  | stack | WG/SM | co-resident (spin-probe) |
|----------------------------|----------|-------|-------|-------|--------------------------|
| baseline portable          | 92       | 8196  | 240   | 2     | 376 / 752 (bimodal)      |
| baseline TF32 (`VMO_NV_PTX`)| 94      | 10244 | 240   | 2     | 376 / 752 (bimodal)      |
| **region PoC portable**    | **88**   | 8196  | 320   | 2     | **376 / 752 (bimodal)**  |
| **region PoC TF32**        | **88**   | 10244 | 320   | 2     | **376 / 752 (bimodal)**  |
| +64 switch-regs (op 99)    | 95       | —     | —     | 2     | (95 < 128 ⇒ 2 WG/SM)     |
| +80 switch-regs            | 182      | —     | —     | 1     | 188                      |
| +96 switch-regs            | 197      | —     | —     | 1     | **188 (always)**         |

Register→occupancy cliff (validated both ends): **≤128 regs → 2 WG/SM = 376;
>128 regs → 1 WG/SM = 188.** The spin-probe is bimodal 376/752 for every in-budget build and never
188; only the over-budget 197-reg build reads 188 (always) — the clean discriminator. Baseline sits
at 92–94 → **34–36 registers of headroom**
before the cliff. Switch-addressed registers are **max-not-sum**: N = 8…64 in-case float
accumulators all stayed ~95–96 regs (they reuse the matmul case's physical registers),
so up to ~64 float (16 float4 slots) are free; the jump is between N=64 (95r) and
N=80 (182r).

**Reconciles 752 vs 376:** poc/13's ~8–24-reg microkernels hit the 4-WG/SM hardware
ceiling (752); the 92–94-reg megakernel is REGISTER-limited to 2 WG/SM (376). The gap is
entirely the megakernel's register pressure.

## Verdict: GO

A properly-structured region op costs **+0 registers to the whole-kernel max (88 vs 92),
+0 SLM, occupancy stays 376** — the §10c floor — and computes GELU f32-exact
(maxerr 5.96e-08). §24 finding-2 was too pessimistic: its "~32 vector registers" and
"32 KB SLM" costs come from (a) treating disjoint switch-case registers as additive, and
(b) declaring FRESH per-workgroup `__local` scratch. Per-thread slots (registers or
local-mem, not SLM) avoid both. See docs/decisions.md §27.

## Run

```
. ./env.sh
cmake -S pjrt_plugin -B pjrt_plugin/build-dev -G Ninja -DPJRT_OCL_BUILD_TESTS=ON
cmake --build pjrt_plugin/build-dev
cd poc/15-region-budget && make
./regprobe                       # baseline vm2 regs
./regprobe -DVMO_REGION_POC      # region case regs
./regprobe -DVMO_PROBE_REGS=96   # over-budget → 197 regs
# residency + correctness:
REGION_POC=1 PJRT_OCL_INFO=1 PJRT_OCL_EXTRA_BUILD=-DVMO_REGION_POC \
  ../../pjrt_plugin/build-dev/runtime_test 2>&1 | grep -iE 'residency|region GELU'
```
