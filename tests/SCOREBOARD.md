# Op coverage scoreboard

Monotone-growing record of StableHLO ops our lowering + VLIW engine handle.
Each op is verified by `tests/test_ops_*.py` through BOTH validators (tensor
numpy interpreter + schedule simulator) against the JAX CPU backend, and the
core families are spot-checked on real NVIDIA + PoCL hardware via the plugin.

Updated 2026-07-15 (Phase 2 fan-out landed).

## Supported

| stablehlo op | tile op | family module | notes |
|---|---|---|---|
| add, multiply, subtract | EW | (core) | M2 baseline |
| constant | — | (core) | const pool |
| divide, maximum, minimum, power | EW | elementwise | |
| negate, exponential, log, sqrt, rsqrt, tanh, abs, floor, ceil, sign | EW | elementwise | unary |
| compare (EQ/NE/LT/LE/GT/GE) | EW(cmp) | elementwise | i1→f32 1.0/0.0 |
| select | EW(select) | elementwise | pred nonzero = true |
| broadcast_in_dim | GATHER | shape | stride-0 stretch |
| transpose | GATHER | shape | permuted strides |
| iota | IOTA_DIM | making | any rank/axis |
| convert | EW(copy) | making | f32→f32 only (jax elides same-dtype; int/bool rejected) |
| reduce (sum/max/min/prod) | REDUCE_PART+COMB | reduce | FULL reductions only (all axes → scalar) |
| dot_general | MMA | dot | plain 2D C=A@B (canonical [1]×[0], no batch) |
| while | (control) | (core) | on-device, runtime_test |

## Not yet supported (next, roughly in demand order)

- reshape, slice, concatenate, reverse, pad (reshape/slice/reverse are GATHER
  variants — extend ops/shape.py)
- partial-axis reduce (needs a permuting gather before REDUCE, or strided
  REDUCE tile op)
- batched / non-canonical dot_general
- sine, cosine, and other transcendentals without a vm2 subop
- if/case control flow (IF entry exists in vm2; needs lowering + scheduler seam)
- integer dtypes (int32/int64) — currently all-f32 policy

## How to add an op family

Copy `python/pjrt_ocl/ops/shape.py` as the template: register a stablehlo
handler (`@L.handles`), the tensor-opcode semantics (`opsem.register`), and the
schedule-simulator tile op (`opsem.register_tile_sim` / `register_ew_sim`). Add
one import line to `ops/__init__.py` and a `tests/test_ops_<family>.py` using
`oputil.check`. No core-file edits required.
