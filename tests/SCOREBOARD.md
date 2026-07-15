# Op coverage scoreboard

Monotone-growing record of StableHLO ops our lowering + VLIW engine handle.
Each op is verified by `tests/test_ops_*.py` through BOTH validators (tensor
numpy interpreter + schedule simulator) against the JAX CPU backend, and the
core families are spot-checked on real NVIDIA + PoCL hardware via the plugin.

Updated 2026-07-15 (Phase 2 fan-out + control flow landed: full dtype matrix,
elementwise math/logical, shape/dynamic-index ops, reduce_window, stablehlo.while).

## Dtypes (byte-addressed arena, per-task dtype dispatch)

Byte-addressed arena, per-task dtype dispatch (result dtype tile_op bits 8-15,
operand dtype bits 16-23). f16/bf16 = 2-byte storage + f32 compute (portable, no
cl_khr_fp16). bool = 1-byte. Shape ops propagate input dtype (element copy width).

| dtype | status |
|---|---|
| f32 | ✅ all ops |
| i32 / u32 | ✅ elementwise arith/compare/select/convert; shape ops |
| bool (i1) | ✅ 1-byte; compare→bool, select pred, bool I/O; convert |
| f64 | ✅ elementwise + convert, **gated behind cl_khr_fp64** |
| i64 | ✅ storage + gather + convert; elementwise arith |
| **f16 / bf16** | ✅ elementwise + compare/select/convert + broadcast/shape (2-byte, f32 compute) |
| i8 / i16 / complex / f8 | ❌ next (i8/i16 byte-addressed; complex = real/imag pairs) |

Reduce covers f32 and i32; shape/dynamic-index ops are dtype-agnostic (copy by
element width). iota emits f32/i32.

## Supported ops

52 stablehlo ops. Each row is verified by `tests/test_ops_*.py` through both
reference validators against the JAX CPU backend.

| stablehlo op | tile op | family module | notes |
|---|---|---|---|
| add, subtract, multiply, divide, remainder, power, maximum, minimum, atan2 | EW | elementwise | binary arith |
| negate, abs, sign, exponential, exponential_minus_one, log, log_plus_one, sqrt, rsqrt, cbrt, sine, cosine, tan, tanh, floor, ceil, round_nearest_even, round_nearest_afz, is_finite | EW | elementwise | unary |
| clamp | EW | elementwise | ternary min/max |
| and, or, xor, not | EW | elementwise | logical / bitwise |
| compare (EQ/NE/LT/LE/GT/GE) | EW(cmp) | elementwise | mixed operand dtype → bool |
| select | EW(select) | elementwise | 1-byte pred |
| convert | EW(convert) | elementwise | any dtype ↔ any dtype |
| bitcast_convert | EW(bitcast) | bitcast | same-width bit reinterpret (union recast) |
| constant | — | (core) | const pool (int/f16/bf16 via splat/iter) |
| broadcast_in_dim | GATHER | shape | stride-0 stretch |
| transpose | GATHER | shape | permuted strides |
| reshape | GATHER | shape | strided view where possible |
| slice | GATHER | shape | strided (start/limit/stride) |
| reverse | GATHER | shape | negative-stride gather |
| concatenate, pad | SCATTER | concat | strided-scatter into dst |
| dynamic_slice, dynamic_update_slice | DYN_GATHER / DYN_SCATTER | dynslice | runtime base offset |
| iota | IOTA_DIM | making | any rank/axis |
| reduce (sum/max/min/prod) | REDUCE_PART+COMB | reduce | FULL reductions (all axes → scalar), f32 + i32 |
| reduce_window (max/sum) | RED_WINDOW | reduce_window | pooling; strides/padding |
| dot_general | MMA | dot | plain 2D C=A@B (canonical [1]×[0], no batch) |
| while | (control) | (core) | on-device frame-stack VM; runtime_test B |

## Not yet supported (next, roughly in demand order)

- partial-axis reduce (needs a permuting gather before REDUCE, or strided
  REDUCE tile op) — currently rejected at lowering
- batched / non-canonical dot_general — currently rejected at lowering
- if/case control flow (IF entry exists in vm2; needs lowering + scheduler seam)
- general gather/scatter (data-dependent indices), sort, top-k
- i8 / i16 / complex / f8 dtypes

## How to add an op family

Copy `python/pjrt_ocl/ops/shape.py` as the template: register a stablehlo
handler (`@L.handles`), the tensor-opcode semantics (`opsem.register`), and the
schedule-simulator tile op (`opsem.register_tile_sim` / `register_ew_sim`). Add
one import line to `ops/__init__.py` and a `tests/test_ops_<family>.py` using
`oputil.check`. No core-file edits required.
