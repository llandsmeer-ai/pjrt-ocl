# JAX coverage progress (lax_test.py)

## After dtype work (2026-07-15): 297 → **405 passing** (2162 → 2054 failing)

The i32/u32/i64/bool/f64/convert implementation cleared EVERY 4-byte + boolean +
i8/i16 dtype rejection from the baseline (i32 528, bool 286, i16 212, i8 178,
u32 102 — all gone). Remaining dtype rejections are only the ones needing the
2-byte / emulated path: **bf16 920, f16 786, complex 358, f8 10**. Next-op
frontier surfaced by dtype support: `or`/`and`/`xor` (138, bitwise/logical now
that int/bool exist), composite (118), reduce_window (102), convolution (94),
log_plus_one (54), bitcast_convert (52), sort (46).

Priority now: (1) f16/bf16 (2-byte; byte-addressed arena is ready — need
cl_khr_fp16 gating + bf16 u16 emulation) = the single biggest bucket; (2) the
easy new ops (or/and/xor, log1p, bitcast_convert); (3) reduce_window/conv/sort.

---

# JAX coverage baseline (lax_test.py, 2026-07-15, pre-dtype)

First run of JAX's own `tests/lax_test.py` against our OpenCL backend
(`tools/jax_coverage.sh`), single-device, jax 0.10.2.

**Result: 297 passed, 2162 failed, 74 skipped** (9m19s). The 297 passing with zero
op/dtype work beyond f32 elementwise+shape+reduce+dot is a useful floor.

## Failures are dominated by DTYPES, not missing ops

Dtype rejections (element type unsupported), ranked:

| dtype | failures | tier |
|---|---|---|
| bf16 | 892 | 3 (emulate: store u16, up-convert to f32) |
| f16 | 766 | 2 (cl_khr_fp16, device-conditional) |
| i32 | 528 | **1 (4-byte reinterpret)** |
| complex<f32> | 358 | 3 (real/imag f32 pairs) |
| i1 / bool | 286 | **1** |
| i16 | 212 | 2/3 (byte-addressed) |
| i8 | 178 | 2/3 |
| ui32 | 102 | **1** |
| ui16, ui8 | 72+ | 2/3 |

→ **Tier 1 (i32 + bool + u32) clears ~900 failures** with the 4-byte-slot
reinterpret approach (no byte-addressing needed). **bf16 + f16 (1658)** are the
biggest prize but need the byte-addressed arena + fp16 extension / bf16 emulation.

## Missing ops (f32 path), ranked

| op | count | notes |
|---|---|---|
| reduce_window | 102 | pooling; windowed reduction — new tile op |
| composite | 96 | wraps a decomposed subgraph; may inline |
| convolution | 86 | conv — big, own kernel eventually |
| complex | 62 | complex construction (Tier 3) |
| log_plus_one / expm1 | 54+12 | EW unary (log1p/expm1) — easy |
| atan2 | 26 | EW binary — easy |
| sort | 20 | new op |
| round_nearest_afz / _even | 24 | EW unary rounding — easy |
| concatenate | 16 | multi-input gather (planned) |
| bitcast_convert | 14 | reinterpret bits — trivial once dtypes exist |
| is_finite | 14 | EW unary → bool |
| cbrt, tan, sine, cosine | ~44 | EW unary transcendentals — easy (need OpenCL fns) |
| clamp | 12 | EW ternary — easy |
| pad | 10 | fill + gather |
| reduce_precision | 10 | dtype rounding |

## Priority (data-driven)

1. **Dtype Tier 1** (i32/bool/u32): ~900 failures, tractable (4-byte reinterpret).
   Unblocks integer indexing, boolean masks, comparisons producing real bools.
   Vertical slice: i32 elementwise end-to-end, then broaden.
2. **Easy EW ops** (log1p, expm1, atan2, rounding, cbrt, tan/sin/cos, clamp,
   is_finite): dozens of failures, each a few lines in ops/elementwise.
3. **concatenate, pad, bitcast_convert**: gather/fill-family.
4. **Dtype Tier 2/3** (f16, bf16, complex): the big ML/scientific prize; needs
   the byte-addressed-arena refactor + extensions/emulation.
5. **reduce_window, convolution, sort**: larger new ops.

Re-run `tools/jax_coverage.sh` after each batch; the pass count is the scoreboard.
