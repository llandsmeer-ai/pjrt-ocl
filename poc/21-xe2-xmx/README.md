# poc/21 — Can we reach Xe2's XMX (DPAS) matrix engine from OpenCL?

**Question.** poc/19 tuned the portable f32 SGEMM on Intel Arc 140V (Xe2) to
**1866 GFLOP/s** at 2048³ (128×128 bk8), with the shipped 128×64 bk8 at ~1522.
That is ~44% of the part's f32 vector peak — close to the portable ceiling.
Xe2 also has Intel's **XMX** matrix engine, but it does bf16/f16/int8/tf32,
**not f32**. So an f32 matmul could only use it by rounding its operands —
exactly the trade the NVIDIA path already makes for TF32 (§35–41, §10c).
Is XMX reachable from plain OpenCL C at all, how fast is it, and how much
precision does it actually cost?

**Method.** Three standalone programs, plain C + `-lOpenCL`, no plugin, no
python, in the style of poc/19 and poc/17:

- `probe21.c` / `probe21.cl` — dump the device extension string, the supported
  sub-group sizes, and compile-test 11 candidate `matrix_mad` signatures at
  sub-group size 8 and 16, reporting the driver's own build log for each.
- `bench21.c` + `xmx21.cl` (bf16/f16) + `xmx21tf32.cl` (tf32) — a real
  `C = A@B`, f32 in / f32 out, sweeping 12 sub-group tile geometries. Times the
  GEMM (min of 12 after a 300 ms warmup) and the f32→tile packing pre-pass
  separately, and checks the result against **two** CPU references: an exact f64
  accumulation of the original operands, and an f64 accumulation of the operands
  after rounding to the engine's input format. The gap to the *rounded*
  reference is the correctness signal (a layout or accumulate bug lands here);
  the gap to the *exact* one is the precision cost. It also brute-force fits the
  output against "round both operands to `mb` mantissa bits, RTE or RTZ" to
  recover the engine's actual operand format from the outside.
- `tf32bits.c` — a single 8×16×8 tile with `B = I`, dumping the f32 bit patterns
  the tf32 engine returns for probe values, to settle its rounding mode.

```bash
cc -O2 -o probe21 probe21.c -lOpenCL -lm && ./probe21
cc -O2 -o bench21 bench21.c -lOpenCL -lm
./bench21 2048                  # [N] [platform-substr] — sweep, 256 sampled cells
./bench21 512 Intel full        # ALL N*N cells vs a full O(N^3) f64 reference
./bench21 2048 Intel contpos    # continuous, all-positive operands (precision study)
cc -O2 -o tf32bits tf32bits.c -lOpenCL -lm && ./tf32bits
```

## Is it reachable? Yes.

`CL_DEVICE_EXTENSIONS` on Intel Arc 140V, `intel-opencl-icd` **26.22.38646.6**,
`OpenCL 3.0 NEO` — the XMX-relevant subset (verbatim, all present):

```
cl_intel_subgroup_matrix_multiply_accumulate
cl_intel_subgroup_matrix_multiply_accumulate_tf32
cl_intel_subgroups   cl_intel_subgroups_short   cl_intel_subgroups_char
cl_intel_required_subgroup_size   cl_intel_bfloat16_conversions   cl_khr_fp16
cl_intel_subgroup_2d_block_io   cl_intel_subgroup_local_block_io
cl_intel_subgroup_buffer_prefetch   cl_khr_integer_dot_product
```

`CL_DEVICE_SUB_GROUP_SIZES_INTEL` = **16, 32**. Requesting 8 — the size most of
Intel's published `matrix_mad` example code assumes — is rejected outright:

```
sg8 V_BF16_M8  FAIL(-11): error: in kernel 'probe': Kernel compiled with
                          required subgroup size 8, which is unsupported on
                          this platform  error: backend compiler failed build.
```

At `intel_reqd_sub_group_size(16)` the working shapes are **M=8, N=16**:

| builtin | a | b | acc | K |
|---|---|---|---|---|
| `intel_sub_group_bf16_bf16_matrix_mad_k16` | `short8` | `int8` | `float8` | 16 |
| `intel_sub_group_f16_f16_matrix_mad_k16` | `short8` | `int8` | `float8` | 16 |
| `intel_sub_group_tf32_tf32_matrix_mad_k8` | `float4` | `float8` | `float8` | 8 |
| `intel_sub_group_i8_i8_matrix_mad_k32` | `int8` | `int8` | `int8` | 32 |

`half8` for the f16 A operand and `float2` for the tf32 A operand do **not**
resolve (`no matching function for call to …`); A is passed as raw 16-bit lanes.
The `int8` A form also compiles at sg16 but is the sub-group-size-8 overload and
is semantically wrong there — the sg16 A operand is `short8`.

Layouts (deduced, then **verified** numerically, see below): for lane `l` and
vector element `i`, A is `A[i][l]` (a plain row-major 8×16 tile), B is
`{B[2i][l], B[2i+1][l]}` (VNNI row-pair pack), C is `C[i][l]`. Each is therefore
exactly one `intel_sub_group_block_read_us8` / `block_read8`, and the f32→16-bit
conversion pass writes straight into that tile order. tf32 needs no VNNI: its A
tile is row-major 8×8 read with `block_read4`, B is row-major 8×16.

## Result — GFLOP/s, Intel Arc 140V, 64 CUs, 128 KB SLM

Best geometry per (type, N); f32 column is poc/19's `bench19` re-run in the same
session on the same machine (best of its sweep).

| N | f32 (poc/19) | **bf16** | **f16** | **tf32** |
|---|---|---|---|---|
| 512 | 1148 | **6885** (6.0×) | **6933** (6.0×) | 4282 (3.7×) |
| 1024 | 1639 | **12805** (7.8×) | 12643 (7.7×) | 7969 (4.9×) |
| 2048 | 1855 | **16550** (8.9×) | 16509 (8.9×) | 8367 (4.5×) |

That is ~49% of Arc 140V's ~33.6 TFLOP/s bf16 XMX peak (8 Xe cores × 2048
bf16 FLOP/clk × ~2.05 GHz) — a plausible, not-too-good-to-be-true fraction.
tf32 is exactly half of bf16/f16, as its K=8-vs-16 shape predicts.

**But f32 operands have to be packed first**, and that is an extra O(N²) round
trip the f32 kernel does not pay. With the (unoptimised) packing pass included:

| N | pack ms (A+B) | bf16 end-to-end | f16 | tf32 |
|---|---|---|---|---|
| 512 | 0.10 | 1890 (1.6×) | 1931 (1.7×) | 1851 (1.6×) |
| 1024 | 0.34 | 4202 (2.6×) | 4202 (2.6×) | 3258 (2.0×) |
| 2048 | 0.57 | 10750 (5.8×) | 10592 (5.7×) | 6164 (3.3×) |

The packer is deliberately naive (one work-item per element, scattered writes):
at 2048 it moves 48 MB in 0.56 ms = 86 GB/s against poc/20's measured **109 GB/s**
achievable ceiling, so a tuned packer buys at most ~20%. The real fix is to fold
the conversion into the GEMM's load path (SLM-staged tiles), which removes the
pass entirely and puts the kernel-only column back in reach. **Not done here.**

Best tile geometry is `SGM×SGN = 2×4` or `4×4` with `RM×RN = 4×2` — i.e. a
64×128 or 128×128 workgroup tile from 128–256 work-items, 8 `float8`
accumulators per lane. `RN=4` (16 accumulators) collapses to ~40% everywhere
(register pressure), and `RM=RN=1` never exceeds 30% of best (no reuse).

## Precision — measured, not assumed

The operand format was recovered from the outside by fitting the GEMM output:

```
operand-format fit (bf16):  7 explicit mantissa bits, round-to-nearest-even (residual 1.50e-05)
operand-format fit (f16):  10 explicit mantissa bits, round-to-nearest-even (residual 7.69e-06)
operand-format fit (tf32): 10 explicit mantissa bits, truncated (RTZ)       (residual 1.67e-05)
```

`tf32bits.c` confirms the RTZ finding directly on single values — the low 13
mantissa bits are always cleared, and never rounded up:

```
input        bits       engine out    bits
1.99999988   3fffffff   1.99902344    3fffe000     (RTE would give 2.0)
1.0007323    3f8017ff   1             3f800000     (RTE would give 3f802000)
1.00076282   3f8018ff   1             3f800000     (RTE would give 3f802000)
3.14159274   40490fdb   3.140625      40490000
```

**Intel's tf32 truncates; NVIDIA's TF32 rounds to nearest.** That is not a
detail. Error at N=2048 with continuous operands, as a fraction of max|C|
(`bias` = mean *signed* error / mean |C|, i.e. systematic drift):

| operands | metric | f32 (poc/19) | bf16 | f16 | tf32 |
|---|---|---|---|---|---|
| sign-symmetric | max rel err | ~7e-7 | 2.5e-3 | **3.5e-4** | 8.1e-4 |
| sign-symmetric | bias | — | −3.9e-4 | −6.4e-7 | −3.7e-5 |
| all-positive | max rel err | ~7e-7 | 1.8e-4 | **2.1e-5** | 7.1e-4 |
| all-positive | bias | — | −2.2e-6 | −3.4e-7 | **−7.1e-4** |

Read the last row carefully. On all-positive data (post-ReLU activations ×
positive weights — a very common shape) tf32's error is **entirely systematic
drift**: max error == bias == −7.1e-4, matching the predicted truncation bias
2 × ½ × 2⁻¹¹ = 4.9e-4. It does not average out with K; it grows with it. bf16,
despite having three *fewer* mantissa bits, is 4× more accurate there because it
rounds to nearest. **On this device tf32 is not the "safe" option it is on
NVIDIA.**

f16 is the most accurate of the three in both regimes *and* ties bf16 for speed.
Its cost is dynamic range: a 5-bit exponent overflows above 65504 and loses
normals below 6.1e-5, where bf16 and tf32 keep f32's full 8-bit exponent.

## Findings

1. **XMX is reachable from plain OpenCL C**, no SPIR-V, no inline asm, no
   IGC-internal builtins — just `cl_intel_subgroup_matrix_multiply_accumulate`
   plus `intel_reqd_sub_group_size(16)`. This is markedly easier than the NVIDIA
   tensor-core path, which needed inline PTX (§35–41).
2. **8.9× over the tuned portable f32 kernel** at 2048 (16.5 vs 1.86 TFLOP/s),
   5.8× once f32→bf16 packing is counted. This is by far the largest remaining
   Xe2 matmul lever — bigger than everything poc/19 found put together.
3. **Sub-group size 16 is mandatory.** Every sg8 build fails with an explicit
   driver error. Any published Xe/DPAS sample written for sg8 must be re-derived,
   including the A-operand type (`short8`, not `int8`).
4. **Intel's tf32 truncates its operands.** Measured two independent ways. It is
   therefore *not* a drop-in equivalent of the NVIDIA TF32 path this project
   already ships by default, and on all-positive data it is less accurate than
   bf16 at half the speed. tf32's only real advantage over f16 is exponent range.
5. **f16 dominates bf16 and tf32 on this device** — same speed as bf16, 7×
   lower error than bf16 and 30× lower than tf32 on all-positive data — provided
   the operands fit in a 5-bit exponent. Nothing in the plugin currently checks
   that, so it cannot just be assumed.
6. **The packing pass is the integration problem, not the GEMM.** It costs 35%
   of the 2048 runtime and *more* than the GEMM itself below 1024, which is why
   the end-to-end win falls to 1.6× at N=512 — precisely the size range where
   most of our workloads' matmuls live.

## Should we ship this?

**Not as it stands, and never in the core path.** Concretely:

- **Core path: no.** `cl_intel_subgroup_matrix_multiply_accumulate` is a vendor
  extension, which CLAUDE.md's portability discipline excludes from the core VM
  and kernel library. It belongs behind the kernel-table override, exactly like
  `VMO_NV_PTX` — a second build of the matmul kernel, feature-detected at init
  by exact-token matching on `CL_DEVICE_EXTENSIONS` (`probe21.c` has the matcher;
  a naive `strstr` matches `..._accumulate` inside `..._accumulate_tf32`).
- **Default-on: no — unlike NVIDIA TF32.** The §10c precedent (TF32 on by
  default, `PJRT_OCL_MEGA_TC=0` to disable) rested on TF32 being "clean, SAFE,
  always-correct" and on JAX-CUDA doing the same thing, so we matched the
  reference backend's numerics. Neither holds here: no Intel backend of JAX
  exists to match, bf16's 2.5e-3 relative error would fail the CPU-comparison
  tolerances the test suite is built on, and Intel's tf32 adds a *systematic*
  −7e-4 drift on non-negative data. Silently making every `dot_general` on Intel
  4–9× faster and 1000× less accurate is not a trade a user can discover.
- **What I would ship: an opt-in `PJRT_OCL_MM_XMX=bf16|f16|tf32` override,
  off by default**, routing only the standalone `mm2` host-dispatch matmul.
  The mechanism is proven and the payoff is large enough (5.8× end-to-end at
  2048) to be worth the flag. If exactly one is picked, pick **f16**: it is the
  fastest *and* the most accurate here, and its exponent-range risk is checkable
  (a max-abs reduction over the operands is O(N²), cheap next to the pack pass
  we already need).
- **Prerequisite before that flag is worth anything:** fold the f32→16-bit
  conversion into the GEMM's SLM staging. Without it the win at N≤1024 is only
  1.6–2.6×, which is not obviously worth a precision cliff; with it the
  kernel-only 8.9× is the honest number.

## How this was verified

- **Full all-cell check**: `./bench21 512 Intel full` compares **all 262,144**
  output cells of all three variants against a full O(N³) f64 CPU reference.
  Max residual vs the exact operand-format model: bf16 6.44e-6, f16 6.06e-6,
  tf32 7.82e-6 — i.e. the GEMM reproduces "round operands, accumulate in f32"
  to f32 round-off. This is what rules out a wrong tile layout: a permuted
  layout would give O(|C|) errors, not 6e-6.
- **Sampled check** on 256 random cells at N=512/1024/2048, every geometry,
  every variant — all `ok`, and identical across geometries (so the sweep is not
  measuring 12 different bugs).
- **Format recovered independently of the spec** by brute-force fitting 36
  rounding models (6–23 mantissa bits × RTE/RTZ) to the output; the fit picks the
  right one with a 10³–10⁴ margin over its neighbours.
- **tf32 rounding mode confirmed a second way**, on single values with `B = I`
  (`tf32bits.c`), including the A-side and B-side separately (identical).
- **Baseline measured in the same session on the same machine**, not quoted:
  `poc/19/bench19` gave 1855 GFLOP/s at 2048, 1639 at 1024, 1148 at 512,
  matching poc/19's recorded 1866/1654/1148.
- **Timing**: min of 12 runs after a 300 ms warmup loop. This matters — with
  poc/19's 3-iteration warmup the same config varied 2× run-to-run on this iGPU
  (4962 vs 10074 GFLOP/s). With the long warmup, three back-to-back full runs at
  N=2048 gave bf16 16497/16146/16391 (2.1% spread), f16 16461/16069/16410 (2.4%),
  tf32 8307/8069/7967 (4.1%). The **pack** timings are noisier (0.56–0.75 ms for
  the same work, ~30%) since that pass is short and bandwidth-bound; treat the
  end-to-end column as ±10%, not ±3%.

**Not verified / not done:** non-square and small/batched shapes (only square
512/1024/2048); the int8 path (compiles, never run); `cl_intel_subgroup_2d_block_io`
(present, unused — it would remove the packing pass for B); the fused
convert-in-GEMM variant that the recommendation above depends on; anything at
all inside the plugin.
