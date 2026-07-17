# OpenMegaKernel (pjrt-ocl)

**Run JAX on any OpenCL device.** `pjrt-ocl` is a [PJRT](https://openxla.org/xla/pjrt)
plugin that lets `jax.jit` execute on OpenCL-capable hardware — Intel, AMD, NVIDIA,
or a CPU via [PoCL](https://portablecl.org/) — with no vendor SDK (no CUDA, no ROCm)
on the execution path.

> ⚠️ **Experimental / work in progress.** A growing subset of StableHLO ops across the
> full JAX dtype matrix (f32/f64/i32/u32/i64/bool/f16/bf16), not yet on PyPI. Validated
> end-to-end on an NVIDIA RTX PRO 6000 (via NVIDIA's OpenCL) and on PoCL (CPU). Not
> affiliated with Google, OpenXLA, or the JAX project.

## How it works

JAX lowers your `jit`-compiled function to StableHLO. Instead of JIT-compiling a kernel
per dispatch, `pjrt-ocl` lowers StableHLO **once** into a compact **bytecode** — a flat
list of tile-ops with a per-lane schedule. At run time a single persistent OpenCL
megakernel (the "VLIW VM") interprets that bytecode: each workgroup is a *lane* running
its own instruction stream, with different lanes running different ops in parallel and
synchronizing at scheduler-placed barriers. The generic OpenCL kernel library is compiled
**once** per device at plugin init — the OpenCL compiler is never invoked on the hot path.

```
jax.jit(f)  ──►  StableHLO  ──►  lowering (Python)  ──►  VMProgram bytecode
                                                              │
                                          device-side bytecode VM (one megakernel)
                                                              ▼
                                                     results on the OpenCL device
```

## Requirements

- Python ≥ 3.10, with `jax` / `jaxlib` installed (developed against **jax 0.10.2**).
- An OpenCL 1.2+ runtime and an ICD for your device. Check with `clinfo -l`.
  - NVIDIA: the CUDA driver ships an OpenCL ICD.
  - CPU dev/testing: `sudo apt install pocl-opencl-icd`.
- To build the plugin: a C++20 compiler, `cmake`, `ninja`, and OpenCL headers
  (`sudo apt install opencl-headers ocl-icd-opencl-dev cmake ninja-build clinfo`).

## Installation

`pip install` builds the C++ plugin (via cmake/scikit-build-core) and bundles it with
the Python package — no manual build step. You need the build prerequisites on the
system first: **cmake, ninja, and OpenCL dev headers** (Ubuntu:
`sudo apt install cmake ninja-build opencl-headers ocl-icd-opencl-dev`), plus an ICD
for your device (see [Requirements](#requirements)).

```bash
pip install "git+https://github.com/llandsmeer-ai/pjrt-ocl.git"
```

That's it — the `JAX_PLATFORMS=opencl` example below works immediately.

<details>
<summary>Development install (editable, no rebuild-on-edit for the C++)</summary>

For hacking on the plugin, build the `.so` and use an editable Python install:

```bash
git clone https://github.com/llandsmeer-ai/pjrt-ocl.git && cd pjrt-ocl
cmake -S pjrt_plugin -B pjrt_plugin/build -G Ninja && cmake --build pjrt_plugin/build
pip install -e python/    # finds the .so in the build tree automatically
```

The loader searches `PJRT_OCL_PLUGIN_PATH` → the `.so` bundled in the package → the
dev build tree, and prints a clear error if none is found.
</details>

Check JAX sees the device:

```bash
JAX_PLATFORMS=opencl python -c "import jax; print(jax.devices())"
# [OclDevice(id=0)]   # .device_kind shows the platform/device name string
```

## Quickstart

```python
import os
os.environ["JAX_PLATFORMS"] = "opencl"      # use the OpenCL backend

import jax, jax.numpy as jnp

@jax.jit
def f(a, b):
    return jnp.maximum(a @ b, 0.0)          # matmul + relu, fused on device

x = jnp.ones((256, 128), jnp.float32)
w = jnp.ones((128, 64),  jnp.float32)
print(f(x, w).shape)                         # (256, 64)
```

### Choosing a device

`pjrt-ocl` picks the first GPU (else the first CPU) by default. Override with a platform
name substring and optional device index:

```bash
PJRT_OCL_DEVICE="NVIDIA"        python your_script.py   # NVIDIA CUDA OpenCL
PJRT_OCL_DEVICE="Portable"      python your_script.py   # PoCL (CPU)
PJRT_OCL_DEVICE="Intel:1"       python your_script.py   # 2nd Intel device
```

## Supported ops

52 StableHLO ops, grown test-first — each verified against JAX's CPU backend. Full
scoreboard: [`tests/SCOREBOARD.md`](tests/SCOREBOARD.md).

- **Elementwise** — add, subtract, multiply, divide, remainder, pow, max, min, atan2;
  negate, abs, sign, exp, expm1, log, log1p, sqrt, rsqrt, cbrt, sin, cos, tan, tanh,
  floor, ceil, round (even & away-from-zero), is_finite; clamp; and/or/xor/not;
  compare (all directions), select.
- **Type** — `convert` (any dtype ↔ any dtype), `bitcast_convert`.
- **Shape** — `broadcast_in_dim`, `transpose`, `reshape`, `slice` (strided), `reverse`,
  `concatenate`, `pad` (as strided gathers/scatters).
- **Dynamic indexing** — `dynamic_slice`, `dynamic_update_slice`.
- **Reductions** — `reduce` (full sum / max / min / prod, f32 & i32), `reduce_window`
  (pooling, with strides & padding).
- **Linear algebra** — `dot_general` (plain 2D matmul, register-blocked tile kernel).
- **Making** — `iota`, `constant`.
- **Control flow** — `while` (interpreted on device by a frame-stack VM).

**Dtypes** — f32, f64 (where `cl_khr_fp64` is present), i32, u32, i64, bool, f16, bf16.
f16/bf16 use 2-byte storage with f32 compute (portable, no `cl_khr_fp16` required).

Anything unsupported raises a clear `LoweringError` naming the op.

## Hardware tested & benchmarks

Correctness comes first; performance is early. Every device below runs the full
test suite (240 tests: op families x the dtype matrix + e2e); benchmarks are
per-op wall-clock vs problem size N so you can judge whether the library is
worth it for *your* sizes and hardware. New devices go through
[`docs/hardware-bringup.md`](docs/hardware-bringup.md).

Reproduce any plot (writes the `.png` + `.csv`; picks native CUDA as reference
when a CUDA jaxlib is installed, else JAX CPU):

```bash
. ./env.sh && python tools/plot_bench.py --device <platform substring>
# end-to-end transformer sweep (one figure: latency + throughput vs model size):
. ./env.sh && python tools/plot_transformer.py --device <platform substring>
```

**Methodology** (docs/decisions.md §21): each op is applied **16× inside one
jitted program as a data-dependent chain** (`optimization_barrier` between
links so XLA can't fuse/CSE the repeats — and so dead links can't be
eliminated); reported time is per op application, min-of-rounds with rounds
auto-sized to ≥50 ms. This amortizes per-call python/PJRT dispatch (~15–25 µs)
out of BOTH sides — run-to-run deviation is 0.1–0.8% — and both backends
execute one program containing 16 real op instances, which is also each
backend's realistic regime (16 VM instructions for us, 16 kernels in one
executable for XLA). Ratios below are therefore *device-work* ratios; earlier
revisions of these plots included the dispatch floor in both columns, which
flattered whichever side was slower per kernel. The NVIDIA per-op plot uses
this methodology; the Xe2/PoCL per-op plots predate it (one op per call) and
will be regenerated on the next bring-up pass for those devices.

### NVIDIA RTX PRO 6000 (Blackwell) — vs native CUDA

**Testbench: full suite green.** Per-op wall-clock, **our OpenCL backend
against JAX's native CUDA (XLA + cuBLAS) on the same GPU** — an
apples-to-apples GPU-vs-GPU comparison of the VM against a production compiler.

![ours (OpenCL) vs JAX CUDA, per-op N-vs-time](docs/bench_plot.png)

Takeaways (higher = slower; both axes log; ratios are per-op device work,
dispatch-free — see methodology above):

- **Elementwise (add / mul): ~1.5x at 4K, ~2.5x at 16M elements.** At 16M we
  sustain ~1.6 TB/s — real HBM bandwidth — so the large-N gap is CUDA's
  L2-friendlier scheduling, not kernel quality. In the mid range the gap is our
  **per-instruction cost (~15 µs: cross-workgroup barrier + instruction
  dispatch) vs CUDA's ~3.3 µs in-graph kernel launch** — that floor, not
  bandwidth, is what the VM pays per bytecode step, and it's the number the
  megakernel design has to beat. (Earlier zero-copy I/O + buffer-pool work
  still stands; the old "at parity / faster than CUDA" read came from ~20 µs
  of per-call dispatch masking CUDA's floor.)
- **`gather` (`dynamic_slice`)** ~4.4x at 16M (ours ~1.3 TB/s); flat
  ~30 µs/op mid-range = the same per-instruction floor (the chain's offset
  arithmetic adds a few scalar instructions per link on both sides).
- **`matrix × vector`** ~46x at 2048 — the honest number for our GEMV kernel:
  it streams the 16.8 MB matrix at ~90 GB/s while cuBLAS keeps it L2-resident
  (~4 µs/op ≈ 4 TB/s effective on Blackwell's 128 MB L2). Biggest per-op gap
  we have; a proper GEMV tile is the known fix.
- **`dot_general`** ~5.9x at 2048³. Large matmul runs a **standalone SGEMM**
  (`mm2`, launched outside the megakernel so an 8×4 register tile doesn't
  inflate the shared register budget). cuBLAS hits **134 TFLOP/s at 4096,
  above Blackwell's f32 peak** — i.e. it is TF32 **tensor-core** bound, which
  a portable f32 kernel can't reach. So there's an **NVIDIA-only TF32
  tensor-core path** (inline-PTX `wmma` behind the `VMO_NV_PTX` build,
  portable f32 fallback intact; note tf32 means ~1e-3 relative precision on
  NVIDIA matmul, same trade cuBLAS makes by default). It stays short of cuBLAS
  because the in-kernel tile is **arithmetic-intensity-capped**: it can't grow
  without dropping below the co-residency the persistent megakernel's
  cross-workgroup barrier requires. See the end-to-end transformer numbers
  below for where this lands on a real workload.
- **`while` loops: ~80–100x FASTER than CUDA below ~256K elements, ~1.9x
  slower at 16M.** The counted-loop pipeline (`OP_FOR` + bytecode unroll, §15) plus
  affine-chain composition folds the 32-step `x*1.5+1` body into a handful of
  instructions at compile time (~2 µs/op), while XLA GPU runs a real
  device-synchronized loop (~175 µs/op regardless of N). Past the unroll
  arena gate (~512K) we run the loop for real too and land at 1.9x — the
  earlier campaign's fixes (affine op folding, in-place carries,
  contention-free barrier spin) are what hold that at 2x rather than the 28x
  it started from.
- **`lax.scan` (stacked outputs) ties XLA CPU at 1M elements.** The
  dynamic_update_slice that stacks each step's output used to re-copy the
  whole ys buffer every iteration (O(T²·n) traffic); the in-place-DUS fold
  (§15a) scatters the row straight into the loop carry instead. Scan-RNN
  1M×T8: 1.11 ms vs XLA CPU's 1.09; 1M×T32: 4.84 vs 4.84 (FOR mode; 2× over
  the copying path on NVIDIA, up to 15× on PoCL host-dispatch).

On the ops closest to parity, the residual mid-size gap is the VM's fixed
per-instruction cost (barrier + dispatch, ~15 µs vs ~3 µs); fusing more work
into fewer instructions (§19) attacks exactly that.

#### End-to-end: a GPT-style transformer (`tools/bench_transformer.py`)

A realistic forward pass — batched multi-head attention, layernorm, GELU FFN,
residuals (random weights) — run through the full plugin and checked against a
JAX-CPU reference (`--check`, f32-exact on the portable path). This is the
apples-to-apples "does a real workload work, and how close are we?" test.
NVIDIA (TF32 tensor cores on), vs native JAX CUDA on the same GPU
(`tools/plot_transformer.py`):

![ours (OpenCL/NVIDIA) vs JAX CUDA, transformer forward pass](docs/bench_transformer.png)

| config (D, ff, layers) | ours | native CUDA | gap | ours throughput |
|------------------------|------|-------------|-----|-----------------|
| base (512, 2048, 6)    | 9.7 ms | 0.44 ms | 22.3× | 2.2 TFLOP/s |
| large_l1 (1024, 4096, 1) | 5.5 ms | 0.57 ms | 9.8× | 10.1 TFLOP/s |
| large (1024, 4096, 6)  | 33.5 ms | 3.7 ms | **9.1×** | **10.0 TFLOP/s** |

The gap **more than halves as the model gets compute-bound** (and holds at full
depth): `base`'s 22× is small-op/overhead-bound, not a matmul deficit — on
compute-heavy work we sustain **~10 TFLOP/s within ~9× of cuBLAS**. Getting here
took a campaign of general mechanisms, each of which helps any workload:
segmented (workgroup-collaborative) reductions for softmax/layernorm; an
**access-map fusion** pass that folds transposes/reshapes/broadcasts into the
consuming operand's strided read (no materialization, no barrier); **arena
liveness-reuse** (bounds device memory by peak live set, not the sum of all
temporaries — cut the `base` arena 716→105 MiB and unblocked `large` entirely,
which otherwise overflowed the address space); and TF32 tensor-core matmul with
a bank-conflict-free staging tile. The residual gap is matmul: the in-megakernel
tile is arithmetic-intensity-capped by the persistent-kernel co-residency the
cross-workgroup barrier needs, so closing it further needs a hybrid
megakernel-plus-standalone-launch split (the documented next lever).

### Intel Arc 140V (Xe2, Lunar Lake iGPU) — vs JAX CPU

**Testbench: full suite green** (both engines; `intel-opencl-icd` 26.22). No
native JAX plugin exists for this iGPU, so the reference is JAX's XLA **CPU**
backend on the same package (Core Ultra 9 288V) — cross-device but honest:
it's the alternative you'd actually use.

![ours (OpenCL/Xe2) vs JAX CPU, per-op N-vs-time](docs/bench_plot_xe2.png)

- **Large arrays are where the iGPU pays off, and the gap grows with N**:
  elementwise and `gather` at 16M run **~6–8x faster** than XLA CPU
  (~100 GB/s effective — the zero-copy I/O ports matter doubly on an
  integrated GPU, where "device memory" is the same LPDDR5X). Crossover vs
  XLA CPU: ~2M elements for elementwise, ~4M for `gather`, ~256K for the
  `while` loop. `dot_general` is faster from N≈256 up — ~1.1 TFLOP/s at
  2048³ via the standalone SGEMM path, ~2.8x XLA CPU. `matrix × vector` is
  at parity at 2048 via the dedicated `gemv` kernel.
- **`while` loops**: ~23 µs/iteration overhead at small N (the affine-fold +
  in-place-carry + barrier-spin fixes cut it ~5x), and 1.6x *faster* than XLA
  CPU at 16M.
- **Small ops are dispatch-bound** (~70–160 µs wall-clock vs XLA CPU's
  ~10–30 µs): expect 2–6x slower below ~1M elements; batch or fuse small
  work.
- Bring-up found and fixed a real portability bug: lane count is now *measured*
  at init (occupancy discovery, `docs/decisions.md` §9) instead of derived from
  the vendor-ambiguous `CL_DEVICE_MAX_COMPUTE_UNITS`.

### CPU via PoCL (Intel Core Ultra 9 288V) — vs JAX native CPU

**Testbench: full suite green.** This is a same-silicon CPU-vs-CPU comparison:
our plugin through PoCL's OpenCL against JAX's native XLA CPU backend on the
same 8 cores. It answers "what does the OpenCL detour cost on a CPU?" —

![ours (OpenCL/PoCL) vs JAX native CPU, per-op N-vs-time](docs/bench_plot_pocl.png)

- **PoCL started as the debug backend and is now genuinely fast on streaming
  ops.** CPU OpenCL runtimes only auto-vectorize the implicit work-item loop,
  which our in-kernel tile loops defeated — every hot tile body now has an
  explicit-`float8` CPU variant selected by a device-keyed build define
  (`poc/09-cpu-kernels`, `docs/decisions.md` §11). Result: at 16M elements
  our OpenCL-on-CPU is **~3x faster than native XLA CPU** on elementwise
  (4.8 vs 16.2 ms), **2.3x** on `dynamic_slice`, `matvec` at parity, `while`
  within 1.3x.
- **`dot_general` remains XLA's win, but by 3x rather than 88x**: the packed
  + KC-blocked CPU SGEMM (`poc/10-cpu-sgemm`) reaches ~156 GFLOP/s at 2048³
  vs Eigen's ~400–600. `PJRT_OCL_MM_CPU=reg` selects the simpler register
  kernel for hardware that prefers it. Below ~1M elements the ~17–50 µs PoCL
  launch floor keeps small ops 2–7x slower (host-dispatch phases are batched
  onto the in-order queue; the remaining floor is one `clFinish` + PoCL's
  per-command cost).
- If your machine has *any* supported GPU — including an iGPU — prefer it
  (see below). PoCL remains the bring-up/debug/CI backend: printf, host
  debuggers, and sanitizers all work there.

### CPU vs GPU, same machine (Lunar Lake), both through pjrt-ocl

The two backends above share silicon and memory (LPDDR5X); here they are
against each other — GPU (red) vs CPU (blue), both running the identical
bytecode through this plugin
(`tools/plot_bench.py --compare docs/bench_plot_xe2.csv docs/bench_plot_pocl.csv`):

![ours Xe2 iGPU vs ours PoCL CPU, per-op N-vs-time](docs/bench_plot_lnl_xe2_vs_pocl.png)

- After the CPU kernel work, the two backends **converge on pure streaming**
  (they share the same LPDDR5X): the iGPU leads ~2x on elementwise/`gather`
  at 16M and ~2x on the `while` loop, and `matvec` is a tie. Compute
  density still separates them: **~7x** on `matmul` at 2048³.
- Practical guidance: on any machine with a working GPU ICD, the default
  device selection (first GPU) is the right choice — it never loses, and wins
  big on compute-dense programs. Select PoCL explicitly
  (`PJRT_OCL_DEVICE=Portable`) for debugging (printf/sanitizers) or
  CI-without-GPU.

## Inside the VM: scheduled vs. measured execution

How does a `jax.jit` function actually run on the device? Take a program with both
parallel and sequential structure — a heavy matmul next to a cheap elementwise chain,
joined at the end:

```python
def f(a, b, c):          # all 256x256 f32
    m = a @ b            # heavy matmul (shaped)    \  runs in parallel with the
    s = c + c            # elementwise               } elementwise chain s,p,q, which
    p = c * c            # elementwise               } fuses onto lanes with NO barrier
    q = s * p            # elementwise (needs s,p)  /  (same-index deps chain per tile)
    return q + m         # needs m (shaped) -> the one real barrier: the join
```

The compile pipeline turns this into the per-lane schedule below. Lowering emits one
**task** per op and splits each into **tiles** (16K elements for elementwise, 64×64
output blocks for matmul). The scheduler groups ops into **phases**: independent ops and
same-index elementwise **chains** share a phase — a chain runs on one lane per tile, so a
dependency like `q = s * p` costs no barrier, only a cheap thread-local re-read. A **shaped**
op (matmul/reduce/gather — its output reshuffles tiles across lanes) is the only thing that
forces a phase boundary. It packs each phase's tiles onto **lanes** (persistent workgroups)
by cost (LPT) and separates phases with **global barriers** — so `s, p, q` run barrier-free
alongside the matmul and only the final join (which needs `m`) pays a barrier. That schedule
*is* the bytecode the engines execute. (`PJRT_OCL_FUSE=0` reverts to one barrier per level.)

A schedule is only as good as its cost model, so the plugin **measures** it rather than
assuming one. On first use it runs a µbenchmark per tile-op family on the actual device
(two tile counts, slope, so launch overhead cancels), caches the result under
`~/.cache/pjrt-ocl/` keyed by device+driver, and every subsequent compile schedules with
those costs — on this CPU `ew=310 mma=5073 reduce=89 gather=201` µs/tile (a matmul tile
really costs **~16×** an elementwise one), on the NVIDIA GPU
`ew=15 mma=27 reduce=13 gather=21` (nearly uniform — same program, opposite balance). It
matters: a naive unit-cost model dedicates whole lanes to the cheap elementwise ops, which
then finish in under a millisecond and stall at the barrier while the matmul lanes grind on.

`tools/plot_schedule.py` draws the calibrated schedule (top, the scheduler's intent on the
device's measured clock), then runs the program through the plugin with per-entry
instrumentation and draws what the device really did (bottom, white gaps = bubbles):

![scheduled vs measured, device-calibrated cost model](docs/schedule_diamond_calibrated.png)

Reading it:

- **Level 0**: independent ops run side by side — the matmul's tiles fan out over all
  lanes while `c+c` and `c*c` run concurrently. With measured costs the packer
  **fans the matmul out over every lane** and **sequentializes the cheap elementwise ops
  behind its chunks** instead of dedicating lanes to them.
- The dashed **barriers** separate levels: `s * p` and the final join each wait for
  every lane, because their inputs were produced across lanes.
- The planned and measured panels agree structurally; the remaining idle lane-time (~20%)
  is the CPU runtime's own per-workgroup jitter, not scheduling.

On shapes where cheap ops would otherwise steal lanes from a matmul (e.g. one matmul
+ seven small elementwise ops), the calibrated schedule is ~1.3× faster end-to-end on
PoCL. Override knobs: `PJRT_OCL_COST_TABLE=<json>` (explicit table),
`PJRT_OCL_CALIBRATE=0|1` (disable / force re-measure).

Reproduce (any of `diamond`, `chain`, `wide`, or your own StableHLO):

```bash
. ./env.sh
python tools/plot_schedule.py --example diamond --device Portable \
    --out docs/schedule_diamond_calibrated.png                 # measured costs (default)
python tools/plot_schedule.py --stablehlo my_program.mlir      # planned timeline only
```

How the measurement works: `PJRT_OCL_VM_TRACE=<file>` switches execution to the
host-dispatch engine and runs **every schedule entry as its own single-workgroup
launch on a per-lane profiling queue** (lanes still run concurrently — verified on
PoCL and NVIDIA), so each entry gets device-clock start/end timestamps via OpenCL
event profiling, appended as JSON per execute. Two caveats: per-entry launches add
overhead (~tens of µs each), so treat it as a timeline, not a benchmark — and the
GPU megakernel path is not per-entry observable from the host (only barrier arrival
ranks), so traces always reflect the host-dispatch engine.

## Development

```bash
# C++ unit test (executes hand-built bytecode on the device)
cmake --build pjrt_plugin/build && ./pjrt_plugin/build/runtime_test

# Python + end-to-end tests (compares every op against JAX's CPU backend)
pip install -e python/ && python -m pytest tests/

# Per-op perf sweep (lane scaling + vs JAX CPU)
tools/bench_ops.sh
```

The codebase targets OpenCL 3.0-core / the common 1.2 subset; no vendor extensions on the
core path, and it never assumes fp64. Design decisions and their rationale live in
[`docs/decisions.md`](docs/decisions.md); the bytecode format is specified in
[`docs/vmprogram.md`](docs/vmprogram.md) and the execution model in
[`docs/tile-isa.md`](docs/tile-isa.md).

## Limitations & roadmap

- **Op coverage is partial** and grows test-first. Now in: innermost-suffix partial
  reductions (softmax/layernorm) and batched / broadcast `dot_general` (attention, `x@W`).
  Not yet: non-suffix reductions and non-canonical `dot_general` (both need an operand
  transpose first), `if`/`case` control flow, general (data-dependent) gather/scatter, sort.
  These raise a clear `LoweringError` today.
- **Dtypes**: the full JAX matrix (f32/f64/i32/u32/i64/bool/f16/bf16) is in; f64 is gated
  on `cl_khr_fp64`. Still to come: i8/i16 and complex.
- **Two execution engines, auto-selected.** GPUs run a persistent megakernel with an
  on-device cross-workgroup barrier (device-scope acquire/release fences — correct even for
  cross-lane data under iteration). CPU/non-GPU devices (e.g. PoCL) use a **host-dispatch**
  engine instead: the host drives control flow and enforces the barrier with one kernel
  launch per phase. This is required on CPU — an in-kernel spin-barrier deadlocks on a
  non-preemptive CPU runtime (imbalance-starvation; it's why OpenCL mandates kernel
  boundaries for cross-group sync). Override with `PJRT_OCL_ENGINE=host|mega|auto`.
- Performance is improving but not yet tuned. Elementwise/gather are near CUDA;
  large matmul uses a standalone SGEMM but full parity needs TF32 tensor cores
  (inline-PTX WMMA behind the kernel-table override — not yet built); `while` is
  down to ~4x via affine folding + in-place carries (see `docs/decisions.md` §9).

## References

Sources this project drew on, by topic. (Design rationale that cites these lives in
[`docs/decisions.md`](docs/decisions.md).)

**PJRT & the XLA plugin interface**
- PJRT C API header — [openxla/xla `xla/pjrt/c/pjrt_c_api.h`](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h) (vendored at a pinned commit)
- [PJRT overview](https://openxla.org/xla/pjrt) and [PJRT integration guide](https://openxla.org/xla/pjrt/pjrt_integration)
- [StableHLO specification](https://openxla.org/stablehlo/spec) — op semantics we lower from
- [JAX](https://github.com/jax-ml/jax) — the frontend; plugins register via a `jax_plugins` entry point

**OpenCL runtime & the cross-workgroup barrier (the project's #1 risk)**
- [PoCL — Portable Computing Language](https://portablecl.org/) (our CPU/debug backend); its CPU pipeline uses **Continuation-Based Synchronization** ([PoCL papers](https://portablecl.org/publications.html)), i.e. kernel-boundary barriers — the basis of our host-dispatch engine
- T. Sorensen, A. F. Donaldson, et al., **"Portable Inter-Workgroup Barrier Synchronisation for GPUs"**, OOPSLA 2016 — occupancy-bound execution; why an in-kernel spin-barrier needs all workgroups co-resident (and deadlocks on a non-preemptive CPU runtime)
- [The OpenCL Specification (Khronos)](https://www.khronos.org/registry/OpenCL/specs/) — memory model, atomics/fences, and the rule that cross-workgroup sync is a kernel boundary

**Persistent-megakernel design (VM + scheduler inspiration)**
- HazyResearch **Megakernels** — [github.com/HazyResearch/Megakernels](https://github.com/HazyResearch/Megakernels) and the blog posts [*No Bubbles* (Llama-1B)](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) and [*Whole GPU* (TP-Llama-70B)](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main) — per-block instruction streams, fine-grained dependency counters instead of a grid barrier, warp specialization
- [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) — the tile-abstraction lineage

**Matmul & TF32 tensor cores from OpenCL (inline PTX)**
- [ihavnoid/hgemmtest](https://github.com/ihavnoid/hgemmtest) — WMMA tensor cores invoked via inline PTX from OpenCL C
- [sschaetz/nvidia-opencl-examples `oclInlinePTX`](https://github.com/sschaetz/nvidia-opencl-examples) — minimal inline-PTX-from-OpenCL syntax
- [alexarmbr/matmul-playground](https://github.com/alexarmbr/matmul-playground) + blog [*How To Write A Fast Matrix Multiplication From Scratch With Tensor Cores*](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) — tiling, swizzling, `ldmatrix`/`mma.sync` fragment layouts
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) (`arch/mma_sm80.h`) — the authoritative TF32 `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` string
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) — `wmma`/`mma.sync`/`ldmatrix`/`cvta` semantics and fragment→register maps

**Compiler fusion (the access-map principle, §13)**
- XLA operator fusion (`kLoop` / `kInput`) — inline elementwise + broadcast up to a reduction boundary; the model our access-map fusion follows
- The polyhedral / loop-fusion view (iteration domains + access relations) — fuse an edge where the access is a *function*, not a many-to-one relation

**Baselines & tooling**
- cuBLAS / XLA:GPU (the CUDA reference in the benchmarks), [Eigen](https://eigen.tuxfamily.org/) (the XLA:CPU matmul baseline)

## License

TBD.
