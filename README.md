# pjrt-ocl

**Run JAX on any OpenCL device.** `pjrt-ocl` is a [PJRT](https://openxla.org/xla/pjrt)
plugin that lets `jax.jit` execute on OpenCL-capable hardware — Intel, AMD, NVIDIA,
or a CPU via [PoCL](https://portablecl.org/) — with no vendor SDK (no CUDA, no ROCm)
on the execution path.

> ⚠️ **Experimental / work in progress.** f32 only, a growing subset of StableHLO ops,
> not yet on PyPI. Validated end-to-end on an NVIDIA RTX PRO 6000 (via NVIDIA's OpenCL)
> and on PoCL (CPU). Not affiliated with Google, OpenXLA, or the JAX project.

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

f32 only (JAX defaults to f32; x64 disabled). Growing test-driven; see
[`tests/SCOREBOARD.md`](tests/SCOREBOARD.md).

- **Elementwise** — add, subtract, multiply, divide, max, min, pow; negate, exp, log,
  sqrt, rsqrt, tanh, abs, floor, ceil, sign; compare (all directions), select.
- **Shape** — `broadcast_in_dim`, `transpose` (as strided gathers).
- **Reductions** — full sum / max / min / prod (over all axes).
- **Linear algebra** — `dot_general` (plain 2D matmul, register-blocked tile kernel).
- **Making** — `iota`, `convert` (f32).
- **Control flow** — `while` (interpreted on device).

Anything unsupported raises a clear `LoweringError` naming the op.

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

- **Dtypes in progress**: f32 works today; full dtype coverage (i32/bool first, then
  i64/f64/f16, then bf16/complex) is the active workstream — the VM is being made
  byte-addressed and dtype-dispatched. f64/f16 will be device-conditional (OpenCL extensions).
- **Op coverage is partial** and grows test-first — reshape, slice, concatenate,
  partial-axis reductions, batched matmul, and `if`/`case` are next.
- **PoCL (CPU) is reliable for correctness spot-checks but not heavy iteration**: the
  device-side cross-workgroup barrier can deadlock under repeated dispatch on PoCL's CPU
  thread pool. Real GPUs (co-resident workgroups) are unaffected. A host-dispatch fallback
  engine is planned (see `docs/decisions.md`).
- Performance is improving but not yet tuned: matmul runs a register-blocked tile kernel;
  memory-bound ops are currently limited by arena-copy traffic (see
  [`docs/perf-findings.md`](docs/perf-findings.md)).

## License

TBD.
