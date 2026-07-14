# poc/02-pjrt-skeleton

Minimal, standalone, hand-rolled PJRT C API plugin (`libpjrt_ocl_skeleton.so`) — just
enough surface that JAX lists our device. No XLA source dependency: the single vendored
header `vendor/pjrt_c_api.h` (openxla/xla @ `5a9e73cbd9`, the commit pinned by jax v0.10.2,
PJRT C API **0.112**) is the entire interface.

**Status: PASSING** against jax/jaxlib 0.10.2 (2026-07-14). See `NOTES.md` for the
incident log and the hand-rolled-vs-XLA-wrappers verdict (verdict: hand-rolled is viable).

## Build

```sh
cd poc/02-pjrt-skeleton
cmake -S . -B build -G Ninja
cmake --build build            # ~3 s, needs g++ (C++20), ocl-icd-opencl-dev
```

## Run

Exit-criterion command (discovery via the local `jax_plugins/` namespace dir; `python -c`
puts the cwd on sys.path, so run from this directory):

```sh
cd poc/02-pjrt-skeleton
JAX_PLATFORMS=opencl /home/ubuntu/project/.venv/bin/python -c "import jax; print(jax.devices())"
# -> [OclDevice(id=0)]
```

Explicit-registration variant (works from any cwd):

```sh
JAX_PLATFORMS=opencl /home/ubuntu/project/.venv/bin/python test_jax_devices.py
```

Device selection (real OpenCL enumeration; default = first GPU, else first CPU):

```sh
PJRT_OCL_DEVICE="Portable" ...   # PoCL CPU;  "NVIDIA" for the GPU; optional ":<device idx>"
```

`PJRT_OCL_LOG=1` prints every implemented-callback hit and every error the plugin returns.

## How it works

- `pjrt_api_fn_list.h`: X-macro list of all 136 `PJRT_Error*`-returning entries of
  `PJRT_Api`, generated from the vendored header. Every entry defaults to a stub returning
  `UNIMPLEMENTED: <callback name>` — so any jax crash names the callback to implement next.
- `plugin.cc`: real implementations for the ~30 callbacks the `jax.devices()` path
  exercises (errors incl. the 0.112 error vtable, plugin init/attributes, client, device
  description, device, one memory space per device incl. the 0.112 memory user-data
  vtable), plus OpenCL platform/device enumeration for the device kind/debug strings.
- Known PoC-only behaviors: unmatched `PJRT_OCL_DEVICE` falls back to a stub device instead
  of erroring; `PJRT_Client_TopologyDescription` is left unimplemented (tolerated).
