# PJRT integration notes

Distilled from https://openxla.org/xla/pjrt/pjrt_integration (fetched 2026-07-14). Verify against
the vendored `pjrt_c_api.h` commit when details matter.

## Two integration paths

- **Path A (ours, try first)**: implement the PJRT C API directly against `pjrt_c_api.h`.
- **Path B (fallback)**: implement C++ `xla::PjRtClient` (examples: `pjrt_stream_executor_client.h`,
  `tfrt_cpu_pjrt_client.h`), then wrap: `pjrt::CreateWrapperClient(std::move(client))` +
  `pjrt::CreatePjrtApi(my_plugin::PJRT_Client_Create)` using `pjrt_c_api_wrapper_impl.h`.
  Reference implementation: `pjrt_c_api_cpu.cc`; GPU option handling: `pjrt_c_api_gpu_internal.cc`.

## Entry points

- Mandatory: `GetPjRtApi()` returning the `PJRT_Api*` function-pointer table; `PJRT_Client_Create`
  (receives framework options).
- Optional: `PJRT_Plugin_Initialize` (one-time setup), `PJRT_Plugin_Attributes`,
  `PJRT_TopologyDescription_Create`.

## Conformance testing

XLA ships `RegisterPjRtCApiTestFactory` (`pjrt_c_api_test.h`) to validate basic C API behavior —
requires building inside the XLA tree, so treat as optional/CI-later; our own pytest-vs-CPU suite
is the primary correctness gate.

## JAX discovery & registration

- Preferred packaging: entry points —
  ```toml
  [project.entry-points.'jax_plugins']
  opencl = 'opencl_pjrt'
  ```
  (Alternative: `jax_plugins/<name>/` namespace package with `__init__.py` + the `.so`.)
- The module must expose `initialize()`:
  ```python
  from jax._src import xla_bridge as xb
  xb.register_plugin('opencl', priority=500, library_path=<path to .so>, options=None)
  ```
  Default priority is 400; >400 wins when `JAX_PLATFORMS` is unset. Dev: `JAX_PLATFORMS=opencl`.

## Versioning

- jaxlib version must match the PJRT C API version it was built with; no ABI compatibility yet.
  Practical rule: vendor `pjrt_c_api.h` from an XLA commit contemporaneous with the pinned
  jax/jaxlib release, and record both in docs/decisions.md §5.
