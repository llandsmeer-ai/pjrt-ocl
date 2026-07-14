# poc/02-pjrt-skeleton — hand-rolled PJRT C API: findings log

Session 2026-07-14. Goal: minimal standalone .so implementing the PJRT C API by hand
(no XLA source dependency) such that `JAX_PLATFORMS=opencl python -c "import jax;
print(jax.devices())"` lists our device. **Result: PASSED** (see Verdict at bottom).

## Version negotiation (do this first, it determines everything)

- jaxlib 0.10.2 bundles no PJRT header and no obvious version string. The reliable way to
  find the matching PJRT C API version: jax pins its XLA commit in
  `third_party/xla/revision.bzl` at the release tag.
  - jax tag `jax-v0.10.2` → `XLA_COMMIT = 5a9e73cbd92530cac2ac36f4736a774b2412afe2`.
  - Vendored header (single self-contained file, only libc includes):
    `https://raw.githubusercontent.com/openxla/xla/5a9e73cbd92530cac2ac36f4736a774b2412afe2/xla/pjrt/c/pjrt_c_api.h`
    → `vendor/pjrt_c_api.h`, **PJRT C API version 0.112** (PJRT_API_MAJOR 0, PJRT_API_MINOR 112).
- Cross-check: `strings libjax_common.so | grep "PJRT C API version"` shows feature gates up
  to 0.102 ("PJRT_Buffer_Bitcast requires PJRT C API version 0.98 or higher", etc.), i.e. the
  framework does *runtime* minor-version gating; exact-match paranoia is only needed for
  `major_version`. `ENABLE_PJRT_COMPATIBILITY` env var exists in the binary for relaxed
  minor matching; we didn't need it since we match 0.112 exactly.
- The version-pinning recipe for future jax upgrades: read `revision.bzl` at the jax tag,
  re-vendor the header, done.

## API-surface facts learned (PJRT C API 0.112)

- `PJRT_Error` and `PJRT_Memory` are **not opaque** in 0.112: the header defines them as
  structs whose first member is a function-table pointer (`PJRT_Error_FunctionTable` with
  destroy/message/get_code/for_each_payload; `PJRT_Memory_FunctionTable` with
  get_user_data/set_user_data + `instance_struct_size`). Implementations must embed the
  vtable'd base as first member. `PJRT_Client`, `PJRT_Device`, `PJRT_DeviceDescription`,
  `PJRT_LoadedExecutable` etc. remain opaque (we define the structs ourselves).
- The `PJRT_Api` table has 138 entries (136 returning `PJRT_Error*`, 2 returning void).
  We generated an X-macro list (`pjrt_api_fn_list.h`) from the header with awk/sed and
  default-stubbed all 136 with `UNIMPLEMENTED: <callback name>`, then implemented only what
  the jax.devices() path actually exercised.

## Incident log (every crash/error, in order)

1. **`NOT_FOUND: GetPjrtApi not found in ...so`** — jaxlib dlsym's **`GetPjrtApi`**
   (lowercase "rt"), while some openxla docs spell it `GetPjRtApi`. Fix: export the
   lowercase-rt spelling.
2. **Infinite error recursion → core dump (SIGABRT under timeout)** — leaving
   `PJRT_Error_ForEachPayload` as an UNIMPLEMENTED stub is fatal: the framework calls it on
   *every* error it receives (status-payload collection); the stub returned a *new* error,
   which triggered another ForEachPayload, ad infinitum. Fix: implement it for real
   (we have no payloads → return nullptr immediately). Lesson: the error-handling trio+
   ForEachPayload must be correct before anything else, exactly as the charter predicted —
   plus ForEachPayload, which the charter didn't predict.
3. **CHECK-crash (abseil LOG(FATAL), not a catchable error) in
   `xla::PjRtCApiDevice::InitAttributes` → `pjrt::LogFatalIfPjrtError`** when
   `PJRT_Device_GetAttributes` returned UNIMPLEMENTED. jaxlib 0.10.2 uses the *newer*
   `PJRT_Device_GetAttributes` entry (with heap holder + deleter callback), NOT
   `PJRT_DeviceDescription_Attributes`, during client construction. Empty attribute list is
   accepted. Fix: implement it returning 0 attributes with a working deleter.
   General lesson: some unimplemented callbacks are survivable errors, others are
   LogFatalIfPjrtError CHECK-crashes — you cannot tell from the header; you find out by
   running. The `UNIMPLEMENTED: <name>` stub message made each culprit identifiable in one
   run.
4. **`ALREADY_EXISTS: PJRT_Api already exists for device type opencl`** (cosmetic) — when the
   explicit `register_plugin` test script runs with `jax_plugins/` also reachable on
   sys.path, discovery registers a second time. Guarded in `jax_plugins/opencl/__init__.py`.

Non-fatal observations:
- `PJRT_Client_TopologyDescription` returning UNIMPLEMENTED is tolerated by
  `PjRtCApiClient` (topology stays unavailable; devices() unaffected).
- `PJRT_Plugin_Attributes` returning 0 attributes is accepted (no xla_version /
  stablehlo_current_version needed for devices(); expect to need these later for compile).

## Callbacks actually required for `jax.devices()` (jaxlib 0.10.2, API 0.112)

Error vtable + `PJRT_Error_Destroy/Message/GetCode/ForEachPayload`;
`PJRT_Plugin_Initialize`, `PJRT_Plugin_Attributes`;
`PJRT_Client_Create/Destroy/PlatformName/PlatformVersion/ProcessIndex/Devices/
AddressableDevices/AddressableMemories`;
`PJRT_DeviceDescription_Id/ProcessIndex/Kind/DebugString/ToString` (+`Attributes` wired but
the framework used `PJRT_Device_GetAttributes` instead);
`PJRT_Device_GetDescription/IsAddressable/LocalHardwareId/AddressableMemories/DefaultMemory/
GetAttributes`;
`PJRT_Memory_Id/Kind/Kind_Id/DebugString/ToString/AddressableByDevices` + memory user-data
vtable. (~30 of 138 entries; LookupDevice/LookupAddressableDevice implemented but not
observed to be hit.)

At least one memory space per device is required structurally (PjRtCApiClient's
InitDevicesAndMemorySpaces enumerates client memories and per-device memories during
client construction).

## Bonus: real OpenCL enumeration

Implemented (clGetPlatformIDs/clGetDeviceIDs, `-lOpenCL`): default = first GPU else first
CPU; `PJRT_OCL_DEVICE=<platform substring>[:<device idx>]` honored. Verified:
- default / `NVIDIA` → device_kind "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
- `Portable` / `Portable:0` → "cpu-haswell-AMD Ryzen 9 3900X 12-Core Processor" (PoCL)
- no match → falls back to "OpenCL stub device" (PoC behavior; real plugin should error).

## Verdict: hand-rolled C API is viable

The hand-rolled route **works and is cheap**: one 3200-line self-contained vendored header,
~650 lines of C++, CMake+Ninja, zero XLA source dependency, builds in ~3 s. Reaching
`jax.devices()` took exactly three real incidents (symbol spelling, ForEachPayload
recursion, Device_GetAttributes CHECK-crash), each diagnosed in a single run thanks to
name-carrying UNIMPLEMENTED stubs; jaxlib's error paths (stack traces name the exact
`xla::PjRtCApi*` caller) made this closer to pleasant than painful. The user's prediction
that it might fail did not materialize at this milestone. Risks that remain for later
milestones (M2+): the CHECK-crash pattern means every newly-exercised callback (Compile,
Execute, buffer transfers, events) must be implemented to spec rather than left erroring,
and the async Event contract is the likeliest source of real pain. No evidence so far that
justifies paying for the XLA C++ wrapper route (full Bazel/XLA build — which the ~5 GB disk
budget cannot fit anyway). Recommendation: **continue hand-rolled**; keep the stub-table
trick permanently so unimplemented surface area always self-identifies.
