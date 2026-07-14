# What bytes does a PJRT plugin receive at `PJRT_Client_Compile` time?

Research against the **installed** jax 0.10.2 / jaxlib 0.10.2 in `.venv/` (python 3.12).
All paths below are relative to `.venv/lib/python3.12/site-packages/`. Everything marked
**[verified]** was executed/inspected on this machine 2026-07-14.

## Headline answer

The plugin's `PJRT_Client_Compile` receives a `PJRT_Program` whose

- `code` = a **StableHLO portable artifact**: the module downgraded to the **VHLO** dialect at a
  negotiated target version and emitted as **MLIR bytecode**. First bytes are the MLIR bytecode
  magic `4D 4C EF 52` (`"ML\xefR"`) followed by the producer string `"StableHLO_v<X.Y.Z>"`
  **[verified** by reproducing the serialization — see below**]**;
- `format` = the string `"mlir"` (`pjrt::kMlirFormat`; the alternative `"hlo"` =
  `HloModuleProto` is only used for the legacy `XlaComputation` path, which jax's
  StableHLO pipeline does not take);

and `PJRT_Client_Compile_Args.compile_options` = a serialized **`xla.CompileOptionsProto`**
(protobuf wire format) **[verified:** `jaxlib._jax.CompileOptions().SerializeAsString()` returns
proto bytes, 963 bytes for the defaults; the plugin-side parse error string
`"PjRtCApiExecutable::GetCompileOptions: Failed to parse CompileOptionsProto"` is present in
`jaxlib/libjax_common.so`**]**.

The target VHLO version is negotiated as follows (see §3):

- if the plugin advertises a `stablehlo_current_version` attribute (vector of 3 int64
  `{major, minor, patch}`) via `PJRT_Plugin_Attributes`, the client serializes at
  `min(plugin_version, client_current_version)`;
- otherwise it uses the version that satisfies a **12-week backward-compatibility window**.
  For this jaxlib: `WEEK_12 → "1.13.7"` **[verified** via
  `stablehlo.get_version_from_compatibility_requirement(StablehloCompatibilityRequirement.WEEK_12)`**]**.
  Client current version is **1.17.0**, minimum deserializable version **0.9.0**, StableHLO
  API version 9 **[verified]**.

Deserializing on the plugin side with jaxlib's own bundled bindings works **[verified]**:

```python
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo
ctx = ir.Context()
module = stablehlo.deserialize_portable_artifact(ctx, artifact_bytes)  # upgrades VHLO -> current StableHLO
```

## 1. Python side: jax never serializes on the compile path

Trace of `jax.jit(f)(args)` (first call) in the installed source:

1. **Lowering to StableHLO** happens in
   `jax/_src/interpreters/mlir.py:1287` — `def lower_jaxpr_to_module(...)`, returning a
   `LoweringResult` (`mlir.py:1130`) that carries a live `ir.Module` (jax's own MLIR context,
   `JaxIrContext`, `mlir.py:615`). The module is plain StableHLO + `func` dialect with a public
   `func.func @main`; module-level attrs `mhlo.num_replicas`/`mhlo.num_partitions` (builtin i32
   attrs under dialect-prefixed *names* — no mhlo dialect ops) **[verified** by dumping
   `jax.jit(f).lower(...).compiler_ir('stablehlo')`**]**.
2. `jax/_src/interpreters/pxla.py:1687` — `UnloadedMeshExecutable.from_hlo(name, hlo: ir.Module, ...)`
   → `_cached_compilation` (`pxla.py:1500`, call at `pxla.py:1733`), which builds
   `xc.CompileOptions` via `create_compile_options` (`pxla.py:1477`) →
   `jax/_src/compiler.py:139` `get_compile_options`.
3. `jax/_src/compiler.py:387` — `compile_or_get_cached(backend, computation: ir.Module, ...)` →
   `backend_compile_and_load` (`compiler.py:293`), which calls
   **`backend.compile_and_load(module, executable_devices=..., compile_options=options)`**
   at `compiler.py:344`/`compiler.py:353`, passing the **live `ir.Module` object**, not bytes.
4. `backend` is a `jaxlib._jax.Client` (nanobind, compiled). Its `compile_and_load` overloads
   accept `object` (an `ir.Module`), `bytes`, or `str` **[verified** via
   `client.compile_and_load.__doc__`**]**. So the MLIR→bytes conversion for a C-API plugin happens
   **inside jaxlib's C++**, not in jax Python.

Note: `mlir.py:603` `module_to_bytecode` (plain MLIR bytecode, *not* a portable artifact) exists
but is used for jax.export / cache keys / host-callback plumbing (e.g. `mlir.py:3479`), **not**
for the plugin compile path. Do not confuse the two: plain `module.operation.write_bytecode()`
output is version-fragile; the portable artifact is the compatibility contract.

## 2. jaxlib side: `PjRtCApiClient::CompileAndLoad`

jaxlib ships no C++ source, but `jaxlib/libjax_common.so` (334 MB, contains XLA + MLIR +
StableHLO) embeds the build paths and error strings of the exact translation units
**[verified** via `strings -a`**]**:

- `external/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc` — the PJRT C-API client wrapper that
  every dynamically-loaded plugin goes through
  (`jax/_src/xla_bridge.py:596 register_plugin` → `xla_bridge.py:557 make_pjrt_c_api_client` →
  `jaxlib/xla_client.py:137 make_c_api_client` / `xla_client.py:114 load_pjrt_plugin_dynamically`
  → `_jax.get_c_api_client`, `xla_bridge.py:212`).
- `external/xla/xla/pjrt/mlir_to_hlo.cc` — where `xla::Serialize(module, target)` lives.
- String evidence of the version negotiation + serialization in the .so:
  - `stablehlo_current_version` (the plugin attribute name it looks up),
  - `WEEK_4`, `WEEK_12` (the `mlir::vhlo::Version::CompatibilityRequirement` names used when the
    attribute is absent),
  - `"Failed to serialize StableHLO to plugin version "` and
    `"Failed to serialize StableHLO with mixed dialects to plugin version "` — the two branches of
    `xla::Serialize`: pure-StableHLO modules go through
    `mlir::stablehlo::serializePortableArtifact(module, target)`; modules with extra dialects
    (chlo/sdy/...) go through the same API with `allow_other_dialects=true`. Either way the wire
    format is a **VHLO portable artifact**, i.e. MLIR bytecode.

Cross-referenced with openxla/xla source of those files at the jaxlib-0.10 pin
(`xla/pjrt/c_api_client/pjrt_c_api_client.cc`, `PjRtCApiClient::CompileAndLoad(mlir::ModuleOp, CompileOptions)`):
it fetches plugin attributes, computes the target version as described in the headline, calls
`xla::Serialize`, then `InitializeArgsAndCompile` fills
`PJRT_Program{code, code_size, format="mlir"}` and
`compile_options = CompileOptions::ToProto().SerializeAsString()` and invokes
`PJRT_Client_Compile`. (We cannot execute this path yet — no plugin exists — so the .so string
evidence + upstream source is the citation; **revisit with a byte-for-byte check once poc/02's
skeleton can log the received buffer.**)

## 3. Version negotiation: what OUR plugin should do

- The artifact's embedded version is the version its VHLO was **downgraded to**; any
  deserializer with StableHLO ≥ that version upgrades it back to its current opset on load.
  Since our lowering subprocess uses **the same jaxlib as the jax that produced the artifact**,
  we are version-matched by construction — both `WEEK_12` (1.13.7) and `current` (1.17.0)
  artifacts round-trip **[verified** for both targets: 661-byte artifacts, identical op walk
  after deserialization**]**.
- Still, the plugin **should advertise** `stablehlo_current_version = {1, 17, 0}` (query it at
  plugin init by asking the lowering subprocess for `stablehlo.get_current_version()`) so the
  client skips the 12-week downgrade. Less downgrade surface = fewer ways to hit VHLO
  legalization bugs.
- Useful binding APIs (all in `jaxlib/mlir/dialects/stablehlo.py:19`, re-exported from
  `jaxlib.mlir._mlir_libs._stablehlo`) **[verified]**:
  - `serialize_portable_artifact(module: ir.Module, target: str, allow_other_dialects=False) -> bytes`
  - `deserialize_portable_artifact(ctx: ir.Context, artifact: bytes|str) -> ir.Module`
  - `get_current_version() -> '1.17.0'`, `get_minimum_version() -> '0.9.0'`,
    `get_api_version() -> 9`, `get_smaller_version(a, b)`,
    `get_version_from_compatibility_requirement(StablehloCompatibilityRequirement.WEEK_4|WEEK_12|NONE|MAX)`
    → `1.15.0 / 1.13.7 / 1.17.0 / 0.9.0`.

## 4. Interpreting `compile_options`

The plugin receives serialized `xla.CompileOptionsProto`
(schema: openxla/xla `xla/pjrt/proto/compile_options.proto`; the proto **descriptors are embedded**
in `libjax_common.so`, e.g. `xla.CompileOptionsProto.compiler_variant` is visible in strings, so
field numbers can be recovered offline if we ever need them without an XLA checkout).
What jax actually populates (`compiler.py:139-289`, `pxla.py:1477-1497`):

- `num_replicas = 1`, `num_partitions = 1` (single device), `device_assignment` (1x1),
- `parameter_is_tupled_arguments` (False for plugins),
- `executable_build_options` incl. `debug_options` (XLA-internal knobs we can ignore),
  `allow_spmd_sharding_propagation_to_{parameters,output}`,
- `env_option_overrides` from user `compiler_options=`.

**For pjrt-ocl M2:** parse nothing; assert single device; treat the options blob as opaque and
log its size. The proto matters only when we implement multi-device or want
`compiler_options=` passthrough to the lowerer. (`PJRT_Executable_GetCompileOptions` may be
asked to return these bytes back — keep them stored with the executable.)

## 5. What this means for the lowering subprocess contract

stdin → the exact `PJRT_Program.code` bytes (VHLO portable artifact; MLIR bytecode magic
`ML\xefR`); stdout → VMProgram binary; stderr → JSON error + nonzero exit. Optionally argv/env
may later carry the compile-options blob; not needed now. Implemented in `lower_service.py`,
proven by `test_poc03.py`.
