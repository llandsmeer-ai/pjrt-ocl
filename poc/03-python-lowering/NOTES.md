# poc/03 NOTES — tried/failed, quirks, open questions

For merge into `docs/decisions.md` (§2 StableHLO ingestion). Everything below observed on
jax/jaxlib 0.10.2, python 3.12, 2026-07-14.

## Findings & gotchas (chronological)

1. **jax Python never serializes on the compile path.** `backend.compile_and_load` receives a
   live `ir.Module`; serialization to plugin bytes happens inside jaxlib C++
   (`PjRtCApiClient::CompileAndLoad` → `xla::Serialize`). So the ONLY faithful way to reproduce
   the plugin-received bytes from Python is to call
   `stablehlo.serialize_portable_artifact(module, target)` ourselves with the same negotiated
   target. Details + citations in research.md.
2. **`serialize_portable_artifact` MUTATES ITS INPUT IN PLACE** (converts the module to VHLO).
   Serializing the module returned by `jax.jit(f).lower(...).compiler_ir('stablehlo')` corrupts
   jax's *cached* lowering: the next `jax.jit(f)(...)` with the same avals fails with
   `INVALID_ARGUMENT: MLIR module must have a main function` (the cached module is now
   `vhlo.func_v1`, no `func.func @main`). Fix: clone first (MLIR-bytecode round-trip
   `module.operation.write_bytecode(io.BytesIO())` → `ir.Module.parse(bytes, context=...)`).
   Only affects tooling that serializes in the *same process* as jax; the real plugin
   subprocess never has jax's module in hand, so this is a dump_stablehlo.py-only hazard.
3. **Artifact bytes are not byte-stable across call sites**: MLIR locations (python tracebacks)
   are serialized too, so the same lambda lowered from different stack depths yields different
   artifact sizes (645 vs 745 bytes for the same 3-op function). Any future compile cache must
   key on semantic content, not raw artifact bytes.
4. **Binding API dead ends** (jaxlib 0.10.2 bundled MLIR python bindings):
   - `ir.RankedTensorType.isinstance(t)` / `ir.F32Type.isinstance(t)` do NOT exist (upstream
     MLIR has them). These bindings **auto-downcast**: `ir.Type.parse("tensor<8xf32>")` already
     returns `RankedTensorType`, and plain python `isinstance()` is the supported check.
   - Iterating `operation.attributes` yields attribute *names* (str), not `NamedAttribute`
     objects; use `dict(op.attributes)` / indexing.
   - `stablehlo.get_version_from_compatibility_requirement` takes the enum
     `stablehlo.StablehloCompatibilityRequirement.{NONE,WEEK_4,WEEK_12,MAX}`, not a string.
   - `np.asarray(ir.DenseFPElementsAttr(attr))` works (buffer protocol), including reading
     splats of 1 element; splat-to-N expansion must be done manually.
5. **`deserialize_portable_artifact(ctx, bytes)` needs no dialect registration** on a fresh
   `ir.Context()` — it registers/loads what it needs and upgrades VHLO → current StableHLO
   opset automatically. Both 1.13.7 (WEEK_12) and 1.17.0 (current) artifacts round-trip.
6. **libjax_common.so string-mining works** as a way to confirm which XLA translation units are
   inside jaxlib and what they do (`external/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc`,
   `mlir_to_hlo.cc`, attribute name `stablehlo_current_version`, `WEEK_4`/`WEEK_12`,
   CompileOptionsProto parse-failure messages). Cheap substitute for reading a matching XLA
   checkout when disk forbids one.
7. `jaxlib._jax.CompileOptions` has `SerializeAsString()`/`ParseFromString()` — we can build
   and parse `xla.CompileOptionsProto` blobs from Python for plugin tests without protobuf
   installed.
8. **Startup cost of the lowering subprocess**: measured 0.14 s wall (1.7 s cpu, parallel
   imports) for `lower_service.py` end-to-end on the tiny add program, warm page cache.
   Comfortably within decision #2's budget (compile-time only); a `--server` stdin-loop mode
   remains an easy optimization if real programs get slow.
9. Only stdlib + numpy + jaxlib used. No new pip installs.

## Open questions for the C++ side (poc/02 / M2)

- **Byte-for-byte confirmation**: research.md's serialization claim is source/strings-derived.
  Once poc/02's skeleton implements `PJRT_Client_Compile`, log `PJRT_Program.{format,code}`
  and diff against `dump_stablehlo.py` output (expect same magic/producer + same module after
  deserialization; raw bytes may differ by locations, see finding 3).
- **Advertise `stablehlo_current_version`?** Recommended (research.md §3): plugin should report
  jaxlib's `get_current_version()` (1.17.0) via `PJRT_Plugin_Attributes` as `int64[3]{1,17,0}`.
  Needs poc/02 to confirm the attribute plumbing and that the client then serializes at 1.17.0
  rather than WEEK_12's 1.13.7. Either version deserializes fine on our side.
- **Where does the venv python path come from?** Decision log says
  `register_plugin(options=...)` → `PJRT_Client_Create` create_options. Confirm in poc/02 that
  string options survive the round trip, and define the fallback when absent (env var
  `PJRT_OCL_PYTHON`?).
- **compile_options blob**: plugin currently may ignore it (single device), but
  `PJRT_Executable_GetCompileOptions` (used by jax in some paths?) may need the original bytes
  echoed back — store them with the executable. Verify whether jax 0.10.2 ever calls it.
- **Error surfacing**: lower_service exit 2 (unsupported op) should map to a clean
  `PJRT_Error` with the JSON message; decide the PJRT error code (UNIMPLEMENTED?).
- **VMProgram format gaps** (fine for strawman, must be settled in M1):
  - dtype enum has only F32 implemented; i32/i1 needed as soon as `while`/`compare` land.
  - No liveness-based buffer reuse (one buffer per SSA value → arena bloat on real programs).
  - WHILE is spec'd (sub-list ids in fields a/b, pred buffer in aux) but the lowerer doesn't
    emit it yet; loop-carried values are planned as in-place buffer writes — needs the
    executor to guarantee write-before-read ordering across sub-list boundary (barrier).
  - Scalars (tensor<f32>, rank 0) get n_elems=1 buffers — VM must handle n_elems=1 instrs
    efficiently (whole grid for 1 element is wasteful but correct).
  - No launch geometry in the instr yet (CLAUDE.md mentions it); grid-stride n_elems is the
    only iteration-space info. Strides/shapes needed once broadcast/transpose land.
