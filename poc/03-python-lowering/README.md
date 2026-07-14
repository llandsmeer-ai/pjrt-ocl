# poc/03-python-lowering

Pure-Python proof (decision #2, docs/decisions.md) that we can go from **the bytes a PJRT
plugin receives at `PJRT_Client_Compile` time** to a strawman **VMProgram** bytecode, using
only jaxlib's bundled StableHLO/MLIR python bindings — no C++ deps, no new pip installs.

**Status: PASSING.** `test_poc03.py` proves serialize → subprocess-lower → parse → numpy
execution == `jax.jit` (exact f32 match), plus the JSON-error path for unsupported programs.

Everything runs with the project venv python: `PY=/home/ubuntu/project/.venv/bin/python`
(jax/jaxlib 0.10.2).

## Files

| file | what |
|---|---|
| `research.md` | **Load-bearing**: what bytes PJRT Compile actually receives (VHLO portable artifact, format `"mlir"`, `xla.CompileOptionsProto` options), cited from installed jax sources + `libjax_common.so` evidence |
| `dump_stablehlo.py` | Baked-in examples (`add`, `while_reduce`, `fma_const`) → textual StableHLO or the exact plugin-received bytes |
| `walk.py` | Deserialize artifact bytes with `stablehlo.deserialize_portable_artifact`, walk ops generically (name/types/attrs, recursing into while/reduce regions) |
| `vmprogram.py` | VMProgram v0 binary format spec (module docstring) + writer/reader + stablehlo→VMProgram lowering for the f32 elementwise subset (`OP_HANDLERS` dict, easy to extend) |
| `lower_service.py` | The subprocess interface the C++ plugin will exec: artifact on stdin → VMProgram on stdout, JSON error on stderr (exit 2 = unsupported program, 1 = internal error) |
| `test_poc03.py` | End-to-end sanity (must pass) |
| `NOTES.md` | Dead ends, version quirks, open questions for the C++ side |

## How to run

```sh
PY=/home/ubuntu/project/.venv/bin/python
cd poc/03-python-lowering

$PY test_poc03.py                          # the whole point; must print PASS

$PY dump_stablehlo.py add --text           # textual stablehlo
$PY dump_stablehlo.py add -o add.vhlo      # exact plugin-received bytes (target: current)
$PY dump_stablehlo.py while_reduce --target week12 -o wr.vhlo   # no-attribute negotiation

$PY walk.py add                            # deserialize + generic op walk
$PY walk.py while_reduce                   # ...including while/reduce region recursion
$PY walk.py --file wr.vhlo

$PY vmprogram.py fma_const                 # lower + disassemble + reparse check
$PY vmprogram.py while_reduce              # demonstrates clean NotImplementedError

$PY lower_service.py < add.vhlo > add.vmp  # the exact C++ plugin exec interface
```

## Supported op set (rejects everything else with `NotImplementedError`)

`stablehlo.add` / `multiply` / `subtract` / `constant`, `func` args/results, static-shape f32
tensors. `stablehlo.while` is specified in the binary format (nested linear instruction
lists — no jumps) but not yet emitted; that's M4.
