#!/usr/bin/env python
"""Produce the exact bytes a PJRT plugin receives at PJRT_Client_Compile time.

Reproduces jaxlib's own serialization path (see research.md):
  jax.jit(f).lower(args) -> StableHLO ir.Module
  -> stablehlo.serialize_portable_artifact(module, target_version)   [= xla::Serialize]
where target_version is what PjRtCApiClient would negotiate:
  - "current": plugin advertises stablehlo_current_version == client's => min() == current
  - "week12":  plugin advertises nothing => 12-week compatibility window version

Usage (always via the project venv python):
  .venv/bin/python dump_stablehlo.py add            --text        # textual stablehlo
  .venv/bin/python dump_stablehlo.py add            -o add.vhlo   # plugin bytes to file
  .venv/bin/python dump_stablehlo.py while_reduce   --target week12 -o wr.vhlo
Examples: add | while_reduce
"""
import argparse
import functools
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Baked-in example functions
# ---------------------------------------------------------------------------

def _example_add():
    import jax.numpy as jnp
    f = lambda a, b: a + b
    args = (jnp.zeros(8, jnp.float32), jnp.zeros(8, jnp.float32))
    return f, args


def _example_while_reduce():
    import jax.numpy as jnp
    from jax import lax

    def f(x):
        def cond(c):
            return c[0] < 3

        def body(c):
            return (c[0] + 1, c[1] * 2.0)

        _, y = lax.while_loop(cond, body, (0, jnp.sum(x)))
        return y

    args = (jnp.zeros(8, jnp.float32),)
    return f, args


def _example_fma_const():
    """a + b*c minus a constant vector: exercises add/multiply/subtract/constant."""
    import jax.numpy as jnp
    k = np.arange(8, dtype=np.float32)

    def f(a, b, c):
        return (a + b * c) - jnp.asarray(k)

    args = (jnp.zeros(8, jnp.float32),) * 3
    return f, args


EXAMPLES = {
    "add": _example_add,
    "while_reduce": _example_while_reduce,
    "fma_const": _example_fma_const,
}


# ---------------------------------------------------------------------------
# The serialization path (mirrors xla::Serialize in jaxlib's C++)
# ---------------------------------------------------------------------------

def lower_to_stablehlo_module(example: str):
    """Returns the live ir.Module jax hands to backend.compile_and_load."""
    import jax
    f, args = EXAMPLES[example]()
    lowered = jax.jit(f).lower(*args)
    return lowered.compiler_ir("stablehlo")  # jaxlib.mlir.ir.Module


@functools.cache
def resolve_target(target: str) -> str:
    """Map a target spec to a VHLO version string, as PjRtCApiClient would."""
    from jaxlib.mlir.dialects import stablehlo
    if target == "current":
        # Plugin advertises stablehlo_current_version = jaxlib's own current version;
        # client computes min(plugin, client) == current.
        return stablehlo.get_current_version()
    if target == "week12":
        # Plugin advertises nothing: client uses the 12-week compatibility window.
        return stablehlo.get_version_from_compatibility_requirement(
            stablehlo.StablehloCompatibilityRequirement.WEEK_12)
    return target  # explicit "X.Y.Z"


def serialize_as_plugin_would_receive(module, target: str = "current") -> bytes:
    """Exact PJRT_Program.code bytes: a VHLO portable artifact (MLIR bytecode)."""
    from jaxlib.mlir.dialects import stablehlo
    return stablehlo.serialize_portable_artifact(module, resolve_target(target))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("example", choices=sorted(EXAMPLES))
    p.add_argument("--target", default="current",
                   help="'current' | 'week12' | explicit 'X.Y.Z' (default: current)")
    p.add_argument("--text", action="store_true", help="print textual stablehlo instead")
    p.add_argument("-o", "--out", help="write serialized artifact bytes to this file "
                                       "(default: stdout if not a tty)")
    a = p.parse_args(argv)

    module = lower_to_stablehlo_module(a.example)
    if a.text:
        print(module)
        return 0

    data = serialize_as_plugin_would_receive(module, a.target)
    ver = resolve_target(a.target)
    print(f"# {a.example}: {len(data)} bytes, VHLO target {ver}, "
          f"magic+producer: {data[:22]!r}", file=sys.stderr)
    if a.out:
        with open(a.out, "wb") as fh:
            fh.write(data)
    elif not sys.stdout.isatty():
        sys.stdout.buffer.write(data)
    else:
        print("refusing to write binary artifact to a tty; use -o FILE", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
