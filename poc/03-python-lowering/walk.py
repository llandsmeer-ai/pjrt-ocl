#!/usr/bin/env python
"""Deserialize plugin-received bytes with jaxlib's bundled MLIR/StableHLO bindings
and walk the module generically: op name, operand/result types, attributes,
recursing into regions (while/reduce bodies).

This is exactly what the lowering subprocess will do on the plugin side.

Usage:
  .venv/bin/python walk.py add                 # serialize baked-in example, then walk
  .venv/bin/python walk.py while_reduce
  .venv/bin/python walk.py --file prog.vhlo    # walk artifact bytes from a file
"""
import argparse
import sys


def deserialize(artifact: bytes):
    """Plugin-side entry: VHLO portable artifact bytes -> stablehlo ir.Module."""
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import stablehlo
    ctx = ir.Context()
    # deserialize_portable_artifact loads+upgrades VHLO and registers what it needs;
    # no explicit dialect registration required (verified, see NOTES.md).
    return stablehlo.deserialize_portable_artifact(ctx, artifact)


def _fmt_attr(attr) -> str:
    s = str(attr)
    return s if len(s) <= 70 else s[:67] + "..."


def walk_op(op, indent: int = 0, out=sys.stdout):
    """Generic recursive walk of one operation (mlir python bindings)."""
    o = op.operation
    pad = "  " * indent
    operands = ", ".join(str(v.type) for v in o.operands) or "-"
    results = ", ".join(str(r.type) for r in o.results) or "-"
    print(f"{pad}{o.name}", file=out)
    print(f"{pad}  operands: {operands}", file=out)
    print(f"{pad}  results:  {results}", file=out)
    # NB: iterating OpAttributeMap yields attribute *names* (str) in jaxlib 0.10.2
    attrs = {name: _fmt_attr(attr) for name, attr in dict(o.attributes).items()}
    if attrs:
        print(f"{pad}  attrs:    {attrs}", file=out)
    for ri, region in enumerate(o.regions):
        for bi, block in enumerate(region.blocks):
            bargs = ", ".join(str(a.type) for a in block.arguments)
            print(f"{pad}  region[{ri}] block[{bi}] args: ({bargs})", file=out)
            for inner in block.operations:
                walk_op(inner, indent + 2, out=out)


def walk_module(module, out=sys.stdout):
    for op in module.body.operations:
        walk_op(op, out=out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("example", nargs="?", help="baked-in example name (see dump_stablehlo.py)")
    p.add_argument("--file", help="read artifact bytes from file instead")
    p.add_argument("--target", default="current")
    a = p.parse_args(argv)

    if a.file:
        with open(a.file, "rb") as fh:
            artifact = fh.read()
    elif a.example:
        import dump_stablehlo
        module = dump_stablehlo.lower_to_stablehlo_module(a.example)
        artifact = dump_stablehlo.serialize_as_plugin_would_receive(module, a.target)
    else:
        p.error("give an example name or --file")

    module = deserialize(artifact)
    print(f"# deserialized {len(artifact)} artifact bytes "
          f"(magic+producer: {artifact[:22]!r})")
    walk_module(module)
    return 0


if __name__ == "__main__":
    sys.exit(main())
