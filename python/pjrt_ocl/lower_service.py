#!/usr/bin/env python
"""Lowering service — the exact interface the C++ plugin execs at
PJRT_Client_Compile time (see docs/decisions.md #2):

    <python_exe> <lower_service.py>  < vhlo_artifact_bytes  > vmprogram_bytes

Both invocations work:
    python -m pjrt_ocl.lower_service
    python /path/to/pjrt_ocl/lower_service.py     (what the C++ plugin does)

  stdin :  PJRT_Program.code bytes (StableHLO/VHLO portable artifact)
  stdout:  VMProgram v1 binary (docs/vmprogram.md) — empty on failure
  stderr:  on failure, one JSON object: {"error": <class>, "message": <str>}
  exit  :  0 ok
           2 unsupported program (valid input, beyond current op coverage —
             the C++ side surfaces a clean PJRT UNIMPLEMENTED-style error)
           3 internal error (lowering bug / bad invocation)
"""
from __future__ import annotations

import json
import os
import sys

EXIT_OK = 0
EXIT_UNSUPPORTED = 2
EXIT_INTERNAL = 3


def _import_lowering():
    if __package__:
        from . import lowering               # python -m pjrt_ocl.lower_service
        return lowering
    try:
        from pjrt_ocl import lowering        # direct script, package installed
    except ImportError:
        # direct script path in a bare source checkout: put the package's
        # parent dir (python/) on sys.path
        sys.path.insert(0, os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        from pjrt_ocl import lowering
    return lowering


def main() -> int:
    artifact = sys.stdin.buffer.read()
    try:
        if not artifact:
            raise ValueError(
                "empty input on stdin (expected VHLO portable artifact bytes)")
        lowering = _import_lowering()
        blob = lowering.lower_artifact(artifact).serialize()
    except Exception as e:  # noqa: BLE001 — service boundary
        json.dump({"error": type(e).__name__, "message": str(e)}, sys.stderr)
        sys.stderr.write("\n")
        # LoweringError subclasses NotImplementedError
        return (EXIT_UNSUPPORTED if isinstance(e, NotImplementedError)
                else EXIT_INTERNAL)
    sys.stdout.buffer.write(blob)
    sys.stdout.buffer.flush()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
