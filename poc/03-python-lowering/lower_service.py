#!/usr/bin/env python
"""Subprocess-mode lowering service — the exact interface the C++ plugin will exec
at PJRT_Client_Compile time:

    .venv/bin/python lower_service.py  < vhlo_artifact_bytes  > vmprogram_bytes

  stdin :  the PJRT_Program.code bytes (StableHLO/VHLO portable artifact)
  stdout:  VMProgram v0 binary (see vmprogram.py docstring)
  stderr:  on failure, one JSON object: {"error": <class>, "message": <str>}
  exit  :  0 ok; 2 unsupported program (NotImplementedError); 1 anything else

The C++ side treats exit 2 as "this program is valid but beyond current op
coverage" (surface a clean PJRT error) vs 1 as an internal lowering bug.
"""
import json
import sys


def main() -> int:
    artifact = sys.stdin.buffer.read()
    try:
        if not artifact:
            raise ValueError("empty input on stdin (expected VHLO artifact bytes)")
        import vmprogram
        prog = vmprogram.lower_artifact(artifact)
        blob = prog.serialize()
    except Exception as e:  # noqa: BLE001 - service boundary
        json.dump({"error": type(e).__name__, "message": str(e)}, sys.stderr)
        sys.stderr.write("\n")
        return 2 if isinstance(e, NotImplementedError) else 1
    sys.stdout.buffer.write(blob)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
