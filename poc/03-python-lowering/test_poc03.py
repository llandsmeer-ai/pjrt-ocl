#!/usr/bin/env python
"""End-to-end sanity for poc/03: run with .venv/bin/python test_poc03.py

Pipeline under test (a + b*c, plus a constant to exercise the const pool):
  1. jax.jit(f).lower(...) -> the exact serialized bytes a plugin receives
     (VHLO portable artifact, reproduced per research.md)
  2. lower_service.py subprocess: artifact on stdin -> VMProgram binary on stdout
  3. vmprogram.parse() reads the binary back
  4. a numpy reference interpreter executes the VMProgram
  5. results compared against jax.jit's own (CPU backend) answer
Also checks the service's JSON-on-stderr error path with an unsupported program.
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import vmprogram  # noqa: E402


# ---------------------------------------------------------------------------
# Numpy reference interpreter for VMProgram (mirrors what the OpenCL VM will do)
# ---------------------------------------------------------------------------

def execute_vmprogram(prog: vmprogram.VMProgram, args: list[np.ndarray]) -> list[np.ndarray]:
    arena = np.zeros(prog.arena_size, dtype=np.uint8)

    def view(buf_id):
        b = prog.buffers[buf_id]
        return arena[b.offset:b.offset + b.size].view(vmprogram.DTYPE_NUMPY[b.dtype])

    # upload consts (once) and args (per execute), as the executor would
    for buf_id, data in prog.consts:
        b = prog.buffers[buf_id]
        arena[b.offset:b.offset + len(data)] = np.frombuffer(data, dtype=np.uint8)
    for i, b in enumerate(prog.buffers):
        if b.kind == vmprogram.ARG:
            view(i)[:] = np.ascontiguousarray(args[b.index], dtype=np.float32).ravel()

    def run_list(instrs):
        for ins in instrs:
            if ins.opcode == vmprogram.OP_ADD:
                view(ins.out)[:] = view(ins.a) + view(ins.b)
            elif ins.opcode == vmprogram.OP_MUL:
                view(ins.out)[:] = view(ins.a) * view(ins.b)
            elif ins.opcode == vmprogram.OP_SUB:
                view(ins.out)[:] = view(ins.a) - view(ins.b)
            elif ins.opcode == vmprogram.OP_COPY:
                view(ins.out)[:] = view(ins.a)
            else:
                raise NotImplementedError(f"opcode {ins.opcode:#x}")

    run_list(prog.lists[0])

    results = {b.index: view(i).copy()
               for i, b in enumerate(prog.buffers) if b.kind == vmprogram.RESULT}
    return [results[i] for i in sorted(results)]


# ---------------------------------------------------------------------------

VENV_PY = "/home/ubuntu/project/.venv/bin/python"


def run_service(artifact: bytes) -> subprocess.CompletedProcess:
    return subprocess.run([VENV_PY, os.path.join(HERE, "lower_service.py")],
                          input=artifact, capture_output=True, check=False)


def main() -> int:
    import jax
    import jax.numpy as jnp
    import dump_stablehlo

    K = np.arange(8, dtype=np.float32)

    def f(a, b, c):
        return (a + b * c) - jnp.asarray(K)  # add, multiply, subtract, constant

    rng = np.random.default_rng(0)
    a, b, c = (rng.standard_normal(8).astype(np.float32) for _ in range(3))

    # 1. the exact bytes a plugin would receive at PJRT_Client_Compile
    module = jax.jit(f).lower(jnp.asarray(a), jnp.asarray(b), jnp.asarray(c)) \
                       .compiler_ir("stablehlo")
    artifact = dump_stablehlo.serialize_as_plugin_would_receive(module, "current")
    assert artifact[:4] == b"ML\xefR", "not MLIR bytecode?"
    print(f"[1] artifact: {len(artifact)} bytes, producer {artifact[4:22]!r}")

    # 2. lowering subprocess (the plugin's exec interface)
    proc = run_service(artifact)
    assert proc.returncode == 0, f"service failed: {proc.stderr.decode()}"
    blob = proc.stdout
    print(f"[2] lower_service.py ok: VMProgram binary {len(blob)} bytes")

    # 3. parse the emitted VMProgram back
    prog = vmprogram.parse(blob)
    print("[3] parsed program:")
    print("    " + prog.dump().replace("\n", "\n    "))

    # 4. execute with the numpy reference interpreter
    (got,) = execute_vmprogram(prog, [a, b, c])

    # 5. compare against jax.jit's own result
    want = np.asarray(jax.jit(f)(a, b, c))
    np.testing.assert_allclose(got, want, rtol=0, atol=0)  # identical f32 op order
    print(f"[4] VMProgram result   : {got}")
    print(f"[5] jax.jit result     : {want}")
    print("    exact match (atol=0): OK")

    # 6. error path: unsupported program must yield JSON on stderr + exit 2
    wr_module = dump_stablehlo.lower_to_stablehlo_module("while_reduce")
    wr_artifact = dump_stablehlo.serialize_as_plugin_would_receive(wr_module)
    proc = run_service(wr_artifact)
    err = json.loads(proc.stderr.decode())
    assert proc.returncode == 2 and proc.stdout == b"", (proc.returncode, err)
    assert "stablehlo.reduce" in err["message"], err
    print(f"[6] unsupported-program path: exit=2, stderr JSON: {err}")

    print("\nPASS: poc/03 end-to-end (serialize -> subprocess lower -> parse -> "
          "numpy execute == jax.jit)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
