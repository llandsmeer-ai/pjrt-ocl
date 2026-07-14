"""pjrt_ocl python-side tests: VMProgram v1 golden layout, jax e2e via the
lower_service subprocess, error paths, and jax_plugins registration smoke.

Run with: /home/ubuntu/project/.venv/bin/python -m pytest tests/
(needs `pip install -e python/` once).
"""
from __future__ import annotations

import io
import json
import os
import struct
import subprocess
import sys
import zlib

import numpy as np
import pytest

import pjrt_ocl
import pjrt_ocl.lowering as L
import pjrt_ocl.vmreader as R

PYTHON = sys.executable

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def serialize_as_plugin_would_receive(fn, *args) -> bytes:
    """The exact PJRT_Program.code bytes: jax.jit lowering serialized to a VHLO
    portable artifact at the negotiated (current) version.

    Clones the module first via MLIR-bytecode round-trip:
    serialize_portable_artifact MUTATES its input to VHLO in place, which would
    corrupt jax's cached lowering in this process (poc/03 NOTES #2).
    """
    import jax
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import stablehlo

    module = jax.jit(fn).lower(*args).compiler_ir("stablehlo")
    buf = io.BytesIO()
    module.operation.write_bytecode(file=buf)
    clone = ir.Module.parse(buf.getvalue(), context=module.context)
    return stablehlo.serialize_portable_artifact(
        clone, stablehlo.get_current_version())


def run_service(artifact: bytes, direct_script: bool = False
                ) -> subprocess.CompletedProcess:
    """Invoke lower_service exactly as the C++ plugin will."""
    if direct_script:
        cmd = [PYTHON, pjrt_ocl.lower_service_path()]
    else:
        cmd = [PYTHON, "-m", "pjrt_ocl.lower_service"]
    return subprocess.run(cmd, input=artifact, capture_output=True, check=False)


def lower_via_service(fn, *args) -> R.Program:
    proc = run_service(serialize_as_plugin_would_receive(fn, *args))
    assert proc.returncode == 0, proc.stderr.decode()
    return R.parse(proc.stdout)


# ---------------------------------------------------------------------------
# (a) golden header/layout checks, raw struct.unpack — independent of vmreader
# ---------------------------------------------------------------------------


def _hand_built_program() -> L.VMProgram:
    """b3 = (b0 + b1) - const[1,2,3], all shape (3,) f32."""
    const = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    return L.VMProgram(
        arena_bytes=256,
        buffers=[L.Buffer(0, 12), L.Buffer(64, 12), L.Buffer(128, 12),
                 L.Buffer(192, 12)],
        inputs=[0, 1], outputs=[3],
        input_shapes=[(3,), (3,)], output_shapes=[(3,)],
        consts=[(2, const.tobytes())],
        instrs=[L.Instr(L.OP_ADD_F32, dst=3, a=0, b=1, n=3),
                L.Instr(L.OP_SUB_F32, dst=3, a=3, b=2, n=3)],
        main_len=2,
    )


def test_golden_layout_exact_bytes():
    """Byte-for-byte spec check of the writer against docs/vmprogram.md."""
    blob = _hand_built_program().serialize()

    expected = bytearray()
    # header (40B): magic, version, n_buffers, n_instrs, n_consts, main_len,
    #               n_inputs, n_outputs, arena_bytes u64
    expected += struct.pack("<IIIIIIIIQ", 0x314D5056, 1, 4, 2, 1, 2, 2, 1, 256)
    # buffer table: {arena_byte_offset u64, size_bytes u64, dtype u32, pad u32}
    for off in (0, 64, 128, 192):
        expected += struct.pack("<QQII", off, 12, 0, 0)
    # IO maps: inputs u32[2] (already 8B), outputs u32[1] + 4B pad
    expected += struct.pack("<II", 0, 1)
    expected += struct.pack("<I", 3) + b"\0" * 4
    # IO shapes: {rank u32, pad u32, dims u64[rank]} x (inputs then outputs)
    for _ in range(3):
        expected += struct.pack("<IIQ", 1, 0, 3)
    # const pool: {buffer_id u32, byte_len u32, data}, padded to 8B
    expected += struct.pack("<II", 2, 12)
    expected += np.array([1.0, 2.0, 3.0], dtype="<f4").tobytes() + b"\0" * 4
    # instructions: {op,dst,a,b,n,imm,pad,pad} u32 x8
    expected += struct.pack("<8I", 1, 3, 0, 1, 3, 0, 0, 0)  # ADD_F32
    expected += struct.pack("<8I", 3, 3, 3, 2, 3, 0, 0, 0)  # SUB_F32

    assert len(blob) == 40 + 4 * 24 + 16 + 48 + 24 + 2 * 32 == 288
    assert blob == bytes(expected)


def test_golden_layout_jax_lowered_add():
    """Field-level struct.unpack on the service output for a + b (f32[8])."""
    import jax.numpy as jnp
    x = jnp.zeros(8, jnp.float32)
    proc = run_service(serialize_as_plugin_would_receive(lambda a, b: a + b, x, x))
    assert proc.returncode == 0, proc.stderr.decode()
    blob = proc.stdout

    (magic, version, n_buffers, n_instrs, n_consts, main_len, n_inputs,
     n_outputs, arena_bytes) = struct.unpack_from("<IIIIIIIIQ", blob, 0)
    assert magic == 0x314D5056
    assert version == 1
    assert (n_buffers, n_instrs, n_consts, main_len) == (3, 1, 0, 1)
    assert (n_inputs, n_outputs) == (2, 1)
    assert arena_bytes == 3 * 64  # three f32[8] buffers, 64B-aligned slots
    pos = 40

    for i in range(n_buffers):
        off, size, dtype, pad = struct.unpack_from("<QQII", blob, pos)
        assert off == i * 64 and off % 64 == 0
        assert size == 8 * 4
        assert dtype == 0 and pad == 0
        pos += 24

    assert pos % 8 == 0
    assert struct.unpack_from("<II", blob, pos) == (0, 1)     # inputs map
    pos += 8                                                  # 2*u32, aligned
    assert struct.unpack_from("<I", blob, pos) == (2,)        # outputs map
    assert blob[pos + 4:pos + 8] == b"\0" * 4                 # pad to 8B
    pos += 8

    for _ in range(n_inputs + n_outputs):                     # IO shapes
        assert pos % 8 == 0
        rank, pad = struct.unpack_from("<II", blob, pos)
        assert (rank, pad) == (1, 0)
        (dim,) = struct.unpack_from("<Q", blob, pos + 8)
        assert dim == 8
        pos += 16

    assert pos % 8 == 0                                       # instructions
    assert struct.unpack_from("<8I", blob, pos) == (1, 2, 0, 1, 8, 0, 0, 0)
    pos += 32
    assert pos == len(blob)


def test_reader_rejects_bad_magic_and_version():
    blob = bytearray(_hand_built_program().serialize())
    good = bytes(blob)
    blob[0] ^= 0xFF
    with pytest.raises(R.FormatError, match="magic"):
        R.parse(bytes(blob))
    blob[0] ^= 0xFF
    struct.pack_into("<I", blob, 4, 2)  # version = 2
    with pytest.raises(R.FormatError, match="version"):
        R.parse(bytes(blob))
    with pytest.raises(R.FormatError, match="trailing"):
        R.parse(good + b"\0" * 8)


# ---------------------------------------------------------------------------
# (b) e2e: jax fn -> artifact -> lower_service subprocess -> vmreader numpy
#     interpreter == jax.jit exactly
# ---------------------------------------------------------------------------

def _f_add(a, b):
    return a + b


def _f_mul_sub(a, b, c):
    return a * b - c


def _f_sum_diff_prod(a, b):
    return (a + b) * (a - b)


def _f_with_const(a):
    import jax.numpy as jnp
    return a + jnp.asarray(np.arange(1, 9, dtype=np.float32))


E2E_CASES = [
    ("add", _f_add, [(8,), (8,)]),
    ("add_2d", _f_add, [(2, 3), (2, 3)]),
    ("mul_sub", _f_mul_sub, [(8,), (8,), (8,)]),
    ("sum_diff_prod", _f_sum_diff_prod, [(16,), (16,)]),
    ("with_const", _f_with_const, [(8,)]),
]


@pytest.mark.parametrize("name,fn,shapes", E2E_CASES,
                         ids=[c[0] for c in E2E_CASES])
def test_e2e_matches_jax(name, fn, shapes):
    import jax
    rng = np.random.default_rng(zlib.crc32(name.encode()))  # stable per case
    args = [rng.standard_normal(s).astype(np.float32) for s in shapes]

    prog = lower_via_service(fn, *args)
    assert prog.input_shapes == [tuple(s) for s in shapes]
    (got,) = R.execute(prog, args)

    want = np.asarray(jax.jit(fn)(*args))
    assert got.shape == want.shape
    np.testing.assert_array_equal(got, want)  # identical f32 op order => exact


def test_e2e_direct_script_invocation():
    """The C++ plugin execs the script by path, not -m: cover that mode."""
    import jax
    rng = np.random.default_rng(7)
    args = [rng.standard_normal(8).astype(np.float32) for _ in range(2)]
    artifact = serialize_as_plugin_would_receive(_f_add, *args)
    proc = run_service(artifact, direct_script=True)
    assert proc.returncode == 0, proc.stderr.decode()
    (got,) = R.execute(R.parse(proc.stdout), args)
    np.testing.assert_array_equal(got, np.asarray(jax.jit(_f_add)(*args)))


# ---------------------------------------------------------------------------
# (c) service error paths
# ---------------------------------------------------------------------------


def test_unsupported_op_exit2_json():
    import jax.numpy as jnp

    def f(x):
        return jnp.sum(x)  # stablehlo.reduce: beyond v1 coverage

    artifact = serialize_as_plugin_would_receive(f, jnp.zeros(8, jnp.float32))
    proc = run_service(artifact)
    assert proc.returncode == 2
    assert proc.stdout == b""
    err = json.loads(proc.stderr.decode())
    assert err["error"] == "LoweringError"
    assert "stablehlo.reduce" in err["message"]
    assert "stablehlo.add" in err["message"]  # known-ops list included


def test_empty_stdin_exit3_json():
    proc = run_service(b"")
    assert proc.returncode == 3
    assert proc.stdout == b""
    err = json.loads(proc.stderr.decode())
    assert err["error"] == "ValueError"


# ---------------------------------------------------------------------------
# (d) registration smoke: no built .so must not break jax
# ---------------------------------------------------------------------------


def test_initialize_without_plugin_so_does_not_crash_jax():
    env = {k: v for k, v in os.environ.items() if k != "PJRT_OCL_PLUGIN_PATH"}
    code = (
        "import jax, pjrt_ocl\n"          # jax import runs plugin discovery
        "pjrt_ocl.initialize()\n"          # explicit call must also be safe
        "print([d.platform for d in jax.devices()])\n"
    )
    proc = subprocess.run([PYTHON, "-c", code], env=env, capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "cpu" in proc.stdout
    if not os.path.isfile(pjrt_ocl._default_plugin_path()):
        assert "skipping registration" in proc.stderr  # graceful, logged


# ---------------------------------------------------------------------------
# (e) vmreader: WHILE / FILL / IOTA / LTS per spec (lowering doesn't emit
#     these yet; the reader+interpreter must still honor the format)
# ---------------------------------------------------------------------------


def test_vmreader_while_loop():
    """counter = 0; data = iota(8); while (counter < 5) { counter += 1;
    data += data; }  => counter == 5, data == iota(8) * 32."""
    prog = L.VMProgram(
        arena_bytes=320,
        buffers=[
            L.Buffer(0, 4),      # 0: counter scalar
            L.Buffer(64, 4),     # 1: const limit = 5.0
            L.Buffer(128, 4),    # 2: const one = 1.0
            L.Buffer(192, 4),    # 3: cond scalar
            L.Buffer(256, 32),   # 4: data f32[8]
        ],
        inputs=[], outputs=[0, 4],
        input_shapes=[], output_shapes=[(), (8,)],
        consts=[(1, np.float32(5.0).tobytes()),
                (2, np.float32(1.0).tobytes())],
        instrs=[
            # main list [0, 3)
            L.Instr(L.OP_FILL_F32, dst=0, n=1, imm=0),          # counter = 0.0
            L.Instr(L.OP_IOTA_F32, dst=4, n=8),                 # data = iota
            L.Instr(L.OP_WHILE, dst=3, a=3, b=1, n=4, imm=2),   # cond [3,4) body [4,6)
            # cond list
            L.Instr(L.OP_LTS_F32, dst=3, a=0, b=1),             # cond = counter < 5
            # body list
            L.Instr(L.OP_ADD_F32, dst=0, a=0, b=2, n=1),        # counter += 1
            L.Instr(L.OP_ADD_F32, dst=4, a=4, b=4, n=8),        # data += data
        ],
        main_len=3,
    )
    parsed = R.parse(prog.serialize())
    counter, data = R.execute(parsed, [])
    assert counter.shape == () and counter == np.float32(5.0)
    np.testing.assert_array_equal(
        data, np.arange(8, dtype=np.float32) * np.float32(32.0))
