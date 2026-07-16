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

# These tests exercise LOWERING against the CPU backend. Once libpjrt_ocl.so
# is built, jax's plugin discovery would otherwise route tracing through the
# plugin itself (priority 500 > cpu). Must be set before jax is imported, and
# propagated to subprocess tests via os.environ.
os.environ["JAX_PLATFORMS"] = "cpu"

import pjrt_ocl
import pjrt_ocl.lowering as L
import pjrt_ocl.scheduler as S
import pjrt_ocl.vmreader as R

PYTHON = sys.executable

# deterministic device config for golden tests (independent of env)
GOLDEN_CFG = S.DeviceConfig(nlanes=8, costs={})

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


def both_validators(prog: R.Program, args) -> np.ndarray:
    """Run BOTH reference validators (semantic tensor interpreter + schedule
    lane simulator) and assert they agree; return the (shared) outputs."""
    sem = R.execute(prog, args)
    sch = R.execute_schedule(prog, args)
    assert len(sem) == len(sch)
    for a, b in zip(sem, sch):
        np.testing.assert_array_equal(a, b)
    return sem


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
    """Byte-for-byte spec check of the v3 writer against docs/vmprogram.md:
    exact bytes for the tensor sections, then struct.unpack of the schedule
    header + one task + one barrier entry."""
    prog = _hand_built_program()
    sched = S.schedule_program(prog, GOLDEN_CFG)
    blob = prog.serialize(sched)

    expected = bytearray()
    # header (48B): magic, version=3, n_buffers, n_instrs, n_consts, main_len,
    #               n_inputs, n_outputs, n_aux, pad, arena_bytes u64
    expected += struct.pack("<IIIIIIIIIIQ",
                            0x314D5056, 3, 4, 2, 1, 2, 2, 1, 0, 0, 256)
    # buffer table: {arena_byte_offset u64, size_bytes u64, dtype u32, pad u32}
    for off in (0, 64, 128, 192):
        expected += struct.pack("<QQII", off, 12, 0, 0)
    # IO maps: inputs u32[2] (already 8B), outputs u32[1] + 4B pad
    expected += struct.pack("<II", 0, 1)
    expected += struct.pack("<I", 3) + b"\0" * 4
    # IO shapes: {rank u32, pad u32, dims u64[rank]} x (inputs then outputs)
    for _ in range(3):
        expected += struct.pack("<IIQ", 1, 0, 3)
    # aux pool: n_aux = 0 -> nothing (8B-aligned already)
    # const pool: {buffer_id u32, byte_len u32, data}, padded to 8B
    expected += struct.pack("<II", 2, 12)
    expected += np.array([1.0, 2.0, 3.0], dtype="<f4").tobytes() + b"\0" * 4
    # instructions: {op,dst,a,b,n,imm,aux,pad1} u32 x8 (aux=0 for EW)
    expected += struct.pack("<8I", 1, 3, 0, 1, 3, 0, 0, 0)  # ADD_F32
    expected += struct.pack("<8I", 3, 3, 3, 2, 3, 0, 0, 0)  # SUB_F32

    tensor_len = 48 + 4 * 24 + 16 + 48 + 24 + 2 * 32
    assert tensor_len == 296
    assert blob[:tensor_len] == bytes(expected)

    # --- schedule sections ---
    FLAG_NONE = 0xFFFFFFFF
    BARRIER = 0xFFFFFFFE
    pos = tensor_len
    n_tasks, n_entries, n_flags, n_lanes = struct.unpack_from("<IIII", blob, pos)
    # add-then-sub => two dataflow levels, each one 1-tile EW task on lane 0;
    # one BARRIER between the levels (no trailing barrier after the last).
    assert (n_tasks, n_flags, n_lanes) == (2, 0, 8)
    assert n_entries == 3 + 7 * 1  # lane0: E,BAR,E ; lanes1-7: BAR
    pos += 16

    # task 0: EW add, dst=3 a=0 b=1 p0=0(add) p1=3(n_elems); p4/p5 (view) = 0
    t0 = struct.unpack_from("<10I", blob, pos)
    assert t0 == (S.TILE_EW, 3, 0, 1, S.EW_ADD, 3, 0, 0, 0, 0)
    pos += 40 * n_tasks  # skip past both tasks (task_t is 40B now)

    # lane table (n_lanes x {entry_off, entry_count}); lane 0 owns E,BAR,E = 3
    off0, cnt0 = struct.unpack_from("<II", blob, pos)
    assert (off0, cnt0) == (0, 3)
    pos += 16 * n_lanes

    # entries: first is task 0 on tiles [0,1); second is a BARRIER
    e0 = struct.unpack_from("<8I", blob, pos)
    assert e0 == (0, 0, 1, FLAG_NONE, 0, FLAG_NONE, 0, 0)
    e1 = struct.unpack_from("<8I", blob, pos + 32)
    assert e1 == (BARRIER, 0, 0, FLAG_NONE, 0, FLAG_NONE, 0, 0)

    assert len(blob) == tensor_len + 16 + 40 * n_tasks + 16 * n_lanes + 32 * n_entries


def test_golden_layout_jax_lowered_add():
    """Field-level struct.unpack on the service output for a + b (f32[8])."""
    import jax.numpy as jnp
    x = jnp.zeros(8, jnp.float32)
    proc = run_service(serialize_as_plugin_would_receive(lambda a, b: a + b, x, x))
    assert proc.returncode == 0, proc.stderr.decode()
    blob = proc.stdout

    (magic, version, n_buffers, n_instrs, n_consts, main_len, n_inputs,
     n_outputs, n_aux, hpad, arena_bytes) = struct.unpack_from(
         "<IIIIIIIIIIQ", blob, 0)
    assert magic == 0x314D5056
    assert version == 3
    assert (n_buffers, n_instrs, n_consts, main_len) == (3, 1, 0, 1)
    assert (n_inputs, n_outputs) == (2, 1)
    assert (n_aux, hpad) == (0, 0)
    # three f32[8] buffers, all live in the single phase (inputs pinned from the
    # start, output to the end) => three distinct 64B-aligned slots. The
    # liveness allocator (§16) assigns offsets by interval, not buffer-id order,
    # so assert the SET of offsets rather than off == i*64.
    assert arena_bytes == 3 * 64
    pos = 48

    offsets = []
    for i in range(n_buffers):
        off, size, dtype, pad = struct.unpack_from("<QQII", blob, pos)
        assert off % 64 == 0 and off + size <= arena_bytes
        assert size == 8 * 4
        assert dtype == 0 and pad == 0
        offsets.append(off)
        pos += 24
    assert sorted(offsets) == [0, 64, 128]        # a permutation of the 3 slots

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

    # aux pool empty (n_aux == 0), so const pool then instructions follow
    assert pos % 8 == 0                                       # instructions
    # {op,dst,a,b,n,imm,aux,pad1}: ADD_F32 dst=2 a=0 b=1 n=8, aux=0
    assert struct.unpack_from("<8I", blob, pos) == (1, 2, 0, 1, 8, 0, 0, 0)
    pos += 32

    # schedule sections: one EW add task, one dataflow level => a single phase
    # with NO barrier (trailing barrier omitted); task on lane 0, others empty.
    n_tasks, n_entries, n_flags, n_lanes = struct.unpack_from("<IIII", blob, pos)
    assert (n_tasks, n_flags, n_lanes) == (1, 0, 8)
    assert n_entries == 1           # one task entry on lane0, no barrier
    pos += 16
    # task_t is 10 u32 words (p4/p5 = MMA view offsets, 0 for this EW task).
    assert struct.unpack_from("<10I", blob, pos) == (S.TILE_EW, 2, 0, 1,
                                                     S.EW_ADD, 8, 0, 0, 0, 0)
    pos += 40 * n_tasks + 16 * n_lanes
    # the single entry = task 0 tiles [0,1)
    assert struct.unpack_from("<8I", blob, pos)[:3] == (0, 0, 1)
    pos += 32 * n_entries
    assert pos == len(blob)


def test_reader_rejects_bad_magic_and_version():
    prog = _hand_built_program()
    sched = S.schedule_program(prog, GOLDEN_CFG)
    blob = bytearray(prog.serialize(sched))
    good = bytes(blob)
    blob[0] ^= 0xFF
    with pytest.raises(R.FormatError, match="magic"):
        R.parse(bytes(blob))
    blob[0] ^= 0xFF
    struct.pack_into("<I", blob, 4, 2)  # version = 2 (want 3)
    with pytest.raises(R.FormatError, match="version"):
        R.parse(bytes(blob))
    # trailing garbage after the (complete) schedule sections is rejected
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
    # Integer-valued f32 inputs: every intermediate is exactly representable,
    # so XLA CPU's mul+add/sub FMA contraction (which changes rounding on
    # general data — see python/NOTES.md) cannot alter the result and the
    # jax.jit comparison is bit-exact.
    args = [rng.integers(-16, 16, size=s).astype(np.float32) for s in shapes]

    prog = lower_via_service(fn, *args)
    assert prog.input_shapes == [tuple(s) for s in shapes]
    assert prog.schedule is not None
    (got,) = both_validators(prog, args)      # semantic + schedule agree

    want = np.asarray(jax.jit(fn)(*args))
    assert got.shape == want.shape
    np.testing.assert_array_equal(got, want)  # identical f32 op order => exact


def test_e2e_real_data_matches_per_op_jax():
    """General real-valued data, compared exactly against *eager* jax (one XLA
    op at a time = the StableHLO per-op semantics our VM implements). jax.jit
    is NOT used as the oracle here: XLA CPU contracts multiply+subtract into
    an FMA under jit, giving 1-ULP differences (measured; see python/NOTES.md).
    """
    import jax.numpy as jnp
    rng = np.random.default_rng(zlib.crc32(b"real"))
    args = [rng.standard_normal(64).astype(np.float32) for _ in range(3)]

    prog = lower_via_service(_f_mul_sub, *args)
    (got,) = both_validators(prog, args)      # semantic + schedule agree

    eager = np.asarray(_f_mul_sub(*(jnp.asarray(x) for x in args)))
    np.testing.assert_array_equal(got, eager)


def test_e2e_direct_script_invocation():
    """The C++ plugin execs the script by path, not -m: cover that mode."""
    import jax
    rng = np.random.default_rng(7)
    args = [rng.standard_normal(8).astype(np.float32) for _ in range(2)]
    artifact = serialize_as_plugin_would_receive(_f_add, *args)
    proc = run_service(artifact, direct_script=True)
    assert proc.returncode == 0, proc.stderr.decode()
    (got,) = both_validators(R.parse(proc.stdout), args)
    np.testing.assert_array_equal(got, np.asarray(jax.jit(_f_add)(*args)))


# ---------------------------------------------------------------------------
# (c) service error paths
# ---------------------------------------------------------------------------


def test_unsupported_op_exit2_json():
    import jax.numpy as jnp
    import jax.lax as lax

    def f(x):
        # stablehlo.sort: beyond current coverage. (The example op here has been
        # updated repeatedly as coverage grew: sin -> concatenate -> sort.)
        return lax.sort(x)

    artifact = serialize_as_plugin_would_receive(f, jnp.zeros(8, jnp.float32))
    proc = run_service(artifact)
    assert proc.returncode == 2
    assert proc.stdout == b""
    err = json.loads(proc.stderr.decode())
    assert err["error"] == "LoweringError"
    assert "stablehlo.sort" in err["message"]
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
    env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
    # Point discovery at a nonexistent .so: initialize() must skip gracefully
    # and jax must fall back to a real backend (cpu, or gpu/cuda if a CUDA
    # jaxlib happens to be installed) — the point is no crash + graceful log.
    env["PJRT_OCL_PLUGIN_PATH"] = "/nonexistent/libpjrt_ocl.so"
    code = (
        "import jax, pjrt_ocl\n"          # jax import runs plugin discovery
        "pjrt_ocl.initialize()\n"          # explicit call must also be safe
        "print([d.platform for d in jax.devices()])\n"
    )
    proc = subprocess.run([PYTHON, "-c", code], env=env, capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert any(p in proc.stdout for p in ("cpu", "gpu", "cuda"))
    assert "not registered" in proc.stderr  # graceful, logged


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
    parsed = R.parse(prog.serialize())        # tensor-only (WHILE not schedulable)
    assert parsed.schedule is None
    counter, data = R.execute(parsed, [])
    assert counter.shape == () and counter == np.float32(5.0)
    np.testing.assert_array_equal(
        data, np.arange(8, dtype=np.float32) * np.float32(32.0))


# ---------------------------------------------------------------------------
# (f) scheduler: dataflow levels, co-scheduling, packing, config
# ---------------------------------------------------------------------------


def _tasks_in_phase(prog: R.Program):
    """Return, per barrier phase, the set of task ids executed in that phase."""
    sched = prog.schedule
    counts = [sum(1 for e in s if e.task == S.TASK_BARRIER)
              for s in sched.lane_streams]
    assert len(set(counts)) == 1, f"barrier counts differ: {counts}"
    n_phases = counts[0] + 1     # B barriers separate B+1 phases (no trailing)
    phases = [set() for _ in range(n_phases)]
    for stream in sched.lane_streams:
        p = 0
        for e in stream:
            if e.task == S.TASK_BARRIER:
                p += 1
            else:
                phases[p].add(e.task)
    return phases


def _f_two_independent(a, b, c, d):
    return a + b, c * d


def test_two_independent_ops_co_scheduled():
    """lambda a,b,c,d: (a+b, c*d) — the two ops share no data, so they MUST
    land in the same dataflow level (same barrier phase) and run on distinct
    lanes. Verifiable directly in the parsed schedule."""
    import jax
    rng = np.random.default_rng(1234)
    args = [rng.integers(-16, 16, size=(8,)).astype(np.float32)
            for _ in range(4)]

    prog = lower_via_service(_f_two_independent, *args)
    assert prog.schedule is not None
    assert len(prog.schedule.tasks) == 2

    phases = _tasks_in_phase(prog)
    # exactly ONE barrier phase containing BOTH tasks => co-scheduled
    assert len(phases) == 1, f"expected a single level, got {len(phases)} phases"
    assert phases[0] == {0, 1}

    # the two task entries are on different lanes within that phase
    lanes_with_tasks = [lane for lane, stream in enumerate(prog.schedule.lane_streams)
                        for e in stream if e.task in (0, 1)]
    assert len(lanes_with_tasks) == 2
    assert len(set(lanes_with_tasks)) == 2, "co-scheduled tasks must use 2 lanes"

    # both validators agree with eager jax (per-op semantics)
    got = both_validators(prog, args)
    want = [np.asarray(x) for x in jax.jit(_f_two_independent)(*args)]
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)


def test_scheduler_dependent_chain_serializes():
    """a*b - c : the subtract depends on the multiply, but both are elementwise
    over the same shape, so the dependency is LANE-LOCAL (output element i reads
    only input element i). The scheduler fuses them into one chain that runs on a
    lane per tile — ONE barrier phase, no cross-workgroup barrier between the two
    ops (that is the whole point: a same-index chain doesn't need a barrier)."""
    rng = np.random.default_rng(9)
    args = [rng.integers(-8, 8, size=(8,)).astype(np.float32) for _ in range(3)]
    prog = lower_via_service(_f_mul_sub, *args)
    phases = _tasks_in_phase(prog)
    assert len(phases) == 1
    assert phases[0] == {0, 1}
    # both tasks land on the SAME lane (the chain), in dependency order
    lanes = {lane for lane, s in enumerate(prog.schedule.lane_streams)
             for e in s if e.task in (0, 1)}
    assert len(lanes) == 1


def test_scheduler_nlanes_config_and_coverage():
    """PJRT_OCL_NLANES controls lane count; every lane carries the barrier
    sequence; a single big EW op spreads its tiles across lanes covering them
    exactly once (checked by execute_schedule's coverage assertions)."""
    # one add of 40000 elems => ceil(40000/16384) = 3 tiles
    prog = L.VMProgram(
        arena_bytes=3 * ((40000 * 4 + 63) // 64 * 64),
        buffers=[L.Buffer(0, 40000 * 4),
                 L.Buffer((40000 * 4 + 63) // 64 * 64, 40000 * 4),
                 L.Buffer(2 * ((40000 * 4 + 63) // 64 * 64), 40000 * 4)],
        inputs=[0, 1], outputs=[2],
        input_shapes=[(40000,), (40000,)], output_shapes=[(40000,)],
        consts=[],
        instrs=[L.Instr(L.OP_ADD_F32, dst=2, a=0, b=1, n=40000)],
        main_len=1,
    )
    cfg = S.DeviceConfig(nlanes=4, costs={})
    sched = S.schedule_program(prog, cfg)
    assert sched.n_lanes == 4
    # 3 tiles, cost-equal => 3 lanes get one tile each, lane 3 unused this level
    entries = [e for stream in sched.lane_streams for e in stream
               if e.task == 0]
    assert sorted((e.tile_lo, e.tile_hi) for e in entries) == [(0, 1), (1, 2), (2, 3)]
    # roundtrip + coverage validation via the reader
    parsed = R.parse(prog.serialize(sched))
    a = np.arange(40000, dtype=np.float32)
    b = np.arange(40000, dtype=np.float32) * 2
    (out,) = R.execute_schedule(parsed, [a, b])
    np.testing.assert_array_equal(out, a + b)


def _two_add_prog(n0: int, n1: int) -> L.VMProgram:
    """Two independent adds with element counts n0, n1 (co-schedule in 1 level).
    Buffers: in0,in1 (n0), in2,in3 (n1), out0 (n0), out1 (n1)."""
    sizes = [n0, n0, n1, n1, n0, n1]
    buffers, off = [], 0
    for n in sizes:
        buffers.append(L.Buffer(off, n * 4))
        off += (n * 4 + 63) // 64 * 64      # 64B-aligned slots
    return L.VMProgram(
        arena_bytes=off,
        buffers=buffers,
        inputs=[0, 1, 2, 3], outputs=[4, 5],
        input_shapes=[(n0,), (n0,), (n1,), (n1,)],
        output_shapes=[(n0,), (n1,)],
        consts=[],
        instrs=[L.Instr(L.OP_ADD_F32, dst=4, a=0, b=1, n=n0),
                L.Instr(L.OP_ADD_F32, dst=5, a=2, b=3, n=n1)],
        main_len=2,
    )


def test_scheduler_lane_allocation_proportional_to_cost():
    """When two independent EW ops co-schedule, chunks are allocated in
    proportion to cost (= tiles x unit cost) and LPT-balanced; a lane may
    carry entries of BOTH tasks (sequentialized) instead of each task owning
    dedicated lanes. 8 tiles vs 2 tiles over 8 lanes: the big op fans out to
    7 lanes, the small one rides on the remaining capacity, and no lane
    carries more than ceil(10 tiles / 8 lanes) = 2 (optimal makespan)."""
    prog = _two_add_prog(8 * S.TILE_SIZE, 2 * S.TILE_SIZE)
    sched = S.schedule_program(prog, S.DeviceConfig(nlanes=8, costs={}))
    lanes0 = {i for i, st in enumerate(sched.lane_streams)
              for e in st if e.task == 0}
    lanes1 = {i for i, st in enumerate(sched.lane_streams)
              for e in st if e.task == 1}
    assert len(lanes0) > len(lanes1)
    loads = [sum(e.tile_hi - e.tile_lo for e in st if e.task != S.TASK_BARRIER)
             for st in sched.lane_streams]
    assert max(loads) == 2                       # LPT: optimal makespan
    # tiles split contiguously and cover each task's tiles exactly once
    R.execute_schedule(
        R.parse(prog.serialize(sched)),
        [np.zeros(8 * S.TILE_SIZE, np.float32)] * 2 +
        [np.zeros(2 * S.TILE_SIZE, np.float32)] * 2)


def test_scheduler_sequentializes_cheap_tasks_with_cost_table():
    """The diamond regression (docs/decisions.md #1 trace findings): with a
    measured cost table where an MMA tile dwarfs an EW tile, the matmul must
    fan out over ALL lanes and the cheap elementwise ops must be stacked
    BEHIND its chunks on shared lanes — not hold dedicated lanes hostage."""
    import jax.numpy as jnp

    def f(a, b, c):
        m = a @ b
        s = c + c
        p = c * c
        q = s * p
        return q + m

    x = jnp.ones((256, 256), jnp.float32)
    artifact = serialize_as_plugin_would_receive(f, x, x, x)
    prog = L.lower_artifact(artifact)
    cfg = S.DeviceConfig(nlanes=8,
                         costs={"mma_tile_us": 25.0, "ew_tile_us": 1.0})
    sched = S.schedule_program(prog, cfg)
    mma_ids = {i for i, t in enumerate(sched.tasks) if t.tile_op == S.TILE_MMA}
    assert len(mma_ids) == 1
    # level 0 = every lane's entries before its first barrier
    level0 = [[] for _ in range(8)]
    for lane, st in enumerate(sched.lane_streams):
        for e in st:
            if e.task == S.TASK_BARRIER:
                break
            level0[lane].append(e)
    mma_lanes = {ln for ln in range(8)
                 if any(e.task in mma_ids for e in level0[ln])}
    assert mma_lanes == set(range(8))            # matmul on every lane
    for ln in range(8):                          # cheap EW rides along, never
        for e in level0[ln]:                     # displaces an MMA chunk
            if e.task not in mma_ids:
                assert any(e2.task in mma_ids for e2 in level0[ln])
    # correctness of the multi-entry-per-lane schedule
    args = [np.full((256, 256), 0.5, np.float32)] * 3
    outs = R.execute_schedule(R.parse(prog.serialize(sched)), args)
    exp = f(*[jnp.asarray(a) for a in args])
    np.testing.assert_allclose(outs[0].reshape(256, 256), np.asarray(exp),
                               rtol=1e-5)


def test_scheduler_equal_cost_even_split():
    prog = _two_add_prog(4 * S.TILE_SIZE, 4 * S.TILE_SIZE)
    sched = S.schedule_program(prog, S.DeviceConfig(nlanes=8, costs={}))
    lanes0 = sum(1 for st in sched.lane_streams for e in st if e.task == 0)
    lanes1 = sum(1 for st in sched.lane_streams for e in st if e.task == 1)
    assert lanes0 == lanes1 == 4


def test_device_config_from_env_and_cost_table(tmp_path):
    """PJRT_OCL_NLANES + PJRT_OCL_COST_TABLE parsing; missing file => 1.0."""
    cost = tmp_path / "cost.json"
    cost.write_text(json.dumps({"ew_tile_us": 2.5, "mma_tile_us": 40.0,
                                "gather_tile_us": 3.0, "reduce_tile_us": 5.0}))
    cfg = S.DeviceConfig.from_env(
        {"PJRT_OCL_NLANES": "16", "PJRT_OCL_COST_TABLE": str(cost)})
    assert cfg.nlanes == 16
    assert cfg.unit_cost(S.TILE_EW) == 2.5
    assert cfg.unit_cost(S.TILE_MMA) == 40.0
    assert cfg.unit_cost(S.TILE_GATHER) == 3.0
    assert cfg.unit_cost(S.TILE_REDUCE_PART) == 5.0

    # defaults: no env => 8 lanes, all unit costs 1.0
    d = S.DeviceConfig.from_env({})
    assert d.nlanes == 8
    assert d.unit_cost(S.TILE_EW) == 1.0
    assert d.unit_cost(S.TILE_MMA) == 1.0

    # missing file => all 1.0
    m = S.DeviceConfig.from_env({"PJRT_OCL_COST_TABLE": str(tmp_path / "no.json")})
    assert m.unit_cost(S.TILE_EW) == 1.0


# ---------------------------------------------------------------------------
# (g) control-flow scheduling seam (WHILE/IF): not implemented yet
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="region-op scheduling (WHILE/IF) is a Phase-1 seam; "
                         "scheduler raises ScheduleError until implemented")
def test_scheduler_while_loop():
    """Skeleton: once region-op scheduling lands, a WHILE tensor instr should
    schedule its cond/body region lists into per-lane sub-ranges and emit a
    WHILE control entry (task=0xFFFFFFFD) uniformly in every lane."""
    raise NotImplementedError


def test_scheduler_emits_while_control_entry():
    """A WHILE instruction now schedules to a uniform control entry per lane,
    with cond/body sub-ranges living beyond each lane's root_len (root_len rule,
    docs/vmprogram.md). Empty cond/body ranges are the degenerate but valid
    shape used here (a real while carries cond/body compute)."""
    # root: [WHILE] only (cond/body empty). cond buffer = buf 1 (const 0 -> exit
    # immediately). This exercises the scheduler's control-entry emission and
    # the root_len bookkeeping without needing a full jax program.
    prog = L.VMProgram(
        arena_bytes=64 * 2,
        buffers=[L.Buffer(0, 4), L.Buffer(64, 4)],
        inputs=[], outputs=[0],
        input_shapes=[], output_shapes=[()],
        consts=[(1, np.float32(0.0).tobytes())],
        instrs=[L.Instr(L.OP_WHILE, dst=1, a=1, b=0, n=1, imm=0)],
        main_len=1,
    )
    sched = S.schedule_program(prog, GOLDEN_CFG, allow_multilane_while=True)
    assert sched.root_lens is not None
    for lane, stream in enumerate(sched.lane_streams):
        root_len = sched.root_lens[lane]
        # every lane's root walk contains exactly one WHILE control entry
        root = stream[:root_len]
        whiles = [e for e in root if e.task == S.TASK_WHILE]
        assert len(whiles) == 1
        w = whiles[0]
        # cond/body sub-ranges (empty here) start at/after root_len
        assert w.tile_lo >= root_len
        assert w.wait_flag >= root_len
        assert w.signal_flag == 1              # cond buffer id (pre-patch)
