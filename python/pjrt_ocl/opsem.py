"""Shared per-tensor-opcode semantics registry.

Each op family module in ``pjrt_ocl.ops.*`` registers, for the tensor opcode(s)
it introduces:

- ``to_task(ins) -> scheduler.Task``   : how the scheduler tiles it
- ``interp(ins, rt)``                  : numpy reference semantics (validator a)
- ``reads(ins) -> set[int]``           : buffer ids read (dependency analysis;
                                          default {a, b})

``scheduler.py`` and ``vmreader.py`` consult these registries so op families can
be added as standalone modules without editing the core files. Built-in EW ops
(add/mul/sub/fill) keep fast paths in the core files; everything else routes
through here.

``rt`` passed to ``interp`` is a small facade exposing:
  ``rt.view(buf_id, n=None) -> np.ndarray`` (f32 view into the arena),
  ``rt.aux`` (list[int], read as signed via ``rt.aux_i32(off)``),
  ``rt.f32_from_bits(imm)``.
"""
from __future__ import annotations

from typing import Callable

# opcode -> callable(ins) -> Task
TO_TASK: dict[int, Callable] = {}
# opcode -> callable(ins, rt) -> None   (writes results into rt.view(...))
INTERP: dict[int, Callable] = {}
# opcode -> callable(ins) -> set[int]   (read buffer ids; default {a, b})
READS: dict[int, Callable] = {}
# tile_op -> callable(task, entry, rt) -> None  (schedule simulator, validator
# b): execute the tiles [entry.tile_lo, entry.tile_hi) of `task`. `rt` exposes
# view/aux/aux_i32/f32_from_bits plus rt.tile_size. Register alongside
# TO_TASK/INTERP for a new tile op.
TILE_SIM: dict[int, Callable] = {}


def register(opcode: int, *, to_task: Callable, interp: Callable,
             reads: Callable | None = None) -> None:
    TO_TASK[opcode] = to_task
    INTERP[opcode] = interp
    if reads is not None:
        READS[opcode] = reads


def register_tile_sim(tile_op: int, sim: Callable) -> None:
    """Register a schedule-simulator (validator b) for a tile op."""
    TILE_SIM[tile_op] = sim


# EW subop -> callable(a, b, task, rt, lo, hi) -> ndarray for output[lo:hi].
# a/b are the operand f32 views already sliced to [lo, hi) (b is None for unary
# subops); task carries p2/p3; rt is the _SchedRT facade and lo/hi the global
# element range (select reads rt.view(task.p3)[lo:hi]). Lets the elementwise
# family add subops without editing the core EW simulator; ADD/MUL/SUB/FILL keep
# builtin fast paths.
EW_SIM: dict[int, Callable] = {}


def register_ew_sim(subop: int, sim: Callable) -> None:
    EW_SIM[subop] = sim


def reads_of(ins) -> set[int]:
    fn = READS.get(ins.op)
    if fn is not None:
        return fn(ins)
    return {ins.a, ins.b}
