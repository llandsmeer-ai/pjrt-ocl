"""Register-resident fused map-region op (§23/§27/§28).

OP_MAP_REGION is emitted ONLY by lowering's `_fuse_region` recognizer (no
stablehlo op): it collapses a maximal run of pure-map f32 EW ops into one op
whose intermediates stay in per-thread float4 slots (one global load per input,
one store). This module registers the scheduler mapping (→ TILE_MAP_REGION), the
two validators (tensor interp + schedule simulator) and the dependency read-set.

The numpy micro-op interpreter here MUST match ops/region.cl's vmo_region_micro
(same builtins) so both validators agree with the device on the re-parsed
bytecode. Descriptor layout (u32 words at ins.aux), mirroring the kernel:
    [0]=in0_slot  [1]=in1_slot (0xFFFF=unused)  [2]=n_micro  [3]=out_slot
    n_micro × { kind, dst_slot, a_slot, b_slot, s_bits, t_bits }
"""
from __future__ import annotations

import numpy as np

from .. import lowering as L
from .. import opsem
from ..scheduler import Task, TILE_MAP_REGION

# SUB_* region micro-op kinds (match vm_common.cl / lowering._REGION_KIND).
_REGION_NONE = 0xFFFF


def _f32(bits: int) -> np.float32:
    return np.frombuffer(np.uint32(bits & 0xFFFFFFFF).tobytes(), "<f4")[0]


def _apply(kind: int, x, y, s, t):
    """One micro-op over float32 ndarrays; matches ops/region.cl builtins."""
    if kind == 0:   return x + y                      # SUB_ADD
    if kind == 1:   return x * y                      # SUB_MUL
    if kind == 2:   return x - y                      # SUB_SUB
    if kind == 3:   return x / y                      # SUB_DIV
    if kind == 4:   return np.maximum(x, y)           # SUB_MAX
    if kind == 5:   return np.minimum(x, y)           # SUB_MIN
    if kind == 8:   return -x                         # SUB_NEG
    if kind == 9:   return np.exp(x)                  # SUB_EXP
    if kind == 10:  return np.log(x)                  # SUB_LOG
    if kind == 11:  return np.sqrt(x)                 # SUB_SQRT
    if kind == 12:  return (np.float32(1.0) / np.sqrt(x)).astype(np.float32)  # SUB_RSQRT
    if kind == 13:  return np.tanh(x)                 # SUB_TANH
    if kind == 14:  return np.abs(x)                  # SUB_ABS
    if kind == 40:  return x * s + t                  # SUB_AFFINE (a*s+t)
    raise NotImplementedError(f"region micro-op kind {kind}")


def _run_region(aux, desc: int, a: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    """Interpret the micro-program at `aux[desc:]` over f32 input arrays a[, b].
    Returns the region output array (f32), same length as a."""
    in0_slot = aux[desc + 0] & 0xFFFF
    in1_slot = aux[desc + 1] & 0xFFFF
    n_micro = aux[desc + 2]
    out_slot = aux[desc + 3]
    R: dict[int, np.ndarray] = {in0_slot: a.astype(np.float32)}
    if in1_slot != _REGION_NONE:
        R[in1_slot] = b.astype(np.float32)
    o = desc + 4
    for _ in range(n_micro):
        kind = aux[o]
        ds, as_, bs = aux[o + 1], aux[o + 2], aux[o + 3]
        s, t = _f32(aux[o + 4]), _f32(aux[o + 5])
        R[ds] = _apply(kind, R[as_], R.get(bs), s, t).astype(np.float32)
        o += 6
    return R[out_slot]


def _to_task(ins) -> Task:
    # a = in0, b = in1 (self-aliases a for single-input regions); p0 = descriptor
    # word offset, p1 = element count. Loader resolves a/b/dst handles.
    return Task(TILE_MAP_REGION, dst=ins.dst, a=ins.a, b=ins.b,
                p0=ins.aux, p1=ins.n)


def _reads(ins) -> set:
    return {ins.a, ins.b}


def _interp(ins, rt) -> None:                # validator a (tensor interpreter)
    a = rt.view(ins.a, ins.n)
    b = rt.view(ins.b, ins.n) if ins.b != ins.a else None
    rt.view(ins.dst, ins.n)[:] = _run_region(rt.aux, ins.aux, a, b)


def _sim(task, entry, rt) -> None:           # validator b (schedule simulator)
    n = task.p1
    ts = rt.tile_size
    lo = entry.tile_lo * ts
    hi = min(entry.tile_hi * ts, n)
    if lo >= hi:
        return
    a = rt.view(task.a)[lo:hi]
    b = rt.view(task.b)[lo:hi] if task.b != task.a else None
    rt.view(task.dst)[lo:hi] = _run_region(rt.aux, task.p0, a, b)


opsem.register(L.OP_MAP_REGION, to_task=_to_task, interp=_interp, reads=_reads)
opsem.register_tile_sim(TILE_MAP_REGION, _sim)
