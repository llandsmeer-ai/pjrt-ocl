"""Register-resident fused map-region op (§23/§27/§28).

OP_MAP_REGION is emitted ONLY by lowering's `_fuse_region` recognizer (no
stablehlo op): it collapses a maximal run of pure-map f32 EW ops into one op
whose intermediates stay in per-thread float4 slots (one global load per input,
one store). This module registers the scheduler mapping (→ TILE_MAP_REGION), the
two validators (tensor interp + schedule simulator) and the dependency read-set.

Multi-input (§28 follow-up): a region takes up to 8 inputs, carried in the FULL
ordered `ins.region_inputs` tuple and mapped onto task fields a, b, p2..p7 (all
loader-patched to byte offsets / ports). This lets a multi-output connected
component split into one single-output sub-region per live output (its fan-in
cone) — the per-iteration EW chains of scan/loop bodies (HH neuron's
new_m/new_h/new_n/new_V) collapse into a few regions instead of ~78 tile-ops.

The numpy micro-op interpreter here MUST match ops/region.cl's vmo_region_micro
(same builtins) so both validators agree with the device on the re-parsed
bytecode. Descriptor layout (u32 words at ins.aux), mirroring the kernel:
    [0]=n_in  [1]=out_slot  [2]=n_micro
    [3 .. 3+n_in) = in_slot[k]     (slot each of the n_in inputs loads into)
    n_micro × { kind, dst_slot, a_slot, b_slot, s_bits, t_bits }
"""
from __future__ import annotations

import numpy as np

from .. import lowering as L
from .. import opsem
from ..scheduler import Task, TILE_MAP_REGION


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


def _run_region(aux, desc: int, ins_arrs: list[np.ndarray]) -> np.ndarray:
    """Interpret the micro-program at `aux[desc:]` over the ordered input arrays
    `ins_arrs` (one per region input). Returns the region output array (f32)."""
    n_in = aux[desc + 0]
    out_slot = aux[desc + 1]
    n_micro = aux[desc + 2]
    sbase = desc + 3
    R: dict[int, np.ndarray] = {}
    for k in range(n_in):
        R[aux[sbase + k]] = ins_arrs[k].astype(np.float32)
    o = sbase + n_in
    for _ in range(n_micro):
        kind = aux[o]
        ds, as_, bs = aux[o + 1], aux[o + 2], aux[o + 3]
        s, t = _f32(aux[o + 4]), _f32(aux[o + 5])
        R[ds] = _apply(kind, R[as_], R.get(bs), s, t).astype(np.float32)
        o += 6
    return R[out_slot]


# input handles ride task fields in this fixed order (matches ops/region.cl).
def _to_task(ins) -> Task:
    inp = list(getattr(ins, "region_inputs", ())) or [ins.a, ins.b]
    fields = [inp[k] if k < len(inp) else inp[0] for k in range(8)]
    a, b, p2, p3, p4, p5, p6, p7 = fields
    # p0 = descriptor word offset, p1 = element count. Loader resolves a/b/p2..p7.
    return Task(TILE_MAP_REGION, dst=ins.dst, a=a, b=b,
                p0=ins.aux, p1=ins.n, p2=p2, p3=p3, p4=p4, p5=p5, p6=p6, p7=p7)


def _reads(ins) -> set:
    return set(getattr(ins, "region_inputs", ())) or {ins.a, ins.b}


def _interp(ins, rt) -> None:                # validator a (tensor interpreter)
    # inputs from region_inputs when present (in-memory), else the trailing
    # handle words of the descriptor (survives serialize/re-parse).
    if getattr(ins, "region_inputs", ()):
        inp = list(ins.region_inputs)
    else:
        desc = ins.aux
        n_in = rt.aux[desc]
        n_micro = rt.aux[desc + 2]
        htail = desc + 3 + n_in + n_micro * 6
        inp = [rt.aux[htail + k] for k in range(n_in)]
    arrs = [rt.view(h, ins.n) for h in inp]
    rt.view(ins.dst, ins.n)[:] = _run_region(rt.aux, ins.aux, arrs)


def _sim(task, entry, rt) -> None:           # validator b (schedule simulator)
    n = task.p1
    ts = rt.tile_size
    lo = entry.tile_lo * ts
    hi = min(entry.tile_hi * ts, n)
    if lo >= hi:
        return
    n_in = rt.aux[task.p0]
    fields = [task.a, task.b, task.p2, task.p3, task.p4, task.p5, task.p6, task.p7]
    arrs = [rt.view(fields[k])[lo:hi] for k in range(n_in)]
    rt.view(task.dst)[lo:hi] = _run_region(rt.aux, task.p0, arrs)


opsem.register(L.OP_MAP_REGION, to_task=_to_task, interp=_interp, reads=_reads)
opsem.register_tile_sim(TILE_MAP_REGION, _sim)
