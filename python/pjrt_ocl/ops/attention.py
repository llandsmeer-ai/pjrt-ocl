"""Fused flash-attention via OP_FLASH_ATTN / TILE_FLASH_ATTN (§34).

The lowering peephole `_fuse_attention` recognizes the batched per-head
attention idiom  DOT(QKᵀ)·scale → softmax(-1) → DOT(AV)  and rewrites it to ONE
OP_FLASH_ATTN. This module supplies its VM semantics: the scheduler mapper, the
two validators (tensor interpreter + schedule simulator), and the read-set.

There is NO stablehlo handler here — flash-attention is never lowered directly
from an op; it only ever arises from the peephole over our own instruction
stream (robust to jaxlib idiom variation; hard-gated; PJRT_OCL_FLASH=0 reverts
to the decomposed DOT→softmax→DOT chain).

Encoding (mirrors ops/attention.cl):
  Instr: a=Q src, b=K src, imm2=V src, dst=out, n=H(=G), imm=T(=M),
         aux=descriptor word-offset.
  descriptor (9 words at ins.aux): [H, T, C, hd, scale_bits, causal, qv, kv, vv]
  Task:  a=Q, b=K, p0=V (loader-patched), dst=out, p1=H, p2=T, p3=aux-offset.

Q/K/V are read through the SAME strided view offsets (qv/kv/vv; +1, 0=contig)
the decomposed matmuls used, addressed with the matmul's own flat index — so the
fused numerics read byte-identical inputs to DOT1/DOT2 (validators mirror this).
"""
from __future__ import annotations

from .. import lowering as L
from .. import opsem
from ..scheduler import Task, TILE_FLASH_ATTN


def _hdr(rt, off):
    """(H, T, C, hd, scale, causal, qv, kv, vv) from the descriptor at `off`."""
    H = rt.aux_i32(off + 0)
    T = rt.aux_i32(off + 1)
    C = rt.aux_i32(off + 2)
    hd = rt.aux_i32(off + 3)
    scale = float(rt.f32_from_bits(rt.aux[off + 4]))
    causal = rt.aux_i32(off + 5)
    qv = rt.aux_i32(off + 6)
    kv = rt.aux_i32(off + 7)
    vv = rt.aux_i32(off + 8)
    return H, T, C, hd, scale, causal, qv, kv, vv


def _softmax_rows(scores):
    """Stable row softmax over the last axis (matches the online kernel's math
    up to float reassociation)."""
    import numpy as np
    m = scores.max(-1, keepdims=True)
    e = np.exp(scores - m)
    return e / e.sum(-1, keepdims=True)


def _attention(Q, Kb, Vb, scale, causal, T, C):
    """Q[H,T,hd], Kb[H,hd,C] (= DOT1 B operand), Vb[H,C,hd] (= DOT2 B operand).
    Returns O[H,T,hd]. Mirrors ops/attention.cl exactly."""
    import numpy as np
    scores = np.einsum("htd,hdc->htc", Q, Kb).astype(np.float32) * np.float32(scale)
    if causal:
        # queries are the last T positions of the C-length window
        qpos = np.arange(T)[:, None] + (C - T)
        kpos = np.arange(C)[None, :]
        scores = np.where(kpos > qpos, np.float32(-np.inf), scores)
    A = _softmax_rows(scores.astype(np.float32)).astype(np.float32)
    return np.einsum("htc,hcd->htd", A, Vb).astype(np.float32)


def _flash_to_task(ins) -> Task:
    # p0 = V buffer id (loader-patched to a byte offset); p1/p2 = H/T (n_tiles);
    # p3 = descriptor aux word-offset (read on-device, NOT patched).
    return Task(TILE_FLASH_ATTN, dst=ins.dst, a=ins.a, b=ins.b,
                p0=ins.imm2, p1=ins.n, p2=ins.imm, p3=ins.aux)


def _flash_reads(ins) -> set:
    # Q (a), K (b), V (imm2) — all read through folded views of these sources.
    return {ins.a, ins.b, ins.imm2}


def _flash_interp(ins, rt) -> None:
    """Tensor validator (a): full attention on the re-parsed bytecode, reading
    Q/K/V through their folded view descriptors exactly like the matmuls."""
    H, T, C, hd, scale, causal, qv, kv, vv = _hdr(rt, ins.aux)
    Q = rt.viewed(ins.a, H * T * hd, qv).reshape(H, T, hd)
    Kb = rt.viewed(ins.b, H * hd * C, kv).reshape(H, hd, C)     # DOT1 B[h,d,c]
    Vb = rt.viewed(ins.imm2, H * C * hd, vv).reshape(H, C, hd)  # DOT2 B[h,c,d]
    O = _attention(Q, Kb, Vb, scale, causal, T, C)
    rt.view(ins.dst, H * T * hd)[:] = O.reshape(-1)


def _flash_sim(task, entry, rt):
    """Schedule simulator (b): one (head g, query row m) per tile; tile = g*T+m.
    Same math as the interp, restricted to this entry's tile range."""
    off = task.p3
    H, T, C, hd, scale, causal, qv, kv, vv = _hdr(rt, off)
    # schedule-sim rt.view() takes only a buffer id (full buffer); use viewed()
    # ONLY for a folded operand (matches ops/dot.py's guarded pattern).
    Q = (rt.viewed(task.a, H * T * hd, qv) if qv
         else rt.view(task.a)).reshape(H, T, hd)
    Kb = (rt.viewed(task.b, H * hd * C, kv) if kv
          else rt.view(task.b)).reshape(H, hd, C)
    Vb = (rt.viewed(task.p0, H * C * hd, vv) if vv
          else rt.view(task.p0)).reshape(H, C, hd)
    out = rt.view(task.dst).reshape(H, T, hd)
    for tile in range(entry.tile_lo, min(entry.tile_hi, H * T)):
        g, m = tile // T, tile % T
        O = _attention(Q[g:g + 1], Kb[g:g + 1], Vb[g:g + 1],
                       scale, causal, T, C)
        out[g, m, :] = O[0, m, :]


opsem.register(L.OP_FLASH_ATTN, to_task=_flash_to_task, interp=_flash_interp,
               reads=_flash_reads)
opsem.register_tile_sim(TILE_FLASH_ATTN, _flash_sim)
