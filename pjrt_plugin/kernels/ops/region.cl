/* §27/§28 register-resident fused map-region tile-op (TOP_MAP_REGION).
 *
 * The megakernel-native form of the §23/§24 fused map-region: a run of pure-map
 * micro-ops whose intermediates never leave the lane. One global load per region
 * input, interpret the straight-line micro sub-list over per-thread float4 slots,
 * one global store. No cross-workgroup barrier (pure map: element i independent),
 * so it composes with both engines (spin-barrier + host-dispatch) unchanged and
 * §18/§19a PoCL barrier rules never bite (there is no barrier).
 *
 * OCCUPANCY (§27, measured — GO): the slots live in a fixed float4[REGION_NSLOTS]
 * array scoped INSIDE the switch case, so its registers overlap the
 * mutually-exclusive matmul/reduce cases' registers (max-not-sum). Dynamic slot
 * indexing lands the array in per-thread local (stack) memory, which consumes
 * neither the register file nor SLM — the whole-kernel register max is unchanged
 * (88 regs, still 2 WG/SM = 376 lanes on the RTX PRO 6000). No __local declared
 * here: the shared As/Bs stay byte-identical to baseline.
 *
 * Region inputs ride the task's own dst/a/b handles (loader-resolved like any op
 * — no aux handle patching, §20). Descriptor (int words at t.p0):
 *   [0]=in0_slot  [1]=in1_slot (0xFFFF = unused)  [2]=n_micro  [3]=out_slot
 *   n_micro × { kind, dst_slot, a_slot, b_slot, s_bits, t_bits }
 * t.p1 = element count n. Single-output v1 (the lowering recognizer splits
 * over-budget / multi-output regions into single-output on-chip sub-regions).
 *
 * `kind` is a vmo_ew SUB_* opcode (a subset: the pure-map ALU the recognizer
 * emits). The builtins MUST match ops/ew.cl so the fused region is numerically
 * identical to the decomposed EW chain it replaces. */
#ifndef REGION_NSLOTS
#define REGION_NSLOTS 8       /* §24/§27 gate: n_slots ≤ 8 (register-file bound) */
#endif
#define REGION_NONE 0xFFFFu

/* vmo_region_micro (the shared straight-line map micro-op interpreter) now lives
 * in vm_common.cl (§33 R2c) so ops/mma.cl's store-epilogue can call it too. */

static void vmo_map_region(__global uchar *arena, __global uchar **iop,
                           __global const int *aux, const task_t t,
                           uint tile, uint lid, uint lsz)
{
    const uint desc = t.p0, n = t.p1;
    const uint in0_slot = (uint)aux[desc + 0];
    const uint in1_slot = (uint)aux[desc + 1];
    const uint n_micro  = (uint)aux[desc + 2];
    const uint out_slot = (uint)aux[desc + 3];
    const uint mbase = desc + 4u;

    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    const uint lo4 = lo >> 2, hi4 = lo4 + ((hi - lo) >> 2);
    __global float4 *d4 = (__global float4 *)VMO_BASE(t.dst);
    __global const float4 *a4 = (__global const float4 *)VMO_BASE(t.a);
    __global const float4 *b4 = (__global const float4 *)VMO_BASE(t.b);

    for (uint i = lo4 + lid; i < hi4; i += lsz) {
        float4 R[REGION_NSLOTS];
        R[in0_slot] = a4[i];
        if (in1_slot != REGION_NONE) R[in1_slot] = b4[i];
        for (uint m = 0; m < n_micro; ++m) {
            const uint o  = mbase + m * 6u;
            const uint k  = (uint)aux[o];
            const uint ds = (uint)aux[o + 1];
            const uint as = (uint)aux[o + 2];
            const uint bs = (uint)aux[o + 3];
            R[ds] = vmo_region_micro(k, R[as], R[bs],
                                     as_float(aux[o + 4]), as_float(aux[o + 5]));
        }
        d4[i] = R[out_slot];
    }
    /* scalar tail (elements past the last full float4 of this tile) */
    __global float *d = (__global float *)VMO_BASE(t.dst);
    __global const float *a = (__global const float *)VMO_BASE(t.a);
    __global const float *b = (__global const float *)VMO_BASE(t.b);
    for (uint j = lo + ((hi - lo) & ~3u) + lid; j < hi; j += lsz) {
        float R1[REGION_NSLOTS];
        R1[in0_slot] = a[j];
        if (in1_slot != REGION_NONE) R1[in1_slot] = b[j];
        for (uint m = 0; m < n_micro; ++m) {
            const uint o  = mbase + m * 6u;
            const uint k  = (uint)aux[o];
            const uint ds = (uint)aux[o + 1], as = (uint)aux[o + 2],
                       bs = (uint)aux[o + 3];
            const float4 v = vmo_region_micro(k, (float4)(R1[as]),
                                              (float4)(R1[bs]),
                                              as_float(aux[o + 4]),
                                              as_float(aux[o + 5]));
            R1[ds] = v.x;
        }
        d[j] = R1[out_slot];
    }
}
