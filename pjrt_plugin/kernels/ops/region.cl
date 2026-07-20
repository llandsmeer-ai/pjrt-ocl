/* §27 register-resident map-region tile-op (INVESTIGATION PoC, off by default).
 *
 * Guarded by -DVMO_REGION_POC so the shipped megakernel is byte-identical unless
 * the flag is set. Proves the megakernel-native form of the §23/§24 fused
 * map-region: a run of pure-map micro-ops whose intermediates never leave the
 * lane — one global load per region input, interpret the straight-line sub-list
 * over per-thread float4 slots, one global store. No cross-workgroup barrier
 * (pure map: element i independent), so it composes with both engines.
 *
 * The whole point of §27 is OCCUPANCY: does this case fit under the megakernel's
 * co-residency register budget (≤128 regs/thread ⇒ 2 workgroups/SM ⇒ 376 lanes
 * on this 188-SM part)?  The slots live in a fixed float4[REGION_NSLOTS] array
 * scoped INSIDE the switch case, so its registers overlap the mutually-exclusive
 * matmul/reduce cases' registers (max-not-sum). Dynamic slot indexing lands the
 * array in per-thread local (stack) memory, which does not consume the register
 * file or SLM at all — occupancy-neutral by construction; the switch-addressed
 * register variant is measured separately (poc/15 case 99).
 *
 * Region inputs ride the task's own dst/a/b handles (loader-resolved like any op
 * — no aux handle patching, §20). Descriptor (int words at t.p0):
 *   [0]=in0_slot  [1]=in1_slot (0xFFFF = unused)  [2]=n_micro  [3]=out_slot
 *   n_micro × { kind, dst_slot, a_slot, b_slot, s_bits, t_bits }
 * t.p1 = element count n. Single-output v1. */
#ifdef VMO_REGION_POC
#ifndef REGION_NSLOTS
#define REGION_NSLOTS 8       /* §24 gate: n_slots ≤ 8 (register-file bound) */
#endif
#define REGION_NONE 0xFFFFu

static float4 vmo_region_micro(const uint kind, const float4 x, const float4 y,
                               const float s, const float t)
{
    switch (kind) {
    case SUB_ADD:    return x + y;
    case SUB_MUL:    return x * y;
    case SUB_SUB:    return x - y;
    case SUB_DIV:    return x / y;
    case SUB_TANH:   return tanh(x);
    case SUB_EXP:    return exp(x);
    case SUB_AFFINE: return mad(x, (float4)(s), (float4)(t));  /* x*s + t */
    default:         return x;
    }
}

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
#endif  /* VMO_REGION_POC */
