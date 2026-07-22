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
 * Region inputs ride the task's own handle fields, loader-resolved like any op
 * (no aux handle patching, §20): up to REGION_MAXIN inputs in the fixed order
 * a, b, p2, p3, p4, p5, p6, p7 (the loader patches all of them to byte offsets /
 * I/O ports for a kTopMapRegion task). Descriptor (int words at t.p0):
 *   [0]=n_in  [1]=out_slot  [2]=n_micro
 *   [3 .. 3+n_in) = in_slot[k]   (slot each of the n_in inputs loads into)
 *   n_micro × { kind, dst_slot, a_slot, b_slot, s_bits, t_bits }
 * t.p1 = element count n. Single-output (the recognizer splits a multi-output
 * connected component into one single-output sub-region per live output — its
 * fan-in cone — each ≤ REGION_NSLOTS slots and ≤ REGION_MAXIN inputs, §28
 * follow-up: this is what lets the long per-iteration EW chains of scan/loop
 * bodies collapse — new_m/new_h/new_n/new_V of the HH neuron become 4 regions).
 *
 * `kind` is a vmo_ew SUB_* opcode (a subset: the pure-map ALU the recognizer
 * emits). The builtins MUST match ops/ew.cl so the fused region is numerically
 * identical to the decomposed EW chain it replaces. */
#ifndef REGION_NSLOTS
#define REGION_NSLOTS 8       /* §24/§27 gate: n_slots ≤ 8 (register-file bound) */
#endif
#define REGION_MAXIN 8        /* a, b, p2, p3, p4, p5, p6, p7 */
#define REGION_NONE 0xFFFFu

/* vmo_region_micro (the shared straight-line map micro-op interpreter) now lives
 * in vm_common.cl (§33 R2c) so ops/mma.cl's store-epilogue can call it too. */

static void vmo_map_region(__global uchar *arena, __global uchar **iop,
                           __global const int *aux, const task_t t,
                           uint tile, uint lid, uint lsz)
{
    const uint desc = t.p0, n = t.p1;
    const uint n_in     = (uint)aux[desc + 0];
    const uint out_slot = (uint)aux[desc + 1];
    const uint n_micro  = (uint)aux[desc + 2];
    const uint sbase = desc + 3u;               /* in_slot[] */
    const uint mbase = sbase + n_in;            /* micro-program */
    /* input handles in fixed order; the loader patched all REGION_MAXIN of
     * them, so unused tail entries just alias in0 (never read: k < n_in). */
    const uint inoff[REGION_MAXIN] =
        { t.a, t.b, t.p2, t.p3, t.p4, t.p5, t.p6, t.p7 };

    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    const uint lo4 = lo >> 2, hi4 = lo4 + ((hi - lo) >> 2);
    __global float4 *d4 = (__global float4 *)VMO_BASE(t.dst);

    for (uint i = lo4 + lid; i < hi4; i += lsz) {
        float4 R[REGION_NSLOTS];
        for (uint k = 0; k < n_in; ++k)
            R[(uint)aux[sbase + k]] =
                ((__global const float4 *)VMO_BASE(inoff[k]))[i];
        for (uint m = 0; m < n_micro; ++m) {
            const uint o  = mbase + m * 6u;
            const uint kd = (uint)aux[o];
            const uint ds = (uint)aux[o + 1];
            const uint as = (uint)aux[o + 2];
            const uint bs = (uint)aux[o + 3];
            R[ds] = vmo_region_micro(kd, R[as], R[bs],
                                     as_float(aux[o + 4]), as_float(aux[o + 5]));
        }
        d4[i] = R[out_slot];
    }
    /* scalar tail (elements past the last full float4 of this tile) */
    __global float *d = (__global float *)VMO_BASE(t.dst);
    for (uint j = lo + ((hi - lo) & ~3u) + lid; j < hi; j += lsz) {
        float R1[REGION_NSLOTS];
        for (uint k = 0; k < n_in; ++k)
            R1[(uint)aux[sbase + k]] =
                ((__global const float *)VMO_BASE(inoff[k]))[j];
        for (uint m = 0; m < n_micro; ++m) {
            const uint o  = mbase + m * 6u;
            const uint kd = (uint)aux[o];
            const uint ds = (uint)aux[o + 1], as = (uint)aux[o + 2],
                       bs = (uint)aux[o + 3];
            const float4 v = vmo_region_micro(kd, (float4)(R1[as]),
                                              (float4)(R1[bs]),
                                              as_float(aux[o + 4]),
                                              as_float(aux[o + 5]));
            R1[ds] = v.x;
        }
        d[j] = R1[out_slot];
    }
}
