/* Strided gather tile op (TOP_GATHER). aux at task.p0:
 *   rank, out_dims[rank], in_strides[rank], src_off  (all i32, elements).
 * Covers broadcast_in_dim / transpose / slice / reverse (via strides+src_off).
 */

/* Gather copies whole elements, so it is dtype-agnostic for a given element
 * size. `esz` (bytes) picks the mover: 4-byte types copy as uint bits; 8-byte
 * as ulong. dst/a are BYTE offsets. */
static void vmo_gather_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                        const task_t t, uint tile, uint esz, uint lid, uint lsz)
{
    __global const int *x = aux + t.p0;
    const int rank = x[0];
    __global const int *dims = x + 1, *strides = x + 1 + rank;
    const int src_off = x[1 + 2 * rank];
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
    /* Contiguous fast path (dynamic_slice / slice with unit stride): the
     * div/mod rank loop below costs ~20 cycles of serial ALU per ELEMENT and
     * dominated the chained bench (~30 us/op flat). 4-wide copy; vloadn only
     * requires element alignment, so a runtime-odd src_off is fine, and dst
     * (arena/port allocation, 16B-aligned) takes an aligned vstore4. */
    if (esz == 4 && rank == 1 && strides[0] == 1) {
        __global uint *d = AP(uint, t.dst);
        __global const uint *a = AP(const uint, t.a) + src_off;
        const uint lo4 = lo >> 2, hi4 = lo4 + ((hi - lo) >> 2);
        uint v = lo4 + lid;
        for (; v + lsz < hi4; v += 2u * lsz) {
            const uint4 x0 = vload4(v, a), x1 = vload4(v + lsz, a);
            vstore4(x0, v, d);
            vstore4(x1, v + lsz, d);
        }
        if (v < hi4) vstore4(vload4(v, a), v, d);
        for (uint j = lo + ((hi - lo) & ~3u) + lid; j < hi; j += lsz)
            d[j] = a[j];
        return;
    }
    if (esz == 8) {
        __global ulong *d = AP(ulong, t.dst);
        __global const ulong *a = AP(const ulong, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = src_off;
            for (int e = rank - 1; e >= 0; --e) {
                off += (rem % dims[e]) * strides[e];
                rem /= dims[e];
            }
            d[i] = a[off];
        }
    } else if (esz == 2) {
        __global ushort *d = AP(ushort, t.dst);
        __global const ushort *a = AP(const ushort, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = src_off;
            for (int e = rank - 1; e >= 0; --e) {
                off += (rem % dims[e]) * strides[e];
                rem /= dims[e];
            }
            d[i] = a[off];
        }
    } else if (esz == 1) {
        __global uchar *d = AP(uchar, t.dst);
        __global const uchar *a = AP(const uchar, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = src_off;
            for (int e = rank - 1; e >= 0; --e) {
                off += (rem % dims[e]) * strides[e];
                rem /= dims[e];
            }
            d[i] = a[off];
        }
    } else {
        __global uint *d = AP(uint, t.dst);
        __global const uint *a = AP(const uint, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = src_off;
            for (int e = rank - 1; e >= 0; --e) {
                off += (rem % dims[e]) * strides[e];
                rem /= dims[e];
            }
            d[i] = a[off];
        }
    }
}
