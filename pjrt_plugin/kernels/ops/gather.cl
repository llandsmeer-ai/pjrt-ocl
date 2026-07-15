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
