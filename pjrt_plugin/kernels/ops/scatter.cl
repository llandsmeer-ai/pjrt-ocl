/* Strided scatter tile op (TOP_SCATTER) — the mirror of gather. For each INPUT
 * element i (row-major over in_dims), write it to the output at an affine
 * position:  dst[out_off + sum_d idx_d(i)*out_stride_d] = a[i].
 * Used by concatenate / pad: each source is scattered into a disjoint region of
 * a preallocated output. aux at task.p0:
 *   rank u32, in_dims i32[rank], out_strides i32[rank], out_off i32
 * dtype-agnostic (copies whole elements): esz picks the width. */

static void scatter_tile(__global uchar *arena, __global const int *aux,
                         const task_t t, uint tile, uint esz, uint lid, uint lsz)
{
    __global const int *x = aux + t.p0;
    const int rank = x[0];
    __global const int *dims = x + 1, *strides = x + 1 + rank;
    const int out_off = x[1 + 2 * rank];
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
    if (esz == 8) {
        __global ulong *d = AP(ulong, t.dst);
        __global const ulong *a = AP(const ulong, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = out_off;
            for (int e = rank - 1; e >= 0; --e) { off += (rem % dims[e]) * strides[e]; rem /= dims[e]; }
            d[off] = a[i];
        }
    } else if (esz == 2) {
        __global ushort *d = AP(ushort, t.dst);
        __global const ushort *a = AP(const ushort, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = out_off;
            for (int e = rank - 1; e >= 0; --e) { off += (rem % dims[e]) * strides[e]; rem /= dims[e]; }
            d[off] = a[i];
        }
    } else if (esz == 1) {
        __global uchar *d = AP(uchar, t.dst);
        __global const uchar *a = AP(const uchar, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = out_off;
            for (int e = rank - 1; e >= 0; --e) { off += (rem % dims[e]) * strides[e]; rem /= dims[e]; }
            d[off] = a[i];
        }
    } else {
        __global int *d = AP(int, t.dst);
        __global const int *a = AP(const int, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = out_off;
            for (int e = rank - 1; e >= 0; --e) { off += (rem % dims[e]) * strides[e]; rem /= dims[e]; }
            d[off] = a[i];
        }
    }
}
