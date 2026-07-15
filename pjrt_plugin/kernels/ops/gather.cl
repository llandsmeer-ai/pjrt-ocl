/* Strided gather tile op (TOP_GATHER). aux at task.p0:
 *   rank, out_dims[rank], in_strides[rank], src_off  (all i32, elements).
 * Covers broadcast_in_dim / transpose / slice / reverse (via strides+src_off).
 */

static void gather_tile(__global float *arena, __global const int *aux,
                        const task_t t, uint tile, uint lid, uint lsz)
{
    __global const int *x = aux + t.p0;
    const int rank = x[0];
    __global const int *dims = x + 1, *strides = x + 1 + rank;
    const int src_off = x[1 + 2 * rank];
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
    for (uint i = lo + lid; i < hi; i += lsz) {
        int rem = (int)i, off = src_off;
        for (int d = rank - 1; d >= 0; --d) {
            off += (rem % dims[d]) * strides[d];
            rem /= dims[d];
        }
        arena[t.dst + i] = arena[t.a + off];
    }
}
