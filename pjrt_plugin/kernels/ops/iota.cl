/* Iota-along-dim tile op (TOP_IOTA_DIM). aux at task.p0:
 *   rank, out_dims[rank], dim.  dst[i] = coordinate of i along `dim`. */

static void iota_tile(__global float *arena, __global const int *aux,
                      const task_t t, uint tile, uint lid, uint lsz)
{
    __global const int *x = aux + t.p0;
    const int rank = x[0];
    __global const int *dims = x + 1;
    const int dim = x[1 + rank];
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
    for (uint i = lo + lid; i < hi; i += lsz) {
        int rem = (int)i, val = 0;
        for (int d = rank - 1; d >= 0; --d) {
            const int idx = rem % dims[d];
            rem /= dims[d];
            if (d == dim) val = idx;
        }
        arena[t.dst + i] = (float)val;
    }
}
