/* Iota-along-dim tile op (TOP_IOTA_DIM). aux at task.p0:
 *   rank, out_dims[rank], dim.  dst[i] = coordinate of i along `dim`. */

static void vmo_iota_tile(__global uchar *arena, __global const int *aux,
                      const task_t t, uint tile, uint lid, uint lsz)
{
    __global const int *x = aux + t.p0;
    const int rank = x[0];
    __global const int *dims = x + 1;
    const int dim = x[1 + rank];
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
    __global float *d = AP(float, t.dst);
    for (uint i = lo + lid; i < hi; i += lsz) {
        int rem = (int)i, val = 0;
        for (int e = rank - 1; e >= 0; --e) {
            const int idx = rem % dims[e];
            rem /= dims[e];
            if (e == dim) val = idx;
        }
        d[i] = (float)val;
    }
}
