/* Iota-along-dim tile op (TOP_IOTA_DIM). aux at task.p0:
 *   rank, out_dims[rank], dim.  dst[i] = coordinate of i along `dim`. */

static void vmo_iota_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                      const task_t t, uint tile, uint dt, uint lid, uint lsz)
{
    __global const int *x = aux + t.p0;
    const int rank = x[0];
    __global const int *dims = x + 1;
    const int dim = x[1 + rank];
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
    /* dtype-aware store: integer iota (int32/uint32/int64 — e.g. the threefry
     * RNG counter is ui64) writes the integer coordinate; float iota writes
     * (float)val. The pointers alias the same bytes; only one is used per dt. */
    const int as_i32 = (dt == DT_I32 || dt == DT_U32);
    const int as_i64 = (dt == DT_I64);
    __global float *df = AP(float, t.dst);
    __global int *di = AP(int, t.dst);
    __global long *dl = AP(long, t.dst);
    for (uint i = lo + lid; i < hi; i += lsz) {
        int rem = (int)i, val = 0;
        for (int e = rank - 1; e >= 0; --e) {
            const int idx = rem % dims[e];
            rem /= dims[e];
            if (e == dim) val = idx;
        }
        if (as_i64) dl[i] = (long)val; else if (as_i32) di[i] = val; else df[i] = (float)val;
    }
}
