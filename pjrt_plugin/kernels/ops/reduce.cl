/* Two-phase reduction tile ops.
 * TOP_RED_PART: one partial per tile (task.p0=n, p1=chunk, p2=kind) via a
 *   workgroup local-memory tree reduce; writes arena[t.dst + tile].
 * TOP_RED_COMB: fold n_parts partials -> final (p0=n_parts, p1=kind).
 * kind: 0 sum, 1 max, 2 min, 3 prod.
 */

/* Integer (i32/u32) partial reduce: integer accumulation; max/min via
 * max()/min(); identities INT_MIN/INT_MAX. The local tree buffer `As` (float)
 * is aliased as int — same 4-byte storage, no numeric use. */
static void vmo_reduce_part_tile_i32(__global uchar *arena, __global uchar **iop, const task_t t,
                                 uint tile, __local float *As, uint lid,
                                 uint lsz)
{
    const uint n = t.p0, chunk = t.p1, kind = t.p2;
    const uint lo = tile * chunk, hi = min(lo + chunk, n);
    __global const int *a = AP(const int, t.a);
    __local int *Ai = (__local int *)As;
    int acc = kind == 0 ? 0 : kind == 1 ? INT_MIN : kind == 2 ? INT_MAX : 1;
    for (uint i = lo + lid; i < hi; i += lsz) {
        const int v = a[i];
        acc = kind == 0 ? acc + v
            : kind == 1 ? max(acc, v)
            : kind == 2 ? min(acc, v) : acc * v;
    }
    Ai[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = lsz / 2; s > 0; s >>= 1) {
        if (lid < s) {
            const int x = Ai[lid], y = Ai[lid + s];
            Ai[lid] = kind == 0 ? x + y
                    : kind == 1 ? max(x, y)
                    : kind == 2 ? min(x, y) : x * y;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) AP(int, t.dst)[tile] = Ai[0];
}

static void vmo_reduce_comb_tile_i32(__global uchar *arena, __global uchar **iop, const task_t t,
                                 uint lid)
{
    if (lid != 0) return;
    const uint n = t.p0, kind = t.p1;
    __global const int *a = AP(const int, t.a);
    int acc = kind == 0 ? 0 : kind == 1 ? INT_MIN : kind == 2 ? INT_MAX : 1;
    for (uint i = 0; i < n; ++i) {
        const int v = a[i];
        acc = kind == 0 ? acc + v
            : kind == 1 ? max(acc, v)
            : kind == 2 ? min(acc, v) : acc * v;
    }
    AP(int, t.dst)[0] = acc;
}

static void vmo_reduce_part_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                             __local float *As, uint dt, uint lid, uint lsz)
{
    if (dt == DT_I32 || dt == DT_U32) {
        vmo_reduce_part_tile_i32(arena, iop, t, tile, As, lid, lsz);
        return;
    }
    const uint n = t.p0, chunk = t.p1, kind = t.p2;
    const uint lo = tile * chunk, hi = min(lo + chunk, n);
    __global const float *a = AP(const float, t.a);
    const float init = kind == 0 ? 0.0f
                     : kind == 1 ? -INFINITY
                     : kind == 2 ? INFINITY : 1.0f;
    float acc = init;
#ifdef VMO_CPU_TILES
    /* CPU shape (poc/09 / decisions.md #11): contiguous chunk per WI, 8-lane
     * vector partials + horizontal combine. Reductions already reassociate
     * (the tree below), so vector partials don't change the contract. */
    {
        const uint cw = (hi - lo + lsz - 1) / lsz;
        const uint clo = min(lo + lid * cw, hi), chi = min(clo + cw, hi);
        float8 a8 = (float8)(init);
        uint i = clo;
        for (; i + 8u <= chi; i += 8u) {
            const float8 v = vload8(0, a + i);
            a8 = kind == 0 ? a8 + v
               : kind == 1 ? fmax(a8, v)
               : kind == 2 ? fmin(a8, v) : a8 * v;
        }
        float h[8] = {a8.s0, a8.s1, a8.s2, a8.s3, a8.s4, a8.s5, a8.s6, a8.s7};
        for (int j = 0; j < 8; ++j)
            acc = kind == 0 ? acc + h[j]
                : kind == 1 ? fmax(acc, h[j])
                : kind == 2 ? fmin(acc, h[j]) : acc * h[j];
        for (; i < chi; ++i) {
            const float v = a[i];
            acc = kind == 0 ? acc + v
                : kind == 1 ? fmax(acc, v)
                : kind == 2 ? fmin(acc, v) : acc * v;
        }
    }
#else
    for (uint i = lo + lid; i < hi; i += lsz) {
        const float v = a[i];
        acc = kind == 0 ? acc + v
            : kind == 1 ? fmax(acc, v)
            : kind == 2 ? fmin(acc, v) : acc * v;
    }
#endif
    As[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = lsz / 2; s > 0; s >>= 1) {
        if (lid < s) {
            const float x = As[lid], y = As[lid + s];
            As[lid] = kind == 0 ? x + y
                    : kind == 1 ? fmax(x, y)
                    : kind == 2 ? fmin(x, y) : x * y;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) AP(float, t.dst)[tile] = As[0];
}

/* TOP_RED_SEG: segmented reduce over the innermost `seg` elements —
 * out[o] = reduce(in[o*seg : (o+1)*seg]), for o in this tile's output range
 * (p0=n_out, p1=seg, p2=kind). One work-item per output element (grid-stride);
 * the per-segment reduce is serial (softmax/layernorm segments are small). This
 * is the partial-axis reduction the two-phase flat model can't express. */
static void vmo_redseg_tile(__global uchar *arena, __global uchar **iop, const task_t t,
                        uint tile, uint dt, uint lid, uint lsz)
{
    const uint n_out = t.p0, seg = t.p1, kind = t.p2;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n_out);
    if (dt == DT_I32 || dt == DT_U32) {
        __global const int *a = AP(const int, t.a);
        __global int *d = AP(int, t.dst);
        for (uint o = lo + lid; o < hi; o += lsz) {
            int acc = kind == 0 ? 0 : kind == 1 ? INT_MIN : kind == 2 ? INT_MAX : 1;
            for (uint j = o * seg; j < o * seg + seg; ++j) {
                const int v = a[j];
                acc = kind == 0 ? acc + v : kind == 1 ? max(acc, v)
                    : kind == 2 ? min(acc, v) : acc * v;
            }
            d[o] = acc;
        }
        return;
    }
    __global const float *a = AP(const float, t.a);
    __global float *d = AP(float, t.dst);
    for (uint o = lo + lid; o < hi; o += lsz) {
        float acc = kind == 0 ? 0.0f : kind == 1 ? -INFINITY
                  : kind == 2 ? INFINITY : 1.0f;
        for (uint j = o * seg; j < o * seg + seg; ++j) {
            const float v = a[j];
            acc = kind == 0 ? acc + v : kind == 1 ? fmax(acc, v)
                : kind == 2 ? fmin(acc, v) : acc * v;
        }
        d[o] = acc;
    }
}

static void vmo_reduce_comb_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint dt,
                             uint lid)
{
    if (dt == DT_I32 || dt == DT_U32) {
        vmo_reduce_comb_tile_i32(arena, iop, t, lid);
        return;
    }
    if (lid != 0) return;
    const uint n = t.p0, kind = t.p1;
    __global const float *a = AP(const float, t.a);
    float acc = kind == 0 ? 0.0f
              : kind == 1 ? -INFINITY
              : kind == 2 ? INFINITY : 1.0f;
    for (uint i = 0; i < n; ++i) {
        const float v = a[i];
        acc = kind == 0 ? acc + v
            : kind == 1 ? fmax(acc, v)
            : kind == 2 ? fmin(acc, v) : acc * v;
    }
    AP(float, t.dst)[0] = acc;
}
