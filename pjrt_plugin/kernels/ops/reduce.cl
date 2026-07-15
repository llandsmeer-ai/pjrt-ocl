/* Two-phase reduction tile ops.
 * TOP_RED_PART: one partial per tile (task.p0=n, p1=chunk, p2=kind) via a
 *   workgroup local-memory tree reduce; writes arena[t.dst + tile].
 * TOP_RED_COMB: fold n_parts partials -> final (p0=n_parts, p1=kind).
 * kind: 0 sum, 1 max, 2 min, 3 prod.
 */

/* Integer (i32/u32) partial reduce: integer accumulation; max/min via
 * max()/min(); identities INT_MIN/INT_MAX. The local tree buffer `As` (float)
 * is aliased as int — same 4-byte storage, no numeric use. */
static void vmo_reduce_part_tile_i32(__global uchar *arena, const task_t t,
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

static void vmo_reduce_comb_tile_i32(__global uchar *arena, const task_t t,
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

static void vmo_reduce_part_tile(__global uchar *arena, const task_t t, uint tile,
                             __local float *As, uint dt, uint lid, uint lsz)
{
    if (dt == DT_I32 || dt == DT_U32) {
        vmo_reduce_part_tile_i32(arena, t, tile, As, lid, lsz);
        return;
    }
    const uint n = t.p0, chunk = t.p1, kind = t.p2;
    const uint lo = tile * chunk, hi = min(lo + chunk, n);
    __global const float *a = AP(const float, t.a);
    float acc = kind == 0 ? 0.0f
              : kind == 1 ? -INFINITY
              : kind == 2 ? INFINITY : 1.0f;
    for (uint i = lo + lid; i < hi; i += lsz) {
        const float v = a[i];
        acc = kind == 0 ? acc + v
            : kind == 1 ? fmax(acc, v)
            : kind == 2 ? fmin(acc, v) : acc * v;
    }
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

static void vmo_reduce_comb_tile(__global uchar *arena, const task_t t, uint dt,
                             uint lid)
{
    if (dt == DT_I32 || dt == DT_U32) {
        vmo_reduce_comb_tile_i32(arena, t, lid);
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
