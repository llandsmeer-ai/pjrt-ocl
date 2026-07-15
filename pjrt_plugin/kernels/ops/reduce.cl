/* Two-phase reduction tile ops.
 * TOP_RED_PART: one partial per tile (task.p0=n, p1=chunk, p2=kind) via a
 *   workgroup local-memory tree reduce; writes arena[t.dst + tile].
 * TOP_RED_COMB: fold n_parts partials -> final (p0=n_parts, p1=kind).
 * kind: 0 sum, 1 max, 2 min, 3 prod.
 */

static void reduce_part_tile(__global float *arena, const task_t t, uint tile,
                             __local float *As, uint lid, uint lsz)
{
    const uint n = t.p0, chunk = t.p1, kind = t.p2;
    const uint lo = tile * chunk, hi = min(lo + chunk, n);
    float acc = kind == 0 ? 0.0f
              : kind == 1 ? -INFINITY
              : kind == 2 ? INFINITY : 1.0f;
    for (uint i = lo + lid; i < hi; i += lsz) {
        const float v = arena[t.a + i];
        acc = kind == 0 ? acc + v
            : kind == 1 ? fmax(acc, v)
            : kind == 2 ? fmin(acc, v) : acc * v;
    }
    As[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = lsz / 2; s > 0; s >>= 1) {
        if (lid < s) {
            const float a = As[lid], b = As[lid + s];
            As[lid] = kind == 0 ? a + b
                    : kind == 1 ? fmax(a, b)
                    : kind == 2 ? fmin(a, b) : a * b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) arena[t.dst + tile] = As[0];
}

static void reduce_comb_tile(__global float *arena, const task_t t, uint lid)
{
    if (lid != 0) return;
    const uint n = t.p0, kind = t.p1;
    float acc = kind == 0 ? 0.0f
              : kind == 1 ? -INFINITY
              : kind == 2 ? INFINITY : 1.0f;
    for (uint i = 0; i < n; ++i) {
        const float v = arena[t.a + i];
        acc = kind == 0 ? acc + v
            : kind == 1 ? fmax(acc, v)
            : kind == 2 ? fmin(acc, v) : acc * v;
    }
    arena[t.dst] = acc;
}
