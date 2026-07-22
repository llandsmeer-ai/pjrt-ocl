/* Two-phase reduction tile ops.
 * TOP_RED_PART: one partial per tile (task.p0=n, p1=chunk, p2=kind) via a
 *   workgroup local-memory tree reduce; writes arena[t.dst + tile].
 * TOP_RED_COMB: fold n_parts partials -> final (p0=n_parts, p1=kind).
 * kind: 0 sum, 1 max, 2 min, 3 prod, 4 and, 5 or, 6 xor.
 */

/* Integer/bool reduction kind table. f32 covers kinds 0..3 (sum/max/min/prod);
 * i32/u32/bool additionally cover the bitwise reducers 4 and / 5 or / 6 xor
 * (from stablehlo.and/or/xor — jp.all / jp.any over int or bool). `is_uns`
 * selects UNSIGNED compare + identities for u32 (signed max()/min() and the
 * INT_MIN/INT_MAX identities are WRONG for u32); the all-ones AND identity is
 * -1 signed / UINT_MAX unsigned (same bit pattern, so vmo_ired_ident returns
 * ~0 for both — the bits are what matter). bool (0/1, stored 1-byte) rides this
 * same int path; its caller loads uchar and masks the stored result to 0/1. */
static inline int vmo_ired_ident(uint kind, int is_uns)
{
    switch (kind) {
    case 0:  return 0;                                  /* sum */
    case 1:  return is_uns ? 0 : INT_MIN;               /* max */
    case 2:  return is_uns ? (int)UINT_MAX : INT_MAX;   /* min */
    case 3:  return 1;                                  /* prod */
    case 4:  return ~0;                                 /* and  (all ones) */
    case 5:  return 0;                                  /* or  */
    default: return 0;                                  /* xor */
    }
}

static inline int vmo_ired_comb(int a, int b, uint kind, int is_uns)
{
    switch (kind) {
    case 0:  return a + b;
    case 1:  return is_uns ? (int)max((uint)a, (uint)b) : max(a, b);
    case 2:  return is_uns ? (int)min((uint)a, (uint)b) : min(a, b);
    case 3:  return a * b;
    case 4:  return a & b;
    case 5:  return a | b;
    default: return a ^ b;
    }
}

/* Integer/bool partial reduce: integer accumulation via the kind table above.
 * The local tree buffer `As` (float) is aliased as int — same 4-byte storage,
 * no numeric use. dt selects unsigned (u32) and bool (1-byte load/store). */
static void vmo_reduce_part_tile_i32(__global uchar *arena, __global uchar **iop, const task_t t,
                                 uint tile, __local float *As, uint lid,
                                 uint lsz, uint dt)
{
    const uint n = t.p0, chunk = t.p1, kind = t.p2;
    const uint lo = tile * chunk, hi = min(lo + chunk, n);
    const int is_uns = (dt == DT_U32), is_bool = (dt == DT_BOOL);
    __global const int *a = AP(const int, t.a);
    __global const uchar *ab = AP(const uchar, t.a);
    __local int *Ai = (__local int *)As;
    int acc = vmo_ired_ident(kind, is_uns);
    for (uint i = lo + lid; i < hi; i += lsz) {
        const int v = is_bool ? (int)ab[i] : a[i];
        acc = vmo_ired_comb(acc, v, kind, is_uns);
    }
    Ai[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = lsz / 2; s > 0; s >>= 1) {
        if (lid < s)
            Ai[lid] = vmo_ired_comb(Ai[lid], Ai[lid + s], kind, is_uns);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        if (is_bool) AP(uchar, t.dst)[tile] = (uchar)(Ai[0] & 1);
        else AP(int, t.dst)[tile] = Ai[0];
    }
}

static void vmo_reduce_comb_tile_i32(__global uchar *arena, __global uchar **iop, const task_t t,
                                 uint lid, uint dt)
{
    if (lid != 0) return;
    const uint n = t.p0, kind = t.p1;
    const int is_uns = (dt == DT_U32), is_bool = (dt == DT_BOOL);
    __global const int *a = AP(const int, t.a);
    __global const uchar *ab = AP(const uchar, t.a);
    int acc = vmo_ired_ident(kind, is_uns);
    for (uint i = 0; i < n; ++i) {
        const int v = is_bool ? (int)ab[i] : a[i];
        acc = vmo_ired_comb(acc, v, kind, is_uns);
    }
    if (is_bool) AP(uchar, t.dst)[0] = (uchar)(acc & 1);
    else AP(int, t.dst)[0] = acc;
}

static void vmo_reduce_part_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                             __local float *As, uint dt, uint lid, uint lsz)
{
    if (dt == DT_I32 || dt == DT_U32 || dt == DT_BOOL) {
        vmo_reduce_part_tile_i32(arena, iop, t, tile, As, lid, lsz, dt);
    } else {
    /* no return above: a return preceding the barriers below breaks
     * PoCL 5.0 parallel-region formation (see vmo_redseg_tile note). */
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
}

/* TOP_RED_SEG: segmented reduce over the innermost `seg` elements —
 * out[o] = reduce(in[o*seg : (o+1)*seg]), for o in this tile's output range
 * (p0=n_out, p1=seg, p2=kind). One work-item per output element (grid-stride);
 * the per-segment reduce is serial (softmax/layernorm segments are small). This
 * is the partial-axis reduction the two-phase flat model can't express. */
/* TOP_RED_SEG: ONE segment per tile, reduced COLLABORATIVELY by the whole
 * workgroup via a local-memory tree — so n_out tiles spread over all lanes and
 * every workgroup is busy (thread-per-output starved on layernorm's n_out=512:
 * one workgroup). tile = segment index o; the reduction runs over the seg
 * contiguous inputs at [o*seg, (o+1)*seg). `As` is the workgroup's local tree. */
static void vmo_redseg_tile(__global uchar *arena, __global uchar **iop, const task_t t,
                        uint tile, uint dt, __local float *As, uint lid, uint lsz)
{
    /* NO early returns in here: a `return` on a path that precedes a
     * barrier() — even a workgroup-UNIFORM one (spec-legal) — crashes PoCL
     * 5.0's parallel-region formation at lazy kernel compile
     * ("region_entry_barrier != NULL", llvmopencl/Kernel.cc). Over-assigned
     * tiles (o >= n_out; the scheduler never emits them, this is defensive)
     * run the tree on init values and skip the store instead. */
    const uint n_out = t.p0, seg = t.p1, kind = t.p2;
    const uint o = tile;
    const int valid = o < n_out;
    const uint base = o * seg;
    /* NOTE: no early `return` here — a conditional return before the workgroup
     * barriers below makes PoCL's parallel-region analysis assert
     * (region_entry_barrier != NULL). All work-items in a workgroup share `tile`,
     * so the barriers are reached uniformly. */
    if (dt == DT_I32 || dt == DT_U32 || dt == DT_BOOL) {
        const int is_uns = (dt == DT_U32), is_bool = (dt == DT_BOOL);
        __global const int *a = AP(const int, t.a);
        __global const uchar *ab = AP(const uchar, t.a);
        __local int *Ai = (__local int *)As;
        int acc = vmo_ired_ident(kind, is_uns);
        if (valid)
            for (uint j = lid; j < seg; j += lsz) {
                const int v = is_bool ? (int)ab[base + j] : a[base + j];
                acc = vmo_ired_comb(acc, v, kind, is_uns);
            }
        Ai[lid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint s = lsz / 2; s > 0; s >>= 1) {
            if (lid < s)
                Ai[lid] = vmo_ired_comb(Ai[lid], Ai[lid + s], kind, is_uns);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        /* No trailing barrier: after the tree's final barrier only lid 0
         * reads As[0], and the next tile's As[lid] writes conflict with no
         * post-barrier read. (Also: a barrier as the last statement of this
         * switch case — right before vmo_exec_tiles' loop backedge — is what
         * crashed PoCL 5.0 region formation.) */
        if (lid == 0 && valid) {
            if (is_bool) AP(uchar, t.dst)[o] = (uchar)(Ai[0] & 1);
            else AP(int, t.dst)[o] = Ai[0];
        }
    } else {
        __global const float *a = AP(const float, t.a);
        /* p3 = dot mode (GEMV routing, ops/dot.py): the segment is a matrix
         * row, multiplied by the shared vector at t.b while accumulating. */
        __global const float *bv =
            t.p3 ? AP(const float, t.b) : (__global const float *)0;
        float acc = kind == 0 ? 0.0f : kind == 1 ? -INFINITY
                  : kind == 2 ? INFINITY : 1.0f;
#ifdef VMO_CPU_TILES
        /* CPU (lsz=1, one WI per segment): lid 0 reduces the WHOLE contiguous
         * segment with a float8 vector accumulator + scalar tail. Other lanes
         * (only exist if PJRT_OCL_CPU_LSZ overrides lsz>1) contribute the
         * identity, so the tree below stays correct for any lsz. Barrier-free
         * per-segment vectorization is the CPU win (poc/09 / decisions #12):
         * the collaborative tree is pure overhead when a work-group is one
         * serial CPU thread. Branch is loop-shaped (PoCL region-former #18). */
        if (valid && lid == 0) {
            float8 a8 = (float8)(kind == 0 ? 0.0f : kind == 1 ? -INFINITY
                                : kind == 2 ? INFINITY : 1.0f);
            uint j = 0;
            if (t.p3) {                       /* GEMV: row . vector, sum only */
                for (; j + 8u <= seg; j += 8u)
                    a8 += vload8(0, a + base + j) * vload8(0, bv + j);
            } else {
                for (; j + 8u <= seg; j += 8u) {
                    const float8 v = vload8(0, a + base + j);
                    a8 = kind == 0 ? a8 + v : kind == 1 ? fmax(a8, v)
                       : kind == 2 ? fmin(a8, v) : a8 * v;
                }
            }
            float h[8] = {a8.s0, a8.s1, a8.s2, a8.s3, a8.s4, a8.s5, a8.s6, a8.s7};
            for (int k = 0; k < 8; ++k)
                acc = (kind == 0 || t.p3) ? acc + h[k] : kind == 1 ? fmax(acc, h[k])
                    : kind == 2 ? fmin(acc, h[k]) : acc * h[k];
            for (; j < seg; ++j) {
                const float v = t.p3 ? a[base + j] * bv[j] : a[base + j];
                acc = (kind == 0 || t.p3) ? acc + v : kind == 1 ? fmax(acc, v)
                    : kind == 2 ? fmin(acc, v) : acc * v;
            }
        }
#else
        if (valid) {
            if (t.p3) {
                uint j = lid;
                if (!((seg & 3u) | (uint)((uintptr_t)(a + base) & 15u) |
                      (uint)((uintptr_t)bv & 15u))) {
                    const uint seg4 = seg >> 2;
                    for (uint v = lid; v < seg4; v += lsz)
                        acc += dot(vload4(v, a + base), vload4(v, bv));
                    j = seg;
                }
                for (; j < seg; j += lsz) acc += a[base + j] * bv[j];
            } else
            for (uint j = lid; j < seg; j += lsz) {
                const float v = a[base + j];
                acc = kind == 0 ? acc + v : kind == 1 ? fmax(acc, v)
                    : kind == 2 ? fmin(acc, v) : acc * v;
            }
        }
#endif
        As[lid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint s = lsz / 2; s > 0; s >>= 1) {
            if (lid < s) {
                const float x = As[lid], y = As[lid + s];
                As[lid] = kind == 0 ? x + y : kind == 1 ? fmax(x, y)
                        : kind == 2 ? fmin(x, y) : x * y;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0 && valid) AP(float, t.dst)[o] = As[0];
    }
}

/* Uniform grid-stride bound: ceil(seg/lsz)*lsz. EVERY seg-loop in the fused
 * norm kernels iterates to SEG_UNIFORM (not seg) so all work-items execute the
 * SAME number of iterations, guarding the body with `j < seg`. A DIVERGENT-trip
 * grid-stride loop (`j < seg`, lanes doing different counts) that follows a
 * barrier and performs a GLOBAL store is MISCOMPILED by PoCL 5.0's work-item
 * loop / parallel-region former — intermittent heap corruption (crash on I/O
 * ports, wrong values on the arena; non-deterministic, ~half of runs). The
 * uniform bound makes the post-barrier region's WI-loop structure identical
 * across lanes and is the §18-class fix (verified 16/16 stable on PoCL, was
 * ~50% crash before). vmo_redseg_tile dodged this by storing from lid 0 only. */
#define SEG_UNIFORM(seg, lsz) ((((seg) + (lsz) - 1u) / (lsz)) * (lsz))

/* TOP_SOFTMAX_SEG (§19): fused softmax over the innermost `seg` elements.
 * out[o*seg + j] = exp(x[o*seg+j] - max_j) / sum_j exp(x[o*seg+j] - max_j).
 * ONE segment per tile, reduced COLLABORATIVELY by the whole workgroup (like
 * vmo_redseg_tile): the seg-wide row is staged into `As` once, the max/sum
 * tree-reduces run in `Bs`, and the result is written once — one global read +
 * one global write, no intermediate global buffers. seg <= MMA_ASZ (gated in
 * the lowering recognizer). Barrier discipline follows §18: no `return` before
 * a barrier, no barrier as the last statement before the tile-loop backedge,
 * and every post-barrier grid-stride loop uses the UNIFORM bound (above). */
static void vmo_softmax_seg(__global uchar *arena, __global uchar **iop, const task_t t,
                        uint tile, __local float *As, __local float *Bs, uint lid, uint lsz)
{
    const uint n_out = t.p0, seg = t.p1;
    const uint o = tile;
    const int valid = o < n_out;          /* over-assigned tiles run the tree
                                           * on init values, skip the store. */
    const uint base = o * seg;
#ifdef VMO_CPU_TILES
    /* CPU (lsz=1): one WI owns the whole segment. Three barrier-free float8
     * passes (max, exp+sum, scale) reading global directly — no __local
     * staging, no tree. seg-wide row is L1-resident (<=1024 f32 = 4 KiB). */
    if (!valid || lid != 0) return;
    __global const float *a = AP(const float, t.a);
    __global float *dst = AP(float, t.dst);
    /* pass 1: max */
    float8 m8 = (float8)(-INFINITY);
    uint j = 0;
    for (; j + 8u <= seg; j += 8u) m8 = fmax(m8, vload8(0, a + base + j));
    float mx = fmax(fmax(fmax(m8.s0, m8.s1), fmax(m8.s2, m8.s3)),
                    fmax(fmax(m8.s4, m8.s5), fmax(m8.s6, m8.s7)));
    for (uint jj = j; jj < seg; ++jj) mx = fmax(mx, a[base + jj]);
    /* pass 2: exp(x - mx) -> dst, accumulate sum */
    float8 s8 = (float8)(0.0f);
    const float8 mx8 = (float8)(mx);
    j = 0;
    for (; j + 8u <= seg; j += 8u) {
        const float8 e = exp(vload8(0, a + base + j) - mx8);
        vstore8(e, 0, dst + base + j);
        s8 += e;
    }
    float sm = s8.s0 + s8.s1 + s8.s2 + s8.s3 + s8.s4 + s8.s5 + s8.s6 + s8.s7;
    for (uint jj = j; jj < seg; ++jj) {
        const float e = exp(a[base + jj] - mx);
        dst[base + jj] = e;
        sm += e;
    }
    /* pass 3: scale by 1/sum */
    const float8 inv8 = (float8)(1.0f / sm);
    j = 0;
    for (; j + 8u <= seg; j += 8u)
        vstore8(vload8(0, dst + base + j) * inv8, 0, dst + base + j);
    for (uint jj = j; jj < seg; ++jj) dst[base + jj] *= 1.0f / sm;
    return;
#else
    const uint jN = SEG_UNIFORM(seg, lsz);
    __global const float *a = AP(const float, t.a);
    __global float *dst = AP(float, t.dst);
    /* stage the segment row into local once (grid-stride if seg > lsz) */
    for (uint j = lid; j < jN; j += lsz)
        if (valid && j < seg)
            As[j] = a[base + j];
    barrier(CLK_LOCAL_MEM_FENCE);
    /* max reduce: per-lane partial over its strided slice, then a local tree */
    float m = -INFINITY;
    for (uint j = lid; j < jN; j += lsz)
        if (valid && j < seg)
            m = fmax(m, As[j]);
    Bs[lid] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = lsz / 2; s > 0; s >>= 1) {
        if (lid < s) Bs[lid] = fmax(Bs[lid], Bs[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float mx = Bs[0];
    barrier(CLK_LOCAL_MEM_FENCE);          /* all lanes read mx before Bs reuse */
    /* exp(x - max) in place, accumulating the per-lane partial sum */
    float sacc = 0.0f;
    for (uint j = lid; j < jN; j += lsz)
        if (valid && j < seg) {
            const float e = exp(As[j] - mx);
            As[j] = e;
            sacc += e;
        }
    Bs[lid] = sacc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = lsz / 2; s > 0; s >>= 1) {
        if (lid < s) Bs[lid] = Bs[lid] + Bs[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float sm = Bs[0];
    /* No trailing barrier: this read of Bs[0]/As[j] follows the sum tree's
     * final barrier; the next tile re-stages As and re-barriers before any
     * shared read (§18). Uniform-trip guarded store (see SEG_UNIFORM). */
    for (uint j = lid; j < jN; j += lsz)
        if (valid && j < seg)
            dst[base + j] = As[j] / sm;
#endif
}

/* TOP_LAYERNORM_SEG (§19): fused layernorm CORE over the innermost `seg` elems.
 * mu = mean(x), var = mean(x^2) - mu^2 (single pass), out = (x-mu)*rsqrt(var+eps).
 * The trailing per-channel affine (*g + b) stays as separate EW ops. eps rides
 * in as_float(t.p2). Same workgroup-per-segment collaborative structure as
 * vmo_softmax_seg; §18 barrier rules + the SEG_UNIFORM grid-stride bound apply. */
static void vmo_layernorm_seg(__global uchar *arena, __global uchar **iop, const task_t t,
                        uint tile, __local float *As, __local float *Bs, uint lid, uint lsz)
{
    const uint n_out = t.p0, seg = t.p1;
    const float eps = as_float(t.p2);
    const uint o = tile;
    const int valid = o < n_out;
    const uint base = o * seg;
#ifdef VMO_CPU_TILES
    /* CPU (lsz=1): one WI owns the whole segment. Two barrier-free float8
     * passes (sum+sumsq, then normalize) reading global directly. */
    if (!valid || lid != 0) return;
    {
        __global const float *a = AP(const float, t.a);
        __global float *dst = AP(float, t.dst);
        float8 p1 = (float8)(0.0f), p2 = (float8)(0.0f);
        uint j = 0;
        for (; j + 8u <= seg; j += 8u) {
            const float8 v = vload8(0, a + base + j);
            p1 += v;
            p2 += v * v;
        }
        float s1 = p1.s0 + p1.s1 + p1.s2 + p1.s3 + p1.s4 + p1.s5 + p1.s6 + p1.s7;
        float s2 = p2.s0 + p2.s1 + p2.s2 + p2.s3 + p2.s4 + p2.s5 + p2.s6 + p2.s7;
        for (uint jj = j; jj < seg; ++jj) {
            const float v = a[base + jj];
            s1 += v;
            s2 += v * v;
        }
        const float n = (float)seg;
        const float mu = s1 / n;
        const float rs = rsqrt(s2 / n - mu * mu + eps);
        const float8 mu8 = (float8)(mu), rs8 = (float8)(rs);
        j = 0;
        for (; j + 8u <= seg; j += 8u)
            vstore8((vload8(0, a + base + j) - mu8) * rs8, 0, dst + base + j);
        for (uint jj = j; jj < seg; ++jj)
            dst[base + jj] = (a[base + jj] - mu) * rs;
    }
    return;
#else
    const uint jN = SEG_UNIFORM(seg, lsz);
    __global const float *a = AP(const float, t.a);
    __global float *dst = AP(float, t.dst);
    for (uint j = lid; j < jN; j += lsz)
        if (valid && j < seg)
            As[j] = a[base + j];
    barrier(CLK_LOCAL_MEM_FENCE);
    /* single pass: per-lane partial sum and sum-of-squares */
    float s1 = 0.0f, s2 = 0.0f;
    for (uint j = lid; j < jN; j += lsz)
        if (valid && j < seg) {
            const float v = As[j];
            s1 += v;
            s2 += v * v;
        }
    Bs[lid] = s1;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = lsz / 2; s > 0; s >>= 1) {
        if (lid < s) Bs[lid] = Bs[lid] + Bs[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float sum = Bs[0];
    barrier(CLK_LOCAL_MEM_FENCE);          /* all lanes read sum before Bs reuse */
    Bs[lid] = s2;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = lsz / 2; s > 0; s >>= 1) {
        if (lid < s) Bs[lid] = Bs[lid] + Bs[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float sumsq = Bs[0];
    const float n = (float)seg;
    const float mu = sum / n;
    const float var = sumsq / n - mu * mu;
    const float rs = rsqrt(var + eps);
    /* No trailing barrier before the backedge (§18). Uniform-trip guarded store. */
    for (uint j = lid; j < jN; j += lsz)
        if (valid && j < seg)
            dst[base + j] = (As[j] - mu) * rs;
#endif
}

/* TOP_RED_STRIDED: partial-axis reduce over a contiguous interior/prefix axis
 * block. Input viewed (outer, red, inner); out[o*inner+i] = reduce_r
 * in[(o*red+r)*inner+i]. p0=n_out (outer*inner), p1=red, p2=inner (stride),
 * p3=kind. THREAD-PER-OUTPUT, EW-style: this tile covers outputs
 * [tile*EW_TS, min((tile+1)*EW_TS, n_out)), work-items grid-stride within it and
 * each fully reduces one output serially over `red` strided elements. No
 * workgroup barriers (dodges the PoCL 5.0 region-formation traps entirely). */
static void vmo_redstrided_tile(__global uchar *arena, __global uchar **iop, const task_t t,
                            uint tile, uint dt, uint lid, uint lsz)
{
    const uint n_out = t.p0, red = t.p1, inner = t.p2, kind = t.p3;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n_out);
    if (dt == DT_I32 || dt == DT_U32 || dt == DT_BOOL) {
        const int is_uns = (dt == DT_U32), is_bool = (dt == DT_BOOL);
        __global const int *a = AP(const int, t.a);
        __global const uchar *ab = AP(const uchar, t.a);
        __global int *d = AP(int, t.dst);
        __global uchar *db = AP(uchar, t.dst);
        for (uint g = lo + lid; g < hi; g += lsz) {
            const uint o = g / inner, i = g % inner;
            const uint base = o * red * inner + i;
            int acc = vmo_ired_ident(kind, is_uns);
            for (uint r = 0; r < red; ++r) {
                const int v = is_bool ? (int)ab[base + r * inner]
                                      : a[base + r * inner];
                acc = vmo_ired_comb(acc, v, kind, is_uns);
            }
            if (is_bool) db[g] = (uchar)(acc & 1);
            else d[g] = acc;
        }
    } else {
        __global const float *a = AP(const float, t.a);
        __global float *d = AP(float, t.dst);
        for (uint g = lo + lid; g < hi; g += lsz) {
            const uint o = g / inner, i = g % inner;
            const uint base = o * red * inner + i;
            float acc = kind == 0 ? 0.0f : kind == 1 ? -INFINITY
                      : kind == 2 ? INFINITY : 1.0f;
            for (uint r = 0; r < red; ++r) {
                const float v = a[base + r * inner];
                acc = kind == 0 ? acc + v : kind == 1 ? fmax(acc, v)
                    : kind == 2 ? fmin(acc, v) : acc * v;
            }
            d[g] = acc;
        }
    }
}

static void vmo_reduce_comb_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint dt,
                             uint lid)
{
    if (dt == DT_I32 || dt == DT_U32 || dt == DT_BOOL) {
        vmo_reduce_comb_tile_i32(arena, iop, t, lid, dt);
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
