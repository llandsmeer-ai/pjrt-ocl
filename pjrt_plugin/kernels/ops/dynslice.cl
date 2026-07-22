/* Dynamic slice/update tile ops — gather/scatter with a RUNTIME base offset.
 *
 * Unlike TOP_GATHER/TOP_SCATTER (whose src/out base is a compile-time aux
 * constant), the start offsets here are SCALAR BUFFERS: the tile op reads
 * them, clamps to the legal range, and forms the affine base offset at
 * runtime. Their locations ride in the aux pool as idx_byteoff[rank], patched
 * at LOAD time by the loader from idx_bufid[rank] (lowering can't know them:
 * its reuse pass moves arena offsets, and an input scalar may be an I/O port).
 * Each patched word is an arena byte offset or a bit-31 port handle — AP()
 * resolves both.
 *
 * aux at task.p0 (identical layout for both ops; `dims`/`strides` name the
 * ITERATED space — output for gather, update for scatter):
 *   rank i32,
 *   dims i32[rank],          out_dims (slice sizes) / upd_dims
 *   strides i32[rank],       in_strides / out_strides (element strides)
 *   clamp_max i32[rank],     max legal start per axis (dim - size)
 *   idx_byteoff i32[rank],   byte offset of each start scalar in the arena
 *   idx_bufid i32[rank],     buffer id (python validators only; unused here)
 *   is64 i32                 0 = i32 start scalars, 1 = i64
 *
 * TOP_DYN_GATHER (dynamic_slice):  dst[i] = a[base + affine(i)]
 * TOP_DYN_SCATTER (dynamic_update_slice): dst[base + affine(i)] = a[i]
 *   with base = sum_d clamp(start_d, 0, clamp_max_d) * strides_d.
 * Both copy whole elements, so they are dtype-agnostic (esz picks the mover).
 */

static int vmo_dyn_base(__global uchar *arena, __global uchar **iop, __global const int *x, int rank,
                    int is64)
{
    __global const int *strides = x + 1 + rank;
    __global const int *clampmax = x + 1 + 2 * rank;
    __global const int *byteoff = x + 1 + 3 * rank;
    int base = 0;
    for (int d = 0; d < rank; ++d) {
        int s = is64 ? (int)AP(const long, byteoff[d])[0]
                     : AP(const int, byteoff[d])[0];
        s = s < 0 ? 0 : (s > clampmax[d] ? clampmax[d] : s);
        base += s * strides[d];
    }
    return base;
}

#define DYN_GATHER_BODY(T)                                                    \
    do {                                                                      \
        __global T *d = AP(T, t.dst);                                         \
        __global const T *a = AP(const T, t.a);                               \
        for (uint i = lo + lid; i < hi; i += lsz) {                           \
            int rem = (int)i, off = base;                                     \
            for (int e = rank - 1; e >= 0; --e) {                             \
                off += (rem % dims[e]) * strides[e]; rem /= dims[e];          \
            }                                                                 \
            d[i] = a[off];                                                    \
        }                                                                     \
    } while (0)

static void vmo_dyn_gather_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                            const task_t t, uint tile, uint esz, uint lid,
                            uint lsz)
{
    __global const int *x = aux + t.p0;
    const int rank = x[0];
    __global const int *dims = x + 1, *strides = x + 1 + rank;
    const int is64 = x[1 + 5 * rank];
    const int base = vmo_dyn_base(arena, iop, x, rank, is64);
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
#ifdef VMO_CPU_TILES
    /* Contiguous rank-1 f32 slice (the common dynamic_slice) is a straight
     * copy: d[i] = a[base+i]. The generic body's per-element div/mod runs
     * scalar on CPU runtimes (poc/09); use the chunk-per-WI float8 mover. */
    if (esz == 4 && rank == 1 && strides[0] == 1) {
        __global float *d = AP(float, t.dst);
        __global const float *a = AP(const float, t.a) + base;
        const uint chunk = (hi - lo + lsz - 1) / lsz;
        const uint clo = min(lo + lid * chunk, hi), chi = min(clo + chunk, hi);
        uint i = clo;
        for (; i + 8u <= chi; i += 8u) vstore8(vload8(0, a + i), 0, d + i);
        for (; i < chi; ++i) d[i] = a[i];
        return;
    }
#else
    /* GPU twin of the CPU fast path: 4-wide copy, 2x unrolled for ILP (the
     * generic body's per-element div/mod chain measured ~30 us per 16K tile).
     * vloadn needs only element alignment, so a runtime-odd `base` is fine. */
    if (esz == 4 && rank == 1 && strides[0] == 1) {
        __global uint *d = AP(uint, t.dst);
        __global const uint *a = AP(const uint, t.a) + base;
        const uint lo4 = lo >> 2, hi4 = lo4 + ((hi - lo) >> 2);
        uint v = lo4 + lid;
        for (; v + lsz < hi4; v += 2u * lsz) {
            const uint4 x0 = vload4(v, a), x1 = vload4(v + lsz, a);
            vstore4(x0, v, d);
            vstore4(x1, v + lsz, d);
        }
        if (v < hi4) vstore4(vload4(v, a), v, d);
        for (uint j = lo + ((hi - lo) & ~3u) + lid; j < hi; j += lsz)
            d[j] = a[j];
        return;
    }
#endif
    if (esz == 8)      DYN_GATHER_BODY(ulong);
    else if (esz == 2) DYN_GATHER_BODY(ushort);
    else if (esz == 1) DYN_GATHER_BODY(uchar);
    else               DYN_GATHER_BODY(uint);
}

#define DYN_SCATTER_BODY(T)                                                   \
    do {                                                                      \
        __global T *d = AP(T, t.dst);                                         \
        __global const T *a = AP(const T, t.a);                               \
        for (uint i = lo + lid; i < hi; i += lsz) {                           \
            int rem = (int)i, off = base;                                     \
            for (int e = rank - 1; e >= 0; --e) {                             \
                off += (rem % dims[e]) * strides[e]; rem /= dims[e];          \
            }                                                                 \
            d[off] = a[i];                                                    \
        }                                                                     \
    } while (0)

static void vmo_dyn_scatter_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                             const task_t t, uint tile, uint esz, uint lid,
                             uint lsz)
{
    __global const int *x = aux + t.p0;
    const int rank = x[0];
    __global const int *dims = x + 1, *strides = x + 1 + rank;
    const int is64 = x[1 + 5 * rank];
    const int base = vmo_dyn_base(arena, iop, x, rank, is64);
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
#ifdef VMO_CPU_TILES
    /* Contiguous rank-1 f32 update (the common dynamic_update_slice) is a
     * straight copy: d[base+i] = a[i] — same CPU mover as dyn_gather. */
    if (esz == 4 && rank == 1 && strides[0] == 1) {
        __global float *d = AP(float, t.dst) + base;
        __global const float *a = AP(const float, t.a);
        const uint chunk = (hi - lo + lsz - 1) / lsz;
        const uint clo = min(lo + lid * chunk, hi), chi = min(clo + chunk, hi);
        uint i = clo;
        for (; i + 8u <= chi; i += 8u) vstore8(vload8(0, a + i), 0, d + i);
        for (; i < chi; ++i) d[i] = a[i];
        return;
    }
#else
    /* GPU twin (in-place DUS carries / scan): 4-wide, 2x unrolled. d+base is
     * runtime-odd in general; vstoren needs only element alignment. */
    if (esz == 4 && rank == 1 && strides[0] == 1) {
        __global uint *d = AP(uint, t.dst) + base;
        __global const uint *a = AP(const uint, t.a);
        const uint lo4 = lo >> 2, hi4 = lo4 + ((hi - lo) >> 2);
        uint v = lo4 + lid;
        for (; v + lsz < hi4; v += 2u * lsz) {
            const uint4 x0 = vload4(v, a), x1 = vload4(v + lsz, a);
            vstore4(x0, v, d);
            vstore4(x1, v + lsz, d);
        }
        if (v < hi4) vstore4(vload4(v, a), v, d);
        for (uint j = lo + ((hi - lo) & ~3u) + lid; j < hi; j += lsz)
            d[j] = a[j];
        return;
    }
#endif
    if (esz == 8)      DYN_SCATTER_BODY(ulong);
    else if (esz == 2) DYN_SCATTER_BODY(ushort);
    else if (esz == 1) DYN_SCATTER_BODY(uchar);
    else               DYN_SCATTER_BODY(uint);
}
