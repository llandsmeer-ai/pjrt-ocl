/* Dynamic slice/update tile ops — gather/scatter with a RUNTIME base offset.
 *
 * Unlike TOP_GATHER/TOP_SCATTER (whose src/out base is a compile-time aux
 * constant), the start offsets here are SCALAR BUFFERS living in the arena: the
 * tile op reads them, clamps to the legal range, and forms the affine base
 * offset at runtime. Their byte offsets ride in the aux pool (the loader can
 * only patch task dst/a/b to byte offsets, not an arbitrary-length index list).
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

static int vmo_dyn_base(__global uchar *arena, __global const int *x, int rank,
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

static void vmo_dyn_gather_tile(__global uchar *arena, __global const int *aux,
                            const task_t t, uint tile, uint esz, uint lid,
                            uint lsz)
{
    __global const int *x = aux + t.p0;
    const int rank = x[0];
    __global const int *dims = x + 1, *strides = x + 1 + rank;
    const int is64 = x[1 + 5 * rank];
    const int base = vmo_dyn_base(arena, x, rank, is64);
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
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

static void vmo_dyn_scatter_tile(__global uchar *arena, __global const int *aux,
                             const task_t t, uint tile, uint esz, uint lid,
                             uint lsz)
{
    __global const int *x = aux + t.p0;
    const int rank = x[0];
    __global const int *dims = x + 1, *strides = x + 1 + rank;
    const int is64 = x[1 + 5 * rank];
    const int base = vmo_dyn_base(arena, x, rank, is64);
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
    if (esz == 8)      DYN_SCATTER_BODY(ulong);
    else if (esz == 2) DYN_SCATTER_BODY(ushort);
    else if (esz == 1) DYN_SCATTER_BODY(uchar);
    else               DYN_SCATTER_BODY(uint);
}
