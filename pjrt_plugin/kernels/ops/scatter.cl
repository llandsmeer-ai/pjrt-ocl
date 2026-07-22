/* Strided scatter tile op (TOP_SCATTER) — the mirror of gather. For each INPUT
 * element i (row-major over in_dims), write it to the output at an affine
 * position:  dst[out_off + sum_d idx_d(i)*out_stride_d] = a[i].
 * Used by concatenate / pad: each source is scattered into a disjoint region of
 * a preallocated output. aux at task.p0:
 *   rank u32, in_dims i32[rank], out_strides i32[rank], out_off i32
 * dtype-agnostic (copies whole elements): esz picks the width. */

static void vmo_scatter_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                         const task_t t, uint tile, uint esz, uint lid, uint lsz)
{
    __global const int *x = aux + t.p0;
    const int rank = x[0];
    __global const int *dims = x + 1, *strides = x + 1 + rank;
    const int out_off = x[1 + 2 * rank];
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
    if (esz == 8) {
        __global ulong *d = AP(ulong, t.dst);
        __global const ulong *a = AP(const ulong, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = out_off;
            for (int e = rank - 1; e >= 0; --e) { off += (rem % dims[e]) * strides[e]; rem /= dims[e]; }
            d[off] = a[i];
        }
    } else if (esz == 2) {
        __global ushort *d = AP(ushort, t.dst);
        __global const ushort *a = AP(const ushort, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = out_off;
            for (int e = rank - 1; e >= 0; --e) { off += (rem % dims[e]) * strides[e]; rem /= dims[e]; }
            d[off] = a[i];
        }
    } else if (esz == 1) {
        __global uchar *d = AP(uchar, t.dst);
        __global const uchar *a = AP(const uchar, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = out_off;
            for (int e = rank - 1; e >= 0; --e) { off += (rem % dims[e]) * strides[e]; rem /= dims[e]; }
            d[off] = a[i];
        }
    } else {
        __global int *d = AP(int, t.dst);
        __global const int *a = AP(const int, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = out_off;
            for (int e = rank - 1; e >= 0; --e) { off += (rem % dims[e]) * strides[e]; rem /= dims[e]; }
            d[off] = a[i];
        }
    }
}

/* --- f32 atomic combines (CL 1.2 atomic_cmpxchg on the uint bit pattern) ---
 * OpenCL has no core float atomics; the standard cmpxchg-retry loop below is
 * exact for add/max/min because those combines are associative+commutative, so
 * any interleaving of concurrent updaters lands the same final value. */
static void vmo_scat_add_f32(__global float *p, float v)
{
    volatile __global uint *up = (volatile __global uint *)p;
    uint old = *up, assumed;
    do { assumed = old;
         old = atomic_cmpxchg(up, assumed, as_uint(as_float(assumed) + v));
    } while (old != assumed);
}
static void vmo_scat_max_f32(__global float *p, float v)
{
    volatile __global uint *up = (volatile __global uint *)p;
    uint old = *up, assumed;
    do { assumed = old;
         old = atomic_cmpxchg(up, assumed, as_uint(fmax(as_float(assumed), v)));
    } while (old != assumed);
}
static void vmo_scat_min_f32(__global float *p, float v)
{
    volatile __global uint *up = (volatile __global uint *)p;
    uint old = *up, assumed;
    do { assumed = old;
         old = atomic_cmpxchg(up, assumed, as_uint(fmin(as_float(assumed), v)));
    } while (old != assumed);
}

/* General data-dependent scatter (TOP_SCATTER_INDEX; stablehlo.scatter). Mirror
 * of vmo_gather_index_tile: iterate over UPDATE elements; each maps to an
 * operand location via window coords (op_stride) + runtime scatter indices
 * (idx_op_stride), and its value is COMBINED into the operand result.
 * `kind` = 0 set / 1 add / 2 max / 3 min. The operand is copied into dst BEFORE
 * this op (identity gather + WAW barrier), so dst already holds the operand and
 * here we only apply updates.
 *
 * Duplicate indices: add/max/min run through global atomics, so any tiling
 * order yields the exact stablehlo result. set (overwrite) is a plain store —
 * last writer wins nondeterministically, matching stablehlo's unspecified order
 * for non-unique overwrite indices; the lowering restricts add/max/min to
 * 4-byte f32/i32/u32 (the only widths with core atomics).
 *
 * aux header at task.p0 (all i32, elements):
 *   [0] out_rank (= update rank)   [1] nidx   [2] si_vec_stride   [3] is64
 *   [4] idx_byteoff (loader-patched from idx_bufid)   [5] idx_bufid   [6] kind
 * then contiguous:
 *   upd_dims[out_rank]      # update shape (row-major decode of i)
 *   op_stride[out_rank]     # operand elem stride each window update dim adds
 *   si_stride[out_rank]     # scatter_indices batch stride each scatter dim adds
 *   idx_op_stride[nidx]     # operand elem stride of dim scatter_dims_to_operand_dims[k]
 *   clamp_max[nidx]         # max legal start per component (operand.dim - window)
 * a = updates, dst = operand result. */
static void vmo_scatter_index_tile(__global uchar *arena, __global uchar **iop,
                              __global const int *aux, const task_t t, uint tile,
                              uint esz, uint dt, uint lid, uint lsz)
{
    __global const int *x = aux + t.p0;
    const int out_rank = x[0];
    const int nidx = x[1];
    const int si_vec_stride = x[2];
    const int is64 = x[3];
    const int idx_byteoff = x[4];
    const int kind = x[6];
    __global const int *upd_dims = x + 7;
    __global const int *op_stride = upd_dims + out_rank;
    __global const int *si_stride = op_stride + out_rank;
    __global const int *idx_op_stride = si_stride + out_rank;
    __global const int *clamp_max = idx_op_stride + nidx;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);

#define SIDX_OFF(i)                                                          \
    int rem = (int)(i), off = 0, si_base = 0;                                \
    for (int e = out_rank - 1; e >= 0; --e) {                                \
        int c = rem % upd_dims[e]; rem /= upd_dims[e];                       \
        off += c * op_stride[e]; si_base += c * si_stride[e];                \
    }                                                                        \
    for (int k = 0; k < nidx; ++k) {                                         \
        int s = is64                                                         \
            ? (int)AP(const long, idx_byteoff)[si_base + k*si_vec_stride]    \
            : AP(const int, idx_byteoff)[si_base + k*si_vec_stride];         \
        s = s < 0 ? 0 : (s > clamp_max[k] ? clamp_max[k] : s);              \
        off += s * idx_op_stride[k];                                         \
    }

    if (kind == 0) {   /* overwrite / set: whole-element copy (dtype-agnostic) */
#define SET_BODY(T) do {                                                     \
        __global T *d = AP(T, t.dst); __global const T *a = AP(const T, t.a);\
        for (uint i = lo + lid; i < hi; i += lsz) { SIDX_OFF(i); d[off] = a[i]; } \
    } while (0)
        if (esz == 8)      SET_BODY(ulong);
        else if (esz == 2) SET_BODY(ushort);
        else if (esz == 1) SET_BODY(uchar);
        else               SET_BODY(uint);
#undef SET_BODY
    } else if (dt == DT_F32) {
        __global float *d = AP(float, t.dst);
        __global const float *a = AP(const float, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            SIDX_OFF(i);
            const float v = a[i];
            if (kind == 1)      vmo_scat_add_f32(d + off, v);
            else if (kind == 2) vmo_scat_max_f32(d + off, v);
            else                vmo_scat_min_f32(d + off, v);
        }
    } else if (dt == DT_U32) {
        __global uint *d = AP(uint, t.dst);
        __global const uint *a = AP(const uint, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            SIDX_OFF(i);
            const uint v = a[i];
            if (kind == 1)      atomic_add(d + off, v);
            else if (kind == 2) atomic_max(d + off, v);
            else                atomic_min(d + off, v);
        }
    } else {   /* i32 */
        __global int *d = AP(int, t.dst);
        __global const int *a = AP(const int, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            SIDX_OFF(i);
            const int v = a[i];
            if (kind == 1)      atomic_add(d + off, v);
            else if (kind == 2) atomic_max(d + off, v);
            else                atomic_min(d + off, v);
        }
    }
#undef SIDX_OFF
}
