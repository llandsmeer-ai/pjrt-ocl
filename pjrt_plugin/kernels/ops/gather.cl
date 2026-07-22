/* Strided gather tile op (TOP_GATHER). aux at task.p0:
 *   rank, out_dims[rank], in_strides[rank], src_off  (all i32, elements).
 * Covers broadcast_in_dim / transpose / slice / reverse (via strides+src_off).
 */

/* Gather copies whole elements, so it is dtype-agnostic for a given element
 * size. `esz` (bytes) picks the mover: 4-byte types copy as uint bits; 8-byte
 * as ulong. dst/a are BYTE offsets. */
static void vmo_gather_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                        const task_t t, uint tile, uint esz, uint lid, uint lsz)
{
    __global const int *x = aux + t.p0;
    const int rank = x[0];
    __global const int *dims = x + 1, *strides = x + 1 + rank;
    const int src_off = x[1 + 2 * rank];
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
    /* Contiguous fast path (dynamic_slice / slice with unit stride): the
     * div/mod rank loop below costs ~20 cycles of serial ALU per ELEMENT and
     * dominated the chained bench (~30 us/op flat). 4-wide copy; vloadn only
     * requires element alignment, so a runtime-odd src_off is fine, and dst
     * (arena/port allocation, 16B-aligned) takes an aligned vstore4. */
    if (esz == 4 && rank == 1 && strides[0] == 1) {
        __global uint *d = AP(uint, t.dst);
        __global const uint *a = AP(const uint, t.a) + src_off;
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
    if (esz == 8) {
        __global ulong *d = AP(ulong, t.dst);
        __global const ulong *a = AP(const ulong, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = src_off;
            for (int e = rank - 1; e >= 0; --e) {
                off += (rem % dims[e]) * strides[e];
                rem /= dims[e];
            }
            d[i] = a[off];
        }
    } else if (esz == 2) {
        __global ushort *d = AP(ushort, t.dst);
        __global const ushort *a = AP(const ushort, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = src_off;
            for (int e = rank - 1; e >= 0; --e) {
                off += (rem % dims[e]) * strides[e];
                rem /= dims[e];
            }
            d[i] = a[off];
        }
    } else if (esz == 1) {
        __global uchar *d = AP(uchar, t.dst);
        __global const uchar *a = AP(const uchar, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = src_off;
            for (int e = rank - 1; e >= 0; --e) {
                off += (rem % dims[e]) * strides[e];
                rem /= dims[e];
            }
            d[i] = a[off];
        }
    } else {
        __global uint *d = AP(uint, t.dst);
        __global const uint *a = AP(const uint, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = src_off;
            for (int e = rank - 1; e >= 0; --e) {
                off += (rem % dims[e]) * strides[e];
                rem /= dims[e];
            }
            d[i] = a[off];
        }
    }
}

/* General data-dependent gather (TOP_GATHER_INDEX; stablehlo.gather). Unlike
 * TOP_GATHER (a compile-time-affine view) each output element's operand base
 * offset depends on index values read at RUNTIME from a start_indices buffer.
 *
 * aux header at task.p0 (all i32, elements):
 *   [0] out_rank
 *   [1] nidx            # index-vector components (= len start_index_map)
 *   [2] si_vec_stride   # stride of index_vector_dim in start_indices (0 if implicit)
 *   [3] is64            # 1 => i64 start_indices, else i32
 *   [4] idx_byteoff     # start_indices location (arena byte-off / port handle),
 *                       #   patched at LOAD time from idx_bufid (lowering can't know it)
 *   [5] idx_bufid       # start_indices buffer id (host-side patch source; unused here)
 * then, contiguous:
 *   out_dims[out_rank]      # output shape (row-major decode of i)
 *   op_stride[out_rank]     # operand elem stride each output dim adds (0 for batch dims)
 *   si_stride[out_rank]     # start_indices batch stride each output dim adds (0 for offset dims)
 *   idx_op_stride[nidx]     # operand elem stride of dim start_index_map[k]
 *   clamp_max[nidx]         # max legal start per component (operand.dim - slice_size)
 *
 *   op_off(i)  = sum_e coord_e(i)*op_stride[e]  +  sum_k clamp(S_k)*idx_op_stride[k]
 *   si_base(i) = sum_e coord_e(i)*si_stride[e]
 *   S_k        = start_indices[si_base(i) + k*si_vec_stride]  clamped to [0, clamp_max[k]]
 *   dst[i]     = operand[op_off(i)]
 * Whole-element copy => dtype-agnostic (esz picks the mover). */
static void vmo_gather_index_tile(__global uchar *arena, __global uchar **iop,
                              __global const int *aux, const task_t t, uint tile,
                              uint esz, uint lid, uint lsz)
{
    __global const int *x = aux + t.p0;
    const int out_rank = x[0];
    const int nidx = x[1];
    const int si_vec_stride = x[2];
    const int is64 = x[3];
    const int idx_byteoff = x[4];
    __global const int *out_dims = x + 6;
    __global const int *op_stride = out_dims + out_rank;
    __global const int *si_stride = op_stride + out_rank;
    __global const int *idx_op_stride = si_stride + out_rank;
    __global const int *clamp_max = idx_op_stride + nidx;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);

#define GIDX_BODY(T)                                                          \
    do {                                                                      \
        __global T *d = AP(T, t.dst);                                         \
        __global const T *a = AP(const T, t.a);                               \
        for (uint i = lo + lid; i < hi; i += lsz) {                           \
            int rem = (int)i, off = 0, si_base = 0;                           \
            for (int e = out_rank - 1; e >= 0; --e) {                         \
                int c = rem % out_dims[e]; rem /= out_dims[e];                \
                off += c * op_stride[e];                                      \
                si_base += c * si_stride[e];                                  \
            }                                                                 \
            for (int k = 0; k < nidx; ++k) {                                  \
                int s = is64                                                  \
                    ? (int)AP(const long, idx_byteoff)[si_base + k*si_vec_stride] \
                    : AP(const int, idx_byteoff)[si_base + k*si_vec_stride];  \
                s = s < 0 ? 0 : (s > clamp_max[k] ? clamp_max[k] : s);        \
                off += s * idx_op_stride[k];                                  \
            }                                                                 \
            d[i] = a[off];                                                    \
        }                                                                     \
    } while (0)

    if (esz == 8)      GIDX_BODY(ulong);
    else if (esz == 2) GIDX_BODY(ushort);
    else if (esz == 1) GIDX_BODY(uchar);
    else               GIDX_BODY(uint);
#undef GIDX_BODY
}
