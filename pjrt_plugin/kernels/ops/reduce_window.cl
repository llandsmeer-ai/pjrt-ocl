/* Windowed reduction tile op (TOP_RED_WINDOW) — pooling.
 *
 * Each output element serially reduces the input window mapped to it. Covers
 * the common case only: no base/window dilation, VALID or explicit padding,
 * kind sum/max/min. Padding elements equal the reduction identity, so they are
 * simply skipped (correct because the init value is asserted to be the identity
 * at lowering time).
 *
 * aux at task.p0:
 *   kind i32 (0 sum, 1 max, 2 min), rank i32,
 *   out_dims i32[rank], win_dims i32[rank], win_strides i32[rank],
 *   pad_low i32[rank], in_dims i32[rank], in_strides i32[rank]
 *
 * out[o] = reduce_{w in window} in[o*stride + w - pad_low]   (in-bounds only).
 * Supports f32 (float compute) and i32/u32 (integer compute); the loader gates
 * other dtypes at lowering time.
 */

static void vmo_redwin_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                        const task_t t, uint tile, uint dt, uint lid, uint lsz)
{
    __global const int *x = aux + t.p0;
    const int kind = x[0];
    const int rank = x[1];
    __global const int *odims = x + 2;
    __global const int *wdims = x + 2 + rank;
    __global const int *wstr  = x + 2 + 2 * rank;
    __global const int *plow  = x + 2 + 3 * rank;
    __global const int *idims = x + 2 + 4 * rank;
    __global const int *istr  = x + 2 + 5 * rank;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);

    int wcount = 1;
    for (int d = 0; d < rank; ++d) wcount *= wdims[d];
    const int isint = (dt == DT_I32 || dt == DT_U32);

    for (uint i = lo + lid; i < hi; i += lsz) {
        float facc = kind == 0 ? 0.0f : kind == 1 ? -INFINITY : INFINITY;
        int iacc = kind == 0 ? 0 : kind == 1 ? INT_MIN : INT_MAX;
        for (int w = 0; w < wcount; ++w) {
            int rem_i = (int)i, rem_w = w, off = 0, inb = 1;
            for (int d = rank - 1; d >= 0; --d) {
                const int oc = rem_i % odims[d]; rem_i /= odims[d];
                const int wc = rem_w % wdims[d]; rem_w /= wdims[d];
                const int ic = oc * wstr[d] + wc - plow[d];
                if (ic < 0 || ic >= idims[d]) inb = 0;
                off += ic * istr[d];
            }
            if (!inb) continue;
            if (isint) {
                const int v = AP(const int, t.a)[off];
                iacc = kind == 0 ? iacc + v
                     : kind == 1 ? max(iacc, v) : min(iacc, v);
            } else {
                const float v = AP(const float, t.a)[off];
                facc = kind == 0 ? facc + v
                     : kind == 1 ? fmax(facc, v) : fmin(facc, v);
            }
        }
        if (isint) AP(int, t.dst)[i] = iacc;
        else AP(float, t.dst)[i] = facc;
    }
}
