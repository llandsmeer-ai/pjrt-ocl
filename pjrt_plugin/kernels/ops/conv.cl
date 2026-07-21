/* Direct N-D convolution tile op (TOP_CONV) — §39.
 *
 * Covers the canonical XLA-canonical layout: input NHWC-style
 * [B, S_0..S_{k-1}, Cin], kernel HWIO-style [W_0..W_{k-1}, Cin, Cout], output
 * [B, O_0..O_{k-1}, Cout]. Spatial rank k in 1..4. Supports window strides,
 * explicit (non-negative) padding, and rhs (kernel) dilation. Groups and lhs
 * (base) dilation are rejected at lowering time (feature_group_count == 1).
 *
 * Each output element serially accumulates:
 *   out[b, osp, oc] = sum_{win, ic}
 *       in[b, osp*stride + win*dil - pad_low, ic] * w[win, ic, oc]
 * skipping window taps whose input coordinate falls in the padding halo
 * (implicit zero padding). f32 compute only (loader gates dtype).
 *
 * aux at task.p0 (all i32):
 *   sdim, Cin, Cout,
 *   out_spatial[sdim], win[sdim], stride[sdim], pad_low[sdim], dil[sdim],
 *   in_spatial[sdim]
 *
 * a=input, b=weights, dst=output; p1 = output element count (EW-style tiling).
 */

static void vmo_conv_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                          const task_t t, uint tile, uint lid, uint lsz)
{
    __global const int *x = aux + t.p0;
    const int sdim = x[0];
    const int Cin  = x[1];
    const int Cout = x[2];
    __global const int *ospat = x + 3;
    __global const int *win   = x + 3 + 1 * sdim;
    __global const int *strd  = x + 3 + 2 * sdim;
    __global const int *plow  = x + 3 + 3 * sdim;
    __global const int *dil   = x + 3 + 4 * sdim;
    __global const int *ispat = x + 3 + 5 * sdim;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);

    /* input spatial strides (row-major over [B, S_0..S_{k-1}, Cin]) */
    int inStride[4];
    int acc = Cin;
    for (int d = sdim - 1; d >= 0; --d) { inStride[d] = acc; acc *= ispat[d]; }
    const int inBatchStride = acc;   /* prod(S) * Cin */

    int wcount = 1;
    for (int d = 0; d < sdim; ++d) wcount *= win[d];

    for (uint i = lo + lid; i < hi; i += lsz) {
        /* decode output index i over [B, O_0..O_{k-1}, Cout] (row-major) */
        int rem = (int)i;
        const int oc = rem % Cout; rem /= Cout;
        int ocoord[4];
        for (int d = sdim - 1; d >= 0; --d) { ocoord[d] = rem % ospat[d]; rem /= ospat[d]; }
        const int b = rem;

        float accv = 0.0f;
        for (int w = 0; w < wcount; ++w) {
            int rw = w, inb = 1, inoff = b * inBatchStride;
            for (int d = sdim - 1; d >= 0; --d) {
                const int wc = rw % win[d]; rw /= win[d];
                const int ic = ocoord[d] * strd[d] + wc * dil[d] - plow[d];
                if (ic < 0 || ic >= ispat[d]) inb = 0;
                inoff += ic * inStride[d];
            }
            if (!inb) continue;
            const int wbase = w * Cin * Cout + oc;   /* w[win=w, ic=0, oc] */
            for (int c = 0; c < Cin; ++c)
                accv += AP(const float, t.a)[inoff + c]
                      * AP(const float, t.b)[wbase + c * Cout];
        }
        AP(float, t.dst)[i] = accv;
    }
}
