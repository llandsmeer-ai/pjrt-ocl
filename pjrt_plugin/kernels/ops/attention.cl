/* ops/attention.cl — fused FLASH-ATTENTION tile-op (TOP_FLASH_ATTN, §34).
 *
 * Collapses the batched per-head attention block  QKᵀ → ×scale → softmax(-1)
 * → AV  into ONE megakernel instruction, so the (T×C) score matrix (and the
 * (1×C) decode score row) NEVER materialize in global memory. This attacks the
 * matmul phases §33's store-epilogue leaves alone (the two attention matmuls +
 * the softmax reduce = 3 barriered phases → 1).
 *
 * ONE workgroup computes attention for ONE (head g, query row m):
 *     out[g,m,:] = softmax( scale · Q[g,m,:] · K[g,:,:]ᵀ ) · V[g,:,:]
 * via ONLINE softmax (flash recurrence): stream the C keys in tiles of `lsz`,
 * keep a running max `run_m`, running denominator `run_l`, and running output
 * accumulator acc[hd]; each tile rescales acc/l by exp(run_m - m_new). O(hd)
 * state, one pass, never a C-wide buffer. Grid = H*T workgroups; this is ONE
 * cross-workgroup phase (like softmax_seg) — only LOCAL barriers inside.
 *
 * READ-THROUGH-VIEW (correct-by-construction): Q/K/V are read through the SAME
 * strided view descriptors the decomposed matmuls used (qv/kv/vv aux offsets in
 * the task's descriptor), addressed with the matmul's own flat index formula.
 * So the fused op reads byte-identical inputs to DOT1/DOT2 for ANY folded
 * transpose/reshape (decode: kv only; prefill: qv,kv,vv), with no shape
 * assumptions — a mismatch is impossible because the addressing is shared.
 *   Q[g,m,d] = A-operand (g,m,d) of DOT1[G,M=T,K=hd] : flat g*T*hd + m*hd + d
 *   K[g,j,d] = B-operand (g,k=d,n=j) of DOT1[G,K=hd,N=C] : flat g*hd*C + d*C + j
 *   V[g,j,d] = B-operand (g,k=j,n=d) of DOT2[G,K=C,N=hd] : flat g*C*hd + j*hd + d
 *   out      = DOT2 output (contiguous) : flat g*T*hd + m*hd + d
 *
 * Descriptor (int words at aux[t.p3]):
 *   [0]=H [1]=T [2]=C [3]=hd [4]=scale_bits [5]=causal
 *   [6]=qv [7]=kv [8]=vv   (view aux-offsets +1; 0 = contiguous)
 * task: a=Q src, b=K src, p0=V src (loader-patched), dst=out, p1=H, p2=T,
 *       p3=descriptor aux word-offset.
 *
 * §18/§19a PoCL barrier discipline: the per-tile reductions are local tree
 * reduces; the key-tile loop trip count is work-item UNIFORM (SEG_UNIFORM over
 * C); the loop's cleanup barrier sits at the TOP of the body (never as the last
 * statement before the backedge); no `return` precedes any barrier.
 *
 * Local (As, Bs from the shared MMA panels): As holds Qs[hd] | acc[hd] |
 * Ps[lsz]  (hd ≤ 256 gated ⇒ 2*hd+lsz ≤ MMA_ASZ); Bs holds the reduction
 * scratch red[lsz]. */

static inline float vmo_fa_load(__global const float *src, uint view,
                                __global const int *aux, size_t idx)
{
    return view ? src[vmo_view_idx(aux, view - 1u, (uint)idx)] : src[idx];
}

static void vmo_flash_attn(__global uchar *arena, __global uchar **iop,
                           __global const int *aux, const task_t t,
                           uint tile, __local float *As, __local float *Bs,
                           uint lid, uint lsz)
{
    const uint desc   = t.p3;
    const uint H      = (uint)aux[desc + 0];
    const uint T      = (uint)aux[desc + 1];
    const uint C      = (uint)aux[desc + 2];
    const uint hd     = (uint)aux[desc + 3];
    const float scale = as_float(aux[desc + 4]);
    const uint causal = (uint)aux[desc + 5];
    const uint qv     = (uint)aux[desc + 6];
    const uint kv     = (uint)aux[desc + 7];
    const uint vv     = (uint)aux[desc + 8];

    const uint g = tile / T;          /* head */
    const uint m = tile % T;          /* query row within the head */
    const int valid = tile < H * T;   /* over-assigned tiles: run trees, no store */

    __global const float *Q = AP(const float, t.a);
    __global const float *K = AP(const float, t.b);
    __global const float *V = AP(const float, t.p0);
    __global float *O = AP(float, t.dst);

    __local float *Qs  = As;            /* [hd]  query row */
    __local float *acc = As + hd;       /* [hd]  running output accumulator */
    __local float *Ps  = As + 2u * hd;  /* [lsz] this tile's softmax weights */

    /* stage the query row + zero the accumulator */
    for (uint d = lid; d < hd; d += lsz) {
        Qs[d] = valid
              ? vmo_fa_load(Q, qv, aux, ((size_t)g * T + m) * hd + d)
              : 0.0f;
        acc[d] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float run_m = -INFINITY;   /* running max  (workgroup-uniform; recomputed/lane) */
    float run_l = 0.0f;        /* running denominator */

    /* global query position for causal masking: queries are the last T of the
     * C-length key window (KV-cache decode / self-attention prefill). */
    const uint qpos = m + (C - T);
    const uint jN = SEG_UNIFORM(C, lsz);   /* uniform trip over key tiles */

    for (uint j0 = 0; j0 < jN; j0 += lsz) {
        /* cleanup barrier at the TOP (not the backedge, §18): the previous
         * iteration's acc update read Ps/red; make it complete before this
         * iteration overwrites them. Harmless extra barrier on iter 0. */
        barrier(CLK_LOCAL_MEM_FENCE);
        const uint j = j0 + lid;

        /* 1) score for this lane's key j = scale · Q·K[j] (−inf if masked/OOB) */
        float s = -INFINITY;
        if (valid && j < C && !(causal && j > qpos)) {
            float dp = 0.0f;
            for (uint d = 0; d < hd; ++d)
                dp += Qs[d] *
                      vmo_fa_load(K, kv, aux, (size_t)g * hd * C + (size_t)d * C + j);
            s = dp * scale;
        }

        /* 2) tile max via a local tree over red[] */
        Bs[lid] = s;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint r = lsz >> 1; r > 0; r >>= 1) {
            if (lid < r) Bs[lid] = fmax(Bs[lid], Bs[lid + r]);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const float tile_max = Bs[0];
        barrier(CLK_LOCAL_MEM_FENCE);          /* all read tile_max before red[] reuse */

        const float new_m = fmax(run_m, tile_max);
        /* run_m = -inf on iter 0 ⇒ corr = 0 (acc is 0, so acc*corr is fine);
         * a fully-OOB later tile has tile_max = -inf ⇒ new_m = run_m ⇒ corr = 1. */
        const float corr = exp(run_m - new_m);

        /* 3) p = exp(s - new_m); persist in Ps, reduce the sum in red[] */
        const float p = (s > -INFINITY) ? exp(s - new_m) : 0.0f;
        Ps[lid] = p;
        Bs[lid] = p;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint r = lsz >> 1; r > 0; r >>= 1) {
            if (lid < r) Bs[lid] = Bs[lid] + Bs[lid + r];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const float tile_sum = Bs[0];

        /* 4) acc[d] = acc[d]·corr + Σ_jj Ps[jj]·V[g, j0+jj, d]  (parallel over d) */
        for (uint d = lid; d < hd; d += lsz) {
            float a = 0.0f;
            for (uint jj = 0; jj < lsz; ++jj) {
                const uint kk = j0 + jj;
                if (kk < C)
                    a += Ps[jj] *
                         vmo_fa_load(V, vv, aux,
                                     (size_t)g * C * hd + (size_t)kk * hd + d);
            }
            acc[d] = acc[d] * corr + a;
        }
        run_l = run_l * corr + tile_sum;
        run_m = new_m;
    }

    /* normalize + store (uniform-trip grid-stride; guarded, §19a) */
    const float inv = 1.0f / run_l;
    for (uint d = lid; d < hd; d += lsz)
        if (valid)
            O[((size_t)g * T + m) * hd + d] = acc[d] * inv;
}
