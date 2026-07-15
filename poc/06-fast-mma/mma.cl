/* poc/06: optimized SGEMM tile function for the persistent-lane VM.
 *
 * A TILE FUNCTION, not a free-standing GEMM: one workgroup (256 threads)
 * computes one TMxTN output tile of C[M,N] = A[M,K] @ B[K,N] (all row-major
 * dense in a single float arena, addressed by element offsets aoff/boff/coff),
 * callable in a loop from an interpreter (cf. exec_tiles in poc/04/vliw.cl).
 *
 * The parametrized kernel `mma_tile_fast` is compiled multiple times with
 * different -D options by bench.c to document the optimization progression:
 *   -DTM -DTN     output tile edges (multiple of 16)
 *   -DBK          K-panel depth staged in local memory (multiple of VECW)
 *   -DVECW        1 = scalar global loads, 4 = vload4/vstore4 (fast path)
 *   -DDB          0 = single-buffered local staging, 1 = double-buffered
 * Fixed: 256 threads == 16x16 thread grid; each thread owns an RMxRN
 * register microtile (RM=TM/16, RN=TN/16).
 *
 * Local-memory footprint (floats): (DB?2:1) * BK * (TM + TN).
 * Register footprint (per thread): RM*RN accumulators + RM+RN operands.
 *
 * Correctness: `full` (runtime) selects the VECW fast path only when the tile
 * is fully interior and 16B-aligned; otherwise a scalar, edge-guarded path
 * runs, so M/N/K need not divide the tile edges.
 */

/* ===================== naive baseline (poc/04 mma_tile) ================= */
#define MMA_T 16
static void mma_tile_naive(__global float *arena, uint aoff, uint boff,
                           uint coff, uint M, uint N, uint K, uint tile,
                           __local float *As, __local float *Bs)
{
    const uint tiles_n = (N + MMA_T - 1) / MMA_T;
    const uint tr = tile / tiles_n, tc = tile % tiles_n;
    const uint lr = get_local_id(0) / MMA_T, lc = get_local_id(0) % MMA_T;
    const uint r = tr * MMA_T + lr, c = tc * MMA_T + lc;
    float acc = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += MMA_T) {
        if (get_local_id(0) < MMA_T * MMA_T) {
            const uint ar = tr * MMA_T + lr, ak = k0 + lc;
            As[lr * MMA_T + lc] =
                (ar < M && ak < K) ? arena[aoff + ar * K + ak] : 0.0f;
            const uint bk = k0 + lr, bc = tc * MMA_T + lc;
            Bs[lr * MMA_T + lc] =
                (bk < K && bc < N) ? arena[boff + bk * N + bc] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (get_local_id(0) < MMA_T * MMA_T)
            for (uint k = 0; k < MMA_T; ++k)
                acc += As[lr * MMA_T + k] * Bs[k * MMA_T + lc];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (get_local_id(0) < MMA_T * MMA_T && r < M && c < N)
        arena[coff + r * N + c] = acc;
}

__kernel void bench_naive(__global float *arena, uint aoff, uint boff,
                          uint coff, uint M, uint N, uint K, uint nlanes,
                          uint full)
{
    (void)full;
    __local float As[MMA_T * MMA_T];
    __local float Bs[MMA_T * MMA_T];
    const uint tiles_m = (M + MMA_T - 1) / MMA_T;
    const uint tiles_n = (N + MMA_T - 1) / MMA_T;
    const uint T = tiles_m * tiles_n;
    const uint lane = get_group_id(0);
    const uint per = (T + nlanes - 1) / nlanes;
    const uint lo = lane * per, hi = min(T, lo + per);
    for (uint tile = lo; tile < hi; ++tile)
        mma_tile_naive(arena, aoff, boff, coff, M, N, K, tile, As, Bs);
}

/* ===================== register-blocked fast tile ====================== */
#ifdef FAST

#define TDIM 16              /* 16x16 = 256 threads */
#define RM   (TM / TDIM)     /* microtile rows per thread */
#define RN   (TN / TDIM)     /* microtile cols per thread */
#ifndef VECW
#define VECW 1
#endif
#ifndef DB
#define DB 0
#endif
#define ASZ (BK * TM)        /* As holds BK x TM, transposed: As[k*TM + m] */
#define BSZ (BK * TN)        /* Bs holds BK x TN:            Bs[k*TN + n] */

/* Stage one BK-deep K-panel of A (transposed) and B into local buffers. */
static void stage_panel(__global float *arena, uint aoff, uint boff,
                        uint M, uint N, uint K, uint row0, uint col0,
                        uint k0, __local float *As, __local float *Bs,
                        uint full)
{
    const uint lid = get_local_id(0);
#if VECW == 4
    if (full) {
        /* A: contiguous float4 along K, scatter-store into transposed As */
        for (uint idx = lid; idx < TM * (BK / 4); idx += 256) {
            const uint m  = idx / (BK / 4);
            const uint kb = (idx % (BK / 4)) * 4;
            const float4 v = vload4(0, arena + aoff + (row0 + m) * K + k0 + kb);
            As[(kb + 0) * TM + m] = v.s0;
            As[(kb + 1) * TM + m] = v.s1;
            As[(kb + 2) * TM + m] = v.s2;
            As[(kb + 3) * TM + m] = v.s3;
        }
        /* B: contiguous float4 along N, vector-store into Bs */
        for (uint idx = lid; idx < BK * (TN / 4); idx += 256) {
            const uint kk = idx / (TN / 4);
            const uint nb = (idx % (TN / 4)) * 4;
            const float4 v = vload4(0, arena + boff + (k0 + kk) * N + col0 + nb);
            vstore4(v, 0, Bs + kk * TN + nb);
        }
        return;
    }
#endif
    /* scalar, edge-guarded path (also the VECW==1 path) */
    for (uint idx = lid; idx < TM * BK; idx += 256) {
        const uint m  = idx / BK;
        const uint kk = idx % BK;
        const uint gr = row0 + m, gk = k0 + kk;
        As[kk * TM + m] = (gr < M && gk < K) ? arena[aoff + gr * K + gk] : 0.0f;
    }
    for (uint idx = lid; idx < BK * TN; idx += 256) {
        const uint kk = idx / TN;
        const uint n  = idx % TN;
        const uint gk = k0 + kk, gc = col0 + n;
        Bs[kk * TN + n] = (gk < K && gc < N) ? arena[boff + gk * N + gc] : 0.0f;
    }
}

/* Multiply the staged BK-panel into the register accumulators. */
static void compute_panel(__local float *As, __local float *Bs,
                          uint ty, uint tx, float acc[RM][RN])
{
    #pragma unroll
    for (uint kk = 0; kk < BK; ++kk) {
        float a[RM], b[RN];
        #pragma unroll
        for (int i = 0; i < RM; i++) a[i] = As[kk * TM + ty * RM + i];
        #pragma unroll
        for (int j = 0; j < RN; j++) b[j] = Bs[kk * TN + tx * RN + j];
        #pragma unroll
        for (int i = 0; i < RM; i++)
            #pragma unroll
            for (int j = 0; j < RN; j++)
                acc[i][j] += a[i] * b[j];
    }
}

static void mma_tile_fast(__global float *arena, uint aoff, uint boff,
                          uint coff, uint M, uint N, uint K, uint tile,
                          __local float *As, __local float *Bs, uint full)
{
    const uint tiles_n = (N + TN - 1) / TN;
    const uint tr = tile / tiles_n, tc = tile % tiles_n;
    const uint row0 = tr * TM, col0 = tc * TN;
    const uint lid = get_local_id(0);
    const uint ty = lid / TDIM, tx = lid % TDIM;

    float acc[RM][RN];
    #pragma unroll
    for (int i = 0; i < RM; i++)
        #pragma unroll
        for (int j = 0; j < RN; j++) acc[i][j] = 0.0f;

#if DB
    /* double-buffered: prefetch next panel while computing the current one */
    __local float *A0 = As, *A1 = As + ASZ;
    __local float *B0 = Bs, *B1 = Bs + BSZ;
    stage_panel(arena, aoff, boff, M, N, K, row0, col0, 0, A0, B0, full);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint k0 = 0; k0 < K; k0 += BK) {
        const uint kn = k0 + BK;
        if (kn < K)
            stage_panel(arena, aoff, boff, M, N, K, row0, col0, kn, A1, B1,
                        full);
        compute_panel(A0, B0, ty, tx, acc);
        barrier(CLK_LOCAL_MEM_FENCE);
        __local float *t;
        t = A0; A0 = A1; A1 = t;
        t = B0; B0 = B1; B1 = t;
    }
#else
    for (uint k0 = 0; k0 < K; k0 += BK) {
        stage_panel(arena, aoff, boff, M, N, K, row0, col0, k0, As, Bs, full);
        barrier(CLK_LOCAL_MEM_FENCE);
        compute_panel(As, Bs, ty, tx, acc);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif

    /* write the RMxRN microtile, edge-guarded */
    #pragma unroll
    for (int i = 0; i < RM; i++) {
        const uint gr = row0 + ty * RM + i;
        if (gr >= M) continue;
        #pragma unroll
        for (int j = 0; j < RN; j++) {
            const uint gc = col0 + tx * RN + j;
            if (gc < N) arena[coff + gr * N + gc] = acc[i][j];
        }
    }
}

__kernel void bench_fast(__global float *arena, uint aoff, uint boff,
                         uint coff, uint M, uint N, uint K, uint nlanes,
                         uint full)
{
    __local float As[(DB ? 2 : 1) * ASZ];
    __local float Bs[(DB ? 2 : 1) * BSZ];
    const uint tiles_m = (M + TM - 1) / TM;
    const uint tiles_n = (N + TN - 1) / TN;
    const uint T = tiles_m * tiles_n;
    const uint lane = get_group_id(0);
    const uint per = (T + nlanes - 1) / nlanes;
    const uint lo = lane * per, hi = min(T, lo + per);
    for (uint tile = lo; tile < hi; ++tile)
        mma_tile_fast(arena, aoff, boff, coff, M, N, K, tile, As, Bs, full);
}

#endif /* FAST */
