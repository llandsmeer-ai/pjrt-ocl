/* Register-blocked SGEMM tile (TOP_MMA), from poc/06 step 2 (portable champion
 * family). One 256-thread workgroup computes one MMA_TM x MMA_TN output tile;
 * each thread owns an RM x RN = 4x4 register microtile. Scalar edge-guarded
 * staging, single-buffered -> portable to PoCL. Local: BK*(TM+TN) floats. The
 * scheduler tiles matmul in MMA_TM x MMA_TN blocks (scheduler.MMA_T==MMA_TM).
 * 4x4 (16 accumulators) chosen to bound the megakernel's occupancy tax
 * (docs/tile-isa.md ceiling-1). */
#define MMA_TM 64
#define MMA_TN 64
#define MMA_BK 16
#define MMA_TDIM 16          /* 16x16 thread grid == 256 threads */
#define MMA_RM (MMA_TM / MMA_TDIM)   /* 4 */
#define MMA_RN (MMA_TN / MMA_TDIM)   /* 4 */
#define MMA_ASZ (MMA_BK * MMA_TM)    /* As[m*BK + k] */
#define MMA_BSZ (MMA_BK * MMA_TN)    /* Bs[k*TN + n] */

static void vmo_mma_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                     __local float *As, __local float *Bs)
{
    const uint M = t.p0, N = t.p1, K = t.p2;
    const uint tiles_n = (N + MMA_TN - 1) / MMA_TN;
    const uint tr = tile / tiles_n, tc = tile % tiles_n;
    const uint row0 = tr * MMA_TM, col0 = tc * MMA_TN;
    const uint lid = get_local_id(0);
    const uint ty = lid / MMA_TDIM, tx = lid % MMA_TDIM;
    __global const float *ga = AP(const float, t.a);
    __global const float *gb = AP(const float, t.b);

    float acc[MMA_RM][MMA_RN];
    for (int i = 0; i < MMA_RM; i++)
        for (int j = 0; j < MMA_RN; j++) acc[i][j] = 0.0f;

    for (uint k0 = 0; k0 < K; k0 += MMA_BK) {
        for (uint idx = lid; idx < MMA_TM * MMA_BK; idx += 256) {
            const uint m = idx / MMA_BK, kk = idx % MMA_BK;
            const uint gr = row0 + m, gk = k0 + kk;
            As[m * MMA_BK + kk] =
                (gr < M && gk < K) ? ga[gr * K + gk] : 0.0f;
        }
        for (uint idx = lid; idx < MMA_BK * MMA_TN; idx += 256) {
            const uint kk = idx / MMA_TN, n = idx % MMA_TN;
            const uint gk = k0 + kk, gc = col0 + n;
            Bs[kk * MMA_TN + n] =
                (gk < K && gc < N) ? gb[gk * N + gc] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint kk = 0; kk < MMA_BK; ++kk) {
            float a[MMA_RM], b[MMA_RN];
            for (int i = 0; i < MMA_RM; i++)
                a[i] = As[(ty * MMA_RM + i) * MMA_BK + kk];
            for (int j = 0; j < MMA_RN; j++)
                b[j] = Bs[kk * MMA_TN + tx * MMA_RN + j];
            for (int i = 0; i < MMA_RM; i++)
                for (int j = 0; j < MMA_RN; j++)
                    acc[i][j] += a[i] * b[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    __global float *gd = AP(float, t.dst);
    for (int i = 0; i < MMA_RM; i++) {
        const uint gr = row0 + ty * MMA_RM + i;
        if (gr >= M) continue;
        for (int j = 0; j < MMA_RN; j++) {
            const uint gc = col0 + tx * MMA_RN + j;
            if (gc < N) gd[gr * N + gc] = acc[i][j];
        }
    }
}
