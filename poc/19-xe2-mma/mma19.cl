/* Parameterized register-blocked SGEMM for Xe2 tile-geometry sweep (poc/19).
 * Mirrors the shipped GPU mm2 kernel (pjrt_plugin/kernels/vm_main.cl) but with
 * TM/TN/TD/BK as build -D options so the harness can sweep register-block shape
 * without editing the plugin. Double-buffered __local staging, RMxRN register
 * microtile per work-item. C = A(MxK) @ B(KxN), row-major, N%TN==M%TM==0.
 *
 * Optional -DUSE_SG=<w>: use Intel subgroup block reads for the B panel (needs
 * cl_intel_subgroups); off by default so the base kernel is portable. */
#ifndef TM
#define TM 128
#endif
#ifndef TN
#define TN 64
#endif
#ifndef TD
#define TD 16
#endif
#ifndef BK
#define BK 16
#endif
#define NT (TD * TD)
#define RM (TM / TD)
#define RN (TN / TD)

__kernel __attribute__((reqd_work_group_size(NT, 1, 1)))
void mma(__global const float *A, __global const float *B, __global float *C,
         const uint M, const uint N, const uint K)
{
    __local float As[2][BK * TM];   /* transposed: As[buf][kk*TM + m] */
    __local float Bs[2][BK * TN];   /* Bs[buf][kk*TN + n] */
    const uint lid = get_local_id(0);
    const uint tiles_n = (N + TN - 1) / TN;
    const uint tile = get_group_id(0);
    const uint tr = tile / tiles_n, tc = tile % tiles_n;
    const uint row0 = tr * TM, col0 = tc * TN;
    const uint ty = lid / TD, tx = lid % TD;

    float acc[RM][RN];
    for (int i = 0; i < RM; i++)
        for (int j = 0; j < RN; j++) acc[i][j] = 0.0f;

#define STAGE(BUF, K0)                                                        \
    do {                                                                      \
        for (uint idx = lid; idx < TM * BK; idx += NT) {                      \
            const uint m = idx / BK, kk = idx % BK;                           \
            const uint gr = row0 + m, gk = (K0) + kk;                         \
            As[BUF][kk * TM + m] = (gr < M && gk < K) ? A[gr * K + gk] : 0.0f; \
        }                                                                     \
        for (uint idx = lid; idx < BK * TN; idx += NT) {                      \
            const uint kk = idx / TN, n = idx % TN;                           \
            const uint gk = (K0) + kk, gc = col0 + n;                         \
            Bs[BUF][kk * TN + n] = (gk < K && gc < N) ? B[gk * N + gc] : 0.0f; \
        }                                                                     \
    } while (0)

    STAGE(0, 0);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint buf = 0;
    for (uint k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K) STAGE(buf ^ 1, k0 + BK);
        for (uint kk = 0; kk < BK; ++kk) {
            float a[RM], b[RN];
            for (int i = 0; i < RM; i++)
                a[i] = As[buf][kk * TM + ty * RM + i];
            for (int j = 0; j < RN; j++)
                b[j] = Bs[buf][kk * TN + tx * RN + j];
            for (int i = 0; i < RM; i++)
                for (int j = 0; j < RN; j++)
                    acc[i][j] += a[i] * b[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        buf ^= 1;
    }
#undef STAGE
    for (int i = 0; i < RM; i++) {
        const uint gr = row0 + ty * RM + i;
        if (gr >= M) continue;
        for (int j = 0; j < RN; j++) {
            const uint gc = col0 + tx * RN + j;
            if (gc < N) C[gr * N + gc] = acc[i][j];
        }
    }
}
