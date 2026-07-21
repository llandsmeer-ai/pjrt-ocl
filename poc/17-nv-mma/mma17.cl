/* poc/17 standalone tensor-core (tf32 m16n16k8) matmul, parametrized, for the
 * §31/§35 ceiling measurement. NO megakernel barrier here, so the tile is free
 * to use a big register accumulator / low occupancy — this measures the true
 * ceiling of the OpenCL->PTX WMMA path on this ICD, decoupled from the
 * megakernel co-residency cap (§10c/§27).
 *
 * Build knobs: -DTM= -DTN= (output tile), -DBK= (K-block), -DNBUF={1,2}
 * (synchronous smem multi-buffer: prefetch next K-block into the other panel
 * with ordinary ld.global/st.shared while the tensor cores consume the current;
 * cp.async is UNAVAILABLE on this driver — proven in probe.c — so staging is
 * synchronous). C = A(MxK) @ B(KxN), row-major, A@aoff B@boff C@coff in arena. */
#ifndef TM
#define TM 64
#endif
#ifndef TN
#define TN 64
#endif
#ifndef BK
#define BK 16
#endif
#ifndef NBUF
#define NBUF 1
#endif
#define LDS (BK + 4)                 /* padded leading dim, bank-conflict mitig */
#define TDIM 16
/* 8 warps in a 4(row) x 2(col) grid; each warp owns RF x TNW 16x16 fragments */
#define WROWS 4
#define WCOLS 2
#define RF   (TM / (WROWS * 16))     /* 16-row frags per warp */
#define TNW  (TN / (WCOLS * 16))     /* 16-col frags per warp */
#define KSUB (BK / 8)

#define TC_LOAD_A(f, ptr, stride)                                              \
    asm volatile("{ .reg .u64 sp; cvta.to.shared.u64 sp, %4;\n"                \
        "wmma.load.a.sync.aligned.m16n16k8.shared.row.tf32 {%0,%1,%2,%3}, [sp], %5; }" \
        : "=r"((f)[0]),"=r"((f)[1]),"=r"((f)[2]),"=r"((f)[3])                  \
        : "l"(ptr),"r"(stride))
#define TC_LOAD_B(f, ptr, stride)                                             \
    asm volatile("{ .reg .u64 sp; cvta.to.shared.u64 sp, %4;\n"                \
        "wmma.load.b.sync.aligned.m16n16k8.shared.col.tf32 {%0,%1,%2,%3}, [sp], %5; }" \
        : "=r"((f)[0]),"=r"((f)[1]),"=r"((f)[2]),"=r"((f)[3])                  \
        : "l"(ptr),"r"(stride))
#define TC_MMA(acc, a, b)                                                     \
    asm volatile("wmma.mma.sync.aligned.row.col.m16n16k8.f32.tf32.tf32.f32\n"  \
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15},\n"     \
        "{%0,%1,%2,%3,%4,%5,%6,%7};"                                           \
        : "+f"((acc)[0]),"+f"((acc)[1]),"+f"((acc)[2]),"+f"((acc)[3]),         \
          "+f"((acc)[4]),"+f"((acc)[5]),"+f"((acc)[6]),"+f"((acc)[7])          \
        : "r"((a)[0]),"r"((a)[1]),"r"((a)[2]),"r"((a)[3]),                     \
          "r"((b)[0]),"r"((b)[1]),"r"((b)[2]),"r"((b)[3]))

kernel void mm(global float *arena, uint aoff, uint boff, uint coff,
               uint M, uint N, uint K, uint nlanes)
{
    __local float As[NBUF * TM * LDS];
    __local float Bs[NBUF * TN * LDS];
    global const float *A = arena + aoff;
    global const float *B = arena + boff;
    global float *C = arena + coff;

    const uint tiles_n = (N + TN - 1) / TN;
    const uint tiles_m = (M + TM - 1) / TM;
    const uint total = tiles_m * tiles_n;
    const uint lid = get_local_id(0);
    const uint warp = lid >> 5, lane = lid & 31;
    const uint wm = warp % WROWS, wn = warp / WROWS;

    for (uint tile = get_group_id(0); tile < total; tile += nlanes) {
        const uint tr = tile / tiles_n, tc = tile % tiles_n;
        const uint row0 = tr * TM, col0 = tc * TN;
        float acc[RF][TNW][8];
        for (int i = 0; i < RF; i++)
            for (int j = 0; j < TNW; j++)
                for (int e = 0; e < 8; e++) acc[i][j][e] = 0.0f;

#define STAGE(BUF, K0)                                                         \
        do {                                                                  \
            __local float *As_ = As + (size_t)(BUF) * TM * LDS;               \
            __local float *Bs_ = Bs + (size_t)(BUF) * TN * LDS;               \
            for (uint idx = lid; idx < TM * BK; idx += 256) {                 \
                const uint m = idx / BK, kk = idx % BK;                       \
                const uint gr = row0 + m, gk = (K0) + kk;                     \
                As_[m * LDS + kk] = (gr < M && gk < K) ? A[gr * K + gk] : 0.f; \
            }                                                                 \
            for (uint idx = lid; idx < TN * BK; idx += 256) {                 \
                const uint n = idx / BK, kk = idx % BK;                       \
                const uint gk = (K0) + kk, gc = col0 + n;                     \
                Bs_[n * LDS + kk] = (gk < K && gc < N) ? B[gk * N + gc] : 0.f; \
            }                                                                 \
        } while (0)

#define COMPUTE(BUF)                                                          \
        do {                                                                  \
            __local float *As_ = As + (size_t)(BUF) * TM * LDS;               \
            __local float *Bs_ = Bs + (size_t)(BUF) * TN * LDS;               \
            for (uint ks = 0; ks < KSUB; ks++) {                              \
                uint af[RF][4], bf[TNW][4];                                    \
                for (int i = 0; i < RF; i++)                                   \
                    TC_LOAD_A(af[i], &As_[(wm*RF*16 + i*16)*LDS + ks*8], LDS); \
                for (int j = 0; j < TNW; j++)                                  \
                    TC_LOAD_B(bf[j], &Bs_[(wn*TNW*16 + j*16)*LDS + ks*8], LDS);\
                for (int i = 0; i < RF; i++)                                   \
                    for (int j = 0; j < TNW; j++)                              \
                        TC_MMA(acc[i][j], af[i], bf[j]);                       \
            }                                                                 \
        } while (0)

#if NBUF == 2
        uint buf = 0;
        STAGE(0, 0);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k0 = 0; k0 < K; k0 += BK) {
            if (k0 + BK < K) STAGE(buf ^ 1u, k0 + BK);
            COMPUTE(buf);
            barrier(CLK_LOCAL_MEM_FENCE);
            buf ^= 1u;
        }
#else
        for (uint k0 = 0; k0 < K; k0 += BK) {
            STAGE(0, k0);
            barrier(CLK_LOCAL_MEM_FENCE);
            COMPUTE(0);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
#endif
#undef STAGE
#undef COMPUTE

        /* masked global store, m16n16k8 D-fragment map (poc/08 / mma.cl) */
        for (int i = 0; i < RF; i++)
        for (int j = 0; j < TNW; j++) {
            const uint gr0 = row0 + wm*RF*16 + i*16;
            const uint gc0 = col0 + wn*TNW*16 + j*16;
            for (int reg = 0; reg < 8; reg++) {
                const uint r = (lane >> 2) + 8u * ((reg >> 1) & 1);
                const uint c = (lane & 3) * 2 + (reg & 1) + 8u * (reg >> 2);
                const uint gr = gr0 + r, gc = gc0 + c;
                if (gr < M && gc < N) C[gr * N + gc] = acc[i][j][reg];
            }
        }
    }
}
