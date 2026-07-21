/* poc/17 standalone tensor-core (tf32 m16n16k8) matmul, fully parametrized, for
 * the §31/§35/§36 ceiling measurement. NO megakernel barrier here, so the tile is
 * free to use a big register accumulator / low occupancy — this measures the true
 * ceiling of the OpenCL->PTX WMMA path on this ICD, decoupled from the megakernel
 * co-residency cap (§10c/§27).
 *
 * Build knobs:
 *   -DTM= -DTN=   output tile (per block)
 *   -DBK=         K-block
 *   -DNBUF=       synchronous smem multi-buffer {1,2,3}: prefetch next K-block
 *                 into another panel with ordinary ld.global/st.shared while the
 *                 tensor cores consume the current (cp.async is DEAD on this
 *                 driver — §35 — so staging is SYNCHRONOUS).
 *   -DWM= -DWN=   warp grid (rows x cols); threads = WM*WN*32; each warp owns
 *                 RF x TNW  16x16 fragments (RF*TNW*8 accumulator regs).
 *   -DVEC4=1      float4 vectorized global->smem staging.
 *   -DPAD=        smem leading-dim pad (bank-conflict mitig, default 4).
 * C = A(MxK) @ B(KxN), row-major, A@aoff B@boff C@coff in arena.
 *
 * KEY §36 FINDING: the original B staging read B[gk*N+gc] with adjacent threads
 * differing in gk (stride N) => fully UNCOALESCED. We stage B coalesced-by-n
 * (adjacent threads -> adjacent gc -> adjacent global addresses). A was already
 * coalesced (adjacent threads -> adjacent gk, contiguous). */
#ifndef TM
#define TM 128
#endif
#ifndef TN
#define TN 128
#endif
#ifndef BK
#define BK 16
#endif
#ifndef NBUF
#define NBUF 2
#endif
#ifndef WM
#define WM 4
#endif
#ifndef WN
#define WN 2
#endif
#ifndef VEC4
#define VEC4 0
#endif
#ifndef PAD
#define PAD 4
#endif
#ifndef PIPE
#define PIPE 0
#endif
#define LDS  (BK + PAD)
#define NTHREADS (WM * WN * 32)
#define RF   (TM / (WM * 16))        /* 16-row frags per warp (M) */
#define TNW  (TN / (WN * 16))        /* 16-col frags per warp (N) */
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

__attribute__((reqd_work_group_size(NTHREADS, 1, 1)))
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
    const uint wm = warp % WM, wn = warp / WM;

    for (uint tile = get_group_id(0); tile < total; tile += nlanes) {
        const uint tr = tile / tiles_n, tc = tile % tiles_n;
        const uint row0 = tr * TM, col0 = tc * TN;
        const int full = (row0 + TM <= M) && (col0 + TN <= N);
        float acc[RF][TNW][8];
        for (int i = 0; i < RF; i++)
            for (int j = 0; j < TNW; j++)
                for (int e = 0; e < 8; e++) acc[i][j][e] = 0.0f;

        /* --- staging: A coalesced-by-k (contiguous), B coalesced-by-n --- */
#define STAGE(BUF, K0)                                                         \
        do {                                                                  \
            __local float *As_ = As + (size_t)(BUF) * TM * LDS;               \
            __local float *Bs_ = Bs + (size_t)(BUF) * TN * LDS;               \
            if (VEC4 && full && ((K0) + BK <= K)) {                           \
                for (uint u = lid; u < TM * BK / 4; u += NTHREADS) {          \
                    const uint m = (u * 4) / BK, kk = (u * 4) % BK;           \
                    float4 v = vload4(0, A + (size_t)(row0 + m) * K + (K0) + kk); \
                    vstore4(v, 0, As_ + m * LDS + kk);                        \
                }                                                             \
                for (uint u = lid; u < TN * BK / 4; u += NTHREADS) {          \
                    const uint kk = u % BK, n0 = (u / BK) * 4;                \
                    float4 v = vload4(0, B + (size_t)((K0) + kk) * N + col0 + n0); \
                    Bs_[(n0 + 0) * LDS + kk] = v.x;                           \
                    Bs_[(n0 + 1) * LDS + kk] = v.y;                           \
                    Bs_[(n0 + 2) * LDS + kk] = v.z;                           \
                    Bs_[(n0 + 3) * LDS + kk] = v.w;                           \
                }                                                             \
            } else {                                                          \
                for (uint idx = lid; idx < TM * BK; idx += NTHREADS) {        \
                    const uint m = idx / BK, kk = idx % BK;                   \
                    const uint gr = row0 + m, gk = (K0) + kk;                 \
                    As_[m * LDS + kk] = (gr < M && gk < K) ? A[(size_t)gr * K + gk] : 0.f; \
                }                                                             \
                for (uint idx = lid; idx < TN * BK; idx += NTHREADS) {        \
                    const uint kk = idx % BK, n = idx / BK;                   \
                    const uint gk = (K0) + kk, gc = col0 + n;                 \
                    Bs_[n * LDS + kk] = (gk < K && gc < N) ? B[(size_t)gk * N + gc] : 0.f; \
                }                                                             \
            }                                                                 \
        } while (0)

#if PIPE
/* fragment-level software pipeline: prefetch next ksub's smem fragments into
 * registers while the tensor cores consume the current ksub, hiding wmma.load
 * (smem) latency in the MMA dependency chain (§36 — the sync analogue of what a
 * cp.async multistage pipeline would do for global latency). */
#define COMPUTE(BUF)                                                          \
        do {                                                                  \
            __local float *As_ = As + (size_t)(BUF) * TM * LDS;               \
            __local float *Bs_ = Bs + (size_t)(BUF) * TN * LDS;               \
            uint af[2][RF][4], bf[2][TNW][4];                                  \
            for (int i = 0; i < RF; i++)                                       \
                TC_LOAD_A(af[0][i], &As_[(wm*RF*16 + i*16)*LDS + 0], LDS);     \
            for (int j = 0; j < TNW; j++)                                      \
                TC_LOAD_B(bf[0][j], &Bs_[(wn*TNW*16 + j*16)*LDS + 0], LDS);    \
            for (uint ks = 0; ks < KSUB; ks++) {                              \
                const uint cur = ks & 1u, nxt = cur ^ 1u;                      \
                if (ks + 1 < KSUB) {                                          \
                    for (int i = 0; i < RF; i++)                               \
                        TC_LOAD_A(af[nxt][i], &As_[(wm*RF*16 + i*16)*LDS + (ks+1)*8], LDS); \
                    for (int j = 0; j < TNW; j++)                              \
                        TC_LOAD_B(bf[nxt][j], &Bs_[(wn*TNW*16 + j*16)*LDS + (ks+1)*8], LDS);\
                }                                                             \
                for (int i = 0; i < RF; i++)                                   \
                    for (int j = 0; j < TNW; j++)                              \
                        TC_MMA(acc[i][j], af[cur][i], bf[cur][j]);             \
            }                                                                 \
        } while (0)
#else
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
#endif

#if NBUF >= 2
        uint buf = 0;
        STAGE(0, 0);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k0 = 0; k0 < K; k0 += BK) {
            if (k0 + BK < K) STAGE((buf + 1u) % NBUF, k0 + BK);
            COMPUTE(buf);
            barrier(CLK_LOCAL_MEM_FENCE);
            buf = (buf + 1u) % NBUF;
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
                if (gr < M && gc < N) C[(size_t)gr * N + gc] = acc[i][j][reg];
            }
        }
    }
}
