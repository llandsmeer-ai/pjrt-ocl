/* poc/17 §39 — f16-INPUT variant of the mma.sync ceiling probe. Identical tile
 * to mma17_mma.cl, but A and B arrive as global `half` arrays (not f32). The
 * STAGE decomposition showed the kernel is bound by global reads (0.49ms of a
 * 1.49ms @4096) NOT by the tensor op (free). The f32 arena reads 4 bytes/elem
 * for a value the tensor core consumes at 2 bytes -> 2x wasted global traffic.
 * This variant halves staging bytes: does it break the 92 TF/s wall (and 134)?
 * C = A(MxK) @ B(KxN), f16 inputs, f32 accumulate, f32 store. Same layouts. */
#ifndef TM
#define TM 256
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
#define WM 8
#endif
#ifndef WN
#define WN 4
#endif
#ifndef PAD
#define PAD 8
#endif
#define LDS  (BK + PAD)
#define NTHREADS (WM * WN * 32)
#define RF   ((TM / WM) / 16)
#define TNW  ((TN / WN) / 8)
#define KSUB (BK / 16)

typedef ushort hstore;

#define LDM_A(f, base_ptr, lds)                                                 \
    do {                                                                        \
        uint _mat = (lane >> 3) & 3u, _wi = lane & 7u;                          \
        __local const hstore *_p = (base_ptr) + ((_mat & 1u) * 8u + _wi) * (lds)\
                                    + (_mat >> 1) * 8u;                          \
        asm volatile("{ .reg .u64 sp; cvta.to.shared.u64 sp, %4;\n"             \
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [sp]; }"   \
            : "=r"((f)[0]),"=r"((f)[1]),"=r"((f)[2]),"=r"((f)[3])               \
            : "l"(_p));                                                         \
    } while (0)
#define LDM_B(f, base_ptr, lds)                                                 \
    do {                                                                        \
        uint _mat = (lane >> 3) & 1u, _wi = lane & 7u;                          \
        __local const hstore *_p = (base_ptr) + _wi * (lds) + _mat * 8u;        \
        asm volatile("{ .reg .u64 sp; cvta.to.shared.u64 sp, %2;\n"             \
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [sp]; }"         \
            : "=r"((f)[0]),"=r"((f)[1])                                         \
            : "l"(_p));                                                         \
    } while (0)
#define MMA(acc, a, b)                                                          \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n"          \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"                 \
        : "+f"((acc)[0]),"+f"((acc)[1]),"+f"((acc)[2]),"+f"((acc)[3])           \
        : "r"((a)[0]),"r"((a)[1]),"r"((a)[2]),"r"((a)[3]),                      \
          "r"((b)[0]),"r"((b)[1]))

/* A, B are half*; C is float*. All three live in the same f32 `arena` buffer;
 * aoff/boff/coff are FLOAT-element offsets from the host (A,B regions hold
 * packed halves: M*K halves occupy M*K/2 floats). */
__attribute__((reqd_work_group_size(NTHREADS, 1, 1)))
kernel void mm(global float *arena, uint aoff, uint boff, uint coff,
               uint M, uint N, uint K, uint nlanes)
{
    __local __attribute__((aligned(16))) hstore As[NBUF * TM * LDS];
    __local __attribute__((aligned(16))) hstore Bs[NBUF * TN * LDS];
    global const half *A = (global const half *)(arena + aoff);
    global const half *B = (global const half *)(arena + boff);
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
        float acc[RF][TNW][4];
        for (int i = 0; i < RF; i++)
            for (int j = 0; j < TNW; j++)
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

/* coalesced vec-staging: read 8 contiguous halves (ushort4 = 8 halves) per lane
 * where the leading dim is contiguous (A along K, B along N). */
#define STAGE(BUF, K0)                                                          \
        do {                                                                    \
            __local hstore *As_ = As + (size_t)(BUF) * TM * LDS;                \
            __local hstore *Bs_ = Bs + (size_t)(BUF) * TN * LDS;                \
            const int full = (row0 + TM <= M) && (col0 + TN <= N) &&            \
                             ((K0) + BK <= K);                                  \
            if (full) {                                                         \
                for (uint q = lid; q < TM * (BK/8); q += NTHREADS) {            \
                    const uint m = q / (BK/8), kk = (q % (BK/8)) * 8u;          \
                    ushort8 v = vload8(0, (global const ushort*)A               \
                                        + (size_t)(row0+m)*K + (K0) + kk);       \
                    vstore8(v, 0, (__local ushort*)&As_[m*LDS+kk]);            \
                }                                                               \
                for (uint q = lid; q < BK * (TN/8); q += NTHREADS) {            \
                    const uint kk = q / (TN/8), n = (q % (TN/8)) * 8u;          \
                    ushort8 v = vload8(0, (global const ushort*)B               \
                                        + (size_t)((K0)+kk)*N + col0 + n);       \
                    for (uint e=0;e<8;e++) Bs_[(n+e)*LDS+kk]=((ushort*)&v)[e];  \
                }                                                               \
            } else {                                                            \
                for (uint idx = lid; idx < TM * BK; idx += NTHREADS) {          \
                    const uint m = idx / BK, kk = idx % BK;                     \
                    const uint gr = row0 + m, gk = (K0) + kk;                   \
                    As_[m*LDS+kk] = (gr<M&&gk<K) ?                              \
                        ((global const ushort*)A)[(size_t)gr*K+gk] : 0;         \
                }                                                               \
                for (uint idx = lid; idx < TN * BK; idx += NTHREADS) {          \
                    const uint kk = idx % BK, n = idx / BK;                     \
                    const uint gk = (K0) + kk, gc = col0 + n;                   \
                    Bs_[n*LDS+kk] = (gk<K&&gc<N) ?                              \
                        ((global const ushort*)B)[(size_t)gk*N+gc] : 0;         \
                }                                                               \
            }                                                                   \
        } while (0)

#define COMPUTE(BUF)                                                            \
        do {                                                                    \
            __local hstore *As_ = As + (size_t)(BUF) * TM * LDS;                \
            __local hstore *Bs_ = Bs + (size_t)(BUF) * TN * LDS;                \
            for (uint ks = 0; ks < KSUB; ks++) {                               \
                uint af[RF][4], bf[TNW][2];                                     \
                for (int i = 0; i < RF; i++)                                    \
                    LDM_A(af[i], &As_[(wm*RF*16 + i*16)*LDS + ks*16], LDS);     \
                for (int j = 0; j < TNW; j++)                                   \
                    LDM_B(bf[j], &Bs_[(wn*TNW*8 + j*8)*LDS + ks*16], LDS);      \
                for (int i = 0; i < RF; i++)                                    \
                    for (int j = 0; j < TNW; j++)                               \
                        MMA(acc[i][j], af[i], bf[j]);                           \
            }                                                                   \
        } while (0)

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

        for (int i = 0; i < RF; i++)
        for (int j = 0; j < TNW; j++) {
            const uint gr0 = row0 + wm*RF*16 + i*16;
            const uint gc0 = col0 + wn*TNW*8 + j*8;
            for (int reg = 0; reg < 4; reg++) {
                const uint r = (lane >> 2) + 8u * (reg >> 1);
                const uint c = (lane & 3) * 2 + (reg & 1);
                const uint gr = gr0 + r, gc = gc0 + c;
                if (gr < M && gc < N) C[(size_t)gr * N + gc] = acc[i][j][reg];
            }
        }
    }
}
