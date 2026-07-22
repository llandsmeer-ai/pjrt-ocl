/* poc/17 §39 — modern mma.sync.m16n8k16 + ldmatrix f16 tensor-core matmul.
 *
 * ADVERSARIAL follow-on to §38 (legacy wmma.mma.m16n16k16 reached ~92 TF/s).
 * §38 was occupancy/latency-bound (win came from a thinner f32 accumulator per
 * thread -> more WG/SM). Legacy WMMA carries a HEAVY fragment register cost:
 * f16 A/B = 8 .b32 regs EACH per k16 warp-tile. The modern path
 *   mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
 * uses A = 4 .b32, B = 2 .b32 (3-4x less fragment RF) and loads fragments with
 * ldmatrix (the same HW path wmma uses internally, but we control the count).
 * Hypothesis: lower fragment pressure -> even higher occupancy -> break 92.
 *
 * Outer tile TMxTN, NTHREADS = WM*WN*32. Each warp owns (TM/WM)x(TN/WN); mma
 * tile is 16x8 so per warp RF=(TM/WM)/16 m-subtiles, TNW=(TN/WN)/8 n-subtiles,
 * acc = RF*TNW*4 f32 regs/thread. C=A(MxK)@B(KxN) row-major, f32 acc, f32 store.
 *   -DBK multiple of 16.  arena stays f32; staging converts f32->f16 into smem.
 */
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
#ifndef VEC4
#define VEC4 0
#endif
#define LDS  (BK + PAD)
#define NTHREADS (WM * WN * 32)
#define RF   ((TM / WM) / 16)          /* m16 subtiles per warp */
#define TNW  ((TN / WN) / 8)           /* n8  subtiles per warp */
#define KSUB (BK / 16)                 /* k16 MMA steps per stage */

typedef ushort hstore;

/* ldmatrix: load a 16x16 A tile (row-major smem [M][K]) -> 4 .b32 regs, and a
 * 16x8 B tile (smem [N][K], N-major) transposed -> 2 .b32 regs. */
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

#ifndef BVAR
#define BVAR 1
#endif
#if BVAR == 0
/* Bs=[N][K]; trans; thread provides an N-row start, 2 mats split K */
#define LDM_B(f, base_ptr, lds)                                                 \
    do {                                                                        \
        uint _mat = (lane >> 3) & 1u, _wi = lane & 7u;                          \
        __local const hstore *_p = (base_ptr) + _wi * (lds) + _mat * 8u;        \
        asm volatile("{ .reg .u64 sp; cvta.to.shared.u64 sp, %2;\n"             \
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [sp]; }"   \
            : "=r"((f)[0]),"=r"((f)[1])                                         \
            : "l"(_p));                                                         \
    } while (0)
#elif BVAR == 1
/* Bs=[N][K]; NO trans; thread provides an N-row start, 2 mats split K */
#define LDM_B(f, base_ptr, lds)                                                 \
    do {                                                                        \
        uint _mat = (lane >> 3) & 1u, _wi = lane & 7u;                          \
        __local const hstore *_p = (base_ptr) + _wi * (lds) + _mat * 8u;        \
        asm volatile("{ .reg .u64 sp; cvta.to.shared.u64 sp, %2;\n"             \
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [sp]; }"         \
            : "=r"((f)[0]),"=r"((f)[1])                                         \
            : "l"(_p));                                                         \
    } while (0)
#elif BVAR == 2
/* Bs=[N][K]; trans; mats split N-rows(0-7,8-15 irrelevant), addr along K */
#define LDM_B(f, base_ptr, lds)                                                 \
    do {                                                                        \
        uint _wi = lane & 7u;                                                    \
        __local const hstore *_p = (base_ptr) + _wi * (lds);                     \
        asm volatile("{ .reg .u64 sp; cvta.to.shared.u64 sp, %2;\n"             \
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [sp]; }"   \
            : "=r"((f)[0]),"=r"((f)[1])                                         \
            : "l"(_p));                                                         \
    } while (0)
#endif

#define MMA(acc, a, b)                                                          \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n"          \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"                 \
        : "+f"((acc)[0]),"+f"((acc)[1]),"+f"((acc)[2]),"+f"((acc)[3])           \
        : "r"((a)[0]),"r"((a)[1]),"r"((a)[2]),"r"((a)[3]),                      \
          "r"((b)[0]),"r"((b)[1]))

#define CVT_STORE(dst, v) vstore_half((v), 0, (__local half *)&(dst))

__attribute__((reqd_work_group_size(NTHREADS, 1, 1)))
kernel void mm(global float *arena, uint aoff, uint boff, uint coff,
               uint M, uint N, uint K, uint nlanes)
{
    __local __attribute__((aligned(16))) hstore As[NBUF * TM * LDS];
    __local __attribute__((aligned(16))) hstore Bs[NBUF * TN * LDS];
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
        float acc[RF][TNW][4];
        for (int i = 0; i < RF; i++)
            for (int j = 0; j < TNW; j++)
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

/* Coalesced float4 staging (§39). The old scalar layout read B UNCOALESCED
 * (consecutive lanes strode global by N). Here consecutive lanes read a
 * contiguous float4: A along K (its contiguous dim), B along N (its contiguous
 * dim). `full` = interior tile (all divisible on the bench), else scalar edge. */
#if VEC4
#define STAGE(BUF, K0)                                                          \
        do {                                                                    \
            __local hstore *As_ = As + (size_t)(BUF) * TM * LDS;                \
            __local hstore *Bs_ = Bs + (size_t)(BUF) * TN * LDS;                \
            const int full = (row0 + TM <= M) && (col0 + TN <= N) &&            \
                             ((K0) + BK <= K);                                  \
            if (full) {                                                         \
                for (uint q = lid; q < TM * (BK/4); q += NTHREADS) {            \
                    const uint m = q / (BK/4), kk = (q % (BK/4)) * 4u;          \
                    float4 v = vload4(0, A + (size_t)(row0+m)*K + (K0) + kk);   \
                    CVT_STORE(As_[m*LDS+kk+0], v.x); CVT_STORE(As_[m*LDS+kk+1], v.y); \
                    CVT_STORE(As_[m*LDS+kk+2], v.z); CVT_STORE(As_[m*LDS+kk+3], v.w); \
                }                                                               \
                for (uint q = lid; q < BK * (TN/4); q += NTHREADS) {            \
                    const uint kk = q / (TN/4), n = (q % (TN/4)) * 4u;          \
                    float4 v = vload4(0, B + (size_t)((K0)+kk)*N + col0 + n);   \
                    CVT_STORE(Bs_[(n+0)*LDS+kk], v.x); CVT_STORE(Bs_[(n+1)*LDS+kk], v.y); \
                    CVT_STORE(Bs_[(n+2)*LDS+kk], v.z); CVT_STORE(Bs_[(n+3)*LDS+kk], v.w); \
                }                                                               \
            } else {                                                            \
                for (uint idx = lid; idx < TM * BK; idx += NTHREADS) {          \
                    const uint m = idx / BK, kk = idx % BK;                     \
                    const uint gr = row0 + m, gk = (K0) + kk;                   \
                    float v = (gr < M && gk < K) ? A[(size_t)gr*K+gk] : 0.f;    \
                    CVT_STORE(As_[m*LDS+kk], v);                                \
                }                                                               \
                for (uint idx = lid; idx < TN * BK; idx += NTHREADS) {          \
                    const uint kk = idx % BK, n = idx / BK;                     \
                    const uint gk = (K0) + kk, gc = col0 + n;                   \
                    float v = (gk < K && gc < N) ? B[(size_t)gk*N+gc] : 0.f;    \
                    CVT_STORE(Bs_[n*LDS+kk], v);                                \
                }                                                               \
            }                                                                   \
        } while (0)
#else
#ifndef NOGLOB
#define NOGLOB 0
#endif
#ifndef NOSMEM
#define NOSMEM 0
#endif
#define STAGE(BUF, K0)                                                          \
        do {                                                                    \
            __local hstore *As_ = As + (size_t)(BUF) * TM * LDS;                \
            __local hstore *Bs_ = Bs + (size_t)(BUF) * TN * LDS;                \
            for (uint idx = lid; idx < TM * BK; idx += NTHREADS) {              \
                const uint m = idx / BK, kk = idx % BK;                         \
                const uint gr = row0 + m, gk = (K0) + kk;                       \
                float v = NOGLOB ? (float)(gr+gk) :                             \
                    ((gr < M && gk < K) ? A[(size_t)gr * K + gk] : 0.f);        \
                if (!NOSMEM) CVT_STORE(As_[m * LDS + kk], v);                   \
                else if (idx==0) As_[0]=(hstore)v;                              \
            }                                                                   \
            for (uint idx = lid; idx < TN * BK; idx += NTHREADS) {              \
                const uint kk = idx % BK, n = idx / BK;                         \
                const uint gk = (K0) + kk, gc = col0 + n;                       \
                float v = NOGLOB ? (float)(gk+gc) :                             \
                    ((gk < K && gc < N) ? B[(size_t)gk * N + gc] : 0.f);        \
                if (!NOSMEM) CVT_STORE(Bs_[n * LDS + kk], v);                   \
                else if (idx==0) Bs_[0]=(hstore)v;                              \
            }                                                                   \
        } while (0)
#endif

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

        /* mma.m16n8k16 D-fragment store map: 4 f32 regs/thread.
         *   row(reg) = (lane>>2) + 8*(reg>>1)
         *   col(reg) = (lane&3)*2 + (reg&1)                                    */
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
