/* poc/17 half-precision tensor-core matmul ceiling probe (ADVERSARIAL follow-on
 * to §35/§36). Prior work fixed the input type to tf32 m16n16k8 and concluded
 * ~57 TF/s is the register-file-capped ceiling. This variant switches the MMA
 * INPUT precision to fp16 / bf16 (m16n16k16), which on Blackwell tensor cores
 * runs at 2x the tf32 rate, while the f32 ACCUMULATOR (the thing that actually
 * caps residency at 2 WG/SM) is byte-for-byte identical. If 57 were a pure
 * latency wall, this changes nothing; if it were the tf32 unit throughput, this
 * should ~double it. The arena stays f32 (so the bench harness + verify are
 * unchanged); staging converts f32 -> {f16,bf16} into smem.
 *
 *   -DHP=1  fp16 inputs (wmma ...m16n16k16.f16)
 *   -DHP=2  bf16 inputs (wmma ...m16n16k16.bf16) — wider exponent, closer to f32
 *   others: TM/TN/BK/NBUF/WM/WN/PAD as in mma17.cl. BK must be a multiple of 16.
 *   C = A(MxK) @ B(KxN) row-major, f32 accumulate, f32 store.
 */
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
#ifndef PAD
#define PAD 8
#endif
#ifndef HP
#define HP 1
#endif
#ifndef NOMMA
#define NOMMA 0
#endif
#ifndef NOLOAD
#define NOLOAD 0
#endif
#if NOMMA
/* diagnostic: skip the tensor-core op, accumulate a cheap smem-derived value so
 * the loads aren't dead-code-eliminated. isolates wmma-vs-staging faults. */
#define DO_MMA(acc, a, b) do { (acc)[0] += as_float((a)[0]) + as_float((b)[0]); } while(0)
#else
#define DO_MMA(acc, a, b) TC_MMA(acc, a, b)
#endif
#define LDS  (BK + PAD)
#define NTHREADS (WM * WN * 32)
#define RF   (TM / (WM * 16))
#define TNW  (TN / (WN * 16))
#define KSUB (BK / 16)                  /* k16 MMA */

/* smem element is a 16-bit half/bf16, stored as ushort bits. */
typedef ushort hstore;

#if HP == 1
/* --- fp16: legacy WMMA f16 fragment = 8 .b32 A/B regs, C/D f32 = 8 regs --- */
#define ABREG 8
#define CVT_STORE(dst, v) vstore_half((v), 0, (__local half *)&(dst))
#define TC_LOAD_A(f, ptr, stride)                                              \
    asm volatile("{ .reg .u64 sp; cvta.to.shared.u64 sp, %8;\n"                \
        "wmma.load.a.sync.aligned.m16n16k16.shared.row.f16"                    \
        " {%0,%1,%2,%3,%4,%5,%6,%7}, [sp], %9; }"                             \
        : "=r"((f)[0]),"=r"((f)[1]),"=r"((f)[2]),"=r"((f)[3]),                 \
          "=r"((f)[4]),"=r"((f)[5]),"=r"((f)[6]),"=r"((f)[7])                  \
        : "l"(ptr),"r"(stride))
#define TC_LOAD_B(f, ptr, stride)                                             \
    asm volatile("{ .reg .u64 sp; cvta.to.shared.u64 sp, %8;\n"                \
        "wmma.load.b.sync.aligned.m16n16k16.shared.col.f16"                    \
        " {%0,%1,%2,%3,%4,%5,%6,%7}, [sp], %9; }"                             \
        : "=r"((f)[0]),"=r"((f)[1]),"=r"((f)[2]),"=r"((f)[3]),                 \
          "=r"((f)[4]),"=r"((f)[5]),"=r"((f)[6]),"=r"((f)[7])                  \
        : "l"(ptr),"r"(stride))
#define TC_MMA(acc, a, b)                                                     \
    asm volatile("wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32\n"           \
        "{%0,%1,%2,%3,%4,%5,%6,%7},\n"                                         \
        "{%8,%9,%10,%11,%12,%13,%14,%15},\n"                                   \
        "{%16,%17,%18,%19,%20,%21,%22,%23},\n"                                 \
        "{%0,%1,%2,%3,%4,%5,%6,%7};"                                           \
        : "+f"((acc)[0]),"+f"((acc)[1]),"+f"((acc)[2]),"+f"((acc)[3]),         \
          "+f"((acc)[4]),"+f"((acc)[5]),"+f"((acc)[6]),"+f"((acc)[7])          \
        : "r"((a)[0]),"r"((a)[1]),"r"((a)[2]),"r"((a)[3]),                     \
          "r"((a)[4]),"r"((a)[5]),"r"((a)[6]),"r"((a)[7]),                     \
          "r"((b)[0]),"r"((b)[1]),"r"((b)[2]),"r"((b)[3]),                     \
          "r"((b)[4]),"r"((b)[5]),"r"((b)[6]),"r"((b)[7]))
#else
/* --- bf16: packed WMMA fragment = 4 .b32 A/B regs (mirrors tf32 kernel) --- */
#define ABREG 4
#define CVT_STORE(dst, v) do {                          \
        uint u_ = as_uint((float)(v));                  \
        u_ += 0x7fffu + ((u_ >> 16) & 1u);              \
        (dst) = (ushort)(u_ >> 16);                     \
    } while (0)
#define TC_LOAD_A(f, ptr, stride)                                              \
    asm volatile("{ .reg .u64 sp; cvta.to.shared.u64 sp, %4;\n"                \
        "wmma.load.a.sync.aligned.m16n16k16.shared.row.bf16"                   \
        " {%0,%1,%2,%3}, [sp], %5; }"                                         \
        : "=r"((f)[0]),"=r"((f)[1]),"=r"((f)[2]),"=r"((f)[3])                  \
        : "l"(ptr),"r"(stride))
#define TC_LOAD_B(f, ptr, stride)                                             \
    asm volatile("{ .reg .u64 sp; cvta.to.shared.u64 sp, %4;\n"                \
        "wmma.load.b.sync.aligned.m16n16k16.shared.col.bf16"                   \
        " {%0,%1,%2,%3}, [sp], %5; }"                                         \
        : "=r"((f)[0]),"=r"((f)[1]),"=r"((f)[2]),"=r"((f)[3])                  \
        : "l"(ptr),"r"(stride))
#define TC_MMA(acc, a, b)                                                     \
    asm volatile("wmma.mma.sync.aligned.row.col.m16n16k16.f32.bf16.bf16.f32\n" \
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15},\n"     \
        "{%0,%1,%2,%3,%4,%5,%6,%7};"                                           \
        : "+f"((acc)[0]),"+f"((acc)[1]),"+f"((acc)[2]),"+f"((acc)[3]),         \
          "+f"((acc)[4]),"+f"((acc)[5]),"+f"((acc)[6]),"+f"((acc)[7])          \
        : "r"((a)[0]),"r"((a)[1]),"r"((a)[2]),"r"((a)[3]),                     \
          "r"((b)[0]),"r"((b)[1]),"r"((b)[2]),"r"((b)[3]))
#endif

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
        float acc[RF][TNW][8];
        for (int i = 0; i < RF; i++)
            for (int j = 0; j < TNW; j++)
                for (int e = 0; e < 8; e++) acc[i][j][e] = 0.0f;

#define STAGE(BUF, K0)                                                         \
        do {                                                                  \
            __local hstore *As_ = As + (size_t)(BUF) * TM * LDS;              \
            __local hstore *Bs_ = Bs + (size_t)(BUF) * TN * LDS;              \
            for (uint idx = lid; idx < TM * BK; idx += NTHREADS) {            \
                const uint m = idx / BK, kk = idx % BK;                       \
                const uint gr = row0 + m, gk = (K0) + kk;                     \
                float v = (gr < M && gk < K) ? A[(size_t)gr * K + gk] : 0.f;  \
                CVT_STORE(As_[m * LDS + kk], v);                              \
            }                                                                 \
            for (uint idx = lid; idx < TN * BK; idx += NTHREADS) {           \
                const uint kk = idx % BK, n = idx / BK;                       \
                const uint gk = (K0) + kk, gc = col0 + n;                     \
                float v = (gk < K && gc < N) ? B[(size_t)gk * N + gc] : 0.f;  \
                CVT_STORE(Bs_[n * LDS + kk], v);                              \
            }                                                                 \
        } while (0)

#define COMPUTE(BUF)                                                          \
        do {                                                                  \
            __local hstore *As_ = As + (size_t)(BUF) * TM * LDS;              \
            __local hstore *Bs_ = Bs + (size_t)(BUF) * TN * LDS;              \
            for (uint ks = 0; ks < KSUB; ks++) {                             \
                uint af[RF][ABREG], bf[TNW][ABREG];                           \
                for (int i = 0; i < RF; i++) af[i][0]=0;                       \
                for (int j = 0; j < TNW; j++) bf[j][0]=0;                      \
                if (!NOLOAD) {                                                 \
                for (int i = 0; i < RF; i++)                                  \
                    TC_LOAD_A(af[i], &As_[(wm*RF*16 + i*16)*LDS + ks*16], LDS);\
                for (int j = 0; j < TNW; j++)                                 \
                    TC_LOAD_B(bf[j], &Bs_[(wn*TNW*16 + j*16)*LDS + ks*16], LDS);\
                }                                                             \
                if (NOLOAD) { for(int i=0;i<RF;i++) af[i][0]=As_[i]; for(int j=0;j<TNW;j++) bf[j][0]=Bs_[j]; } \
                for (int i = 0; i < RF; i++)                                  \
                    for (int j = 0; j < TNW; j++)                             \
                        DO_MMA(acc[i][j], af[i], bf[j]);                      \
            }                                                                 \
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
