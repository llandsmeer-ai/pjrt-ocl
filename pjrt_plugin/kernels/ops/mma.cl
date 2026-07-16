/* Register-blocked SGEMM tile (TOP_MMA), from poc/06 step 2 (portable champion
 * family). One 256-thread workgroup computes one MMA_TM x MMA_TN output tile;
 * each thread owns an RM x RN = 4x4 register microtile. Scalar edge-guarded
 * staging, single-buffered -> portable to PoCL. Local: BK*(TM+TN) floats. The
 * scheduler tiles matmul in MMA_TM x MMA_TN blocks (scheduler.MMA_T==MMA_TM).
 * 4x4 (16 accumulators) chosen to bound the megakernel's occupancy tax
 * (docs/tile-isa.md ceiling-1).
 *
 * NVIDIA TF32 tensor-core variant (docs/decisions.md §10b): when the megakernel
 * program is built with -DVMO_NV_PTX (a NVIDIA-only build variant, see
 * runtime.cc Create — inline PTX is rejected by PoCL/AMD/Intel and MUST NOT
 * enter the portable program), vmo_mma_tile computes the SAME 64x64 output tile
 * with wmma.mma.sync m16n16k8 TF32 tensor cores instead of the scalar 4x4
 * microtile. Same As/Bs local footprint (64*16 each) and a comparable register
 * count (acc[2][8]+af[4]+bf[4] ~= 24 vs the scalar acc[4][4]+a[4]+b[4]), so the
 * whole-megakernel occupancy is preserved (measured — the register budget is a
 * max over all op paths, docs/decisions.md §10b). Mechanism proven in
 * poc/08-tensor-core-mma: A .row / B .col (col-major staged), __local ptrs need
 * cvta.to.shared, wmma.store.d.shared is broken so edges use a hand-mapped
 * direct global store. Batch (t.p3>1, attention) handled by the shared tile/g
 * setup, so batched matmul gets tensor cores too. */
#define MMA_TM 64
#define MMA_TN 64
#define MMA_BK 16
#define MMA_TDIM 16          /* 16x16 thread grid == 256 threads */
#define MMA_RM (MMA_TM / MMA_TDIM)   /* 4 */
#define MMA_RN (MMA_TN / MMA_TDIM)   /* 4 */
#define MMA_ASZ (MMA_BK * MMA_TM)    /* As[m*BK + k] */
#define MMA_BSZ (MMA_BK * MMA_TN)    /* Bs[k*TN+n] portable / Bs[n*BK+k] TC */

#ifdef VMO_NV_PTX
/* --- inline-PTX WMMA helpers (tf32 m16n16k8, from poc/08 / tc_mma.cl) ------ */
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

#define TC_TNW 2             /* 16-wide col subtiles per warp (2*16 == 32) */
#define TC_KSUB (MMA_BK / 8) /* wmma k-substeps per staged BK block */

static void vmo_mma_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                     const task_t t, uint tile,
                     __local float *As, __local float *Bs)
{
    const uint M = t.p0, N = t.p1, K = t.p2;
    const uint tiles_n = (N + MMA_TN - 1) / MMA_TN;
    const uint tiles_m = (M + MMA_TM - 1) / MMA_TM;
    const uint per = tiles_m * tiles_n;
    const uint g = tile / per, loc = tile % per;
    const uint tr = loc / tiles_n, tc = loc % tiles_n;
    const uint row0 = tr * MMA_TM, col0 = tc * MMA_TN;
    const uint lid = get_local_id(0);
    const uint warp = lid >> 5, lane = lid & 31;
    const uint wm = warp & 3;       /* 0..3  -> row block wm*16 (4*16 == 64) */
    const uint wn = warp >> 2;      /* 0..1  -> col block wn*32 (2*32 == 64) */
    /* VIEW-folded operands (docs/decisions.md §13 applied to matmul): when
     * av/bv != 0 the operand read is a strided gather (folded transpose/reshape/
     * broadcast) over the pre-transpose SOURCE buffer, so element (g,m,k) reads
     * base[view_idx(flat)] with flat = the contiguous [G,M,K] index. av==0 keeps
     * the contiguous fast path (base + g*M*K, common case). */
    const uint av = t.p4, bv = t.p5;
    __global const float *ba = AP(const float, t.a);
    __global const float *bb = AP(const float, t.b);
    __global const float *ga = ba + (size_t)g * M * K;
    __global const float *gb = bb + (size_t)g * K * N;

    float acc[TC_TNW][8];
    for (int j = 0; j < TC_TNW; j++)
        for (int e = 0; e < 8; e++) acc[j][e] = 0.0f;

    for (uint k0 = 0; k0 < K; k0 += MMA_BK) {
        for (uint idx = lid; idx < MMA_TM * MMA_BK; idx += 256) {
            const uint m = idx / MMA_BK, kk = idx % MMA_BK;   /* As row-major */
            const uint gr = row0 + m, gk = k0 + kk;
            const bool in = (gr < M && gk < K);
            As[m * MMA_BK + kk] = !in ? 0.0f
                : av ? ba[vmo_view_idx(aux, av - 1u,
                          (uint)((size_t)g * M * K + (size_t)gr * K + gk))]
                     : ga[gr * K + gk];
        }
        for (uint idx = lid; idx < MMA_TN * MMA_BK; idx += 256) {
            const uint n = idx / MMA_BK, kk = idx % MMA_BK;   /* Bs col-major */
            const uint gk = k0 + kk, gc = col0 + n;
            const bool in = (gk < K && gc < N);
            Bs[n * MMA_BK + kk] = !in ? 0.0f
                : bv ? bb[vmo_view_idx(aux, bv - 1u,
                          (uint)((size_t)g * K * N + (size_t)gk * N + gc))]
                     : gb[gk * N + gc];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint ks = 0; ks < TC_KSUB; ks++) {
            uint af[4], bf[TC_TNW][4];
            __local float *ap = &As[(wm * 16) * MMA_BK + ks * 8];
            TC_LOAD_A(af, ap, MMA_BK);
            for (int j = 0; j < TC_TNW; j++) {
                __local float *bp = &Bs[(wn * 32 + j * 16) * MMA_BK + ks * 8];
                TC_LOAD_B(bf[j], bp, MMA_BK);
            }
            for (int j = 0; j < TC_TNW; j++) TC_MMA(acc[j], af, bf[j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Masked direct global store via the derived m16n16k8 D-fragment map
     * (wmma.store.d.shared is broken on this driver — poc/08):
     *   row = (lane>>2) + 8*((reg>>1)&1)
     *   col = (lane&3)*2 + (reg&1) + 8*(reg>>2)                              */
    __global float *gd = AP(float, t.dst) + (size_t)g * M * N;
    for (int j = 0; j < TC_TNW; j++) {
        const uint gr0 = row0 + wm * 16;
        const uint gc0 = col0 + wn * 32 + j * 16;
        for (int reg = 0; reg < 8; reg++) {
            const uint r = (lane >> 2) + 8u * ((reg >> 1) & 1);
            const uint c = (lane & 3) * 2 + (reg & 1) + 8u * (reg >> 2);
            const uint gr = gr0 + r, gc = gc0 + c;
            if (gr < M && gc < N) gd[gr * N + gc] = acc[j][reg];
        }
    }
}

#else  /* portable scalar 4x4 microtile */

static void vmo_mma_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                     const task_t t, uint tile,
                     __local float *As, __local float *Bs)
{
    const uint M = t.p0, N = t.p1, K = t.p2;
    const uint tiles_n = (N + MMA_TN - 1) / MMA_TN;
    const uint tiles_m = (M + MMA_TM - 1) / MMA_TM;
    const uint per = tiles_m * tiles_n;          /* tiles per batch slice */
    const uint g = tile / per, loc = tile % per; /* g = batch index (p3) */
    const uint tr = loc / tiles_n, tc = loc % tiles_n;
    const uint row0 = tr * MMA_TM, col0 = tc * MMA_TN;
    const uint lid = get_local_id(0);
    const uint ty = lid / MMA_TDIM, tx = lid % MMA_TDIM;
    /* batched matmul: each slice g is a contiguous M×K / K×N / M×N sub-matrix.
     * av/bv (t.p4/p5) != 0 => the operand is a folded shape op (transpose/
     * reshape/broadcast), read strided from the pre-transpose SOURCE via
     * vmo_view_idx over the contiguous [G,M,K] flat index (docs/decisions.md §13
     * for matmul). av==0 keeps the contiguous fast path. */
    const uint av = t.p4, bv = t.p5;
    __global const float *ba = AP(const float, t.a);
    __global const float *bb = AP(const float, t.b);
    __global const float *ga = ba + (size_t)g * M * K;
    __global const float *gb = bb + (size_t)g * K * N;

    float acc[MMA_RM][MMA_RN];
    for (int i = 0; i < MMA_RM; i++)
        for (int j = 0; j < MMA_RN; j++) acc[i][j] = 0.0f;

    for (uint k0 = 0; k0 < K; k0 += MMA_BK) {
        for (uint idx = lid; idx < MMA_TM * MMA_BK; idx += 256) {
            const uint m = idx / MMA_BK, kk = idx % MMA_BK;
            const uint gr = row0 + m, gk = k0 + kk;
            const bool in = (gr < M && gk < K);
            As[m * MMA_BK + kk] = !in ? 0.0f
                : av ? ba[vmo_view_idx(aux, av - 1u,
                          (uint)((size_t)g * M * K + (size_t)gr * K + gk))]
                     : ga[gr * K + gk];
        }
        for (uint idx = lid; idx < MMA_BK * MMA_TN; idx += 256) {
            const uint kk = idx / MMA_TN, n = idx % MMA_TN;
            const uint gk = k0 + kk, gc = col0 + n;
            const bool in = (gk < K && gc < N);
            Bs[kk * MMA_TN + n] = !in ? 0.0f
                : bv ? bb[vmo_view_idx(aux, bv - 1u,
                          (uint)((size_t)g * K * N + (size_t)gk * N + gc))]
                     : gb[gk * N + gc];
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

    __global float *gd = AP(float, t.dst) + (size_t)g * M * N;
    for (int i = 0; i < MMA_RM; i++) {
        const uint gr = row0 + ty * MMA_RM + i;
        if (gr >= M) continue;
        for (int j = 0; j < MMA_RN; j++) {
            const uint gc = col0 + tx * MMA_RN + j;
            if (gc < N) gd[gr * N + gc] = acc[i][j];
        }
    }
}

#endif  /* VMO_NV_PTX */
