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
#ifdef VMO_NV_PTX
/* Pad the TF32 smem leading dim: the wmma A/B fragment loads read 16 rows at
 * stride LDS; with LDS==16 consecutive rows collide on the same 32 banks (8-way
 * conflict). LDS=20 makes gcd(20,32)=4 -> 8 distinct bank offsets, conflict-
 * free-ish. Costs 25% more staging smem; no accumulator registers. */
#define TC_LDS (MMA_BK + 4)
#define MMA_ASZ (MMA_TM * TC_LDS)    /* As[m*LDS + k] */
#define MMA_BSZ (MMA_TN * TC_LDS)    /* Bs[n*LDS + k] col-major, padded */
#else
#define MMA_ASZ (MMA_BK * MMA_TM)    /* As[m*BK + k] */
#define MMA_BSZ (MMA_BK * MMA_TN)    /* Bs[k*TN+n] portable */
#endif

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
    /* L2-friendly threadblock swizzle (CUTLASS/Triton "group-M"): instead of
     * row-major (tr=loc/tiles_n), walk GROUP_M row-blocks per column strip so a
     * wave of co-resident workgroups reuses the same B column panel (and A row
     * panels) from L2 before moving on. The 64x64 tile has only ~16 FLOP/byte
     * arithmetic intensity, so at large K it is GLOBAL-BANDWIDTH bound; raising
     * the L2 hit rate on the re-streamed B is the only knob left without growing
     * the register tile (which the megakernel occupancy cap forbids, §10c).
     * Pure index remap — every (tr,tc) still covered exactly once, bit-exact. */
    const uint GROUP_M = 8u;
    const uint in_grp = GROUP_M * tiles_n;
    const uint gid = loc / in_grp;
    const uint first_m = gid * GROUP_M;
    const uint gsz_m = min(tiles_m - first_m, GROUP_M);
    const uint locg = loc % in_grp;
    const uint tr = first_m + locg % gsz_m;
    const uint tc = locg / gsz_m;
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
            As[m * TC_LDS + kk] = !in ? 0.0f
                : av ? ba[vmo_view_idx(aux, av - 1u,
                          (uint)((size_t)g * M * K + (size_t)gr * K + gk))]
                     : ga[gr * K + gk];
        }
        for (uint idx = lid; idx < MMA_TN * MMA_BK; idx += 256) {
            const uint n = idx / MMA_BK, kk = idx % MMA_BK;   /* Bs col-major */
            const uint gk = k0 + kk, gc = col0 + n;
            const bool in = (gk < K && gc < N);
            Bs[n * TC_LDS + kk] = !in ? 0.0f
                : bv ? bb[vmo_view_idx(aux, bv - 1u,
                          (uint)((size_t)g * K * N + (size_t)gk * N + gc))]
                     : gb[gk * N + gc];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint ks = 0; ks < TC_KSUB; ks++) {
            uint af[4], bf[TC_TNW][4];
            __local float *ap = &As[(wm * 16) * TC_LDS + ks * 8];
            TC_LOAD_A(af, ap, TC_LDS);
            for (int j = 0; j < TC_TNW; j++) {
                __local float *bp = &Bs[(wn * 32 + j * 16) * TC_LDS + ks * 8];
                TC_LOAD_B(bf[j], bp, TC_LDS);
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

#elif defined(VMO_CPU_TILES)
/* fwd decl: the staging body is defined below the variant chain. */
static void vmo_mma_tile_stage(__global uchar *arena, __global uchar **iop,
                               __global const int *aux, const task_t t,
                               uint tile, __local float *As, __local float *Bs);
/* CPU mma tile (poc/09 [b3], decisions.md #11): the poc/09-b2 4x16 float8
 * register microkernel embedded in the tile interface. WIs 0..63 each own a
 * 4-row x 16-col block of the 64x64 output tile (idle WIs measured ~4%
 * overhead); NO __local staging, NO barriers — both are pure loss on CPU
 * OpenCL runtimes (b1 8.0 vs b3 35.4 GFLOP/s under identical load). Viewed
 * (av/bv) operands and edge blocks fall back to a guarded scalar loop with
 * identical semantics to the portable variant below. */
static void vmo_mma_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                     const task_t t, uint tile,
                     __local float *As, __local float *Bs)
{
    (void)As; (void)Bs;
    const uint M = t.p0, N = t.p1, K = t.p2;
    const uint tiles_n = (N + MMA_TN - 1) / MMA_TN;
    const uint tiles_m = (M + MMA_TM - 1) / MMA_TM;
    const uint per = tiles_m * tiles_n;
    const uint g = tile / per, loc = tile % per;
    const uint tr = loc / tiles_n, tc = loc % tiles_n;
    const uint row0 = tr * MMA_TM, col0 = tc * MMA_TN;
    const uint lid = get_local_id(0);
    const uint av = t.p4, bv = t.p5;
    __global const float *ba = AP(const float, t.a);
    __global const float *bb = AP(const float, t.b);
    __global const float *ga = ba + (size_t)g * M * K;
    __global const float *gb = bb + (size_t)g * K * N;
    __global float *gd = AP(float, t.dst) + (size_t)g * M * N;
    if (av || bv) {
        /* Folded transpose/reshape operands: staging body (uniform per-tile
         * condition — the whole workgroup takes this branch and reaches its
         * barriers together). */
        vmo_mma_tile_stage(arena, iop, aux, t, tile, As, Bs);
    } else {
    /* lid gate is an if-wrap, not a return: PoCL 5.0 region formation is
     * fragile around returns in functions inlined next to barrier-bearing
     * tile cases (decisions.md #18), even without a barrier here. */
    if (lid < 64u) {
    const uint r0 = row0 + (lid / 4u) * 4u;    /* this WI's 4-row block */
    const uint c0 = col0 + (lid % 4u) * 16u;   /* this WI's 16-col strip */
    if (r0 + 4u <= M && c0 + 16u <= N) {
        float8 a0[4], a1[4];
        for (int i = 0; i < 4; ++i) { a0[i] = (float8)(0.0f); a1[i] = (float8)(0.0f); }
        for (uint k = 0; k < K; ++k) {
            const float8 b0 = vload8(0, gb + k * N + c0);
            const float8 b1 = vload8(0, gb + k * N + c0 + 8u);
            for (int i = 0; i < 4; ++i) {
                const float8 avv = (float8)(ga[(r0 + i) * K + k]);
                a0[i] = mad(avv, b0, a0[i]);
                a1[i] = mad(avv, b1, a1[i]);
            }
        }
        for (int i = 0; i < 4; ++i) {
            vstore8(a0[i], 0, gd + (r0 + i) * N + c0);
            vstore8(a1[i], 0, gd + (r0 + i) * N + c0 + 8u);
        }
    } else {                                   /* edges / viewed operands */
        for (uint i = 0; i < 4u; ++i) {
            const uint gr = r0 + i;
            if (gr >= M) continue;
            for (uint j = 0; j < 16u; ++j) {
                const uint gc = c0 + j;
                if (gc >= N) continue;
                float s = 0.0f;
                for (uint k = 0; k < K; ++k) {
                    const float x = av
                        ? ba[vmo_view_idx(aux, av - 1u,
                              (uint)((size_t)g * M * K + (size_t)gr * K + k))]
                        : ga[gr * K + k];
                    const float y = bv
                        ? bb[vmo_view_idx(aux, bv - 1u,
                              (uint)((size_t)g * K * N + (size_t)k * N + gc))]
                        : gb[k * N + gc];
                    s = mad(x, y, s);
                }
                gd[gr * N + gc] = s;
            }
        }
    }
    }
    }
}
#endif  /* variant-specific vmo_mma_tile above */
/* Portable staging matmul tile (4x4 register microtile over __local panels).
 * Compiled on every device: non-NV, non-CPU builds alias it as THE mma tile
 * (wrapper at end of file); the CPU build dispatches viewed-operand tiles
 * here — staging amortizes vmo_view_idx across the workgroup once per
 * K-block, where a scalar per-element fallback measured 25-35% slower
 * end-to-end on the transformer. */

static void vmo_mma_tile_stage(__global uchar *arena, __global uchar **iop, __global const int *aux,
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


#if !defined(VMO_NV_PTX) && !defined(VMO_CPU_TILES)
static void vmo_mma_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                     const task_t t, uint tile,
                     __local float *As, __local float *Bs)
{
    vmo_mma_tile_stage(arena, iop, aux, t, tile, As, Bs);
}
#endif
