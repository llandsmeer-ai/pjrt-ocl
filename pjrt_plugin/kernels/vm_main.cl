/* pjrt-ocl VLIW engine — dispatcher + interpreter (concatenated last).
 * vmo_exec_tiles routes a task to its op-family tile function (ops/ *.cl above);
 * vm2 is the per-lane interpreter over the schedule stream. */

/* tile_op packs the base op in bits 0-7 and the dtype in bits 8-15. */
static void vmo_exec_tiles(__global uchar *arena, __global uchar **iop,
                       __global const int *aux,
                       const task_t t, uint tile_lo, uint tile_hi,
                       __local float *As, __local float *Bs)
{
    const uint lid = get_local_id(0);
    const uint lsz = get_local_size(0);
    const uint op = t.tile_op & 0xFFu;
    const uint dt = (t.tile_op >> 8) & 0xFFu;    /* result dtype */
    const uint adt = (t.tile_op >> 16) & 0xFFu;  /* operand dtype */
    const uint esz = (dt == DT_I64 || dt == DT_F64) ? 8u
                   : (dt == DT_BOOL) ? 1u
                   : (dt == DT_F16 || dt == DT_BF16) ? 2u : 4u;
    for (uint tile = tile_lo; tile < tile_hi; ++tile) {
        switch (op) {
        case TOP_EW:       vmo_ew_tile(arena, iop, aux, t, tile, dt, adt, lid, lsz); break;
        case TOP_MMA:      vmo_mma_tile(arena, iop, aux, t, tile, As, Bs); break;
        case TOP_GATHER:   vmo_gather_tile(arena, iop, aux, t, tile, esz, lid, lsz); break;
        case TOP_RED_PART: vmo_reduce_part_tile(arena, iop, t, tile, As, dt, lid, lsz); break;
        case TOP_RED_COMB: vmo_reduce_comb_tile(arena, iop, t, dt, lid); break;
        case TOP_IOTA_DIM: vmo_iota_tile(arena, iop, aux, t, tile, lid, lsz); break;
        case TOP_SCATTER:  vmo_scatter_tile(arena, iop, aux, t, tile, esz, lid, lsz); break;
        case TOP_DYN_GATHER:  vmo_dyn_gather_tile(arena, iop, aux, t, tile, esz, lid, lsz); break;
        case TOP_DYN_SCATTER: vmo_dyn_scatter_tile(arena, iop, aux, t, tile, esz, lid, lsz); break;
        case TOP_RED_WINDOW:  vmo_redwin_tile(arena, iop, aux, t, tile, dt, lid, lsz); break;
        case TOP_RED_SEG:  vmo_redseg_tile(arena, iop, t, tile, dt, As, lid, lsz); break;
        case TOP_SOFTMAX_SEG:   vmo_softmax_seg(arena, iop, t, tile, As, Bs, lid, lsz); break;
        case TOP_LAYERNORM_SEG: vmo_layernorm_seg(arena, iop, t, tile, As, Bs, lid, lsz); break;
#ifdef VMO_REGION_POC
        case TOP_MAP_REGION: vmo_map_region(arena, iop, aux, t, tile, lid, lsz); break;
#endif
#ifdef VMO_PROBE_REGS
        /* §27 register-budget probe. VMO_PROBE_REGS float accumulators kept
         * SIMULTANEOUSLY live across the k-loop, seeded from and reduced back to
         * arena so ptxas cannot DCE them. Scoped INSIDE this case → measures
         * whether the megakernel's whole-kernel register count is
         * max-over-mutually-exclusive-cases (disjoint live ranges) or a sum.
         * Never emitted by lowering (op id 99 is unused); build-flag only. */
        case 99: {
            __global float *d = (__global float *)(arena + t.dst);
            const uint n = t.p0;
            float acc[VMO_PROBE_REGS];
            #pragma unroll
            for (int i = 0; i < VMO_PROBE_REGS; ++i)
                acc[i] = (float)(lid + i) + d[i & 63];
            for (uint k = lid; k < n; k += lsz) {
                const float s = d[k & 255];
                #pragma unroll
                for (int i = 0; i < VMO_PROBE_REGS; ++i)
                    acc[i] = fma(acc[i], s, (float)i);
            }
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < VMO_PROBE_REGS; ++i)
                sum += acc[i];
            d[lid & 255] = sum;
            break;
        }
#endif
        default: break;
        }
    }
}

/* Per-lane interpreter with a frame stack over the lane's OWN stream. */
#define MAX_DEPTH 8
#define WIDX_ROOT 0xFFFFFFFFu
typedef struct { uint pc, end, widx, phase; } frame_t; /* phase: 0 cond,1 body,2 if */

__kernel void vm2(__global uchar *arena,
                  __global const int *aux,
                  __global const task_t *tasks,
                  __global const entry_t *entries,   /* flattened */
                  __global const uint4 *lane_tab,    /* {off,count,root_len,pad} */
                  volatile __global uint *bar,       /* [0,1] barrier, [2] rank */
                  const uint nlanes,
                  __global uint *stats,              /* arrival rank per
                                                        [barrier_i*nlanes+lane] */
                  VMO_IO_PARAMS)                     /* direct I/O buffers (ports) */
{
    VMO_IO_ARRAY;
    /* nlanes == 0: occupancy-probe mode (runtime.cc ProbeResidency; poc/08).
     * Must run before any buffer argument is dereferenced — probe launches
     * pass a dummy buffer for everything except bar (VMO_IO_ARRAY above only
     * copies pointers). The probe lives INSIDE vm2 so it inherits vm2's exact
     * compiled footprint (SIMD width, GRF mode, SLM), which is what determines
     * co-residency: on Xe2 a lookalike probe kernel over-reported 64 where the
     * real vm2's limit is 32 (poc/08). */
    if (nlanes == 0u) {
        if (get_local_id(0) == 0u)
            vmo_discover(bar);
        return;
    }

    const uint lane = get_group_id(0);
    const uint lid = get_local_id(0);
    /* Shared local scratch: MMA staging (As/Bs panels) and REDUCE_PART tree
     * (As[lid], lid<256). Sized for the 64x64 MMA panels. */
    __local float As[MMA_ASZ];
    __local float Bs[MMA_BSZ];

    const uint4 span = lane_tab[lane];   /* .x off, .y count, .z root_len */
    uint barrier_i = 0;

    frame_t st[MAX_DEPTH];
    int sp = 0;
    st[0].pc = 0; st[0].end = span.z; st[0].widx = WIDX_ROOT; st[0].phase = 0;

    for (;;) {
        if (st[sp].pc >= st[sp].end) {
            if (st[sp].widx == WIDX_ROOT)
                break;
            const entry_t w = entries[span.x + st[sp].widx];
            if (w.task == ENT_IF) {            /* branch done */
                sp--;
                st[sp].pc++;
                continue;
            }
            if (w.task == ENT_FOR) {           /* fixed-trip iteration done */
                /* Barrier publishes loop carries the next iteration reads
                 * across lanes. phase counts REMAINING iterations. */
                vmo_barrier(bar, nlanes);
                barrier_i++;
                if (--st[sp].phase != 0u) {
                    st[sp].pc = w.tile_lo;
                    st[sp].end = w.tile_lo + w.tile_hi;
                } else {
                    sp--;
                    st[sp].pc++;
                }
                continue;
            }
            if (st[sp].phase == 0) {           /* while-cond range done */
                vmo_barrier(bar, nlanes);
                barrier_i++;
                const uint cbits = atomic_add(
                    (volatile __global uint *)(arena + w.signal_flag), 0u);
                if (cbits != 0u) {
                    st[sp].pc = w.wait_flag;
                    st[sp].end = w.wait_flag + w.wait_count;
                    st[sp].phase = 1;
                } else {
                    sp--;
                    st[sp].pc++;
                }
            } else {                           /* while-body done: recheck */
                vmo_barrier(bar, nlanes);
                barrier_i++;
                st[sp].pc = w.tile_lo;
                st[sp].end = w.tile_lo + w.tile_hi;
                st[sp].phase = 0;
            }
            continue;
        }

        const uint epc = st[sp].pc;
        const entry_t en = entries[span.x + epc];

        if (en.task == ENT_BARRIER) {
            if (lid == 0 && barrier_i < 4096u)
                stats[barrier_i * nlanes + lane] = atomic_inc(&bar[2]) % nlanes;
            vmo_barrier(bar, nlanes);
            barrier_i++;
            st[sp].pc++;
            continue;
        }
        if (en.task == ENT_WHILE) {
            sp++;
            st[sp].pc = en.tile_lo;
            st[sp].end = en.tile_lo + en.tile_hi;
            st[sp].widx = epc;
            st[sp].phase = 0;
            continue;
        }
        if (en.task == ENT_FOR) {
            if (en.wait_flag == 0u) {          /* trip 0: skip the loop */
                st[sp].pc++;
                continue;
            }
            sp++;
            st[sp].pc = en.tile_lo;
            st[sp].end = en.tile_lo + en.tile_hi;
            st[sp].widx = epc;
            st[sp].phase = en.wait_flag;       /* remaining iterations */
            continue;
        }
        if (en.task == ENT_IF) {
            const uint cbits = atomic_add(
                (volatile __global uint *)(arena + en.signal_flag), 0u);
            const uint start = cbits != 0u ? en.tile_lo : en.wait_flag;
            const uint len = cbits != 0u ? en.tile_hi : en.wait_count;
            if (len == 0) { st[sp].pc++; continue; }
            sp++;
            st[sp].pc = start;
            st[sp].end = start + len;
            st[sp].widx = epc;
            st[sp].phase = 2;
            continue;
        }
        if (en.task != ENT_NOP) {
            /* wait_flag/signal_flag per-op counters are reserved (v0 emits
             * FLAG_NONE); wire a flags buffer through before enabling. */
            vmo_exec_tiles(arena, iop, aux, tasks[en.task], en.tile_lo, en.tile_hi,
                       As, Bs);
        }
        st[sp].pc++;
    }
}

/* HOST-DISPATCH engine (CPU / non-GPU devices, docs/decisions.md #1): the host
 * drives control flow and the cross-workgroup barrier via clFinish between
 * launches, so there is NO in-kernel barrier and no co-residency requirement
 * (a finished workgroup exits and frees its CPU thread — immune to the
 * imbalance-starvation deadlock the persistent spin-barrier hits on PoCL,
 * poc/07). This kernel runs ONE barrier-free segment: each workgroup (lane)
 * executes its contiguous run of tile entries [seg.x, seg.x+seg.y) — the host
 * has already resolved all BARRIER/WHILE/IF control, so a segment holds only
 * tile (or NOP) entries. */
__kernel void vm2_seg(__global uchar *arena,
                      __global const int *aux,
                      __global const task_t *tasks,
                      __global const entry_t *entries,
                      __global const uint2 *seg_tab,   /* per-lane {off, count},
                                                          ring of phase slots */
                      VMO_IO_PARAMS,                    /* direct I/O buffers */
                      const uint seg_base)  /* this phase's slot: uint2 index */
{
    VMO_IO_ARRAY;
    const uint lane = get_group_id(0);
    __local float As[MMA_ASZ];
    __local float Bs[MMA_BSZ];
    const uint2 seg = seg_tab[seg_base + lane];
    for (uint i = 0; i < seg.y; ++i) {
        const entry_t en = entries[seg.x + i];
        if (en.task != ENT_NOP)
            vmo_exec_tiles(arena, iop, aux, tasks[en.task], en.tile_lo, en.tile_hi,
                           As, Bs);
    }
}

/* TRACE mode (PJRT_OCL_VM_TRACE): one entry per launch, so OpenCL event
 * profiling yields a per-entry start/end timestamp. Launched with a single
 * workgroup on the entry's lane queue — same one-workgroup-per-entry execution
 * as vm2_seg, just one entry at a time. */
__kernel void vm2_one(__global uchar *arena,
                      __global const int *aux,
                      __global const task_t *tasks,
                      __global const entry_t *entries,
                      const uint entry_idx,
                      VMO_IO_PARAMS)                    /* direct I/O buffers */
{
    VMO_IO_ARRAY;
    __local float As[MMA_ASZ];
    __local float Bs[MMA_BSZ];
    const entry_t en = entries[entry_idx];
    if (en.task != ENT_NOP)
        vmo_exec_tiles(arena, iop, aux, tasks[en.task], en.tile_lo, en.tile_hi,
                       As, Bs);
}

/* ---- Standalone SGEMM (pure-matmul fast path) --------------------------------
 * Launched OUTSIDE the megakernel for programs that are a single TILE_MMA with
 * no barriers/control (the common `a@b`). Being its own kernel gives it an
 * independent register budget, so an 8x8 register microtile (64 accumulators)
 * stays in registers instead of spilling to global memory — inside the shared
 * megakernel the same tile spills catastrophically (docs/decisions.md #9b) and
 * the megakernel's launch is also occupancy-capped to ~2 workgroups/SM. Here we
 * launch one 256-thread workgroup per 128x128 output tile, filling the GPU.
 * As is stored TRANSPOSED (As[kk*TM+m]) so each thread's a[] is contiguous for
 * a 128-bit vload4. dst/a/b arrive as VM buffer handles (arena offset or I/O
 * port), resolved by the same VMO_BASE macro the tiles use. */
/* 64x64 output tile, 8x8 threads (64 per workgroup), each thread an 8x8 register
 * microtile. Small tile + few threads => 4x more workgroups than a 128x128 tile
 * (e.g. 1024 vs 256 at N=2048) for far higher SM occupancy, while keeping the
 * 8x8 arithmetic intensity a standalone kernel can afford (no megakernel
 * register sharing). */
#ifdef VMO_CPU_TILES
/* CPU-shaped mm2 (poc/09 b2, decisions.md #11): one work-item per workgroup
 * (ceil(M/4) groups — the host sets the geometry per engine), a 4-row x
 * 16-column register block of float8 accumulators, no __local, no barriers.
 * On a CPU OpenCL runtime __local staging is an extra memcpy and every WG
 * barrier forces a loop-split; this shape measured 4x the MMA-tile GFLOP/s
 * standalone and ~11x through the VM. Full 4-row blocks take the unrolled
 * path; edge rows/columns fall to guarded scalar loops. */
__kernel void mm2(__global uchar *arena, VMO_IO_PARAMS,
                  const uint M, const uint N, const uint K,
                  const uint dsth, const uint ah, const uint bh)
{
    VMO_IO_ARRAY;
    __global const float *ga = AP(const float, ah);
    __global const float *gb = AP(const float, bh);
    __global float *gd = AP(float, dsth);
    const uint r0 = get_group_id(0) * 4u;
    const uint nr = min(4u, M - r0);
    uint c0 = 0;
    if (nr == 4u) {
        for (; c0 + 16u <= N; c0 += 16u) {
            float8 acc0[4], acc1[4];
            for (int i = 0; i < 4; ++i) {
                acc0[i] = (float8)(0.0f); acc1[i] = (float8)(0.0f);
            }
            for (uint k = 0; k < K; ++k) {
                const float8 b0 = vload8(0, gb + k * N + c0);
                const float8 b1 = vload8(0, gb + k * N + c0 + 8u);
                for (int i = 0; i < 4; ++i) {
                    const float8 av = (float8)(ga[(r0 + i) * K + k]);
                    acc0[i] = mad(av, b0, acc0[i]);
                    acc1[i] = mad(av, b1, acc1[i]);
                }
            }
            for (int i = 0; i < 4; ++i) {
                vstore8(acc0[i], 0, gd + (r0 + i) * N + c0);
                vstore8(acc1[i], 0, gd + (r0 + i) * N + c0 + 8u);
            }
        }
    }
    for (; c0 < N; ++c0)               /* N tail (and nr<4 edge blocks) */
        for (uint i = 0; i < nr; ++i) {
            float s = 0.0f;
            for (uint k = 0; k < K; ++k)
                s = mad(ga[(r0 + i) * K + k], gb[k * N + c0], s);
            gd[(r0 + i) * N + c0] = s;
        }
}
/* Packed+blocked CPU SGEMM (poc/10, the default; PJRT_OCL_MM_CPU=reg selects
 * the register kernel above instead). mm2_pack reorders B into 16-column
 * panels (Bp[p*K*16 + k*16 + j]) so mm2p's k-loop reads sequentially — the
 * stride-N B read was the dominant cost (poc/10 v1: 1.6x). mm2p is a 6x16
 * float8 register block (12 accs + 2 B + 1 A broadcast = 15/16 ymm) run over
 * [kc0,kc1) K-sweeps; kc0>0 accumulates into C from memory. The host enqueues
 * pack + all sweeps back-to-back (in-order queue; no syncs). Requires
 * N % 16 == 0 — the host falls back to mm2 otherwise. */
__kernel void mm2_pack(__global uchar *arena, VMO_IO_PARAMS,
                       const uint N, const uint K,
                       const uint bh, __global float *Bp)
{
    VMO_IO_ARRAY;
    __global const float *B = AP(const float, bh);
    const uint gid = get_global_id(0);
    const uint p = gid / K, k = gid % K;
    if (p >= N / 16u) return;
    vstore8(vload8(0, B + k * N + p * 16u), 0,
            Bp + (ulong)p * K * 16u + k * 16u);
    vstore8(vload8(0, B + k * N + p * 16u + 8u), 0,
            Bp + (ulong)p * K * 16u + k * 16u + 8u);
}

__kernel void mm2p(__global uchar *arena, VMO_IO_PARAMS,
                   const uint M, const uint N, const uint K,
                   const uint dsth, const uint ah,
                   __global const float *gbp,
                   const uint kc0, const uint kc1)
{
    VMO_IO_ARRAY;
    __global const float *ga = AP(const float, ah);
    __global float *gd = AP(float, dsth);
    const uint r0 = get_group_id(0) * 6u;
    if (r0 >= M) return;
    const uint nr = min(6u, M - r0);
    if (nr == 6u) {
        for (uint p = 0; p < N / 16u; ++p) {
            __global const float *panel = gbp + (ulong)p * K * 16u;
            float8 a0[6], a1[6];
            if (kc0 == 0u) {
                for (int i = 0; i < 6; ++i) {
                    a0[i] = (float8)(0.0f); a1[i] = (float8)(0.0f);
                }
            } else {
                for (int i = 0; i < 6; ++i) {
                    a0[i] = vload8(0, gd + (r0 + i) * N + p * 16u);
                    a1[i] = vload8(0, gd + (r0 + i) * N + p * 16u + 8u);
                }
            }
            for (uint k = kc0; k < kc1; ++k) {
                const float8 b0 = vload8(0, panel + k * 16u);
                const float8 b1 = vload8(0, panel + k * 16u + 8u);
                for (int i = 0; i < 6; ++i) {
                    const float8 av = (float8)(ga[(r0 + i) * K + k]);
                    a0[i] = mad(av, b0, a0[i]);
                    a1[i] = mad(av, b1, a1[i]);
                }
            }
            for (int i = 0; i < 6; ++i) {
                vstore8(a0[i], 0, gd + (r0 + i) * N + p * 16u);
                vstore8(a1[i], 0, gd + (r0 + i) * N + p * 16u + 8u);
            }
        }
    } else if (kc0 == 0u) {   /* edge rows: one full-K scalar pass */
        for (uint c0 = 0; c0 < N; ++c0)
            for (uint i = 0; i < nr; ++i) {
                float s = 0.0f;
                for (uint k = 0; k < K; ++k)
                    s = mad(ga[(r0 + i) * K + k],
                            gbp[(ulong)(c0 / 16u) * K * 16u + k * 16u +
                                (c0 % 16u)], s);
                gd[(r0 + i) * N + c0] = s;
            }
    }
}
#else
#define MM2_TM 128
#define MM2_TN 64
#define MM2_BK 16
#define MM2_TD 16                 /* 16x16 threads == 256 */
#define MM2_NT (MM2_TD * MM2_TD)  /* 256 threads/workgroup */
#define MM2_RM (MM2_TM / MM2_TD)  /* 8 */
#define MM2_RN (MM2_TN / MM2_TD)  /* 4 */

__kernel void mm2(__global uchar *arena, VMO_IO_PARAMS,
                  const uint M, const uint N, const uint K,
                  const uint dsth, const uint ah, const uint bh)
{
    VMO_IO_ARRAY;
    /* DOUBLE-BUFFERED: two smem panels; prefetch the next K-block into the idle
     * panel while the current one is consumed, so global-load latency overlaps
     * compute (one barrier/iter instead of load->barrier->compute->barrier). */
    __local float As[2][MM2_BK * MM2_TM];   /* transposed: As[buf][kk*TM + m] */
    __local float Bs[2][MM2_BK * MM2_TN];   /* Bs[buf][kk*TN + n] */
    const uint lid = get_local_id(0);
    const uint tiles_n = (N + MM2_TN - 1) / MM2_TN;
    const uint tile = get_group_id(0);
    const uint tr = tile / tiles_n, tc = tile % tiles_n;
    const uint row0 = tr * MM2_TM, col0 = tc * MM2_TN;
    const uint ty = lid / MM2_TD, tx = lid % MM2_TD;
    __global const float *ga = AP(const float, ah);
    __global const float *gb = AP(const float, bh);

    float acc[MM2_RM][MM2_RN];
    for (int i = 0; i < MM2_RM; i++)
        for (int j = 0; j < MM2_RN; j++) acc[i][j] = 0.0f;

#define MM2_STAGE(BUF, K0)                                                      \
    do {                                                                       \
        for (uint idx = lid; idx < MM2_TM * MM2_BK; idx += MM2_NT) {          \
            const uint m = idx / MM2_BK, kk = idx % MM2_BK;                    \
            const uint gr = row0 + m, gk = (K0) + kk;                         \
            As[BUF][kk * MM2_TM + m] =                                         \
                (gr < M && gk < K) ? ga[gr * K + gk] : 0.0f;                   \
        }                                                                      \
        for (uint idx = lid; idx < MM2_BK * MM2_TN; idx += MM2_NT) {          \
            const uint kk = idx / MM2_TN, n = idx % MM2_TN;                    \
            const uint gk = (K0) + kk, gc = col0 + n;                         \
            Bs[BUF][kk * MM2_TN + n] =                                         \
                (gk < K && gc < N) ? gb[gk * N + gc] : 0.0f;                   \
        }                                                                      \
    } while (0)

    MM2_STAGE(0, 0);
    barrier(CLK_LOCAL_MEM_FENCE);
    uint buf = 0;
    for (uint k0 = 0; k0 < K; k0 += MM2_BK) {
        if (k0 + MM2_BK < K) MM2_STAGE(buf ^ 1, k0 + MM2_BK);
        for (uint kk = 0; kk < MM2_BK; ++kk) {
            float a[MM2_RM], b[MM2_RN];
            for (int i = 0; i < MM2_RM; i += 4) {
                float4 v = vload4(0, &As[buf][kk * MM2_TM + ty * MM2_RM + i]);
                a[i] = v.x; a[i + 1] = v.y; a[i + 2] = v.z; a[i + 3] = v.w;
            }
            for (int j = 0; j < MM2_RN; j += 4) {
                float4 v = vload4(0, &Bs[buf][kk * MM2_TN + tx * MM2_RN + j]);
                b[j] = v.x; b[j + 1] = v.y; b[j + 2] = v.z; b[j + 3] = v.w;
            }
            for (int i = 0; i < MM2_RM; i++)
                for (int j = 0; j < MM2_RN; j++)
                    acc[i][j] += a[i] * b[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        buf ^= 1;
    }
#undef MM2_STAGE
    __global float *gd = AP(float, dsth);
    for (int i = 0; i < MM2_RM; i++) {
        const uint gr = row0 + ty * MM2_RM + i;
        if (gr >= M) continue;
        for (int j = 0; j < MM2_RN; j++) {
            const uint gc = col0 + tx * MM2_RN + j;
            if (gc < N) gd[gr * N + gc] = acc[i][j];
        }
    }
}
#endif  /* VMO_CPU_TILES */

/* GEMV fast path: y[M] = A[MxK] . x[K] — a matmul task whose N is 1. The MMA
 * tile wastes 63/64 of its work on a width-1 RHS (README flags this on
 * NVIDIA); a one-row-per-WI float8 dot wins on BOTH device classes (poc/09
 * c2: PoCL 12.7 vs ~0.6 GB/s in-VM; Xe2 73 vs 37). Same handle resolution as
 * mm2 (VMO_BASE: arena offset or I/O port). */
__kernel void gemv(__global uchar *arena, VMO_IO_PARAMS,
                   const uint M, const uint K,
                   const uint dsth, const uint ah, const uint bh)
{
    VMO_IO_ARRAY;
    const uint r = get_global_id(0);
    if (r >= M) return;
    __global const float *ga = AP(const float, ah) + (ulong)r * K;
    __global const float *gx = AP(const float, bh);
    float8 s8 = (float8)(0.0f);
    uint k = 0;
    for (; k + 8u <= K; k += 8u)
        s8 = mad(vload8(0, ga + k), vload8(0, gx + k), s8);
    float s = s8.s0 + s8.s1 + s8.s2 + s8.s3 + s8.s4 + s8.s5 + s8.s6 + s8.s7;
    for (; k < K; ++k) s = mad(ga[k], gx[k], s);
    AP(float, dsth)[r] = s;
}
