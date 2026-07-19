/* poc/13 — async / prefetched DRAM loads: do they hide memory latency in our
 * tile loops? (docs/decisions.md §22 = per-tile execution is latency-bound.)
 *
 * Two representative loop shapes, each implemented THREE ways:
 *   (a) baseline  — direct global loads, exactly the shipped tile body
 *   (b) manual DB — software double-buffering: prefetch tile N+1's global loads
 *                   into registers while computing on tile N
 *   (c) async     — async_work_group_copy into __local + wait_group_events
 *
 * LOOP A: streaming elementwise (the §22 headline). Pure streaming, no reuse:
 *   d[i] = a[i]*s + t  (affine, memory-bound). Grid-stride over EW_TS tiles.
 * LOOP B: matmul K-loop global->local stage (the classic double-buffer case,
 *   real data reuse). 64x64 tile, BK=16, 256 lanes (16x16), 4x4 microtile —
 *   mirrors the shipped portable vmo_mma_tile footprint (As/Bs = 8 KB single).
 *
 * Persistent-grid faithful: launched with a FIXED grid (2*CU groups, the
 * megakernel cap) and grid-strides over tiles, like the real VM.
 */

/* ============================ LOOP A: streaming EW ======================= */
#ifndef EW_TS
#define EW_TS 4096
#endif

/* (a) baseline scalar direct loads — one dependent round trip per element. */
__kernel void ew_scalar(__global const float *a, __global float *d,
                        uint n, float s, float t)
{
    const uint lid = get_local_id(0), lsz = get_local_size(0);
    const uint ntiles = (n + EW_TS - 1) / EW_TS;
    for (uint tile = get_group_id(0); tile < ntiles; tile += get_num_groups(0)) {
        const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
        for (uint i = lo + lid; i < hi; i += lsz)
            d[i] = a[i] * s + t;
    }
}

/* (b) manual register double-buffering: float4 + 2x unroll = 8 independent
 * in-flight loads issued before any is consumed. This IS the shipped §22 fast
 * path (ew.cl float4 path) — register-level prefetch, no __local, no occupancy
 * cost. Assumes 16B-aligned, n multiple of 8 (padded on host). */
__kernel void ew_regdb(__global const float *a, __global float *d,
                       uint n, float s, float t)
{
    const uint lid = get_local_id(0), lsz = get_local_size(0);
    const uint ntiles = (n + EW_TS - 1) / EW_TS;
    __global const float4 *a4 = (__global const float4 *)a;
    __global float4 *d4 = (__global float4 *)d;
    const float4 s4 = (float4)(s), t4 = (float4)(t);
    for (uint tile = get_group_id(0); tile < ntiles; tile += get_num_groups(0)) {
        const uint lo4 = (tile * EW_TS) / 4, hi4 = min(lo4 + EW_TS / 4, n / 4);
        uint i = lo4 + lid;
        for (; i + lsz < hi4; i += 2 * lsz) {
            const float4 x0 = a4[i], x1 = a4[i + lsz];   /* two loads issued... */
            d4[i]       = mad(x0, s4, t4);               /* ...before consumed  */
            d4[i + lsz] = mad(x1, s4, t4);
        }
        for (; i < hi4; i += lsz)
            d4[i] = mad(a4[i], s4, t4);
    }
}

/* (c) async_work_group_copy: stage a __local chunk, compute from local, copy
 * back. For pure streaming there is NO reuse, so this adds a local buffer (=>
 * occupancy cost) and two extra local<->global hops for nothing — the honest
 * "does async help streaming?" test. LCHUNK floats staged per pass. */
#ifndef LCHUNK
#define LCHUNK 1024
#endif
__kernel void ew_async(__global const float *a, __global float *d,
                      uint n, float s, float t)
{
    const uint lid = get_local_id(0), lsz = get_local_size(0);
    __local float buf[LCHUNK];
    const uint ntiles = (n + EW_TS - 1) / EW_TS;
    for (uint tile = get_group_id(0); tile < ntiles; tile += get_num_groups(0)) {
        const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
        for (uint base = lo; base < hi; base += LCHUNK) {
            const uint cnt = min((uint)LCHUNK, hi - base);
            event_t e = async_work_group_copy(buf, a + base, cnt, 0);
            wait_group_events(1, &e);
            for (uint i = lid; i < cnt; i += lsz)
                buf[i] = buf[i] * s + t;
            barrier(CLK_LOCAL_MEM_FENCE);
            event_t e2 = async_work_group_copy(d + base, buf, cnt, 0);
            wait_group_events(1, &e2);
        }
    }
}

/* ============================ LOOP B: matmul K-loop ====================== */
/* A: M x K row-major.  B: col-major (Bcm[n*K+k]) so a BK-run of a column is
 * contiguous (async can stage it).  C = A@B, M x N row-major.
 * Tile 64x64, BK=16, 256 lanes (16x16), 4x4 microtile — mirrors the shipped
 * portable vmo_mma_tile footprint. M,N,K padded to tile multiples on host =>
 * no partial-tile bounds in the hot loop. Local As/Bs = 8 KB (like §10c). */
#define TM 64
#define TN 64
#define BK 16
#define RM 4
#define RN 4
#define ASZ (TM * BK)   /* 1024 floats */
#define BSZ (TN * BK)   /* 1024 floats */

/* 4x4 microtile MAC from local As[m*BK+kk] / Bs[n*BK+kk] (both k-inner). */
static void mma_compute(__local const float *As, __local const float *Bs,
                       uint ty, uint tx, float acc[RM][RN])
{
    for (uint kk = 0; kk < BK; ++kk) {
        float av[RM], bv[RN];
        for (uint i = 0; i < RM; ++i) av[i] = As[(ty * RM + i) * BK + kk];
        for (uint j = 0; j < RN; ++j) bv[j] = Bs[(tx * RN + j) * BK + kk];
        for (uint i = 0; i < RM; ++i)
            for (uint j = 0; j < RN; ++j) acc[i][j] += av[i] * bv[j];
    }
}
static void mma_store(__global float *C, uint M, uint N, uint m0, uint n0,
                     uint ty, uint tx, float acc[RM][RN])
{
    for (uint i = 0; i < RM; ++i)
        for (uint j = 0; j < RN; ++j)
            C[(m0 + ty * RM + i) * N + (n0 + tx * RN + j)] = acc[i][j];
}

/* (a) baseline single-buffered: load block -> barrier -> compute -> barrier. */
__kernel void mma_single(__global const float *A, __global const float *B,
                        __global float *C, uint M, uint N, uint K)
{
    __local float As[ASZ], Bs[BSZ];
    const uint lid = get_local_id(0);
    const uint ty = lid / 16, tx = lid % 16;
    const uint mt = (M + TM - 1) / TM, nt = (N + TN - 1) / TN, nblk = mt * nt;
    for (uint blk = get_group_id(0); blk < nblk; blk += get_num_groups(0)) {
        const uint m0 = (blk / nt) * TM, n0 = (blk % nt) * TN;
        float acc[RM][RN];
        for (uint i = 0; i < RM; ++i) for (uint j = 0; j < RN; ++j) acc[i][j] = 0.0f;
        for (uint k0 = 0; k0 < K; k0 += BK) {
            for (uint idx = lid; idx < ASZ; idx += 256) {
                const uint m = idx / BK, kk = idx % BK;
                As[idx] = A[(m0 + m) * K + k0 + kk];
            }
            for (uint idx = lid; idx < BSZ; idx += 256) {
                const uint n = idx / BK, kk = idx % BK;
                Bs[idx] = B[(n0 + n) * K + k0 + kk];       /* B col-major */
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            mma_compute(As, Bs, ty, tx, acc);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        mma_store(C, M, N, m0, n0, ty, tx, acc);
    }
}

/* (b) manual double-buffer: prefetch NEXT K-block's global loads into registers
 * (rA/rB) while computing the current block from __local, then commit regs.
 * ASZ/256 = 4 elems per lane each. Single local pair reused (regs are the
 * second buffer) => same 8 KB local as (a): NO extra occupancy cost. */
__kernel void mma_double(__global const float *A, __global const float *B,
                        __global float *C, uint M, uint N, uint K)
{
    __local float As[ASZ], Bs[BSZ];
    const uint lid = get_local_id(0);
    const uint ty = lid / 16, tx = lid % 16;
    const uint mt = (M + TM - 1) / TM, nt = (N + TN - 1) / TN, nblk = mt * nt;
    for (uint blk = get_group_id(0); blk < nblk; blk += get_num_groups(0)) {
        const uint m0 = (blk / nt) * TM, n0 = (blk % nt) * TN;
        float acc[RM][RN];
        for (uint i = 0; i < RM; ++i) for (uint j = 0; j < RN; ++j) acc[i][j] = 0.0f;
        float rA[ASZ / 256], rB[BSZ / 256];
        /* prime block 0 into regs, commit to local */
        for (uint u = 0; u < ASZ / 256; ++u) {
            const uint idx = lid + u * 256, m = idx / BK, kk = idx % BK;
            rA[u] = A[(m0 + m) * K + 0 + kk];
        }
        for (uint u = 0; u < BSZ / 256; ++u) {
            const uint idx = lid + u * 256, n = idx / BK, kk = idx % BK;
            rB[u] = B[(n0 + n) * K + 0 + kk];
        }
        for (uint u = 0; u < ASZ / 256; ++u) As[lid + u * 256] = rA[u];
        for (uint u = 0; u < BSZ / 256; ++u) Bs[lid + u * 256] = rB[u];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k0 = 0; k0 < K; k0 += BK) {
            const uint nk = k0 + BK;
            /* issue next block's global loads NOW (latency overlaps compute) */
            if (nk < K) {
                for (uint u = 0; u < ASZ / 256; ++u) {
                    const uint idx = lid + u * 256, m = idx / BK, kk = idx % BK;
                    rA[u] = A[(m0 + m) * K + nk + kk];
                }
                for (uint u = 0; u < BSZ / 256; ++u) {
                    const uint idx = lid + u * 256, n = idx / BK, kk = idx % BK;
                    rB[u] = B[(n0 + n) * K + nk + kk];
                }
            }
            mma_compute(As, Bs, ty, tx, acc);
            barrier(CLK_LOCAL_MEM_FENCE);           /* done reading this block */
            if (nk < K) {
                for (uint u = 0; u < ASZ / 256; ++u) As[lid + u * 256] = rA[u];
                for (uint u = 0; u < BSZ / 256; ++u) Bs[lid + u * 256] = rB[u];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        mma_store(C, M, N, m0, n0, ty, tx, acc);
    }
}

/* (c) async_work_group_copy staging: chained contiguous row/column copies
 * (A row -> As, B col -> Bs), one wait_group_events per block. Single-buffered
 * (same 8 KB local as a) — tests whether the driver's DMA path beats hand
 * strided loads. Requires K % BK == 0 and M,N tile-multiples (host pads). */
__kernel void mma_async(__global const float *A, __global const float *B,
                       __global float *C, uint M, uint N, uint K)
{
    __local float As[ASZ], Bs[BSZ];
    const uint lid = get_local_id(0);
    const uint ty = lid / 16, tx = lid % 16;
    const uint mt = (M + TM - 1) / TM, nt = (N + TN - 1) / TN, nblk = mt * nt;
    for (uint blk = get_group_id(0); blk < nblk; blk += get_num_groups(0)) {
        const uint m0 = (blk / nt) * TM, n0 = (blk % nt) * TN;
        float acc[RM][RN];
        for (uint i = 0; i < RM; ++i) for (uint j = 0; j < RN; ++j) acc[i][j] = 0.0f;
        for (uint k0 = 0; k0 < K; k0 += BK) {
            event_t e = 0;
            for (uint u = 0; u < TM; ++u)
                e = async_work_group_copy(As + u * BK, A + (m0 + u) * K + k0, BK, e);
            for (uint u = 0; u < TN; ++u)
                e = async_work_group_copy(Bs + u * BK, B + (n0 + u) * K + k0, BK, e);
            wait_group_events(1, &e);
            mma_compute(As, Bs, ty, tx, acc);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        mma_store(C, M, N, m0, n0, ty, tx, acc);
    }
}

/* ==================== occupancy discovery (poc/08 handshake) ============= */
/* Deadlock-free co-resident-group count, to measure the §10c occupancy cost of
 * the extra __local a double-buffered / async-local kernel would need. */
#define NOT_RESIDENT 0xFFFFFFFFu
static uint discover(volatile __global uint *dd)
{
    uint tk = NOT_RESIDENT;
    while (atomic_cmpxchg(&dd[0], 0u, 1u) != 0u);
    if (atomic_add(&dd[1], 0u) == 1u) tk = atomic_inc(&dd[2]);
    atomic_xchg(&dd[0], 0u);
    if (tk == 0u) {
        uint last = 1u, stable = 0u;
        for (uint i = 0u; i < 50000000u && stable < 100000u; ++i) {
            uint c = atomic_add(&dd[2], 0u);
            if (c == last) stable++; else { stable = 0u; last = c; }
        }
        while (atomic_cmpxchg(&dd[0], 0u, 1u) != 0u);
        atomic_xchg(&dd[1], 0u);
        atomic_xchg(&dd[0], 0u);
    } else if (tk != NOT_RESIDENT) {
        while (atomic_add(&dd[1], 0u) == 1u);
    }
    return tk;
}
/* probe with LOCAL_FLOATS __local footprint + a small live reg accumulator, so
 * both local AND register pressure resemble the corresponding real kernel. */
#ifndef LOCAL_FLOATS
#define LOCAL_FLOATS 2048
#endif
__kernel void probe(volatile __global uint *dd, __global uint *sink, uint never)
{
    __local float scratch[LOCAL_FLOATS];
    float acc[16];
    for (int i = 0; i < 16; i++) acc[i] = (float)(get_group_id(0) + i);
    for (int k = 0; k < 16; k++)
        for (int i = 0; i < 16; i++) acc[i] = acc[i] * 1.0000001f + acc[(i + k) & 15];
    scratch[get_local_id(0) & (LOCAL_FLOATS - 1)] = acc[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float v = scratch[(get_local_id(0) + 1) & (LOCAL_FLOATS - 1)] + acc[15];
    if (never == 1234567u) sink[get_group_id(0)] = (uint)v;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0u) discover(dd);
}
