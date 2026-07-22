/* poc/21 — XMX GEMM via cl_intel_subgroup_matrix_multiply_accumulate_tf32.
 *
 * SG=16 shape is M=8, N=16, K=8, operands typed f32 (the engine rounds to
 * tf32 = 10 explicit mantissa bits internally):
 *     float8 = intel_sub_group_tf32_tf32_matrix_mad_k8(float4 a, float8 b, float8 acc)
 *   A (8x8):  the 64 values are laid out flat in GRF order, so lane l element e
 *             is A[flat/8][flat%8] with flat = e*16 + l  ->  a plain row-major
 *             8x8 tile read with intel_sub_group_block_read4.
 *   B (8x16): lane n, element k -> B[k][n]  -> row-major 8x16 tile, block_read8.
 *   C (8x16): lane n, element m -> C[m][n].
 * No VNNI packing: tf32 operands are 32-bit, so "packing" is pure retiling.
 */

#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#define SG 16

/* A[M][K] -> tile (m/8, k/8), 64 floats, index (m%8)*8 + (k%8) */
__kernel void pack_a(__global const float *A, __global float *Ap, uint K) {
    uint i = get_global_id(0);
    uint m = i / K, k = i - m * K;
    uint tile = (m >> 3) * (K >> 3) + (k >> 3);
    Ap[tile * 64u + (m & 7u) * 8u + (k & 7u)] = A[i];
}

/* B[K][N] -> tile (k/8, n/16), 128 floats, index (k%8)*16 + (n%16) */
__kernel void pack_b(__global const float *B, __global float *Bp, uint N) {
    uint i = get_global_id(0);
    uint k = i / N, n = i - k * N;
    uint tile = (k >> 3) * (N >> 4) + (n >> 4);
    Bp[tile * 128u + (k & 7u) * 16u + (n & 15u)] = B[i];
}

#ifndef SGM
#define SGM 2
#endif
#ifndef SGN
#define SGN 4
#endif
#ifndef RM
#define RM 4
#endif
#ifndef RN
#define RN 2
#endif

#define WM (SGM * RM * 8)
#define WN (SGN * RN * 16)
#define NWI (SGM * SGN * 16)

#define WRROW(q, v)                                                            \
    do {                                                                       \
        intel_sub_group_block_write((q) + 0u * N, as_uint((v).s0));            \
        intel_sub_group_block_write((q) + 1u * N, as_uint((v).s1));            \
        intel_sub_group_block_write((q) + 2u * N, as_uint((v).s2));            \
        intel_sub_group_block_write((q) + 3u * N, as_uint((v).s3));            \
        intel_sub_group_block_write((q) + 4u * N, as_uint((v).s4));            \
        intel_sub_group_block_write((q) + 5u * N, as_uint((v).s5));            \
        intel_sub_group_block_write((q) + 6u * N, as_uint((v).s6));            \
        intel_sub_group_block_write((q) + 7u * N, as_uint((v).s7));            \
    } while (0)

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(NWI, 1, 1)))
__kernel void gemm(__global const float *Ap, __global const float *Bp,
                   __global float *C, uint M, uint N, uint K) {
    const uint KT = K >> 3;              /* k-tiles of 8 */
    const uint NT = N >> 4;
    const uint sg = get_sub_group_id();
    const uint sgm = sg / SGN, sgn = sg % SGN;
    const uint wgN = N / WN;
    const uint wg = get_group_id(0);
    const uint wgm = wg / wgN, wgn = wg - wgm * wgN;

    const uint m8 = (wgm * SGM + sgm) * RM;
    const uint n16 = (wgn * SGN + sgn) * RN;

    float8 acc[RM][RN];
#pragma unroll
    for (int i = 0; i < RM; i++)
#pragma unroll
        for (int j = 0; j < RN; j++) acc[i][j] = (float8)(0.f);

    for (uint kt = 0; kt < KT; kt++) {
        float4 a[RM];
        float8 b[RN];
#pragma unroll
        for (int i = 0; i < RM; i++)
            a[i] = as_float4(intel_sub_group_block_read4(
                (const __global uint *)(Ap + ((m8 + i) * KT + kt) * 64u)));
#pragma unroll
        for (int j = 0; j < RN; j++)
            b[j] = as_float8(intel_sub_group_block_read8(
                (const __global uint *)(Bp + (kt * NT + n16 + j) * 128u)));
#pragma unroll
        for (int i = 0; i < RM; i++)
#pragma unroll
            for (int j = 0; j < RN; j++)
                acc[i][j] = intel_sub_group_tf32_tf32_matrix_mad_k8(a[i], b[j], acc[i][j]);
    }

    __global uint *Cu = (__global uint *)C;
#pragma unroll
    for (int i = 0; i < RM; i++)
#pragma unroll
        for (int j = 0; j < RN; j++) {
            __global uint *q = Cu + (size_t)((m8 + i) * 8u) * N + (n16 + j) * 16u;
            WRROW(q, acc[i][j]);
        }
}
