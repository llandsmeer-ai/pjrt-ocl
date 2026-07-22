/* poc/21 — XMX (DPAS) GEMM via cl_intel_subgroup_matrix_multiply_accumulate.
 *
 * Sub-group size is 16 on Xe2 (sg8 is rejected by the driver, see probe21).
 * For SG=16 the bf16/f16 builtin shape is  M=8, N=16, K=16:
 *     float8 = intel_sub_group_<t>_<t>_matrix_mad_k16(short8 a, int8 b, float8 acc)
 *   A (8x16): lane l, element m -> A[m][l]            (row-major 8x16 tile)
 *   B (16x16): lane n, int j    -> {B[2j][n], B[2j+1][n]}   (VNNI row-pair pack)
 *   C (8x16): lane n, element m -> C[m][n]
 * Both tiles are therefore exactly one intel_sub_group_block_read{_us}8 each,
 * provided the operands are pre-packed into that tile order — which pack_a /
 * pack_b below do while converting f32 -> bf16/f16.
 *
 * Build options: -DVAR_BF16 | -DVAR_F16, -DSGM= -DSGN= -DRM= -DRN=
 *   workgroup tile = (SGM*RM*8) rows x (SGN*RN*16) cols, SGM*SGN*16 work-items
 */

#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#define SG 16

#if defined(VAR_F16)
#  define CVT(f)  as_ushort(convert_half(f))
#  define MAD(a,b,c) intel_sub_group_f16_f16_matrix_mad_k16(a,b,c)
#else
#  define CVT(f)  intel_convert_bfloat16_as_ushort(f)
#  define MAD(a,b,c) intel_sub_group_bf16_bf16_matrix_mad_k16(a,b,c)
#endif

/* ---- packing: f32 row-major -> 16-bit tile order (one work-item per element) */

/* A[M][K] -> tile (m/8, k/16), 128 ushorts, element (m%8)*16 + (k%16) */
__kernel void pack_a(__global const float *A, __global ushort *Ap, uint K) {
    uint i = get_global_id(0);
    uint m = i / K, k = i - m * K;
    uint tile = (m >> 3) * (K >> 4) + (k >> 4);
    Ap[tile * 128u + (m & 7u) * 16u + (k & 15u)] = CVT(A[i]);
}

/* B[K][N] -> tile (k/16, n/16), 256 ushorts; as uints: uint[j*16+lane] holds
 * B[16tk+2j][16tn+lane] in the low half, B[16tk+2j+1][...] in the high half. */
__kernel void pack_b(__global const float *B, __global ushort *Bp, uint N) {
    uint i = get_global_id(0);
    uint k = i / N, n = i - k * N;
    uint tile = (k >> 4) * (N >> 4) + (n >> 4);
    uint j = (k & 15u) >> 1, hi = k & 1u, lane = n & 15u;
    Bp[tile * 256u + (j * 16u + lane) * 2u + hi] = CVT(B[i]);
}

/* ---- the XMX GEMM ---- */

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

#define WM (SGM * RM * 8)     /* workgroup output rows */
#define WN (SGN * RN * 16)    /* workgroup output cols */
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
__kernel void gemm(__global const ushort *Ap, __global const ushort *Bp,
                   __global float *C, uint M, uint N, uint K) {
    const uint KT = K >> 4;              /* k-tiles */
    const uint NT = N >> 4;              /* n-tiles per B row-block */
    const uint sg = get_sub_group_id();
    const uint sgm = sg / SGN, sgn = sg % SGN;
    const uint wgN = N / WN;
    const uint wg = get_group_id(0);
    const uint wgm = wg / wgN, wgn = wg - wgm * wgN;

    const uint m8 = (wgm * SGM + sgm) * RM;    /* first row-tile  (units of 8) */
    const uint n16 = (wgn * SGN + sgn) * RN;   /* first col-tile  (units of 16)*/

    float8 acc[RM][RN];
#pragma unroll
    for (int i = 0; i < RM; i++)
#pragma unroll
        for (int j = 0; j < RN; j++) acc[i][j] = (float8)(0.f);

    for (uint kt = 0; kt < KT; kt++) {
        short8 a[RM];
        int8 b[RN];
#pragma unroll
        for (int i = 0; i < RM; i++)
            a[i] = as_short8(intel_sub_group_block_read_us8(
                Ap + ((m8 + i) * KT + kt) * 128u));
#pragma unroll
        for (int j = 0; j < RN; j++)
            b[j] = as_int8(intel_sub_group_block_read8(
                (const __global uint *)(Bp + (kt * NT + n16 + j) * 256u)));
#pragma unroll
        for (int i = 0; i < RM; i++)
#pragma unroll
            for (int j = 0; j < RN; j++) acc[i][j] = MAD(a[i], b[j], acc[i][j]);
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
