/* poc/21 probe: which cl_intel_subgroup_matrix_multiply_accumulate builtin
 * signatures actually COMPILE on this device/driver? One variant per build,
 * selected by -DV_<name>; the host tries them all and reports the build log.
 *
 * Naming per the extension spec:
 *   intel_sub_group_<Atype>_<Btype>_matrix_mad_k<K>(a, b, acc)
 *     N = subgroup size (set by intel_reqd_sub_group_size)
 *     M = vector width of `acc` / result (1,2,4,8)
 *     K = in the name (bf16/f16: 16, i8: 32, tf32: 8)
 *   A is MxK packed in `a`, B is KxN packed in `b` (as ints), acc is MxN f32.
 */

#ifndef SG
#define SG 16
#endif

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void probe(__global int *ai, __global int *bi, __global float *co) {
    int lid = get_local_id(0);

#if defined(V_BF16_M1)
    int  a = ai[lid]; int8 b = ((__global int8 *)bi)[lid];
    float acc = 0.f;
    acc = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, acc);
    co[lid] = acc;

#elif defined(V_BF16_M8)
    int8 a = ((__global int8 *)ai)[lid]; int8 b = ((__global int8 *)bi)[lid];
    float8 acc = (float8)(0.f);
    acc = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, acc);
    ((__global float8 *)co)[lid] = acc;

#elif defined(V_BF16_M8_SHORT)
    /* subgroup-size-16 form: A row-pair packed as short8 rather than int8 */
    short8 a = ((__global short8 *)ai)[lid]; int8 b = ((__global int8 *)bi)[lid];
    float8 acc = (float8)(0.f);
    acc = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, acc);
    ((__global float8 *)co)[lid] = acc;

#elif defined(V_F16_M8)
    int8 a = ((__global int8 *)ai)[lid]; int8 b = ((__global int8 *)bi)[lid];
    float8 acc = (float8)(0.f);
    acc = intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc);
    ((__global float8 *)co)[lid] = acc;

#elif defined(V_F16_M8_SHORT)
    short8 a = ((__global short8 *)ai)[lid]; int8 b = ((__global int8 *)bi)[lid];
    float8 acc = (float8)(0.f);
    acc = intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc);
    ((__global float8 *)co)[lid] = acc;

#elif defined(V_F16_M8_HALF)
    half8 a = ((__global half8 *)ai)[lid]; int8 b = ((__global int8 *)bi)[lid];
    float8 acc = (float8)(0.f);
    acc = intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc);
    ((__global float8 *)co)[lid] = acc;

#elif defined(V_TF32_M8)
    float4 a = ((__global float4 *)ai)[lid]; float8 b = ((__global float8 *)bi)[lid];
    float8 acc = (float8)(0.f);
    acc = intel_sub_group_tf32_tf32_matrix_mad_k8(a, b, acc);
    ((__global float8 *)co)[lid] = acc;

#elif defined(V_TF32_M8_F2)
    float2 a = ((__global float2 *)ai)[lid]; float8 b = ((__global float8 *)bi)[lid];
    float8 acc = (float8)(0.f);
    acc = intel_sub_group_tf32_tf32_matrix_mad_k8(a, b, acc);
    ((__global float8 *)co)[lid] = acc;

#elif defined(V_I8_M8)
    int8 a = ((__global int8 *)ai)[lid]; int8 b = ((__global int8 *)bi)[lid];
    int8 acc = (int8)(0);
    acc = intel_sub_group_i8_i8_matrix_mad_k32(a, b, acc);
    ((__global int8 *)co)[lid] = acc;

#elif defined(V_BF16CVT)
    /* cl_intel_bfloat16_conversions: f32 <-> bf16 scalar/vector converts */
    float f = as_float(ai[lid]);
    ushort h = intel_convert_bfloat16_as_ushort(f);
    co[lid] = intel_convert_as_bfloat16_float(h);

#else
    /* control: no extension used at all */
    co[lid] = (float)(ai[lid] + bi[lid]);
#endif
}
