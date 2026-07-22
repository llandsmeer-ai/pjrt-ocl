/* poc/17 cp.async feasibility probe (docs/decisions.md §35).
 *
 * Question the whole multi-stage-async matmul thesis hinges on: does the NVIDIA
 * OpenCL ICD's inline-PTX path actually EXECUTE cp.async.cg.shared.global (the
 * Ampere async global->shared copy) + its group completion? If not, the deep
 * software pipeline that cuBLAS-class GEMM needs cannot be built on this ICD.
 *
 * One 256-thread workgroup async-copies a 1024-float global block into shared
 * (16 bytes / thread), completes, barriers, writes shared back to global; host
 * checks identity. V_STSHARED is the control: a SYNCHRONOUS st.shared through
 * the IDENTICAL cvta-derived shared address — if that is correct but every
 * cp.async form is wrong, the address mapping is fine and cp.async itself is the
 * broken primitive. */
kernel void probe(global float *g_in, global float *g_out)
{
    local float sm[256 * 4];
    const uint lid = get_local_id(0);
    global const float *src = g_in + lid * 4;
    local float *dst = sm + lid * 4;

    /* generic->shared, narrowed to the 32-bit shared-window address cp.async
     * wants (the CUDA __cvta_generic_to_shared idiom). */
    unsigned sa;
    asm volatile("{ .reg .u64 s64; cvta.to.shared.u64 s64, %1;"
                 " cvt.u32.u64 %0, s64; }" : "=r"(sa) : "l"(dst));

#if defined(V_STSHARED)             /* CONTROL: synchronous scalar shared stores */
    float v0 = src[0], v1 = src[1], v2 = src[2], v3 = src[3];
    asm volatile("st.shared.f32 [%0],    %1;" :: "r"(sa), "f"(v0) : "memory");
    asm volatile("st.shared.f32 [%0+4],  %1;" :: "r"(sa), "f"(v1) : "memory");
    asm volatile("st.shared.f32 [%0+8],  %1;" :: "r"(sa), "f"(v2) : "memory");
    asm volatile("st.shared.f32 [%0+12], %1;" :: "r"(sa), "f"(v3) : "memory");
#elif defined(V_CG)                 /* cp.async .cg + commit/wait_group */
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(sa), "l"(src) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#elif defined(V_CA)                 /* cp.async .ca */
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :: "r"(sa), "l"(src) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#elif defined(V_FENCE)              /* + async proxy fence before readback */
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(sa), "l"(src) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#elif defined(V_SPIN)               /* wait + 200k-iter spin: not a wait bug? */
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(sa), "l"(src) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    volatile float acc = 0; for (int i = 0; i < 200000; i++) acc += i * 0.5f;
    if (acc == -1.0f) sm[0] = acc;
#elif defined(V_MBAR)               /* mbarrier completion path */
    __local ulong mbar[1];
    unsigned mba;
    asm volatile("{ .reg .u64 s64; cvta.to.shared.u64 s64, %1;"
                 " cvt.u32.u64 %0, s64; }" : "=r"(mba) : "l"((__local void*)mbar));
    if (lid == 0)
        asm volatile("mbarrier.init.shared.b64 [%0], 256;" :: "r"(mba) : "memory");
    barrier(CLK_LOCAL_MEM_FENCE);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(sa), "l"(src) : "memory");
    asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];" :: "r"(mba) : "memory");
    asm volatile("{ .reg .pred p;\n"
                 "L_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], 0;\n"
                 "@!p bra L_%=; }" :: "r"(mba) : "memory");
#else
    for (int i = 0; i < 4; i++) dst[i] = src[i];
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    global float *o = g_out + lid * 4;
    for (int i = 0; i < 4; i++) o[i] = sm[lid * 4 + i];
}
