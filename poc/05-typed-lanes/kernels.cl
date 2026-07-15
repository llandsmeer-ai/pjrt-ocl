/* poc/05: typed-lanes cross-kernel co-residency test (docs/tile-isa.md
 * ceiling-1). Two DIFFERENT __kernel functions, launched on two SEPARATE
 * in-order queues of the same device/context, ping-pong through atomic
 * counters in a shared flags buffer. If the driver does not co-schedule
 * both kernels' workgroups at once, the first WAIT spins forever.
 *
 * Sync idiom is exactly poc/04's WAIT/SIGNAL entry_t pattern:
 *   barrier(local)                          -- all threads see own writes
 *   if (lid==0) atomic_add(&flags[mine],1)  -- SIGNAL this round done
 *   if (lid==0) spin until atomic_add(&flags[theirs],0) >= threshold  -- WAIT
 *   barrier(local, global fence)            -- whole group sees partner data
 *
 * kernel_a always owns flags[0] (signals) / flags[1] (waits on).
 * kernel_b always owns flags[1] (signals) / flags[0] (waits on).
 * Threshold accounts for GA != GB: kernel_a waits for flags[1] >= (r+1)*gb,
 * kernel_b waits for flags[0] >= (r+1)*ga -- correct regardless of the
 * GA/GB ratio since each flag is only ever incremented by its owner kernel's
 * groups, exactly once per group per round.
 */

#define MMA_ITERS 64   /* "mma-ish": heavier per-round dummy compute */
#define EW_ITERS  8    /* "ew-ish": lighter per-round dummy compute */

static float dummy_compute(float v, int iters)
{
    for (int k = 0; k < iters; ++k)
        v = v * 1.0000001f + 0.0000001f;
    return v;
}

__kernel void mma_ish(__global float *scratch,
                      volatile __global uint *flags,
                      const uint rounds, const uint ga, const uint gb)
{
    const uint gid = get_group_id(0);
    const uint lid = get_local_id(0);
    __local float ls[256];
    float v = (float)(gid + 1);
    uint r;
    for (r = 0; r < rounds; ++r) {
        v = dummy_compute(v, MMA_ITERS);
        ls[lid] = v;
        barrier(CLK_LOCAL_MEM_FENCE);
        v = ls[(lid + 1) % 256];               /* touch local mem region */
        barrier(CLK_GLOBAL_MEM_FENCE);          /* our writes are visible */
        if (lid == 0) {
            atomic_add(&flags[0], 1);           /* SIGNAL round r done */
            while (atomic_add(&flags[1], 0) < (r + 1) * gb)
                ;                                /* WAIT for partner's round r */
        }
        barrier(CLK_GLOBAL_MEM_FENCE);          /* whole group sees partner data */
    }
    scratch[gid * 256 + lid] = (float)r;        /* correctness: must == rounds */
}

__kernel void ew_ish(__global float *scratch,
                     volatile __global uint *flags,
                     const uint rounds, const uint ga, const uint gb)
{
    const uint gid = get_group_id(0);
    const uint lid = get_local_id(0);
    __local float ls[256];
    float v = (float)(gid + 1);
    uint r;
    for (r = 0; r < rounds; ++r) {
        v = dummy_compute(v, EW_ITERS);
        ls[lid] = v;
        barrier(CLK_LOCAL_MEM_FENCE);
        v = ls[(lid + 1) % 256];
        barrier(CLK_GLOBAL_MEM_FENCE);
        if (lid == 0) {
            atomic_add(&flags[1], 1);           /* SIGNAL round r done */
            while (atomic_add(&flags[0], 0) < (r + 1) * ga)
                ;                                /* WAIT for partner's round r */
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    scratch[gid * 256 + lid] = (float)r;
}
