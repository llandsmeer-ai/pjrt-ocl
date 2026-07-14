/* poc/04: tick-synchronous VLIW VM (docs/tile-isa.md).
 *
 * Persistent lane-interpreters (one workgroup each). A schedule table
 * (ticks x lanes) assigns each lane a RANGE of tiles of a task descriptor per
 * tick; different lanes run different ops in the same tick. Tick boundary =
 * validated inter-workgroup barrier (poc/01).
 *
 * Instrumentation (logical-clock mode): lane 0 of each group records a global
 * completion rank after finishing its cell; rank spread per tick ~= imbalance.
 */

#define EW_TS 16384      /* elements per EW tile */
#define MMA_T 16         /* MMA tile edge: 16x16 output tile per cell slot */

enum { T_NOP = 0, T_EW = 1, T_MMA = 2, T_RED_PART = 3, T_RED_COMB = 4,
       T_FILL = 5 };
enum { EW_ADD = 0, EW_MUL = 1, EW_SUB = 2 };

typedef struct {
    uint op;             /* T_* */
    uint dst, a, b;      /* arena element offsets */
    uint p0, p1, p2, p3; /* EW: subop,n ; MMA: M,N,K ; RED_PART: n,chunk ;
                            RED_COMB: nparts ; FILL: n,f32bits */
} task_t;

typedef struct {
    uint task;           /* index into tasks[], 0xFFFFFFFF = NOP */
    uint tile_lo, tile_hi, pad;
} cell_t;

static void tick_barrier(volatile __global uint *bar, const uint ngroups)
{
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
        const uint phase = atomic_add(&bar[1], 0);
        if (atomic_inc(&bar[0]) == ngroups - 1) {
            bar[0] = 0;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            atomic_inc(&bar[1]);
        } else {
            while (atomic_add(&bar[1], 0) == phase)
                ;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/* one 16x16 output tile of dst[MxN] = a[MxK] @ b[KxN], local-memory staged */
static void mma_tile(__global float *arena, const task_t t, uint tile,
                     __local float *As, __local float *Bs)
{
    const uint M = t.p0, N = t.p1, K = t.p2;
    const uint tiles_n = (N + MMA_T - 1) / MMA_T;
    const uint tr = tile / tiles_n, tc = tile % tiles_n;
    const uint lr = get_local_id(0) / MMA_T, lc = get_local_id(0) % MMA_T;
    const uint r = tr * MMA_T + lr, c = tc * MMA_T + lc;
    float acc = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += MMA_T) {
        /* stage 16x16 blocks of A and B (first 256 threads participate) */
        if (get_local_id(0) < MMA_T * MMA_T) {
            const uint ar = tr * MMA_T + lr, ak = k0 + lc;
            As[lr * MMA_T + lc] =
                (ar < M && ak < K) ? arena[t.a + ar * K + ak] : 0.0f;
            const uint bk = k0 + lr, bc = tc * MMA_T + lc;
            Bs[lr * MMA_T + lc] =
                (bk < K && bc < N) ? arena[t.b + bk * N + bc] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (get_local_id(0) < MMA_T * MMA_T)
            for (uint k = 0; k < MMA_T; ++k)
                acc += As[lr * MMA_T + k] * Bs[k * MMA_T + lc];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (get_local_id(0) < MMA_T * MMA_T && r < M && c < N)
        arena[t.dst + r * N + c] = acc;
}

/* ---- async engine: PER-LANE instruction streams (model of record) --------
 * Each lane interprets its own stream; no global barrier. Cross-lane deps =
 * per-op completion counters: WAIT (atomic poll by lane-thread 0) before an
 * entry, SIGNAL (atomic_add of executed tile count) after. Producer's data
 * writes are fenced group-wide before the signal; consumer fences after
 * observing — same memory discipline as the validated tick barrier. */

#define FLAG_NONE 0xFFFFFFFFu
#define TASK_NOP 0xFFFFFFFFu
#define TASK_BARRIER 0xFFFFFFFEu  /* scheduler-placed GLOBAL sync point; every
                                     lane's stream must contain the same number
                                     of these, in the same dataflow order */

typedef struct {
    uint task, tile_lo, tile_hi;
    uint wait_flag, wait_count, signal_flag;
    uint pad0, pad1;
} entry_t;

static void exec_tiles(__global float *arena, const task_t t,
                       uint tile_lo, uint tile_hi,
                       __local float *As, __local float *Bs);

__kernel void vliw_async(__global float *arena,
                         __global const task_t *tasks,
                         __global const entry_t *streams,   /* flattened */
                         __global const uint2 *lane_tab,    /* {off,count}/lane */
                         volatile __global uint *flags,
                         volatile __global uint *bar,       /* [0,1]=barrier [2]=rank ctr */
                         const uint nlanes,
                         __global uint *ranks)              /* arrival rank per
                                                               [barrier_i * nlanes + lane] */
{
    const uint lane = get_group_id(0);
    const uint lid = get_local_id(0);
    __local float As[MMA_T * MMA_T];
    __local float Bs[MMA_T * MMA_T];
    uint barrier_i = 0;

    const uint2 span = lane_tab[lane];
    for (uint e = 0; e < span.y; ++e) {
        const entry_t en = streams[span.x + e];
        if (en.task == TASK_BARRIER) {
            /* bubble instrumentation: arrival order at this sync point */
            if (lid == 0)
                ranks[barrier_i * nlanes + lane] = atomic_inc(&bar[2]) % nlanes;
            tick_barrier(bar, nlanes);
            barrier_i++;
            continue;
        }
        if (en.wait_flag != FLAG_NONE) {
            if (lid == 0)
                while (atomic_add(&flags[en.wait_flag], 0) < en.wait_count)
                    ;
            barrier(CLK_GLOBAL_MEM_FENCE);   /* whole group sees dep data */
        }
        if (en.task != TASK_NOP)
            exec_tiles(arena, tasks[en.task], en.tile_lo, en.tile_hi, As, Bs);
        if (en.signal_flag != FLAG_NONE) {
            barrier(CLK_GLOBAL_MEM_FENCE);   /* all our writes done */
            if (lid == 0) {
                mem_fence(CLK_GLOBAL_MEM_FENCE);
                atomic_add(&flags[en.signal_flag], en.tile_hi - en.tile_lo);
            }
        }
    }
}

__kernel void vliw(__global float *arena,
                   __global const task_t *tasks,
                   __global const cell_t *sched,   /* [nticks * nlanes] */
                   const uint nticks,
                   const uint nlanes,
                   volatile __global uint *bar,    /* [0]=arrive [1]=phase
                                                      [2]=order counter */
                   __global uint *inst)            /* [nticks*nlanes] ranks,
                                                      or NULL-sized noop */
{
    const uint lane = get_group_id(0);
    const uint lid = get_local_id(0);
    const uint lsz = get_local_size(0);
    __local float As[MMA_T * MMA_T];
    __local float Bs[MMA_T * MMA_T];

    for (uint tick = 0; tick < nticks; ++tick) {
        const cell_t cell = sched[tick * nlanes + lane];
        if (cell.task != 0xFFFFFFFFu)
            exec_tiles(arena, tasks[cell.task], cell.tile_lo, cell.tile_hi,
                       As, Bs);
        /* instrumentation: completion rank of this lane within the run */
        if (lid == 0)
            inst[tick * nlanes + lane] = atomic_inc(&bar[2]);
        tick_barrier(bar, nlanes);
    }
}

static void exec_tiles(__global float *arena, const task_t t,
                       uint tile_lo, uint tile_hi,
                       __local float *As, __local float *Bs)
{
    const uint lid = get_local_id(0);
    const uint lsz = get_local_size(0);
    {
        {
            for (uint tile = tile_lo; tile < tile_hi; ++tile) {
                switch (t.op) {
                case T_EW: {
                    const uint lo = tile * EW_TS;
                    const uint hi = min(lo + EW_TS, t.p1);
                    for (uint i = lo + lid; i < hi; i += lsz) {
                        const float x = arena[t.a + i], y = arena[t.b + i];
                        arena[t.dst + i] = t.p0 == EW_ADD ? x + y
                                         : t.p0 == EW_MUL ? x * y : x - y;
                    }
                    break;
                }
                case T_FILL: {
                    const uint lo = tile * EW_TS;
                    const uint hi = min(lo + EW_TS, t.p0);
                    for (uint i = lo + lid; i < hi; i += lsz)
                        arena[t.dst + i] = as_float(t.p1);
                    break;
                }
                case T_MMA:
                    mma_tile(arena, t, tile, As, Bs);
                    break;
                case T_RED_PART: {
                    /* partial sum of chunk `tile` -> dst[tile]; local tree
                     * reduce (requires lsz power-of-two, <= 256) */
                    const uint lo = tile * t.p1;
                    const uint hi = min(lo + t.p1, t.p0);
                    __local float *red = As;  /* reuse 256-float scratch */
                    float acc = 0.0f;
                    for (uint i = lo + lid; i < hi; i += lsz)
                        acc += arena[t.a + i];
                    red[lid] = acc;
                    barrier(CLK_LOCAL_MEM_FENCE);
                    for (uint s = lsz / 2; s > 0; s >>= 1) {
                        if (lid < s) red[lid] += red[lid + s];
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (lid == 0) arena[t.dst + tile] = red[0];
                    break;
                }
                case T_RED_COMB: {
                    if (lid == 0) {
                        float acc = 0.0f;
                        for (uint i = 0; i < t.p0; ++i)
                            acc += arena[t.a + i];
                        arena[t.dst] = acc;
                    }
                    break;
                }
                default:
                    break;
                }
            }
        }
    }
}
