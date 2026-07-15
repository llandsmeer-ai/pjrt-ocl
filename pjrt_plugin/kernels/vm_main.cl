/* pjrt-ocl VLIW engine — dispatcher + interpreter (concatenated last).
 * exec_tiles routes a task to its op-family tile function (ops/*.cl above);
 * vm2 is the per-lane interpreter over the schedule stream. */

/* tile_op packs the base op in bits 0-7 and the dtype in bits 8-15. */
static void exec_tiles(__global uchar *arena, __global const int *aux,
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
        case TOP_EW:       ew_tile(arena, t, tile, dt, adt, lid, lsz); break;
        case TOP_MMA:      mma_tile(arena, t, tile, As, Bs); break;
        case TOP_GATHER:   gather_tile(arena, aux, t, tile, esz, lid, lsz); break;
        case TOP_RED_PART: reduce_part_tile(arena, t, tile, As, dt, lid, lsz); break;
        case TOP_RED_COMB: reduce_comb_tile(arena, t, dt, lid); break;
        case TOP_IOTA_DIM: iota_tile(arena, aux, t, tile, lid, lsz); break;
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
                  __global uint *stats)              /* arrival rank per
                                                        [barrier_i*nlanes+lane] */
{
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
            if (st[sp].phase == 0) {           /* while-cond range done */
                global_barrier(bar, nlanes);
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
                global_barrier(bar, nlanes);
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
            global_barrier(bar, nlanes);
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
            exec_tiles(arena, aux, tasks[en.task], en.tile_lo, en.tile_hi,
                       As, Bs);
        }
        st[sp].pc++;
    }
}
