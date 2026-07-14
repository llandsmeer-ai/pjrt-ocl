/* pjrt-ocl device VM. Executes VMProgram v1 (docs/vmprogram.md) instruction
 * lists inside one persistent kernel launch. Validated design: poc/01.
 *
 * Host-side loader patches dst/a/b from buffer-table indices to f32-element
 * offsets into the arena (WHILE keeps program ranges in a/b/n/imm; only its
 * dst is patched). Launch: ngroups <= co-resident capacity (see poc/01).
 */

typedef struct {
    uint op, dst, a, b, n, imm, pad0, pad1;
} instr_t;

enum {
    OP_NOP = 0, OP_ADD_F32 = 1, OP_MUL_F32 = 2, OP_SUB_F32 = 3,
    OP_FILL_F32 = 4, OP_IOTA_F32 = 5, OP_LTS_F32 = 6, OP_WHILE = 7
};

/* Inter-workgroup barrier (poc/01): bar[0] arrival counter, bar[1] phase.
 * atomic_add(p,0) = coherent read. Safe only with all groups co-resident. */
static void global_barrier(volatile __global uint *bar, const uint ngroups)
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

#define MAX_DEPTH 8
#define WIDX_ROOT 0xFFFFFFFFu
typedef struct { uint pc, end, widx, phase; } frame_t;

__kernel void vm(__global float *arena,
                 __global const instr_t *prog,
                 const uint main_len,
                 volatile __global uint *bar,
                 const uint ngroups)
{
    const uint gid = get_global_id(0);
    const uint gsz = get_global_size(0);

    frame_t st[MAX_DEPTH];
    int sp = 0;
    st[0].pc = 0; st[0].end = main_len; st[0].widx = WIDX_ROOT; st[0].phase = 0;

    for (;;) {
        if (st[sp].pc >= st[sp].end) {
            if (st[sp].widx == WIDX_ROOT)
                break;
            const instr_t w = prog[st[sp].widx];
            if (st[sp].phase == 0) {
                /* Cond scalar MUST be read atomically: plain loads can hit
                 * stale per-CU cache, diverging workgroups on the loop
                 * decision and deadlocking the barrier (poc/01, NVIDIA). */
                const uint cbits = atomic_add(
                    (volatile __global uint *)arena + w.dst, 0u);
                if (cbits != 0u) {
                    st[sp].pc = w.n; st[sp].end = w.n + w.imm; st[sp].phase = 1;
                } else {
                    sp--;
                    st[sp].pc++;
                }
            } else {
                st[sp].pc = w.a; st[sp].end = w.a + w.b; st[sp].phase = 0;
            }
            continue;
        }

        const instr_t ins = prog[st[sp].pc];
        if (ins.op == OP_WHILE) {
            sp++;
            st[sp].pc = ins.a; st[sp].end = ins.a + ins.b;
            st[sp].widx = st[sp - 1].pc; st[sp].phase = 0;
            continue;
        }

        switch (ins.op) {
        case OP_ADD_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = arena[ins.a + i] + arena[ins.b + i];
            break;
        case OP_MUL_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = arena[ins.a + i] * arena[ins.b + i];
            break;
        case OP_SUB_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = arena[ins.a + i] - arena[ins.b + i];
            break;
        case OP_FILL_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = as_float(ins.imm);
            break;
        case OP_IOTA_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = (float)i;
            break;
        case OP_LTS_F32:
            if (gid == 0)
                arena[ins.dst] = (arena[ins.a] < arena[ins.b]) ? 1.0f : 0.0f;
            break;
        default:
            break;
        }
        global_barrier(bar, ngroups);
        st[sp].pc++;
    }
}
