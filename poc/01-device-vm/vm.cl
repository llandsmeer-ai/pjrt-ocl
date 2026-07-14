/* poc/01: device-side bytecode VM — persistent kernel, linear instruction list.
 *
 * Design under test:
 *  - single f32 arena; "buffers" are element offsets into it
 *  - fixed-size instructions, strictly linear (no jumps)
 *  - grid-stride loops per instruction, sized independently of launch size
 *  - cross-workgroup barrier between instructions (Xiao & Feng style:
 *    atomic arrival counter + phase flag; OpenCL 1.2 atomics only)
 *
 * The barrier is only safe if ALL workgroups are co-resident. Launch size
 * must come from occupancy, not problem size.
 */

typedef struct {
    uint op;
    uint dst;   /* arena element offset */
    uint a;     /* arena element offset */
    uint b;     /* arena element offset */
    uint n;     /* element count */
    uint imm;   /* f32 bits for FILL */
    uint pad0, pad1;
} instr_t;      /* 32 bytes, mirrored on host */

enum { OP_NOP = 0, OP_ADD, OP_MUL, OP_SUB, OP_FILL, OP_IOTA, OP_REVADD,
       /* scalar compare: arena[dst] = arena[a] < arena[b] (n must be 1) */
       OP_LTS,
       /* while: cond list = prog[a, a+b), body list = prog[n, n+imm),
        * loop while arena[dst] != 0 after running cond list */
       OP_WHILE };

/* bar[0] = arrival counter, bar[1] = phase. atomic_add(,0) = coherent read. */
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

/* Control flow: no jumps. A WHILE instruction references nested (linear)
 * instruction lists; the interpreter drives them with an explicit frame
 * stack. Control decisions depend only on program constants and one arena
 * scalar read by every work-item after a barrier, so all work-items stay
 * converged and execute identical barrier sequences. */
#define MAX_DEPTH 8
#define WIDX_ROOT 0xFFFFFFFFu
typedef struct { uint pc, end, widx, phase; } frame_t; /* phase 0=cond 1=body */

__kernel void vm(__global float *arena,
                 __global const instr_t *prog,
                 const uint n_instr,
                 volatile __global uint *bar,
                 const uint ngroups)
{
    const uint gid = get_global_id(0);
    const uint gsz = get_global_size(0);

    frame_t st[MAX_DEPTH];
    int sp = 0;
    st[0].pc = 0; st[0].end = n_instr; st[0].widx = WIDX_ROOT; st[0].phase = 0;

    for (;;) {
        if (st[sp].pc >= st[sp].end) {         /* current range exhausted */
            if (st[sp].widx == WIDX_ROOT)
                break;
            const instr_t w = prog[st[sp].widx];
            if (st[sp].phase == 0) {           /* cond list done: test scalar */
                /* atomic read: plain loads can hit stale per-CU cache (seen on
                 * NVIDIA), making workgroups diverge on the loop decision and
                 * deadlocking the barrier. Bit-test: producers write 1.0/0.0. */
                const uint cbits = atomic_add(
                    (volatile __global uint *)arena + w.dst, 0u);
                if (cbits != 0u) {
                    st[sp].pc = w.n; st[sp].end = w.n + w.imm; st[sp].phase = 1;
                } else {
                    sp--;
                    st[sp].pc++;               /* step past the WHILE */
                }
            } else {                           /* body done: re-run cond */
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
        case OP_ADD:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = arena[ins.a + i] + arena[ins.b + i];
            break;
        case OP_MUL:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = arena[ins.a + i] * arena[ins.b + i];
            break;
        case OP_SUB:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = arena[ins.a + i] - arena[ins.b + i];
            break;
        case OP_FILL:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = as_float(ins.imm);
            break;
        case OP_IOTA:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = (float)i;
            break;
        case OP_REVADD:
            /* dst[i] = a[n-1-i] + b[i] — cross-workgroup data dependency on
             * the PREVIOUS instruction's output: fails loudly if the global
             * barrier is broken. */
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = arena[ins.a + (ins.n - 1 - i)] + arena[ins.b + i];
            break;
        case OP_LTS:
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

/* reference single-op kernel for the dispatch-overhead comparison */
__kernel void add1(__global float *arena, const uint dst, const uint a,
                   const uint b, const uint n)
{
    const uint gid = get_global_id(0);
    const uint gsz = get_global_size(0);
    for (uint i = gid; i < n; i += gsz)
        arena[dst + i] = arena[a + i] + arena[b + i];
}
