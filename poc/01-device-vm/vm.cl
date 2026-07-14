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

enum { OP_NOP = 0, OP_ADD, OP_MUL, OP_SUB, OP_FILL, OP_IOTA, OP_REVADD };

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

__kernel void vm(__global float *arena,
                 __global const instr_t *prog,
                 const uint n_instr,
                 volatile __global uint *bar,
                 const uint ngroups)
{
    const uint gid = get_global_id(0);
    const uint gsz = get_global_size(0);

    for (uint pc = 0; pc < n_instr; ++pc) {
        const instr_t ins = prog[pc];
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
        default:
            break;
        }
        global_barrier(bar, ngroups);
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
