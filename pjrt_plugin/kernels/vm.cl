/* pjrt-ocl device VM. Executes VMProgram v1 (docs/vmprogram.md) instruction
 * lists inside one persistent kernel launch. Validated design: poc/01.
 *
 * Host-side loader patches dst/a/b from buffer-table indices to f32-element
 * offsets into the arena (WHILE keeps program ranges in a/b/n/imm; only its
 * dst is patched). Launch: ngroups <= co-resident capacity (see poc/01).
 */

typedef struct {
    uint op, dst, a, b, n, imm, aux, pad1;
} instr_t;

enum {
    OP_NOP = 0, OP_ADD_F32 = 1, OP_MUL_F32 = 2, OP_SUB_F32 = 3,
    OP_FILL_F32 = 4, OP_IOTA_F32 = 5, OP_LTS_F32 = 6, OP_WHILE = 7,
    /* v2 */
    OP_DIV_F32 = 8, OP_MAX_F32 = 9, OP_MIN_F32 = 10, OP_POW_F32 = 11,
    OP_COPY_F32 = 12, OP_NEG_F32 = 13, OP_EXP_F32 = 14, OP_LOG_F32 = 15,
    OP_SQRT_F32 = 16, OP_RSQRT_F32 = 17, OP_TANH_F32 = 18, OP_ABS_F32 = 19,
    OP_FLOOR_F32 = 20, OP_CEIL_F32 = 21, OP_SIGN_F32 = 22, OP_CMP_F32 = 23,
    OP_SELECT_F32 = 24, OP_GATHER_STRIDED = 25, OP_REDUCE = 26, OP_DOT = 27,
    OP_IOTA_DIM = 28, OP_IF = 29
};

#define MAX_RANK 8

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
                 const uint ngroups,
                 __global const int *aux)
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
            if (st[sp].phase == 2) {           /* IF branch done */
                sp--;
                st[sp].pc++;
                continue;
            }
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
        if (ins.op == OP_IF) {
            /* Cond read atomically (same coherence rule as WHILE). */
            const uint cbits = atomic_add(
                (volatile __global uint *)arena + ins.dst, 0u);
            const uint start = cbits != 0u ? ins.a : ins.n;
            const uint len = cbits != 0u ? ins.b : ins.imm;
            if (len == 0) { st[sp].pc++; continue; }
            sp++;
            st[sp].pc = start; st[sp].end = start + len;
            st[sp].widx = st[sp - 1].pc; st[sp].phase = 2;
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
        case OP_DIV_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = arena[ins.a + i] / arena[ins.b + i];
            break;
        case OP_MAX_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = fmax(arena[ins.a + i], arena[ins.b + i]);
            break;
        case OP_MIN_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = fmin(arena[ins.a + i], arena[ins.b + i]);
            break;
        case OP_POW_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = pow(arena[ins.a + i], arena[ins.b + i]);
            break;
        case OP_COPY_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = arena[ins.a + i];
            break;
        case OP_NEG_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = -arena[ins.a + i];
            break;
        case OP_EXP_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = exp(arena[ins.a + i]);
            break;
        case OP_LOG_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = log(arena[ins.a + i]);
            break;
        case OP_SQRT_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = sqrt(arena[ins.a + i]);
            break;
        case OP_RSQRT_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = rsqrt(arena[ins.a + i]);
            break;
        case OP_TANH_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = tanh(arena[ins.a + i]);
            break;
        case OP_ABS_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = fabs(arena[ins.a + i]);
            break;
        case OP_FLOOR_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = floor(arena[ins.a + i]);
            break;
        case OP_CEIL_F32:
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = ceil(arena[ins.a + i]);
            break;
        case OP_SIGN_F32:
            for (uint i = gid; i < ins.n; i += gsz) {
                const float v = arena[ins.a + i];
                arena[ins.dst + i] = v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : v);
            }
            break;
        case OP_CMP_F32:
            for (uint i = gid; i < ins.n; i += gsz) {
                const float x = arena[ins.a + i], y = arena[ins.b + i];
                int r;
                switch (ins.imm) {
                case 0: r = x == y; break;
                case 1: r = x != y; break;
                case 2: r = x < y;  break;
                case 3: r = x <= y; break;
                case 4: r = x > y;  break;
                default: r = x >= y; break;
                }
                arena[ins.dst + i] = r ? 1.0f : 0.0f;
            }
            break;
        case OP_SELECT_F32:
            /* imm = pred element offset (loader-patched like dst/a/b) */
            for (uint i = gid; i < ins.n; i += gsz)
                arena[ins.dst + i] = arena[ins.imm + i] != 0.0f
                                         ? arena[ins.a + i]
                                         : arena[ins.b + i];
            break;
        case OP_GATHER_STRIDED: {
            __global const int *x = aux + ins.aux;
            const int rank = x[0];
            __global const int *dims = x + 1, *strides = x + 1 + rank;
            const int src_off = x[1 + 2 * rank];
            for (uint i = gid; i < ins.n; i += gsz) {
                int rem = (int)i, off = src_off;
                for (int d = rank - 1; d >= 0; --d) {
                    off += (rem % dims[d]) * strides[d];
                    rem /= dims[d];
                }
                arena[ins.dst + i] = arena[ins.a + off];
            }
            break;
        }
        case OP_REDUCE: {
            __global const int *x = aux + ins.aux;
            const int kind = x[0], out_rank = x[1];
            __global const int *odims = x + 2, *kstr = x + 2 + out_rank;
            const int red_rank = x[2 + 2 * out_rank];
            __global const int *rdims = x + 3 + 2 * out_rank;
            __global const int *rstr = rdims + red_rank;
            const int src_off = rdims[2 * red_rank];
            for (uint i = gid; i < ins.n; i += gsz) {
                int rem = (int)i, base = src_off;
                for (int d = out_rank - 1; d >= 0; --d) {
                    base += (rem % odims[d]) * kstr[d];
                    rem /= odims[d];
                }
                int red_n = 1;
                for (int d = 0; d < red_rank; ++d) red_n *= rdims[d];
                float acc = kind == 0 ? 0.0f
                          : kind == 1 ? -INFINITY
                          : kind == 2 ? INFINITY : 1.0f;
                for (int r = 0; r < red_n; ++r) {
                    int rr = r, off = base;
                    for (int d = red_rank - 1; d >= 0; --d) {
                        off += (rr % rdims[d]) * rstr[d];
                        rr /= rdims[d];
                    }
                    const float v = arena[ins.a + off];
                    acc = kind == 0 ? acc + v
                        : kind == 1 ? fmax(acc, v)
                        : kind == 2 ? fmin(acc, v) : acc * v;
                }
                arena[ins.dst + i] = acc;
            }
            break;
        }
        case OP_DOT: {
            __global const int *x = aux + ins.aux;
            const int N = x[1], K = x[2];
            for (uint i = gid; i < ins.n; i += gsz) {
                const int r = (int)i / N, c = (int)i % N;
                float acc = 0.0f;
                for (int k = 0; k < K; ++k)
                    acc += arena[ins.a + r * K + k] * arena[ins.b + k * N + c];
                arena[ins.dst + i] = acc;
            }
            break;
        }
        case OP_IOTA_DIM: {
            __global const int *x = aux + ins.aux;
            const int rank = x[0];
            __global const int *dims = x + 1;
            const int dim = x[1 + rank];
            for (uint i = gid; i < ins.n; i += gsz) {
                int rem = (int)i, val = 0;
                for (int d = rank - 1; d >= 0; --d) {
                    const int idx = rem % dims[d];
                    rem /= dims[d];
                    if (d == dim) val = idx;
                }
                arena[ins.dst + i] = (float)val;
            }
            break;
        }
        default:
            break;
        }
        global_barrier(bar, ngroups);
        st[sp].pc++;
    }
}
