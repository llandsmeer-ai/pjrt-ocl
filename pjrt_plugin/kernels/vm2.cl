/* pjrt-ocl VLIW engine (docs/tile-isa.md, docs/vmprogram.md v2.1).
 *
 * One persistent workgroup per lane; each lane interprets ITS OWN entry
 * stream. Global sync only at scheduler-placed BARRIER entries (uniform
 * across lanes). WHILE/IF entries: per-lane frame stack over the lane's own
 * stream; cond scalars read atomically (poc/01 rule); loop/branch decisions
 * are uniform, so barrier sequences stay uniform.
 *
 * Loader pre-patches: task dst/a/b (+ select p3, iota/gather aux stay word
 * offsets) buffer ids -> f32 element offsets; WHILE/IF entry cond
 * (signal_flag field) likewise.
 */

#define EW_TS 16384u
#define MMA_T 16

enum { TOP_EW = 0, TOP_MMA = 1, TOP_GATHER = 2, TOP_RED_PART = 3,
       TOP_RED_COMB = 4, TOP_IOTA_DIM = 5 };
enum { SUB_ADD = 0, SUB_MUL, SUB_SUB, SUB_DIV, SUB_MAX, SUB_MIN, SUB_POW,
       SUB_COPY, SUB_NEG, SUB_EXP, SUB_LOG, SUB_SQRT, SUB_RSQRT, SUB_TANH,
       SUB_ABS, SUB_FLOOR, SUB_CEIL, SUB_SIGN, SUB_FILL, SUB_IOTA_FLAT,
       SUB_CMP, SUB_SELECT, SUB_LTS };

#define ENT_NOP     0xFFFFFFFFu
#define ENT_BARRIER 0xFFFFFFFEu
#define ENT_WHILE   0xFFFFFFFDu
#define ENT_IF      0xFFFFFFFCu
#define FLAG_NONE   0xFFFFFFFFu

typedef struct {
    uint tile_op, dst, a, b, p0, p1, p2, p3;
} task_t;

typedef struct {
    uint task, tile_lo, tile_hi, wait_flag, wait_count, signal_flag,
         slots, pad;
} entry_t;

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

static float ew_bin(const uint sub, const float x, const float y)
{
    switch (sub) {
    case SUB_ADD: return x + y;
    case SUB_MUL: return x * y;
    case SUB_SUB: return x - y;
    case SUB_DIV: return x / y;
    case SUB_MAX: return fmax(x, y);
    case SUB_MIN: return fmin(x, y);
    case SUB_POW: return pow(x, y);
    default:      return 0.0f;
    }
}

static float ew_un(const uint sub, const float x)
{
    switch (sub) {
    case SUB_COPY:  return x;
    case SUB_NEG:   return -x;
    case SUB_EXP:   return exp(x);
    case SUB_LOG:   return log(x);
    case SUB_SQRT:  return sqrt(x);
    case SUB_RSQRT: return rsqrt(x);
    case SUB_TANH:  return tanh(x);
    case SUB_ABS:   return fabs(x);
    case SUB_FLOOR: return floor(x);
    case SUB_CEIL:  return ceil(x);
    case SUB_SIGN:  return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : x);
    default:        return 0.0f;
    }
}

/* Register-blocked SGEMM tile (from poc/06 step 2 — portable champion family).
 * One 256-thread workgroup computes one MMA_TM x MMA_TN output tile; each
 * thread owns an RM x RN = 4x4 register microtile. Scalar edge-guarded staging
 * (VECW=1, single-buffered) — portable to PoCL. Local: BK*(TM+TN) floats. The
 * scheduler tiles matmul in MMA_TM x MMA_TN blocks (scheduler.MMA_T == MMA_TM).
 * Register footprint ~ RM*RN accumulators (16) + operands — chosen to bound the
 * shared megakernel's occupancy tax (docs/tile-isa.md ceiling-1). */
#define MMA_TM 64
#define MMA_TN 64
#define MMA_BK 16
#define MMA_TDIM 16          /* 16x16 thread grid == 256 threads */
#define MMA_RM (MMA_TM / MMA_TDIM)   /* 4 */
#define MMA_RN (MMA_TN / MMA_TDIM)   /* 4 */
#define MMA_ASZ (MMA_BK * MMA_TM)    /* As[m*BK + k] */
#define MMA_BSZ (MMA_BK * MMA_TN)    /* Bs[k*TN + n] */

static void mma_tile(__global float *arena, const task_t t, uint tile,
                     __local float *As, __local float *Bs)
{
    const uint M = t.p0, N = t.p1, K = t.p2;
    const uint tiles_n = (N + MMA_TN - 1) / MMA_TN;
    const uint tr = tile / tiles_n, tc = tile % tiles_n;
    const uint row0 = tr * MMA_TM, col0 = tc * MMA_TN;
    const uint lid = get_local_id(0);
    const uint ty = lid / MMA_TDIM, tx = lid % MMA_TDIM;

    float acc[MMA_RM][MMA_RN];
    for (int i = 0; i < MMA_RM; i++)
        for (int j = 0; j < MMA_RN; j++) acc[i][j] = 0.0f;

    for (uint k0 = 0; k0 < K; k0 += MMA_BK) {
        /* stage a BK-deep panel of A (un-transposed) and B, edge-guarded */
        for (uint idx = lid; idx < MMA_TM * MMA_BK; idx += 256) {
            const uint m = idx / MMA_BK, kk = idx % MMA_BK;
            const uint gr = row0 + m, gk = k0 + kk;
            As[m * MMA_BK + kk] =
                (gr < M && gk < K) ? arena[t.a + gr * K + gk] : 0.0f;
        }
        for (uint idx = lid; idx < MMA_BK * MMA_TN; idx += 256) {
            const uint kk = idx / MMA_TN, n = idx % MMA_TN;
            const uint gk = k0 + kk, gc = col0 + n;
            Bs[kk * MMA_TN + n] =
                (gk < K && gc < N) ? arena[t.b + gk * N + gc] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint kk = 0; kk < MMA_BK; ++kk) {
            float a[MMA_RM], b[MMA_RN];
            for (int i = 0; i < MMA_RM; i++)
                a[i] = As[(ty * MMA_RM + i) * MMA_BK + kk];
            for (int j = 0; j < MMA_RN; j++)
                b[j] = Bs[kk * MMA_TN + tx * MMA_RN + j];
            for (int i = 0; i < MMA_RM; i++)
                for (int j = 0; j < MMA_RN; j++)
                    acc[i][j] += a[i] * b[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < MMA_RM; i++) {
        const uint gr = row0 + ty * MMA_RM + i;
        if (gr >= M) continue;
        for (int j = 0; j < MMA_RN; j++) {
            const uint gc = col0 + tx * MMA_RN + j;
            if (gc < N) arena[t.dst + gr * N + gc] = acc[i][j];
        }
    }
}

static void exec_tiles(__global float *arena, __global const int *aux,
                       const task_t t, uint tile_lo, uint tile_hi,
                       __local float *As, __local float *Bs)
{
    const uint lid = get_local_id(0);
    const uint lsz = get_local_size(0);

    for (uint tile = tile_lo; tile < tile_hi; ++tile) {
        switch (t.tile_op) {
        case TOP_EW: {
            const uint sub = t.p0, n = t.p1;
            const uint lo = tile * EW_TS;
            const uint hi = min(lo + EW_TS, n);
            if (sub <= SUB_POW) {
                for (uint i = lo + lid; i < hi; i += lsz)
                    arena[t.dst + i] = ew_bin(sub, arena[t.a + i], arena[t.b + i]);
            } else if (sub <= SUB_SIGN) {
                for (uint i = lo + lid; i < hi; i += lsz)
                    arena[t.dst + i] = ew_un(sub, arena[t.a + i]);
            } else if (sub == SUB_FILL) {
                for (uint i = lo + lid; i < hi; i += lsz)
                    arena[t.dst + i] = as_float(t.p2);
            } else if (sub == SUB_IOTA_FLAT) {
                for (uint i = lo + lid; i < hi; i += lsz)
                    arena[t.dst + i] = (float)i;
            } else if (sub == SUB_CMP) {
                for (uint i = lo + lid; i < hi; i += lsz) {
                    const float x = arena[t.a + i], y = arena[t.b + i];
                    int r;
                    switch (t.p2) {
                    case 0:  r = x == y; break;
                    case 1:  r = x != y; break;
                    case 2:  r = x < y;  break;
                    case 3:  r = x <= y; break;
                    case 4:  r = x > y;  break;
                    default: r = x >= y; break;
                    }
                    arena[t.dst + i] = r ? 1.0f : 0.0f;
                }
            } else if (sub == SUB_SELECT) {
                for (uint i = lo + lid; i < hi; i += lsz)
                    arena[t.dst + i] = arena[t.p3 + i] != 0.0f ? arena[t.a + i]
                                                               : arena[t.b + i];
            } else if (sub == SUB_LTS) {
                if (lid == 0 && lo == 0)
                    arena[t.dst] = (arena[t.a] < arena[t.b]) ? 1.0f : 0.0f;
            }
            break;
        }
        case TOP_MMA:
            mma_tile(arena, t, tile, As, Bs);
            break;
        case TOP_GATHER: {
            __global const int *x = aux + t.p0;
            const int rank = x[0];
            __global const int *dims = x + 1, *strides = x + 1 + rank;
            const int src_off = x[1 + 2 * rank];
            const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
            for (uint i = lo + lid; i < hi; i += lsz) {
                int rem = (int)i, off = src_off;
                for (int d = rank - 1; d >= 0; --d) {
                    off += (rem % dims[d]) * strides[d];
                    rem /= dims[d];
                }
                arena[t.dst + i] = arena[t.a + off];
            }
            break;
        }
        case TOP_RED_PART: {
            const uint n = t.p0, chunk = t.p1, kind = t.p2;
            const uint lo = tile * chunk, hi = min(lo + chunk, n);
            float acc = kind == 0 ? 0.0f
                      : kind == 1 ? -INFINITY
                      : kind == 2 ? INFINITY : 1.0f;
            for (uint i = lo + lid; i < hi; i += lsz) {
                const float v = arena[t.a + i];
                acc = kind == 0 ? acc + v
                    : kind == 1 ? fmax(acc, v)
                    : kind == 2 ? fmin(acc, v) : acc * v;
            }
            As[lid] = acc;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint s = lsz / 2; s > 0; s >>= 1) {
                if (lid < s) {
                    const float a = As[lid], b = As[lid + s];
                    As[lid] = kind == 0 ? a + b
                            : kind == 1 ? fmax(a, b)
                            : kind == 2 ? fmin(a, b) : a * b;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) arena[t.dst + tile] = As[0];
            break;
        }
        case TOP_RED_COMB: {
            if (lid == 0) {
                const uint n = t.p0, kind = t.p1;
                float acc = kind == 0 ? 0.0f
                          : kind == 1 ? -INFINITY
                          : kind == 2 ? INFINITY : 1.0f;
                for (uint i = 0; i < n; ++i) {
                    const float v = arena[t.a + i];
                    acc = kind == 0 ? acc + v
                        : kind == 1 ? fmax(acc, v)
                        : kind == 2 ? fmin(acc, v) : acc * v;
                }
                arena[t.dst] = acc;
            }
            break;
        }
        case TOP_IOTA_DIM: {
            __global const int *x = aux + t.p0;
            const int rank = x[0];
            __global const int *dims = x + 1;
            const int dim = x[1 + rank];
            const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
            for (uint i = lo + lid; i < hi; i += lsz) {
                int rem = (int)i, val = 0;
                for (int d = rank - 1; d >= 0; --d) {
                    const int idx = rem % dims[d];
                    rem /= dims[d];
                    if (d == dim) val = idx;
                }
                arena[t.dst + i] = (float)val;
            }
            break;
        }
        default:
            break;
        }
    }
}

/* Per-lane interpreter with a frame stack over the lane's OWN stream. */
#define MAX_DEPTH 8
#define WIDX_ROOT 0xFFFFFFFFu
typedef struct { uint pc, end, widx, phase; } frame_t; /* phase: 0 cond, 1 body, 2 if */

__kernel void vm2(__global float *arena,
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
                    (volatile __global uint *)arena + w.signal_flag, 0u);
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
                (volatile __global uint *)arena + en.signal_flag, 0u);
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
