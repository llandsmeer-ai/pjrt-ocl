/* pjrt-ocl VLIW engine — shared header (concatenated first).
 *
 * The kernel program is assembled by concatenating, in order (see
 * CMakeLists.txt VM_CL_SOURCES):
 *   vm_common.cl  (this file: defines, structs, helpers, barrier)
 *   ops/ew.cl ops/gather.cl ops/reduce.cl ops/mma.cl ops/iota.cl
 *   vm_main.cl    (vmo_exec_tiles dispatch + the vm2 interpreter kernel)
 * One translation unit, functions inlined — file-level modularity for parallel
 * op work (mirrors python/pjrt_ocl/ops/), no clLinkProgram needed.
 *
 * Arena is BYTE-addressed: `__global uchar *arena`, and the loader patches task
 * dst/a/b (+ select p3) and WHILE/IF cond to BYTE offsets. A tile op reaches
 * element `i` of a buffer at byte base `base` via AP(T, base)[i] (base is
 * 64B-aligned, so any T is naturally aligned). This lets one arena hold mixed
 * dtypes; ops dispatch on a per-task dtype (packed in tile_op's high byte).
 */

/* Enable fp64 where the device supports it (feature-detected at init; the
 * runtime only builds this program on such devices, and only f64 programs use
 * it). Harmless #pragma on devices that expose the extension. */
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

/* EW tile size (elements). Overridden per device via -DEW_TS at program build
 * (runtime.cc chooses it; the python scheduler reads the same value from
 * PJRT_OCL_EW_TS so host tiling and kernel tile->range mapping agree). GPUs
 * use a smaller tile: a 16K tile is one workgroup's serial latency chain, so
 * ops < ~lanes*EW_TS elements leave most lanes idle and each lane
 * latency-bound (measured flat 15 us/op for any N in 16K..2M on Blackwell). */
#ifndef EW_TS
#define EW_TS 16384u
#endif

/* Buffer addressing. A buffer's 32-bit `base` is EITHER an arena byte offset
 * (intermediates, consts) OR — with bit 31 set — an I/O PORT: the low bits index
 * `iop[]`, a small array of input/output buffers passed straight to the kernel
 * so the VM reads inputs and writes outputs in place, with no arena copy (the
 * copies dominated memory-bound ops — profiled ~70% of `a+b` time). Every tile
 * fn takes `arena` and `iop` in scope, so VMO_BASE resolves either kind. */
#define VMO_IO_BIT 0x80000000u
#define VMO_BASE(base) \
    (((base) & VMO_IO_BIT) ? iop[(base) & 0x7Fu] : (arena + (base)))
#define AP(T, base) ((__global T *)VMO_BASE(base))
#define VMO_N_IO 8   /* # of I/O buffers passed direct to the kernel as ports */
/* The kernel entry points take VMO_N_IO buffer args and pack them into `iop`;
 * unused ports get a dummy buffer from the host. */
#define VMO_IO_PARAMS                                                    \
    __global uchar *io0, __global uchar *io1, __global uchar *io2,        \
    __global uchar *io3, __global uchar *io4, __global uchar *io5,        \
    __global uchar *io6, __global uchar *io7
#define VMO_IO_ARRAY                                                     \
    __global uchar *iop[VMO_N_IO] =                                       \
        {io0, io1, io2, io3, io4, io5, io6, io7}

/* dtype enum (matches python DT_* / runtime.h). */
enum { DT_F32 = 0, DT_I32 = 1, DT_U32 = 2, DT_BOOL = 3,
       DT_I64 = 4, DT_F64 = 5, DT_F16 = 6, DT_BF16 = 7 };

/* f16 and bf16 are 2-byte storage + f32 compute (portable, no cl_khr_fp16):
 * f16 via core vload_half/vstore_half; bf16 via bit shift (top 16 bits of the
 * f32) with round-to-nearest-even. */
#define LDH(base, i) vload_half((i), (const __global half *)VMO_BASE(base))
#define STH(base, i, v) vstore_half((v), (i), (__global half *)VMO_BASE(base))
static float vmo_bf16_to_f32(ushort b) { return as_float(((uint)b) << 16); }
static ushort vmo_f32_to_bf16(float f)
{
    uint u = as_uint(f);
    return (ushort)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);  /* round-nearest-even */
}
#define LDB(base, i) vmo_bf16_to_f32(AP(const ushort, (base))[i])
#define STB(base, i, v) (AP(ushort, (base))[i] = vmo_f32_to_bf16(v))

enum { TOP_EW = 0, TOP_MMA = 1, TOP_GATHER = 2, TOP_RED_PART = 3,
       TOP_RED_COMB = 4, TOP_IOTA_DIM = 5, TOP_SCATTER = 6,
       TOP_DYN_GATHER = 7, TOP_DYN_SCATTER = 8, TOP_RED_WINDOW = 9,
       TOP_RED_SEG = 10,
       /* Fused segmented norms (§19): one segment per tile, whole workgroup
        * collaborates in local memory (one global read + one write).
        * SOFTMAX (p0=n_out, p1=seg); LAYERNORM core (p0=n_out, p1=seg,
        * p2=as_float eps). Kernels: vmo_softmax_seg / vmo_layernorm_seg. */
       TOP_SOFTMAX_SEG = 11, TOP_LAYERNORM_SEG = 12,
       /* §27/§28 register-resident fused map-region: a run of pure-map EW
        * micro-ops interpreted over per-thread float4 slots (one global load per
        * input, one store), collapsing a K-op EW chain into one barrier-free
        * phase. p0=aux descriptor word-offset, p1=n. Kernel: vmo_map_region
        * (ops/region.cl); recognizer: lowering _fuse_region. */
       TOP_MAP_REGION = 13,
       /* §34 fused flash-attention: one workgroup per (head,query) streams the
        * KV cache with online softmax — QKᵀ→scale→softmax→AV in ONE phase, no
        * materialized score matrix. a=Q b=K p0=V dst=out; p1=H p2=T; p3=aux
        * descriptor [H,T,C,hd,scale,causal,qv,kv,vv]. Kernel: vmo_flash_attn
        * (ops/attention.cl); recognizer: lowering _fuse_attention. */
       TOP_FLASH_ATTN = 14,
       /* §38 general data-dependent gather (stablehlo.gather): each output
        * element reads its operand base offset from a runtime start_indices
        * buffer. aux header [out_rank, nidx, si_vec_stride, is64, idx_byteoff,
        * idx_bufid] then out_dims/op_stride/si_stride[out_rank],
        * idx_op_stride/clamp_max[nidx]. Kernel: vmo_gather_index_tile
        * (ops/gather.cl); handler: ops/gather_index.py. */
       TOP_GATHER_INDEX = 15 };
enum { SUB_ADD = 0, SUB_MUL, SUB_SUB, SUB_DIV, SUB_MAX, SUB_MIN, SUB_POW,
       SUB_COPY, SUB_NEG, SUB_EXP, SUB_LOG, SUB_SQRT, SUB_RSQRT, SUB_TANH,
       SUB_ABS, SUB_FLOOR, SUB_CEIL, SUB_SIGN, SUB_FILL, SUB_IOTA_FLAT,
       SUB_CMP, SUB_SELECT, SUB_LTS, SUB_CONVERT, SUB_BITCAST,
       /* new float binary (routed through vmo_ew_bin; vmo_ew_is_bin() range-checks
        * SUB_ATAN2..SUB_REMAINDER) */
       SUB_ATAN2, SUB_REMAINDER,
       /* new float unary (routed through vmo_ew_un; vmo_ew_is_un() range-checks
        * SUB_LOG1P..SUB_ROUND) */
       SUB_LOG1P, SUB_EXPM1, SUB_CBRT, SUB_SIN, SUB_COS, SUB_TAN,
       SUB_RINT /* round_nearest_even */, SUB_ROUND /* round_nearest_afz */,
       /* bitwise int32/bool — dedicated dispatch in vmo_ew_tile_i32/vmo_ew_tile_bool */
       SUB_AND, SUB_OR, SUB_XOR, SUB_NOT,
       /* mixed-dtype: float operand -> bool result (own dispatch in vmo_ew_tile) */
       SUB_ISFINITE,
       /* fused affine: d = a*s + t, s=as_float(p2), t=as_float(p3). Folds a
        * scalar-const scale/bias (and composed chains) into one in-place pass;
        * see python lowering _fold_scalar / _compose_affines. */
       SUB_AFFINE,
       /* fused GELU tanh-approx unary (§19b/§24): computes the whole
        * 0.5*x*(1+tanh(0.7978845608*(x+0.044715*x^3))) per element in registers,
        * one global read + one write. Routed as a unary subop (vmo_ew_is_un()
        * range-extended) so it rides the existing TILE_EW float4 fast path. */
       SUB_GELU };

#define ENT_NOP     0xFFFFFFFFu
#define ENT_BARRIER 0xFFFFFFFEu
#define ENT_WHILE   0xFFFFFFFDu
#define ENT_IF      0xFFFFFFFCu
/* Fixed-trip loop (counted while, e.g. lax.scan/fori_loop): body entry range
 * in tile_lo/tile_hi, trip count in wait_flag. No cond sub-list, no cond-flag
 * read — one barrier per iteration (data deps across iterations) instead of
 * WHILE's two plus the cond compute. */
#define ENT_FOR     0xFFFFFFFBu
#define FLAG_NONE   0xFFFFFFFFu

typedef struct {
    uint tile_op, dst, a, b, p0, p1, p2, p3, p4, p5, p6, p7;
} task_t;   /* p4/p5: MMA operand VIEW aux-offsets (+1; 0 = contiguous).
             * p6/p7 (§33 R2c matmul epilogue): p6 = epilogue descriptor aux
             * word-offset (+1; 0 = no epilogue); p7 = the epilogue's second-input
             * buffer handle (residual/bias), loader-patched to a byte offset. */

/* §33 R2c: shared straight-line map micro-op interpreter over a per-thread
 * value. Used by BOTH ops/region.cl (OP_MAP_REGION, float4-vectorized) and
 * ops/mma.cl (the matmul store-epilogue, scalar per accumulator element). The
 * builtins MUST byte-match ops/ew.cl so a fused region/epilogue is numerically
 * identical to the decomposed EW chain it replaces. `kind` is a SUB_* opcode
 * (the pure-map ALU subset the recognizers emit). Defined here (concatenated
 * first) so it precedes both callers in the single translation unit. */
static float4 vmo_region_micro(const uint kind, const float4 x, const float4 y,
                               const float s, const float t)
{
    switch (kind) {
    case SUB_ADD:    return x + y;
    case SUB_MUL:    return x * y;
    case SUB_SUB:    return x - y;
    case SUB_DIV:    return x / y;
    case SUB_MAX:    return fmax(x, y);
    case SUB_MIN:    return fmin(x, y);
    case SUB_NEG:    return -x;
    case SUB_EXP:    return exp(x);
    case SUB_LOG:    return log(x);
    case SUB_SQRT:   return sqrt(x);
    case SUB_RSQRT:  return rsqrt(x);
    case SUB_TANH:   return tanh(x);
    case SUB_ABS:    return fabs(x);
    case SUB_AFFINE: return mad(x, (float4)(s), (float4)(t));  /* x*s + t */
    /* GELU tanh-approx — byte-identical to ops/ew.cl VMO_GELU_BODY / vmo_gelu4. */
    case SUB_GELU:   return 0.5f * x * (1.0f + tanh(0.7978845608f *
                            (x + 0.044715f * x * x * x)));
    default:         return x;
    }
}

typedef struct {
    uint task, tile_lo, tile_hi, wait_flag, wait_count, signal_flag,
         slots, pad;
} entry_t;

/* Value-level bit-recast (bitcast_convert, NaN-safe integer handling). OpenCL
 * C defines union type-punning; keeps integer bit patterns out of float
 * registers where a GPU might canonicalize a NaN. */
typedef union { float f; int i; uint u; } slot_t;

/* Cross-workgroup barrier: persistent-thread spin over a global arrival
 * counter (bar[0]) + phase flag (bar[1]).
 *
 * MEMORY MODEL (poc/07): a plain `mem_fence(CLK_GLOBAL_MEM_FENCE)` is only
 * work-group-scoped, so non-atomic data a lane writes before the barrier is NOT
 * guaranteed visible to a DIFFERENT lane after it — on NVIDIA that read is
 * ~100% stale from a warm per-SM L1 once a program iterates (measured; this is
 * what forced n_lanes=1 for while). The fix is OpenCL-2.0 DEVICE-SCOPE
 * acquire/release fences: release our data device-wide before signalling
 * arrival, acquire peers' data device-wide after the phase flips. NVIDIA
 * honours memory_scope_device even though clinfo advertises only work-group
 * scope (poc/07 test E), and it's native on PoCL/AMD/Intel. Devices that lack
 * it need the host-dispatch engine (Plan B) — which also solves PoCL liveness.
 * This does NOT fix liveness (co-residency); that is a separate axis.
 *
 * DIALECT: the fence builtins only exist in OpenCL C 2.0+, and strict
 * compilers (Intel) reject them under the 1.2 default that empty clBuildProgram
 * options select. The runtime probes -cl-std variants at init (runtime.cc) and
 * defines VMO_NO_DEVICE_FENCE for the last-resort strict-1.2 build; that build
 * compiles the fences out, so vm2's spin-barrier is UNSAFE there and the
 * runtime forces the host-dispatch engine (vm2_seg never calls vmo_barrier).
 * Feature macros (__opencl_c_atomic_*) can't be used instead: NVIDIA accepts
 * the builtins under -cl-std=CL3.0 without defining the macros (verified). */
#ifdef VMO_NO_DEVICE_FENCE
#define VMO_FENCE_DEV_REL()
#define VMO_FENCE_DEV_ACQ()
/* strict-1.2 fallback: the spin-barrier is unsafe here anyway (host-dispatch is
 * forced), so the phase read only needs to compile — a volatile load suffices. */
#define VMO_LOAD_PHASE(p) (*(volatile __global uint *)(p))
#else
#define VMO_FENCE_DEV_REL() \
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, \
                           memory_scope_device)
#define VMO_FENCE_DEV_ACQ() \
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, \
                           memory_scope_device)
/* Coherent LOAD (not an atomic RMW) of the phase word. The old spin used
 * atomic_add(&bar[1],0) — a read-modify-write — which forces every spinning
 * group to acquire the cache line EXCLUSIVE, so the line ping-pongs among all
 * groups and each spin costs an L2 round-trip under full contention (~38us per
 * barrier across ~hundreds of workgroups; the small-N `while` floor). An
 * acquire LOAD keeps the line in Shared state across all readers — it is only
 * invalidated once, when the last arriver flips the phase — so the spin is
 * near-free until release. Ordering of the payload is still the ACQ fence
 * below; this load is device-scoped so it observes the L2-coherent flip. */
#define VMO_LOAD_PHASE(p) atomic_load_explicit( \
    (volatile __global atomic_uint *)(p), memory_order_relaxed, \
    memory_scope_device)
#endif

/* §29 investigation: per-phase device timestamps. Each lane (workgroup) records
 * the GPU-global nanosecond clock (%globaltimer — one counter shared across all
 * SMs, so arrival times ARE comparable across workgroups, unlike per-SM clock64)
 * into stats[barrier_i*nlanes+lane] at every barrier arrival. Host reads it back
 * (PJRT_OCL_PHASE_TS) → per-phase wall time (max-arrival delta) + idle-at-barrier
 * skew (max-min arrival). Low 32 bits of ns: wraps at ~4.29 s, never within a
 * phase. Only the VMO_NV_PTX build has globaltimer; portable build records 0. */
#ifdef VMO_PHASE_TS
#ifdef VMO_NV_PTX
static inline uint vmo_now_ns(void) {
    ulong t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return (uint)t;
}
#else
static inline uint vmo_now_ns(void) { return 0u; }
#endif
#define VMO_TS_REC(stats, bi, lane, nl)                                        \
    do {                                                                       \
        if (get_local_id(0) == 0u && (bi) < 4096u)                             \
            (stats)[(bi) * (nl) + (lane)] = vmo_now_ns();                       \
    } while (0)
#else
#define VMO_TS_REC(stats, bi, lane, nl) do {} while (0)
#endif

static void vmo_barrier(volatile __global uint *bar, const uint ngroups)
{
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
        VMO_FENCE_DEV_REL();
        const uint phase = VMO_LOAD_PHASE(&bar[1]);
        if (atomic_inc(&bar[0]) == ngroups - 1) {
            bar[0] = 0;
            atomic_inc(&bar[1]);
        } else {
            while (VMO_LOAD_PHASE(&bar[1]) == phase)
                ;
        }
        VMO_FENCE_DEV_ACQ();
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/* Occupancy DISCOVERY (poc/08; Sorensen & Donaldson, OOPSLA 2016): count the
 * workgroups that are SIMULTANEOUSLY resident, deadlock-free no matter how
 * oversized the launch. d[0]=lock, d[1]=gate (init 1=open), d[2]=count.
 * A leader takes a ticket while the gate is open; ticket holders spin until
 * the gate closes (holding their residency slot — without this the scheduler
 * backfills exited groups and the count inflates); ticketless groups exit
 * immediately and never wait on anyone. Ticket 0 closes the gate once the
 * count has been stable for a window. Uses only OpenCL 1.2 atomics on a
 * single buffer — safe even on the strict-1.2 VMO_NO_DEVICE_FENCE build. */
static uint vmo_discover(volatile __global uint *d)
{
    uint t = 0xFFFFFFFFu;
    while (atomic_cmpxchg(&d[0], 0u, 1u) != 0u)
        ;
    if (atomic_add(&d[1], 0u) == 1u)
        t = atomic_inc(&d[2]);
    atomic_xchg(&d[0], 0u);
    if (t == 0u) {
        uint last = 1u, stable = 0u;
        for (uint i = 0u; i < 50000000u && stable < 100000u; ++i) {
            const uint c = atomic_add(&d[2], 0u);
            if (c == last) stable++; else { stable = 0u; last = c; }
        }
        while (atomic_cmpxchg(&d[0], 0u, 1u) != 0u)
            ;
        atomic_xchg(&d[1], 0u);
        atomic_xchg(&d[0], 0u);
    } else if (t != 0xFFFFFFFFu) {
        while (atomic_add(&d[1], 0u) == 1u)
            ;
    }
    return t;
}
