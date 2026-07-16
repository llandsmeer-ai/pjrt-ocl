// Generated from kernels/*.cl — do not edit.
static const char kVmClSource[] = R"CLSRC(
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

#define EW_TS 16384u

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
       TOP_DYN_GATHER = 7, TOP_DYN_SCATTER = 8, TOP_RED_WINDOW = 9 };
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
       SUB_ISFINITE };

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
#else
#define VMO_FENCE_DEV_REL() \
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, \
                           memory_scope_device)
#define VMO_FENCE_DEV_ACQ() \
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, \
                           memory_scope_device)
#endif

static void vmo_barrier(volatile __global uint *bar, const uint ngroups)
{
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
        VMO_FENCE_DEV_REL();
        const uint phase = atomic_add(&bar[1], 0);
        if (atomic_inc(&bar[0]) == ngroups - 1) {
            bar[0] = 0;
            atomic_inc(&bar[1]);
        } else {
            while (atomic_add(&bar[1], 0) == phase)
                ;
        }
        VMO_FENCE_DEV_ACQ();
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/* Elementwise tile op (TOP_EW). subop in task.p0; n in p1; cmp pred / fill bits
 * in p2; select pred BYTE offset in p3. dst/a/b are BYTE offsets.
 * Dispatched by vmo_exec_tiles on result dtype `dt` and operand dtype `adt`
 * (compare: operands adt -> bool result; select: bool pred -> operands dt).
 * bool is 1-byte (uchar 0/1), matching jax PRED. */

static float vmo_ew_bin(const uint sub, const float x, const float y)
{
    switch (sub) {
    case SUB_ADD: return x + y;   case SUB_MUL: return x * y;
    case SUB_SUB: return x - y;   case SUB_DIV: return x / y;
    case SUB_MAX: return fmax(x, y); case SUB_MIN: return fmin(x, y);
    case SUB_POW: return pow(x, y);
    case SUB_ATAN2: return atan2(x, y);
    case SUB_REMAINDER: return fmod(x, y);   /* C fmod: sign of dividend x */
    default: return 0.0f;
    }
}
static float vmo_ew_un(const uint sub, const float x)
{
    switch (sub) {
    case SUB_COPY: return x;      case SUB_NEG: return -x;
    case SUB_EXP: return exp(x);  case SUB_LOG: return log(x);
    case SUB_SQRT: return sqrt(x); case SUB_RSQRT: return rsqrt(x);
    case SUB_TANH: return tanh(x); case SUB_ABS: return fabs(x);
    case SUB_FLOOR: return floor(x); case SUB_CEIL: return ceil(x);
    case SUB_SIGN: return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : x);
    case SUB_LOG1P: return log1p(x);
    case SUB_EXPM1: return expm1(x);
    case SUB_CBRT: return cbrt(x);
    case SUB_SIN: return sin(x);
    case SUB_COS: return cos(x);
    case SUB_TAN: return tan(x);
    case SUB_RINT: return rint(x);    /* round to nearest, ties to even */
    case SUB_ROUND: return round(x);  /* round to nearest, ties away from 0 */
    default: return 0.0f;
    }
}
/* Binary/unary EW dispatch predicates. New subops keep ADD..POW / COPY..SIGN
 * contiguous with their new siblings (ATAN2..REMAINDER / LOG1P..ROUND) so
 * this stays a pair of range checks — see the vm_common.cl enum comment. */
static int vmo_ew_is_bin(const uint sub)
{
    return sub <= SUB_POW || (sub >= SUB_ATAN2 && sub <= SUB_REMAINDER);
}
static int vmo_ew_is_un(const uint sub)
{
    return (sub >= SUB_COPY && sub <= SUB_SIGN) ||
           (sub >= SUB_LOG1P && sub <= SUB_ROUND);
}
#define CMP(p, x, y) ((p)==0?(x)==(y):(p)==1?(x)!=(y):(p)==2?(x)<(y): \
                      (p)==3?(x)<=(y):(p)==4?(x)>(y):(x)>=(y))

static float vmo_load_f(__global uchar *arena, __global uchar **iop, uint base, uint dt, uint i);  /* fwd */

/* compare: operands read as adt, result written as 1-byte bool (uchar 0/1). */
static void vmo_cmp_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                     uint adt, uint lid, uint lsz)
{
    const uint n = t.p1, p = t.p2;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global uchar *d = AP(uchar, t.dst);
    if (adt == DT_I32 || adt == DT_U32 || adt == DT_BOOL) {
        __global const int *a = AP(const int, t.a), *b = AP(const int, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = CMP(p, a[i], b[i]) ? 1 : 0;
#ifdef cl_khr_fp64
    } else if (adt == DT_F64) {
        __global const double *a = AP(const double, t.a), *b = AP(const double, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = CMP(p, a[i], b[i]) ? 1 : 0;
#endif
    } else if (adt == DT_F16 || adt == DT_BF16) {
        for (uint i = lo + lid; i < hi; i += lsz) {
            float x = vmo_load_f(arena, iop, t.a, adt, i), y = vmo_load_f(arena, iop, t.b, adt, i);
            d[i] = CMP(p, x, y) ? 1 : 0;
        }
    } else {
        __global const float *a = AP(const float, t.a), *b = AP(const float, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = CMP(p, a[i], b[i]) ? 1 : 0;
    }
}

/* select: pred (p3) is 1-byte bool; a/b/dst read/written as dt. */
static void vmo_select_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                        uint dt, uint lid, uint lsz)
{
    const uint n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global const uchar *p = AP(const uchar, t.p3);
    if (dt == DT_I32 || dt == DT_U32) {
        __global int *d = AP(int, t.dst);
        __global const int *a = AP(const int, t.a), *b = AP(const int, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = p[i] ? a[i] : b[i];
    } else if (dt == DT_BOOL) {
        __global uchar *d = AP(uchar, t.dst);
        __global const uchar *a = AP(const uchar, t.a), *b = AP(const uchar, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = p[i] ? a[i] : b[i];
    } else if (dt == DT_F16 || dt == DT_BF16) {
        /* 2-byte select = copy the chosen 2-byte element (no arithmetic) */
        __global ushort *d = AP(ushort, t.dst);
        __global const ushort *a = AP(const ushort, t.a), *b = AP(const ushort, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = p[i] ? a[i] : b[i];
#ifdef cl_khr_fp64
    } else if (dt == DT_F64) {
        __global double *d = AP(double, t.dst);
        __global const double *a = AP(const double, t.a), *b = AP(const double, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = p[i] ? a[i] : b[i];
#endif
    } else {
        __global float *d = AP(float, t.dst);
        __global const float *a = AP(const float, t.a), *b = AP(const float, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = p[i] ? a[i] : b[i];
    }
}

static void vmo_ew_tile_f32(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                        uint lid, uint lsz)
{
    const uint sub = t.p0, n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global float *d = AP(float, t.dst);
    __global const float *a = AP(const float, t.a), *b = AP(const float, t.b);
    if (vmo_ew_is_bin(sub))
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = vmo_ew_bin(sub, a[i], b[i]);
    else if (vmo_ew_is_un(sub))
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = vmo_ew_un(sub, a[i]);
    else if (sub == SUB_FILL)
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = as_float(t.p2);
    else if (sub == SUB_IOTA_FLAT)
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = (float)i;
    else if (sub == SUB_LTS) { if (lid == 0 && lo == 0) d[0] = (a[0] < b[0]) ? 1.0f : 0.0f; }
}

#ifdef cl_khr_fp64
static void vmo_ew_tile_f64(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                        uint lid, uint lsz)
{
    const uint sub = t.p0, n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global double *d = AP(double, t.dst);
    __global const double *a = AP(const double, t.a), *b = AP(const double, t.b);
    for (uint i = lo + lid; i < hi; i += lsz) {
        const double x = a[i], y = vmo_ew_is_bin(sub) ? b[i] : 0.0;
        double r;
        switch (sub) {
        case SUB_ADD: r = x + y; break;   case SUB_MUL: r = x * y; break;
        case SUB_SUB: r = x - y; break;   case SUB_DIV: r = x / y; break;
        case SUB_MAX: r = fmax(x, y); break; case SUB_MIN: r = fmin(x, y); break;
        case SUB_POW: r = pow(x, y); break;  case SUB_COPY: r = x; break;
        case SUB_NEG: r = -x; break;      case SUB_EXP: r = exp(x); break;
        case SUB_LOG: r = log(x); break;  case SUB_SQRT: r = sqrt(x); break;
        case SUB_RSQRT: r = rsqrt(x); break; case SUB_TANH: r = tanh(x); break;
        case SUB_ABS: r = fabs(x); break; case SUB_FLOOR: r = floor(x); break;
        case SUB_CEIL: r = ceil(x); break;
        case SUB_SIGN: r = x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : x); break;
        case SUB_ATAN2: r = atan2(x, y); break;
        case SUB_REMAINDER: r = fmod(x, y); break;
        case SUB_LOG1P: r = log1p(x); break;
        case SUB_EXPM1: r = expm1(x); break;
        case SUB_CBRT: r = cbrt(x); break;
        case SUB_SIN: r = sin(x); break;
        case SUB_COS: r = cos(x); break;
        case SUB_TAN: r = tan(x); break;
        case SUB_RINT: r = rint(x); break;
        case SUB_ROUND: r = round(x); break;
        default: r = x; break;
        }
        d[i] = r;
    }
}
#endif

static void vmo_ew_tile_i32(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                        uint lid, uint lsz)
{
    const uint sub = t.p0, n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global int *d = AP(int, t.dst);
    __global const int *a = AP(const int, t.a), *b = AP(const int, t.b);
    if (sub == SUB_FILL) { for (uint i = lo + lid; i < hi; i += lsz) d[i] = (int)t.p2; return; }
    if (sub == SUB_IOTA_FLAT) { for (uint i = lo + lid; i < hi; i += lsz) d[i] = (int)i; return; }
    const int needs_y = (sub <= SUB_POW) || sub == SUB_AND || sub == SUB_OR || sub == SUB_XOR;
    for (uint i = lo + lid; i < hi; i += lsz) {
        const int x = a[i], y = needs_y ? b[i] : 0;
        int r;
        switch (sub) {
        case SUB_ADD: r = x + y; break;   case SUB_MUL: r = x * y; break;
        case SUB_SUB: r = x - y; break;   case SUB_DIV: r = y ? x / y : 0; break;
        case SUB_MAX: r = max(x, y); break; case SUB_MIN: r = min(x, y); break;
        case SUB_COPY: r = x; break;      case SUB_NEG: r = -x; break;
        case SUB_ABS: r = abs(x); break;
        case SUB_SIGN: r = x > 0 ? 1 : (x < 0 ? -1 : 0); break;
        case SUB_AND: r = x & y; break;   case SUB_OR: r = x | y; break;
        case SUB_XOR: r = x ^ y; break;   case SUB_NOT: r = ~x; break;
        default: r = x; break;
        }
        d[i] = r;
    }
}

/* bool (1-byte) elementwise: copy/fill/iota + and/or/xor/not (jax logical_*
 * on bool -> stablehlo.and/or/xor/not). Bool is stored as uchar 0/1, so
 * AND/OR/XOR are plain bitwise ops on the byte (only bit 0 is ever set), but
 * NOT must flip 0<->1 rather than bitwise-complement the whole byte. */
static void vmo_ew_tile_bool(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                         uint lid, uint lsz)
{
    const uint sub = t.p0, n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global uchar *d = AP(uchar, t.dst);
    __global const uchar *a = AP(const uchar, t.a);
    if (sub == SUB_FILL) {
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = (uchar)(t.p2 != 0);
    } else if (sub == SUB_NOT) {
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = (uchar)(a[i] == 0);
    } else if (sub == SUB_AND || sub == SUB_OR || sub == SUB_XOR) {
        __global const uchar *b = AP(const uchar, t.b);
        for (uint i = lo + lid; i < hi; i += lsz) {
            const uchar x = a[i] != 0, y = b[i] != 0;
            d[i] = (sub == SUB_AND) ? (uchar)(x & y)
                 : (sub == SUB_OR)  ? (uchar)(x | y) : (uchar)(x ^ y);
        }
    } else {
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = a[i];  /* copy */
    }
}

/* f16/bf16: 2-byte storage, f32 arithmetic. LOAD/STORE are the half or bf16
 * accessors; only the floating subops apply (jax never emits int ops on them).*/
#define EW_HALF_TILE(NAME, LOAD, STORE)                                        \
static void NAME(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,             \
                 uint lid, uint lsz) {                                         \
    const uint sub = t.p0, n = t.p1;                                           \
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);                     \
    if (vmo_ew_is_bin(sub))                                                        \
        for (uint i = lo + lid; i < hi; i += lsz)                             \
            STORE(t.dst, i, vmo_ew_bin(sub, LOAD(t.a, i), LOAD(t.b, i)));          \
    else if (vmo_ew_is_un(sub))                                                    \
        for (uint i = lo + lid; i < hi; i += lsz)                             \
            STORE(t.dst, i, vmo_ew_un(sub, LOAD(t.a, i)));                         \
    else if (sub == SUB_FILL)                                                  \
        for (uint i = lo + lid; i < hi; i += lsz) STORE(t.dst, i, as_float(t.p2)); \
    else if (sub == SUB_IOTA_FLAT)                                             \
        for (uint i = lo + lid; i < hi; i += lsz) STORE(t.dst, i, (float)i);   \
}
EW_HALF_TILE(vmo_ew_tile_f16, LDH, STH)
EW_HALF_TILE(vmo_ew_tile_bf16, LDB, STB)

/* read element i of buffer `base` as a float, for any float-domain dtype. */
static float vmo_load_f(__global uchar *arena, __global uchar **iop, uint base, uint dt, uint i)
{
    switch (dt) {
    case DT_F16:  return LDH(base, i);
    case DT_BF16: return LDB(base, i);
    case DT_I32: case DT_U32: return (float)AP(const int, base)[i];
    case DT_BOOL: return (float)AP(const uchar, base)[i];
#ifdef cl_khr_fp64
    case DT_F64:  return (float)AP(const double, base)[i];
#endif
    default:      return AP(const float, base)[i];
    }
}

/* convert: read a[i] as adt, write dst[i] as dt with a C cast (float->int
 * truncates toward zero, matching stablehlo). Via a double intermediate where
 * fp64 exists (exact to 2^53); a float intermediate otherwise (4-byte types).
 * i64 beyond 2^53 loses precision — acceptable for now. */
static void vmo_convert_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                         uint dt, uint adt, uint lid, uint lsz)
{
    const uint n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    for (uint i = lo + lid; i < hi; i += lsz) {
#ifdef cl_khr_fp64
        double v;
        switch (adt) {
        case DT_I32: case DT_U32: v = (double)AP(const int, t.a)[i]; break;
        case DT_BOOL: v = (double)AP(const uchar, t.a)[i]; break;
        case DT_I64: v = (double)AP(const long, t.a)[i]; break;
        case DT_F64: v = AP(const double, t.a)[i]; break;
        case DT_F16: v = (double)LDH(t.a, i); break;
        case DT_BF16: v = (double)LDB(t.a, i); break;
        default: v = (double)AP(const float, t.a)[i]; break;
        }
        switch (dt) {
        case DT_I32: case DT_U32: AP(int, t.dst)[i] = (int)v; break;
        case DT_BOOL: AP(uchar, t.dst)[i] = (uchar)(v != 0.0); break;
        case DT_I64: AP(long, t.dst)[i] = (long)v; break;
        case DT_F64: AP(double, t.dst)[i] = v; break;
        case DT_F16: STH(t.dst, i, (float)v); break;
        case DT_BF16: STB(t.dst, i, (float)v); break;
        default: AP(float, t.dst)[i] = (float)v; break;
        }
#else
        float v = vmo_load_f(arena, iop, t.a, adt, i);
        switch (dt) {
        case DT_I32: case DT_U32: AP(int, t.dst)[i] = (int)v; break;
        case DT_BOOL: AP(uchar, t.dst)[i] = (uchar)(v != 0.0f); break;
        case DT_F16: STH(t.dst, i, v); break;
        case DT_BF16: STB(t.dst, i, v); break;
        default: AP(float, t.dst)[i] = v; break;
        }
#endif
    }
}

/* bitcast_convert: reinterpret the BITS of each element as the result dtype
 * (same element size). A typed memcpy of the 2/4/8-byte word — NOT a numeric
 * conversion (f32<->i32<->u32; f64<->i64). Width comes from the result dtype;
 * source and dest share the byte width by construction (checked in lowering). */
static void vmo_bitcast_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                         uint dt, uint lid, uint lsz)
{
    const uint n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    if (dt == DT_I64 || dt == DT_F64) {
        __global long *d = AP(long, t.dst);
        __global const long *a = AP(const long, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = a[i];
    } else if (dt == DT_F16 || dt == DT_BF16) {
        __global ushort *d = AP(ushort, t.dst);
        __global const ushort *a = AP(const ushort, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = a[i];
    } else {   /* 4-byte: f32/i32/u32 */
        __global int *d = AP(int, t.dst);
        __global const int *a = AP(const int, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = a[i];
    }
}

/* is_finite: operand read as adt (float-domain), result written as 1-byte
 * bool (uchar 0/1) — same mixed-dtype shape as vmo_cmp_tile. */
static void vmo_isfinite_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                          uint adt, uint lid, uint lsz)
{
    const uint n = t.p1;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, n);
    __global uchar *d = AP(uchar, t.dst);
#ifdef cl_khr_fp64
    if (adt == DT_F64) {
        __global const double *a = AP(const double, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) d[i] = isfinite(a[i]) ? 1 : 0;
        return;
    }
#endif
    if (adt == DT_F16 || adt == DT_BF16) {
        for (uint i = lo + lid; i < hi; i += lsz)
            d[i] = isfinite(vmo_load_f(arena, iop, t.a, adt, i)) ? 1 : 0;
        return;
    }
    __global const float *a = AP(const float, t.a);
    for (uint i = lo + lid; i < hi; i += lsz) d[i] = isfinite(a[i]) ? 1 : 0;
}

static void vmo_ew_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                    uint dt, uint adt, uint lid, uint lsz)
{
    if (t.p0 == SUB_CMP) { vmo_cmp_tile(arena, iop, t, tile, adt, lid, lsz); return; }
    if (t.p0 == SUB_SELECT) { vmo_select_tile(arena, iop, t, tile, dt, lid, lsz); return; }
    if (t.p0 == SUB_CONVERT) { vmo_convert_tile(arena, iop, t, tile, dt, adt, lid, lsz); return; }
    if (t.p0 == SUB_BITCAST) { vmo_bitcast_tile(arena, iop, t, tile, dt, lid, lsz); return; }
    if (t.p0 == SUB_ISFINITE) { vmo_isfinite_tile(arena, iop, t, tile, adt, lid, lsz); return; }
    switch (dt) {
    case DT_I32: case DT_U32: vmo_ew_tile_i32(arena, iop, t, tile, lid, lsz); break;
    case DT_BOOL:             vmo_ew_tile_bool(arena, iop, t, tile, lid, lsz); break;
    case DT_F16:              vmo_ew_tile_f16(arena, iop, t, tile, lid, lsz); break;
    case DT_BF16:             vmo_ew_tile_bf16(arena, iop, t, tile, lid, lsz); break;
#ifdef cl_khr_fp64
    case DT_F64:              vmo_ew_tile_f64(arena, iop, t, tile, lid, lsz); break;
#endif
    default:                  vmo_ew_tile_f32(arena, iop, t, tile, lid, lsz); break;
    }
}

/* Strided gather tile op (TOP_GATHER). aux at task.p0:
 *   rank, out_dims[rank], in_strides[rank], src_off  (all i32, elements).
 * Covers broadcast_in_dim / transpose / slice / reverse (via strides+src_off).
 */

/* Gather copies whole elements, so it is dtype-agnostic for a given element
 * size. `esz` (bytes) picks the mover: 4-byte types copy as uint bits; 8-byte
 * as ulong. dst/a are BYTE offsets. */
static void vmo_gather_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                        const task_t t, uint tile, uint esz, uint lid, uint lsz)
{
    __global const int *x = aux + t.p0;
    const int rank = x[0];
    __global const int *dims = x + 1, *strides = x + 1 + rank;
    const int src_off = x[1 + 2 * rank];
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
    if (esz == 8) {
        __global ulong *d = AP(ulong, t.dst);
        __global const ulong *a = AP(const ulong, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = src_off;
            for (int e = rank - 1; e >= 0; --e) {
                off += (rem % dims[e]) * strides[e];
                rem /= dims[e];
            }
            d[i] = a[off];
        }
    } else if (esz == 2) {
        __global ushort *d = AP(ushort, t.dst);
        __global const ushort *a = AP(const ushort, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = src_off;
            for (int e = rank - 1; e >= 0; --e) {
                off += (rem % dims[e]) * strides[e];
                rem /= dims[e];
            }
            d[i] = a[off];
        }
    } else if (esz == 1) {
        __global uchar *d = AP(uchar, t.dst);
        __global const uchar *a = AP(const uchar, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = src_off;
            for (int e = rank - 1; e >= 0; --e) {
                off += (rem % dims[e]) * strides[e];
                rem /= dims[e];
            }
            d[i] = a[off];
        }
    } else {
        __global uint *d = AP(uint, t.dst);
        __global const uint *a = AP(const uint, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = src_off;
            for (int e = rank - 1; e >= 0; --e) {
                off += (rem % dims[e]) * strides[e];
                rem /= dims[e];
            }
            d[i] = a[off];
        }
    }
}

/* Two-phase reduction tile ops.
 * TOP_RED_PART: one partial per tile (task.p0=n, p1=chunk, p2=kind) via a
 *   workgroup local-memory tree reduce; writes arena[t.dst + tile].
 * TOP_RED_COMB: fold n_parts partials -> final (p0=n_parts, p1=kind).
 * kind: 0 sum, 1 max, 2 min, 3 prod.
 */

/* Integer (i32/u32) partial reduce: integer accumulation; max/min via
 * max()/min(); identities INT_MIN/INT_MAX. The local tree buffer `As` (float)
 * is aliased as int — same 4-byte storage, no numeric use. */
static void vmo_reduce_part_tile_i32(__global uchar *arena, __global uchar **iop, const task_t t,
                                 uint tile, __local float *As, uint lid,
                                 uint lsz)
{
    const uint n = t.p0, chunk = t.p1, kind = t.p2;
    const uint lo = tile * chunk, hi = min(lo + chunk, n);
    __global const int *a = AP(const int, t.a);
    __local int *Ai = (__local int *)As;
    int acc = kind == 0 ? 0 : kind == 1 ? INT_MIN : kind == 2 ? INT_MAX : 1;
    for (uint i = lo + lid; i < hi; i += lsz) {
        const int v = a[i];
        acc = kind == 0 ? acc + v
            : kind == 1 ? max(acc, v)
            : kind == 2 ? min(acc, v) : acc * v;
    }
    Ai[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = lsz / 2; s > 0; s >>= 1) {
        if (lid < s) {
            const int x = Ai[lid], y = Ai[lid + s];
            Ai[lid] = kind == 0 ? x + y
                    : kind == 1 ? max(x, y)
                    : kind == 2 ? min(x, y) : x * y;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) AP(int, t.dst)[tile] = Ai[0];
}

static void vmo_reduce_comb_tile_i32(__global uchar *arena, __global uchar **iop, const task_t t,
                                 uint lid)
{
    if (lid != 0) return;
    const uint n = t.p0, kind = t.p1;
    __global const int *a = AP(const int, t.a);
    int acc = kind == 0 ? 0 : kind == 1 ? INT_MIN : kind == 2 ? INT_MAX : 1;
    for (uint i = 0; i < n; ++i) {
        const int v = a[i];
        acc = kind == 0 ? acc + v
            : kind == 1 ? max(acc, v)
            : kind == 2 ? min(acc, v) : acc * v;
    }
    AP(int, t.dst)[0] = acc;
}

static void vmo_reduce_part_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                             __local float *As, uint dt, uint lid, uint lsz)
{
    if (dt == DT_I32 || dt == DT_U32) {
        vmo_reduce_part_tile_i32(arena, iop, t, tile, As, lid, lsz);
        return;
    }
    const uint n = t.p0, chunk = t.p1, kind = t.p2;
    const uint lo = tile * chunk, hi = min(lo + chunk, n);
    __global const float *a = AP(const float, t.a);
    float acc = kind == 0 ? 0.0f
              : kind == 1 ? -INFINITY
              : kind == 2 ? INFINITY : 1.0f;
    for (uint i = lo + lid; i < hi; i += lsz) {
        const float v = a[i];
        acc = kind == 0 ? acc + v
            : kind == 1 ? fmax(acc, v)
            : kind == 2 ? fmin(acc, v) : acc * v;
    }
    As[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = lsz / 2; s > 0; s >>= 1) {
        if (lid < s) {
            const float x = As[lid], y = As[lid + s];
            As[lid] = kind == 0 ? x + y
                    : kind == 1 ? fmax(x, y)
                    : kind == 2 ? fmin(x, y) : x * y;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) AP(float, t.dst)[tile] = As[0];
}

static void vmo_reduce_comb_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint dt,
                             uint lid)
{
    if (dt == DT_I32 || dt == DT_U32) {
        vmo_reduce_comb_tile_i32(arena, iop, t, lid);
        return;
    }
    if (lid != 0) return;
    const uint n = t.p0, kind = t.p1;
    __global const float *a = AP(const float, t.a);
    float acc = kind == 0 ? 0.0f
              : kind == 1 ? -INFINITY
              : kind == 2 ? INFINITY : 1.0f;
    for (uint i = 0; i < n; ++i) {
        const float v = a[i];
        acc = kind == 0 ? acc + v
            : kind == 1 ? fmax(acc, v)
            : kind == 2 ? fmin(acc, v) : acc * v;
    }
    AP(float, t.dst)[0] = acc;
}

/* Register-blocked SGEMM tile (TOP_MMA), from poc/06 step 2 (portable champion
 * family). One 256-thread workgroup computes one MMA_TM x MMA_TN output tile;
 * each thread owns an RM x RN = 4x4 register microtile. Scalar edge-guarded
 * staging, single-buffered -> portable to PoCL. Local: BK*(TM+TN) floats. The
 * scheduler tiles matmul in MMA_TM x MMA_TN blocks (scheduler.MMA_T==MMA_TM).
 * 4x4 (16 accumulators) chosen to bound the megakernel's occupancy tax
 * (docs/tile-isa.md ceiling-1). */
#define MMA_TM 64
#define MMA_TN 64
#define MMA_BK 16
#define MMA_TDIM 16          /* 16x16 thread grid == 256 threads */
#define MMA_RM (MMA_TM / MMA_TDIM)   /* 4 */
#define MMA_RN (MMA_TN / MMA_TDIM)   /* 4 */
#define MMA_ASZ (MMA_BK * MMA_TM)    /* As[m*BK + k] */
#define MMA_BSZ (MMA_BK * MMA_TN)    /* Bs[k*TN + n] */

static void vmo_mma_tile(__global uchar *arena, __global uchar **iop, const task_t t, uint tile,
                     __local float *As, __local float *Bs)
{
    const uint M = t.p0, N = t.p1, K = t.p2;
    const uint tiles_n = (N + MMA_TN - 1) / MMA_TN;
    const uint tr = tile / tiles_n, tc = tile % tiles_n;
    const uint row0 = tr * MMA_TM, col0 = tc * MMA_TN;
    const uint lid = get_local_id(0);
    const uint ty = lid / MMA_TDIM, tx = lid % MMA_TDIM;
    __global const float *ga = AP(const float, t.a);
    __global const float *gb = AP(const float, t.b);

    float acc[MMA_RM][MMA_RN];
    for (int i = 0; i < MMA_RM; i++)
        for (int j = 0; j < MMA_RN; j++) acc[i][j] = 0.0f;

    for (uint k0 = 0; k0 < K; k0 += MMA_BK) {
        for (uint idx = lid; idx < MMA_TM * MMA_BK; idx += 256) {
            const uint m = idx / MMA_BK, kk = idx % MMA_BK;
            const uint gr = row0 + m, gk = k0 + kk;
            As[m * MMA_BK + kk] =
                (gr < M && gk < K) ? ga[gr * K + gk] : 0.0f;
        }
        for (uint idx = lid; idx < MMA_BK * MMA_TN; idx += 256) {
            const uint kk = idx / MMA_TN, n = idx % MMA_TN;
            const uint gk = k0 + kk, gc = col0 + n;
            Bs[kk * MMA_TN + n] =
                (gk < K && gc < N) ? gb[gk * N + gc] : 0.0f;
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

    __global float *gd = AP(float, t.dst);
    for (int i = 0; i < MMA_RM; i++) {
        const uint gr = row0 + ty * MMA_RM + i;
        if (gr >= M) continue;
        for (int j = 0; j < MMA_RN; j++) {
            const uint gc = col0 + tx * MMA_RN + j;
            if (gc < N) gd[gr * N + gc] = acc[i][j];
        }
    }
}

/* Iota-along-dim tile op (TOP_IOTA_DIM). aux at task.p0:
 *   rank, out_dims[rank], dim.  dst[i] = coordinate of i along `dim`. */

static void vmo_iota_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                      const task_t t, uint tile, uint lid, uint lsz)
{
    __global const int *x = aux + t.p0;
    const int rank = x[0];
    __global const int *dims = x + 1;
    const int dim = x[1 + rank];
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
    __global float *d = AP(float, t.dst);
    for (uint i = lo + lid; i < hi; i += lsz) {
        int rem = (int)i, val = 0;
        for (int e = rank - 1; e >= 0; --e) {
            const int idx = rem % dims[e];
            rem /= dims[e];
            if (e == dim) val = idx;
        }
        d[i] = (float)val;
    }
}

/* Strided scatter tile op (TOP_SCATTER) — the mirror of gather. For each INPUT
 * element i (row-major over in_dims), write it to the output at an affine
 * position:  dst[out_off + sum_d idx_d(i)*out_stride_d] = a[i].
 * Used by concatenate / pad: each source is scattered into a disjoint region of
 * a preallocated output. aux at task.p0:
 *   rank u32, in_dims i32[rank], out_strides i32[rank], out_off i32
 * dtype-agnostic (copies whole elements): esz picks the width. */

static void vmo_scatter_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                         const task_t t, uint tile, uint esz, uint lid, uint lsz)
{
    __global const int *x = aux + t.p0;
    const int rank = x[0];
    __global const int *dims = x + 1, *strides = x + 1 + rank;
    const int out_off = x[1 + 2 * rank];
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
    if (esz == 8) {
        __global ulong *d = AP(ulong, t.dst);
        __global const ulong *a = AP(const ulong, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = out_off;
            for (int e = rank - 1; e >= 0; --e) { off += (rem % dims[e]) * strides[e]; rem /= dims[e]; }
            d[off] = a[i];
        }
    } else if (esz == 2) {
        __global ushort *d = AP(ushort, t.dst);
        __global const ushort *a = AP(const ushort, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = out_off;
            for (int e = rank - 1; e >= 0; --e) { off += (rem % dims[e]) * strides[e]; rem /= dims[e]; }
            d[off] = a[i];
        }
    } else if (esz == 1) {
        __global uchar *d = AP(uchar, t.dst);
        __global const uchar *a = AP(const uchar, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = out_off;
            for (int e = rank - 1; e >= 0; --e) { off += (rem % dims[e]) * strides[e]; rem /= dims[e]; }
            d[off] = a[i];
        }
    } else {
        __global int *d = AP(int, t.dst);
        __global const int *a = AP(const int, t.a);
        for (uint i = lo + lid; i < hi; i += lsz) {
            int rem = (int)i, off = out_off;
            for (int e = rank - 1; e >= 0; --e) { off += (rem % dims[e]) * strides[e]; rem /= dims[e]; }
            d[off] = a[i];
        }
    }
}

/* Dynamic slice/update tile ops — gather/scatter with a RUNTIME base offset.
 *
 * Unlike TOP_GATHER/TOP_SCATTER (whose src/out base is a compile-time aux
 * constant), the start offsets here are SCALAR BUFFERS living in the arena: the
 * tile op reads them, clamps to the legal range, and forms the affine base
 * offset at runtime. Their byte offsets ride in the aux pool (the loader can
 * only patch task dst/a/b to byte offsets, not an arbitrary-length index list).
 *
 * aux at task.p0 (identical layout for both ops; `dims`/`strides` name the
 * ITERATED space — output for gather, update for scatter):
 *   rank i32,
 *   dims i32[rank],          out_dims (slice sizes) / upd_dims
 *   strides i32[rank],       in_strides / out_strides (element strides)
 *   clamp_max i32[rank],     max legal start per axis (dim - size)
 *   idx_byteoff i32[rank],   byte offset of each start scalar in the arena
 *   idx_bufid i32[rank],     buffer id (python validators only; unused here)
 *   is64 i32                 0 = i32 start scalars, 1 = i64
 *
 * TOP_DYN_GATHER (dynamic_slice):  dst[i] = a[base + affine(i)]
 * TOP_DYN_SCATTER (dynamic_update_slice): dst[base + affine(i)] = a[i]
 *   with base = sum_d clamp(start_d, 0, clamp_max_d) * strides_d.
 * Both copy whole elements, so they are dtype-agnostic (esz picks the mover).
 */

static int vmo_dyn_base(__global uchar *arena, __global uchar **iop, __global const int *x, int rank,
                    int is64)
{
    __global const int *strides = x + 1 + rank;
    __global const int *clampmax = x + 1 + 2 * rank;
    __global const int *byteoff = x + 1 + 3 * rank;
    int base = 0;
    for (int d = 0; d < rank; ++d) {
        int s = is64 ? (int)AP(const long, byteoff[d])[0]
                     : AP(const int, byteoff[d])[0];
        s = s < 0 ? 0 : (s > clampmax[d] ? clampmax[d] : s);
        base += s * strides[d];
    }
    return base;
}

#define DYN_GATHER_BODY(T)                                                    \
    do {                                                                      \
        __global T *d = AP(T, t.dst);                                         \
        __global const T *a = AP(const T, t.a);                               \
        for (uint i = lo + lid; i < hi; i += lsz) {                           \
            int rem = (int)i, off = base;                                     \
            for (int e = rank - 1; e >= 0; --e) {                             \
                off += (rem % dims[e]) * strides[e]; rem /= dims[e];          \
            }                                                                 \
            d[i] = a[off];                                                    \
        }                                                                     \
    } while (0)

static void vmo_dyn_gather_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                            const task_t t, uint tile, uint esz, uint lid,
                            uint lsz)
{
    __global const int *x = aux + t.p0;
    const int rank = x[0];
    __global const int *dims = x + 1, *strides = x + 1 + rank;
    const int is64 = x[1 + 5 * rank];
    const int base = vmo_dyn_base(arena, iop, x, rank, is64);
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
    if (esz == 8)      DYN_GATHER_BODY(ulong);
    else if (esz == 2) DYN_GATHER_BODY(ushort);
    else if (esz == 1) DYN_GATHER_BODY(uchar);
    else               DYN_GATHER_BODY(uint);
}

#define DYN_SCATTER_BODY(T)                                                   \
    do {                                                                      \
        __global T *d = AP(T, t.dst);                                         \
        __global const T *a = AP(const T, t.a);                               \
        for (uint i = lo + lid; i < hi; i += lsz) {                           \
            int rem = (int)i, off = base;                                     \
            for (int e = rank - 1; e >= 0; --e) {                             \
                off += (rem % dims[e]) * strides[e]; rem /= dims[e];          \
            }                                                                 \
            d[off] = a[i];                                                    \
        }                                                                     \
    } while (0)

static void vmo_dyn_scatter_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                             const task_t t, uint tile, uint esz, uint lid,
                             uint lsz)
{
    __global const int *x = aux + t.p0;
    const int rank = x[0];
    __global const int *dims = x + 1, *strides = x + 1 + rank;
    const int is64 = x[1 + 5 * rank];
    const int base = vmo_dyn_base(arena, iop, x, rank, is64);
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);
    if (esz == 8)      DYN_SCATTER_BODY(ulong);
    else if (esz == 2) DYN_SCATTER_BODY(ushort);
    else if (esz == 1) DYN_SCATTER_BODY(uchar);
    else               DYN_SCATTER_BODY(uint);
}

/* Windowed reduction tile op (TOP_RED_WINDOW) — pooling.
 *
 * Each output element serially reduces the input window mapped to it. Covers
 * the common case only: no base/window dilation, VALID or explicit padding,
 * kind sum/max/min. Padding elements equal the reduction identity, so they are
 * simply skipped (correct because the init value is asserted to be the identity
 * at lowering time).
 *
 * aux at task.p0:
 *   kind i32 (0 sum, 1 max, 2 min), rank i32,
 *   out_dims i32[rank], win_dims i32[rank], win_strides i32[rank],
 *   pad_low i32[rank], in_dims i32[rank], in_strides i32[rank]
 *
 * out[o] = reduce_{w in window} in[o*stride + w - pad_low]   (in-bounds only).
 * Supports f32 (float compute) and i32/u32 (integer compute); the loader gates
 * other dtypes at lowering time.
 */

static void vmo_redwin_tile(__global uchar *arena, __global uchar **iop, __global const int *aux,
                        const task_t t, uint tile, uint dt, uint lid, uint lsz)
{
    __global const int *x = aux + t.p0;
    const int kind = x[0];
    const int rank = x[1];
    __global const int *odims = x + 2;
    __global const int *wdims = x + 2 + rank;
    __global const int *wstr  = x + 2 + 2 * rank;
    __global const int *plow  = x + 2 + 3 * rank;
    __global const int *idims = x + 2 + 4 * rank;
    __global const int *istr  = x + 2 + 5 * rank;
    const uint lo = tile * EW_TS, hi = min(lo + EW_TS, t.p1);

    int wcount = 1;
    for (int d = 0; d < rank; ++d) wcount *= wdims[d];
    const int isint = (dt == DT_I32 || dt == DT_U32);

    for (uint i = lo + lid; i < hi; i += lsz) {
        float facc = kind == 0 ? 0.0f : kind == 1 ? -INFINITY : INFINITY;
        int iacc = kind == 0 ? 0 : kind == 1 ? INT_MIN : INT_MAX;
        for (int w = 0; w < wcount; ++w) {
            int rem_i = (int)i, rem_w = w, off = 0, inb = 1;
            for (int d = rank - 1; d >= 0; --d) {
                const int oc = rem_i % odims[d]; rem_i /= odims[d];
                const int wc = rem_w % wdims[d]; rem_w /= wdims[d];
                const int ic = oc * wstr[d] + wc - plow[d];
                if (ic < 0 || ic >= idims[d]) inb = 0;
                off += ic * istr[d];
            }
            if (!inb) continue;
            if (isint) {
                const int v = AP(const int, t.a)[off];
                iacc = kind == 0 ? iacc + v
                     : kind == 1 ? max(iacc, v) : min(iacc, v);
            } else {
                const float v = AP(const float, t.a)[off];
                facc = kind == 0 ? facc + v
                     : kind == 1 ? fmax(facc, v) : fmin(facc, v);
            }
        }
        if (isint) AP(int, t.dst)[i] = iacc;
        else AP(float, t.dst)[i] = facc;
    }
}

/* pjrt-ocl VLIW engine — dispatcher + interpreter (concatenated last).
 * vmo_exec_tiles routes a task to its op-family tile function (ops/ *.cl above);
 * vm2 is the per-lane interpreter over the schedule stream. */

/* tile_op packs the base op in bits 0-7 and the dtype in bits 8-15. */
static void vmo_exec_tiles(__global uchar *arena, __global uchar **iop,
                       __global const int *aux,
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
        case TOP_EW:       vmo_ew_tile(arena, iop, t, tile, dt, adt, lid, lsz); break;
        case TOP_MMA:      vmo_mma_tile(arena, iop, t, tile, As, Bs); break;
        case TOP_GATHER:   vmo_gather_tile(arena, iop, aux, t, tile, esz, lid, lsz); break;
        case TOP_RED_PART: vmo_reduce_part_tile(arena, iop, t, tile, As, dt, lid, lsz); break;
        case TOP_RED_COMB: vmo_reduce_comb_tile(arena, iop, t, dt, lid); break;
        case TOP_IOTA_DIM: vmo_iota_tile(arena, iop, aux, t, tile, lid, lsz); break;
        case TOP_SCATTER:  vmo_scatter_tile(arena, iop, aux, t, tile, esz, lid, lsz); break;
        case TOP_DYN_GATHER:  vmo_dyn_gather_tile(arena, iop, aux, t, tile, esz, lid, lsz); break;
        case TOP_DYN_SCATTER: vmo_dyn_scatter_tile(arena, iop, aux, t, tile, esz, lid, lsz); break;
        case TOP_RED_WINDOW:  vmo_redwin_tile(arena, iop, aux, t, tile, dt, lid, lsz); break;
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
                  __global uint *stats,              /* arrival rank per
                                                        [barrier_i*nlanes+lane] */
                  VMO_IO_PARAMS)                     /* direct I/O buffers (ports) */
{
    VMO_IO_ARRAY;
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
                vmo_barrier(bar, nlanes);
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
                vmo_barrier(bar, nlanes);
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
            vmo_barrier(bar, nlanes);
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
            vmo_exec_tiles(arena, iop, aux, tasks[en.task], en.tile_lo, en.tile_hi,
                       As, Bs);
        }
        st[sp].pc++;
    }
}

/* HOST-DISPATCH engine (CPU / non-GPU devices, docs/decisions.md #1): the host
 * drives control flow and the cross-workgroup barrier via clFinish between
 * launches, so there is NO in-kernel barrier and no co-residency requirement
 * (a finished workgroup exits and frees its CPU thread — immune to the
 * imbalance-starvation deadlock the persistent spin-barrier hits on PoCL,
 * poc/07). This kernel runs ONE barrier-free segment: each workgroup (lane)
 * executes its contiguous run of tile entries [seg.x, seg.x+seg.y) — the host
 * has already resolved all BARRIER/WHILE/IF control, so a segment holds only
 * tile (or NOP) entries. */
__kernel void vm2_seg(__global uchar *arena,
                      __global const int *aux,
                      __global const task_t *tasks,
                      __global const entry_t *entries,
                      __global const uint2 *seg_tab,   /* per-lane {off, count} */
                      VMO_IO_PARAMS)                    /* direct I/O buffers */
{
    VMO_IO_ARRAY;
    const uint lane = get_group_id(0);
    __local float As[MMA_ASZ];
    __local float Bs[MMA_BSZ];
    const uint2 seg = seg_tab[lane];
    for (uint i = 0; i < seg.y; ++i) {
        const entry_t en = entries[seg.x + i];
        if (en.task != ENT_NOP)
            vmo_exec_tiles(arena, iop, aux, tasks[en.task], en.tile_lo, en.tile_hi,
                           As, Bs);
    }
}

)CLSRC";
