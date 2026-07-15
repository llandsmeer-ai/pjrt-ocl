/* pjrt-ocl VLIW engine — shared header (concatenated first).
 *
 * The kernel program is assembled by concatenating, in order (see
 * CMakeLists.txt VM_CL_SOURCES):
 *   vm_common.cl  (this file: defines, structs, helpers, barrier)
 *   ops/ew.cl ops/gather.cl ops/reduce.cl ops/mma.cl ops/iota.cl
 *   vm_main.cl    (exec_tiles dispatch + the vm2 interpreter kernel)
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

/* Typed element pointer at byte base `base` into the byte-addressed arena. */
#define AP(T, base) ((__global T *)(arena + (base)))

/* dtype enum (matches python DT_* / runtime.h). Tier 1 = 4-byte. */
enum { DT_F32 = 0, DT_I32 = 1, DT_U32 = 2, DT_BOOL = 3,
       DT_I64 = 4, DT_F64 = 5 };

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

/* Value-level bit-recast (bitcast_convert, NaN-safe integer handling). OpenCL
 * C defines union type-punning; keeps integer bit patterns out of float
 * registers where a GPU might canonicalize a NaN. */
typedef union { float f; int i; uint u; } slot_t;

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
