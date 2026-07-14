/* poc/01 host: load vm.cl, run correctness + barrier-stress + overhead bench.
 *
 * env:
 *   OCL_PLATFORM  substring of platform name (default "Portable" = PoCL)
 *   VM_GROUPS     number of workgroups (default: device max compute units)
 *   VM_LOCAL      workgroup size (default 64)
 *   VM_STRESS     barrier stress iterations (default 200)
 *   VM_BENCH_N    elements per bench op (default 1<<20)
 *   VM_BENCH_K    instructions per bench program (default 200)
 */
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define CHK(err, what) \
    do { if ((err) != CL_SUCCESS) { fprintf(stderr, "FAIL %s: %d (%s:%d)\n", what, (int)(err), __FILE__, __LINE__); exit(1); } } while (0)

typedef struct {
    cl_uint op, dst, a, b, n, imm, pad0, pad1;
} instr_t;

enum { OP_NOP = 0, OP_ADD, OP_MUL, OP_SUB, OP_FILL, OP_IOTA, OP_REVADD };

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static const char *envs(const char *k, const char *dflt)
{
    const char *v = getenv(k);
    return v && *v ? v : dflt;
}

static long envi(const char *k, long dflt)
{
    const char *v = getenv(k);
    return v && *v ? atol(v) : dflt;
}

static char *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(*len + 1);
    if (fread(buf, 1, *len, f) != *len) { perror("fread"); exit(1); }
    buf[*len] = 0;
    fclose(f);
    return buf;
}

int main(void)
{
    cl_int err;

    /* --- pick platform/device ------------------------------------------ */
    cl_platform_id plats[8];
    cl_uint nplat = 0;
    CHK(clGetPlatformIDs(8, plats, &nplat), "clGetPlatformIDs");
    const char *want = envs("OCL_PLATFORM", "Portable");
    cl_platform_id plat = NULL;
    char pname[256] = {0};
    for (cl_uint i = 0; i < nplat; i++) {
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof pname, pname, NULL);
        if (strstr(pname, want)) { plat = plats[i]; break; }
    }
    if (!plat) { fprintf(stderr, "no platform matching '%s'\n", want); exit(1); }

    cl_device_id dev;
    CHK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, NULL), "clGetDeviceIDs");
    char dname[256];
    cl_uint cus;
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof dname, dname, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cus, &cus, NULL);

    const cl_uint ngroups = (cl_uint)envi("VM_GROUPS", cus);
    const size_t lsz = (size_t)envi("VM_LOCAL", 64);
    const size_t gsz = ngroups * lsz;
    printf("platform: %s\ndevice:   %s (%u CUs)\nlaunch:   %u groups x %zu = %zu work-items\n",
           pname, dname, cus, ngroups, lsz, gsz);

    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    CHK(err, "clCreateContext");
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    CHK(err, "clCreateCommandQueue");

    size_t srclen;
    char *src = read_file("vm.cl", &srclen);
    cl_program clprog = clCreateProgramWithSource(ctx, 1, (const char **)&src, &srclen, &err);
    CHK(err, "clCreateProgramWithSource");
    err = clBuildProgram(clprog, 1, &dev, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[16384];
        clGetProgramBuildInfo(clprog, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
        fprintf(stderr, "build log:\n%s\n", log);
        exit(1);
    }
    cl_kernel kvm = clCreateKernel(clprog, "vm", &err);
    CHK(err, "clCreateKernel vm");
    cl_kernel kadd1 = clCreateKernel(clprog, "add1", &err);
    CHK(err, "clCreateKernel add1");

    /* --- arena + program + barrier buffers ----------------------------- */
    const size_t ARENA_ELEMS = 8u << 20; /* 32 MB of f32 */
    cl_mem arena = clCreateBuffer(ctx, CL_MEM_READ_WRITE, ARENA_ELEMS * 4, NULL, &err);
    CHK(err, "arena");
    const size_t MAX_INSTR = 4096;
    cl_mem progbuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, MAX_INSTR * sizeof(instr_t), NULL, &err);
    CHK(err, "progbuf");
    cl_uint barinit[2] = {0, 0};
    cl_mem barbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof barinit, barinit, &err);
    CHK(err, "barbuf");

    instr_t *prog = calloc(MAX_INSTR, sizeof(instr_t));
    cl_uint n_instr = 0;
#define EMIT(...) do { prog[n_instr++] = (instr_t){__VA_ARGS__}; } while (0)

    /* run helper */
    void run_vm(cl_uint count) {
        CHK(clEnqueueWriteBuffer(q, progbuf, CL_TRUE, 0, count * sizeof(instr_t), prog, 0, NULL, NULL), "write prog");
        clSetKernelArg(kvm, 0, sizeof arena, &arena);
        clSetKernelArg(kvm, 1, sizeof progbuf, &progbuf);
        clSetKernelArg(kvm, 2, sizeof count, &count);
        clSetKernelArg(kvm, 3, sizeof barbuf, &barbuf);
        clSetKernelArg(kvm, 4, sizeof ngroups, &ngroups);
        CHK(clEnqueueNDRangeKernel(q, kvm, 1, NULL, &gsz, &lsz, 0, NULL, NULL), "enqueue vm");
        CHK(clFinish(q), "finish vm");
    }

    /* =========== test 1: correctness of a small linear program ========= */
    /* b0=iota(N); b1=fill(2.0); b2=b0*b1; b3=b2-b0; b4=rev(b3)+b0 */
    const cl_uint N = 1 << 16;
    const cl_uint b0 = 0, b1 = N, b2 = 2 * N, b3 = 3 * N, b4 = 4 * N;
    union { float f; cl_uint u; } two = {.f = 2.0f};
    n_instr = 0;
    EMIT(OP_IOTA, b0, 0, 0, N, 0);
    EMIT(OP_FILL, b1, 0, 0, N, two.u);
    EMIT(OP_MUL, b2, b0, b1, N, 0);
    EMIT(OP_SUB, b3, b2, b0, N, 0);
    EMIT(OP_REVADD, b4, b3, b0, N, 0);
    run_vm(n_instr);

    float *out = malloc(N * 4);
    CHK(clEnqueueReadBuffer(q, arena, CL_TRUE, b4 * 4, N * 4, out, 0, NULL, NULL), "read");
    int bad = 0;
    for (cl_uint i = 0; i < N; i++) {
        /* b3[j] = j; expected out[i] = (N-1-i) + i = N-1 */
        if (out[i] != (float)(N - 1)) { bad++; if (bad < 4) fprintf(stderr, "  out[%u]=%g want %g\n", i, out[i], (float)(N - 1)); }
    }
    printf("test1 correctness: %s (%d bad of %u)\n", bad ? "FAIL" : "PASS", bad, N);

    /* =========== test 2: barrier stress ================================ */
    /* Repeated REVADD ping-pong: x_{k+1}[i] = x_k[n-1-i] + zero[i].
     * Any missed barrier reads a half-updated buffer => value corruption. */
    const cl_uint S = (cl_uint)envi("VM_STRESS", 200);
    union { float f; cl_uint u; } zero = {.f = 0.0f};
    n_instr = 0;
    EMIT(OP_IOTA, b0, 0, 0, N, 0);
    EMIT(OP_FILL, b1, 0, 0, N, zero.u);
    run_vm(n_instr);
    n_instr = 0;
    for (cl_uint s = 0; s < S; s += 2) {
        EMIT(OP_REVADD, b2, b0, b1, N, 0);   /* b2 = rev(b0) */
        EMIT(OP_REVADD, b0, b2, b1, N, 0);   /* b0 = rev(b2) = original */
        if (n_instr > MAX_INSTR - 2) break;
    }
    cl_uint stress_count = n_instr;
    double t0 = now_ms();
    run_vm(stress_count);
    double t_stress = now_ms() - t0;
    CHK(clEnqueueReadBuffer(q, arena, CL_TRUE, b0 * 4, N * 4, out, 0, NULL, NULL), "read");
    bad = 0;
    for (cl_uint i = 0; i < N; i++)
        if (out[i] != (float)i) bad++;
    printf("test2 barrier stress (%u instrs): %s (%d bad) [%.1f ms, %.1f us/instr]\n",
           stress_count, bad ? "FAIL" : "PASS", bad, t_stress, 1e3 * t_stress / stress_count);

    /* =========== test 3: dispatch overhead ============================= */
    const cl_uint BN = (cl_uint)envi("VM_BENCH_N", 1 << 20);
    const cl_uint BK = (cl_uint)envi("VM_BENCH_K", 200);
    n_instr = 0;
    for (cl_uint k = 0; k < BK && n_instr < MAX_INSTR; k++)
        EMIT(OP_ADD, 2 * BN, 0, BN, BN, 0);
    run_vm(n_instr); /* warm */
    t0 = now_ms();
    run_vm(n_instr);
    double t_vm = now_ms() - t0;

    /* same work as BK separate launches */
    cl_uint dst = 2 * BN, a0 = 0, a1 = BN;
    clSetKernelArg(kadd1, 0, sizeof arena, &arena);
    clSetKernelArg(kadd1, 1, sizeof dst, &dst);
    clSetKernelArg(kadd1, 2, sizeof a0, &a0);
    clSetKernelArg(kadd1, 3, sizeof a1, &a1);
    clSetKernelArg(kadd1, 4, sizeof BN, &BN);
    for (cl_uint k = 0; k < 8; k++)
        CHK(clEnqueueNDRangeKernel(q, kadd1, 1, NULL, &gsz, &lsz, 0, NULL, NULL), "warm add1");
    clFinish(q);
    t0 = now_ms();
    for (cl_uint k = 0; k < BK; k++)
        CHK(clEnqueueNDRangeKernel(q, kadd1, 1, NULL, &gsz, &lsz, 0, NULL, NULL), "enqueue add1");
    clFinish(q);
    double t_launch = now_ms() - t0;
    printf("test3 overhead: %u x add(%u elems)\n  vm megakernel: %.1f ms (%.1f us/op)\n  %u launches:   %.1f ms (%.1f us/op)\n  ratio: %.2fx\n",
           BK, BN, t_vm, 1e3 * t_vm / BK, BK, t_launch, 1e3 * t_launch / BK, t_launch / t_vm);

    printf("done\n");
    return bad ? 1 : 0;
}
