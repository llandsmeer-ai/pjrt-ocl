/* poc/06: standalone SGEMM tile-function benchmark harness.
 *
 * Loops output tiles across persistent-style workgroups (lanes = 2 x CU count,
 * each lane a contiguous tile range) exactly as the VM would drive the tile
 * function from poc/04's exec_tiles. Reports best-of-5 GFLOP/s at M=N=K=2048
 * and 4096, and verifies correctness vs a host reference at 512^3 with
 * integer-valued floats (exact compare).
 *
 * Each progression step is one build of mma.cl with different -D options.
 *
 * env: OCL_PLATFORM (default "NVIDIA"), VM_LANES (default 2 x CU count).
 */
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static void chk(cl_int e, const char *m) {
    if (e != CL_SUCCESS) { fprintf(stderr, "ERR %d @ %s\n", e, m); exit(1); }
}
static long envi(const char *k, long d) {
    const char *v = getenv(k); return v && *v ? atol(v) : d;
}
static const char *envs(const char *k, const char *d) {
    const char *v = getenv(k); return v && *v ? v : d;
}
static char *read_file(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(*n + 1); if (fread(b, 1, *n, f) != *n) exit(1);
    b[*n] = 0; fclose(f); return b;
}

static cl_context ctx; static cl_command_queue q;
static cl_device_id dev; static char *src; static size_t srclen;
static cl_mem arena;
static cl_uint nlanes;

typedef struct { const char *name, *kern, *opts; } step_t;

/* build program with opts, return the named kernel */
static cl_kernel build(const char *opts, const char *kern) {
    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **)&src,
                                                &srclen, &err);
    chk(err, "prog");
    if (clBuildProgram(prog, 1, &dev, opts, 0, 0) != CL_SUCCESS) {
        char log[16384];
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log,
                              log, 0);
        fprintf(stderr, "build failed (%s):\n%s\n", opts, log);
        exit(1);
    }
    cl_kernel k = clCreateKernel(prog, kern, &err); chk(err, "kernel");
    return k;
}

/* run one matmul C = A@B on the arena. A@0, B@M*K, C@M*K+K*N. */
static double run_once(cl_kernel k, cl_uint M, cl_uint N, cl_uint K,
                       int tm, int tn, int bk) {
    cl_uint aoff = 0, boff = M * K, coff = M * K + K * N;
    cl_uint full = (M % tm == 0 && N % tn == 0 && K % bk == 0) ? 1u : 0u;
    clSetKernelArg(k, 0, sizeof arena, &arena);
    clSetKernelArg(k, 1, sizeof aoff, &aoff);
    clSetKernelArg(k, 2, sizeof boff, &boff);
    clSetKernelArg(k, 3, sizeof coff, &coff);
    clSetKernelArg(k, 4, sizeof M, &M);
    clSetKernelArg(k, 5, sizeof N, &N);
    clSetKernelArg(k, 6, sizeof K, &K);
    clSetKernelArg(k, 7, sizeof nlanes, &nlanes);
    clSetKernelArg(k, 8, sizeof full, &full);
    size_t gsz = (size_t)nlanes * 256, lsz = 256;
    struct timespec a, b;
    clFinish(q);
    clock_gettime(CLOCK_MONOTONIC, &a);
    chk(clEnqueueNDRangeKernel(q, k, 1, NULL, &gsz, &lsz, 0, 0, 0), "launch");
    chk(clFinish(q), "finish");
    clock_gettime(CLOCK_MONOTONIC, &b);
    return (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
}

/* correctness at 512^3 with integer-valued floats, exact compare */
static int verify(cl_kernel k, int tm, int tn, int bk) {
    const cl_uint S = 512;
    const cl_uint aoff = 0, boff = S * S, coff = 2 * S * S;
    float *A = malloc(S * S * 4), *B = malloc(S * S * 4), *C = malloc(S * S * 4);
    for (cl_uint i = 0; i < S; i++)
        for (cl_uint j = 0; j < S; j++) {
            A[i * S + j] = (float)((i + 2 * j) % 5);      /* 0..4 */
            B[i * S + j] = (float)((3 * i + j) % 4);      /* 0..3 */
        }
    chk(clEnqueueWriteBuffer(q, arena, CL_TRUE, aoff * 4, S * S * 4, A, 0, 0, 0),
        "wA");
    chk(clEnqueueWriteBuffer(q, arena, CL_TRUE, boff * 4, S * S * 4, B, 0, 0, 0),
        "wB");
    run_once(k, S, S, S, tm, tn, bk);
    chk(clEnqueueReadBuffer(q, arena, CL_TRUE, coff * 4, S * S * 4, C, 0, 0, 0),
        "rC");
    int bad = 0;
    for (cl_uint i = 0; i < S && bad < 4; i++)
        for (cl_uint j = 0; j < S; j++) {
            float ref = 0.0f;
            for (cl_uint p = 0; p < S; p++) ref += A[i * S + p] * B[p * S + j];
            if (C[i * S + j] != ref) {
                if (bad < 4)
                    fprintf(stderr, "  mismatch @(%u,%u): got %g want %g\n",
                            i, j, C[i * S + j], ref);
                bad++;
                break;
            }
        }
    free(A); free(B); free(C);
    return bad == 0;
}

static double bench(cl_kernel k, cl_uint S, int tm, int tn, int bk) {
    double best = 1e30;
    for (int r = 0; r < 5; r++) {
        double ms = run_once(k, S, S, S, tm, tn, bk);
        if (ms < best) best = ms;
    }
    double gflop = 2.0 * S * S * S / 1e9;
    return gflop / (best / 1e3);   /* GFLOP/s */
}

int main(void) {
    cl_int err;
    cl_platform_id plats[8]; cl_uint np = 0;
    chk(clGetPlatformIDs(8, plats, &np), "platforms");
    const char *want = envs("OCL_PLATFORM", "NVIDIA");
    cl_platform_id plat = NULL; char pname[256];
    for (cl_uint i = 0; i < np; i++) {
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof pname, pname, 0);
        if (strstr(pname, want)) { plat = plats[i]; break; }
    }
    if (!plat) { fprintf(stderr, "no platform '%s'\n", want); return 1; }
    chk(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, 0), "device");
    char dname[256]; cl_uint cus;
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof dname, dname, 0);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cus, &cus, 0);
    nlanes = (cl_uint)envi("VM_LANES", 2 * cus);
    printf("%s | %s | %u CUs | %u lanes x 256 threads\n",
           pname, dname, cus, nlanes);

    ctx = clCreateContext(0, 1, &dev, 0, 0, &err); chk(err, "ctx");
    q = clCreateCommandQueue(ctx, dev, 0, &err); chk(err, "q");
    src = read_file("mma.cl", &srclen);

    /* arena big enough for the largest run (4096^3: 3 x 16M floats) */
    const size_t AR = 3ull * 4096 * 4096;
    arena = clCreateBuffer(ctx, CL_MEM_READ_WRITE, AR * 4, 0, &err);
    chk(err, "arena");

    /* progression steps: {label, kernel, build opts, TM, TN, BK} */
    struct { step_t s; int tm, tn, bk; } steps[] = {
      {{"1 naive 16x16 (poc/04)",       "bench_naive", ""},                                          16, 16, 16},
      {{"2 reg 64x64 4x4  BK16",        "bench_fast",  "-DFAST -DTM=64  -DTN=64  -DBK=16"},          64, 64, 16},
      {{"3 reg 128x128 8x8 BK8",        "bench_fast",  "-DFAST -DTM=128 -DTN=128 -DBK=8"},          128,128,  8},
      {{"4 +float4 loads",              "bench_fast",  "-DFAST -DTM=128 -DTN=128 -DBK=8  -DVECW=4"},128,128,  8},
      {{"5 +double buffer",             "bench_fast",  "-DFAST -DTM=128 -DTN=128 -DBK=8  -DVECW=4 -DDB=1"},128,128,8},
    };
    int nsteps = (int)(sizeof steps / sizeof steps[0]);
    const char *only = getenv("ONLY");   /* substring filter on label */

    printf("\n%-30s  %8s  %10s  %10s\n", "step", "512 OK", "2048 GF/s",
           "4096 GF/s");
    printf("--------------------------------------------------------------------\n");
    for (int i = 0; i < nsteps; i++) {
        if (only && !strstr(steps[i].s.name, only)) continue;
        cl_kernel k = build(steps[i].s.opts, steps[i].s.kern);
        int ok = verify(k, steps[i].tm, steps[i].tn, steps[i].bk);
        double g2 = bench(k, 2048, steps[i].tm, steps[i].tn, steps[i].bk);
        double g4 = bench(k, 4096, steps[i].tm, steps[i].tn, steps[i].bk);
        printf("%-30s  %8s  %10.1f  %10.1f\n", steps[i].s.name,
               ok ? "yes" : "NO", g2, g4);
        clReleaseKernel(k);
    }
    printf("\npeak (clpeak FP32) = 105900 GF/s (105.9 TFLOPS)\n");
    return 0;
}
