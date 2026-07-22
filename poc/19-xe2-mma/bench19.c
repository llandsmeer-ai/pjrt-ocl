/* poc/19: Xe2 f32 SGEMM tile-geometry sweep. Builds mma19.cl once per (TM,TN,
 * TD,BK) config via -D options, times C=A@B at N, reports GFLOP/s, and spot-
 * checks correctness against a CPU reference. Picks the first platform whose
 * name contains argv-substr (default "Intel"). No plugin, no python.
 *
 * Usage: ./bench19 [N] [platform-substr]                                    */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(n + 1); fread(b, 1, n, f); b[n] = 0; fclose(f); return b;
}

struct cfg { int tm, tn, td, bk; };

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 2048;
    const char *psub = argc > 2 ? argv[2] : "Intel";
    int M = N, K = N;

    cl_platform_id plats[8]; cl_uint np = 0;
    clGetPlatformIDs(8, plats, &np);
    cl_platform_id plat = 0; char pname[256];
    for (cl_uint i = 0; i < np; i++) {
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof pname, pname, 0);
        if (strstr(pname, psub)) { plat = plats[i]; break; }
    }
    if (!plat) { fprintf(stderr, "no platform matching '%s'\n", psub); return 1; }
    cl_device_id dev; clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, 0);
    char dname[256]; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof dname, dname, 0);
    cl_uint cu = 0; size_t maxwg = 0; cl_ulong slm = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, 0);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof maxwg, &maxwg, 0);
    clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof slm, &slm, 0);
    printf("device: %s | CUs=%u maxWG=%zu SLM=%lluKB | N=%d\n",
           dname, cu, maxwg, (unsigned long long)(slm / 1024), N);

    cl_int e;
    cl_context ctx = clCreateContext(0, 1, &dev, 0, 0, &e);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &e);

    size_t bytes = (size_t)N * N * 4;
    float *hA = malloc(bytes), *hB = malloc(bytes), *hC = malloc(bytes);
    srand(1);
    for (size_t i = 0; i < (size_t)N * N; i++) {
        hA[i] = (rand() % 17 - 8) * 0.1f; hB[i] = (rand() % 17 - 8) * 0.1f;
    }
    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, hA, &e);
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, hB, &e);
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, 0, &e);

    char *src = slurp("mma19.cl");

    /* CPU reference for a handful of output cells (full O(N^3) ref is too slow) */
    #define NCHK 24
    int cr[NCHK], cc[NCHK]; float cref[NCHK];
    for (int t = 0; t < NCHK; t++) {
        cr[t] = rand() % M; cc[t] = rand() % N; double s = 0;
        for (int k = 0; k < K; k++) s += (double)hA[cr[t]*K+k] * hB[(size_t)k*N+cc[t]];
        cref[t] = (float)s;
    }

    struct cfg cfgs[] = {
        {128, 64, 16, 16},   /* == shipped mm2 (baseline)             RM8 RN4 */
        {128, 64, 16, 8},    /* baseline shape, bk8                            */
        {128, 64, 16, 4},
        {64,  64, 16, 8},    /* smaller reg block, bk8                RM4 RN4 */
        {64,  64, 16, 4},
        {128, 128,16, 8},    /* 8x8, bk8                                       */
        {128, 128,16, 4},
        {64,  128,16, 8},    /* RM4 RN8                                        */
        {128, 64, 16, 32},   /* deeper K stage                                */
    };
    int ncfg = sizeof cfgs / sizeof cfgs[0];

    printf("  %-22s %8s %8s  %s\n", "TMxTN td bk (RMxRN)", "ms", "GFLOP/s", "chk");
    for (int c = 0; c < ncfg; c++) {
        struct cfg g = cfgs[c];
        int nt = g.td * g.td, rm = g.tm / g.td, rn = g.tn / g.td;
        if (nt > (int)maxwg) { printf("  skip %dx%d td%d (WG %d>max)\n", g.tm,g.tn,g.td,nt); continue; }
        if (N % g.tm || N % g.tn) { continue; }
        char opts[256];
        snprintf(opts, sizeof opts,
                 "-cl-fast-relaxed-math -DTM=%d -DTN=%d -DTD=%d -DBK=%d",
                 g.tm, g.tn, g.td, g.bk);
        cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **)&src, 0, &e);
        e = clBuildProgram(prog, 1, &dev, opts, 0, 0);
        if (e != CL_SUCCESS) {
            size_t ln; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, 0, &ln);
            char *log = malloc(ln + 1); clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, ln, log, 0);
            log[ln] = 0; printf("  build FAIL %dx%d td%d bk%d: %.180s\n", g.tm,g.tn,g.td,g.bk, log);
            free(log); clReleaseProgram(prog); continue;
        }
        cl_kernel k = clCreateKernel(prog, "mma", &e);
        cl_uint um = M, un = N, uk = K;
        clSetKernelArg(k, 0, sizeof dA, &dA); clSetKernelArg(k, 1, sizeof dB, &dB);
        clSetKernelArg(k, 2, sizeof dC, &dC); clSetKernelArg(k, 3, sizeof um, &um);
        clSetKernelArg(k, 4, sizeof un, &un); clSetKernelArg(k, 5, sizeof uk, &uk);
        size_t tiles = (size_t)(N / g.tm) * (N / g.tn);
        size_t gws = tiles * nt, lws = nt;

        for (int w = 0; w < 3; w++) clEnqueueNDRangeKernel(q, k, 1, 0, &gws, &lws, 0,0,0);
        clFinish(q);
        double best = 1e30;
        for (int r = 0; r < 8; r++) {
            double t0 = now_ms();
            clEnqueueNDRangeKernel(q, k, 1, 0, &gws, &lws, 0, 0, 0);
            clFinish(q);
            double dt = now_ms() - t0; if (dt < best) best = dt;
        }
        clEnqueueReadBuffer(q, dC, CL_TRUE, 0, bytes, hC, 0, 0, 0);
        float maxerr = 0;
        for (int t = 0; t < NCHK; t++) {
            float got = hC[(size_t)cr[t] * N + cc[t]];
            float er = fabsf(got - cref[t]); if (er > maxerr) maxerr = er;
        }
        double gf = 2.0 * N * N * N / (best * 1e-3) / 1e9;
        char tag[48]; snprintf(tag, sizeof tag, "%dx%d td%d bk%d (%dx%d)", g.tm,g.tn,g.td,g.bk,rm,rn);
        printf("  %-22s %8.3f %8.1f  %s(%.1e)\n", tag, best, gf,
               maxerr < 1e-2 ? "ok" : "BAD", maxerr);
        clReleaseKernel(k); clReleaseProgram(prog);
    }
    return 0;
}
