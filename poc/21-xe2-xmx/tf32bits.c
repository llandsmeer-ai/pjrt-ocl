/* poc/21 — what does the tf32 XMX path actually do to its operands?
 *
 * The GEMM sweep showed the tf32 result matching NO "round both operands to mb
 * mantissa bits, accumulate exactly" model, while bf16/f16 matched theirs to
 * ~6e-6. This isolates the question: run a single 8x8x16 tile with
 *     A[0][k] = v_k  (k=0..7, all other rows 0)
 *     B[k][n] = 1 if k==n else 0        (identity in the first 8 columns)
 * so that C[0][n] == engine_round(v_n) * engine_round(1.0) exactly, and dump
 * the resulting f32 bit patterns next to the inputs.
 *
 * Uses xmx21tf32.cl at its smallest geometry (SGM=4 SGN=4 RM=1 RN=1 =>
 * 32x64 workgroup tile), M=32 N=64 K=8: one workgroup, one k-tile.
 *
 * Usage: ./tf32bits [platform-substr]                                        */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static char *slurp(const char *p) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(n + 1); if (fread(b, 1, n, f) != (size_t)n) exit(2);
    b[n] = 0; fclose(f); return b;
}
static unsigned bits(float f) { union { float f; unsigned u; } v = { .f = f }; return v.u; }
static float frombits(unsigned u) { union { float f; unsigned u; } v = { .u = u }; return v.f; }

int main(int argc, char **argv) {
    const char *psub = argc > 1 ? argv[1] : "Intel";
    cl_platform_id plats[8]; cl_uint np = 0;
    clGetPlatformIDs(8, plats, &np);
    cl_platform_id plat = 0; char pn[256];
    for (cl_uint i = 0; i < np; i++) {
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof pn, pn, 0);
        if (strstr(pn, psub)) { plat = plats[i]; break; }
    }
    if (!plat) { fprintf(stderr, "no platform '%s'\n", psub); return 2; }
    cl_device_id dev; clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, 0);
    cl_int e; cl_context ctx = clCreateContext(0, 1, &dev, 0, 0, &e);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &e);

    const int M = 32, N = 64, K = 8;
    size_t na = (size_t)M * K, nb = (size_t)K * N, nc = (size_t)M * N;
    float *hA = calloc(na, 4), *hB = calloc(nb, 4), *hC = calloc(nc, 4);

    /* 8 probe values with full 23-bit mantissas, spanning tie cases */
    float v[8];
    v[0] = frombits(bits(1.0f) | 0x7fffffu);        /* 1 + (1-2^-23)          */
    v[1] = frombits(bits(1.0f) | 0x001000u);        /* exact tie at 10 bits   */
    v[2] = frombits(bits(1.0f) | 0x003000u);        /* exact tie at 10 bits   */
    v[3] = frombits(bits(1.0f) | 0x000800u);        /* exact tie at 11 bits   */
    v[4] = frombits(bits(1.0f) | 0x0017ffu);
    v[5] = frombits(bits(1.0f) | 0x0018ffu);
    v[6] = 3.14159265f;
    v[7] = 0.7853981634f;
    for (int k = 0; k < 8; k++) hA[k] = v[k];        /* row 0 of A */
    for (int k = 0; k < 8; k++) hB[(size_t)k * N + k] = 1.0f;

    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, na * 4, hA, &e);
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nb * 4, hB, &e);
    cl_mem dAp = clCreateBuffer(ctx, CL_MEM_READ_WRITE, na * 4, 0, &e);
    cl_mem dBp = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nb * 4, 0, &e);
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, nc * 4, 0, &e);

    char *src = slurp("xmx21tf32.cl");
    cl_program pr = clCreateProgramWithSource(ctx, 1, (const char **)&src, 0, &e);
    e = clBuildProgram(pr, 1, &dev, "-cl-std=CL2.0 -DSGM=4 -DSGN=4 -DRM=1 -DRN=1", 0, 0);
    if (e != CL_SUCCESS) {
        size_t ln = 0; clGetProgramBuildInfo(pr, dev, CL_PROGRAM_BUILD_LOG, 0, 0, &ln);
        char *log = malloc(ln + 1);
        clGetProgramBuildInfo(pr, dev, CL_PROGRAM_BUILD_LOG, ln, log, 0); log[ln] = 0;
        printf("build failed: %s\n", log); return 2;
    }
    cl_kernel ka = clCreateKernel(pr, "pack_a", &e), kb = clCreateKernel(pr, "pack_b", &e),
              kg = clCreateKernel(pr, "gemm", &e);
    cl_uint uK = K, uN = N, uM = M;
    clSetKernelArg(ka, 0, sizeof dA, &dA); clSetKernelArg(ka, 1, sizeof dAp, &dAp);
    clSetKernelArg(ka, 2, sizeof uK, &uK);
    clSetKernelArg(kb, 0, sizeof dB, &dB); clSetKernelArg(kb, 1, sizeof dBp, &dBp);
    clSetKernelArg(kb, 2, sizeof uN, &uN);
    clSetKernelArg(kg, 0, sizeof dAp, &dAp); clSetKernelArg(kg, 1, sizeof dBp, &dBp);
    clSetKernelArg(kg, 2, sizeof dC, &dC);   clSetKernelArg(kg, 3, sizeof uM, &uM);
    clSetKernelArg(kg, 4, sizeof uN, &uN);   clSetKernelArg(kg, 5, sizeof uK, &uK);
    size_t gpa = na, gpb = nb, lp = 8, gws = 256, lws = 256;
    clEnqueueNDRangeKernel(q, ka, 1, 0, &gpa, &lp, 0, 0, 0);
    clEnqueueNDRangeKernel(q, kb, 1, 0, &gpb, &lp, 0, 0, 0);
    if (clEnqueueNDRangeKernel(q, kg, 1, 0, &gws, &lws, 0, 0, 0) != CL_SUCCESS) {
        printf("gemm launch failed\n"); return 2;
    }
    clFinish(q);
    clEnqueueReadBuffer(q, dC, CL_TRUE, 0, nc * 4, hC, 0, 0, 0);

    printf("tf32 XMX operand rounding (C[0][n] = engine(v_n * 1.0)):\n");
    printf("  %-14s %-10s %-14s %-10s  %s\n", "input", "bits", "engine out", "bits", "kept mantissa bits");
    int minkept = 24;
    for (int n = 0; n < 8; n++) {
        float got = hC[n];
        unsigned bi = bits(v[n]), bo = bits(got);
        /* how many low mantissa bits were zeroed? */
        int kept = 23; unsigned m = bo & 0x7fffffu;
        while (kept > 0 && !(m & (1u << (23 - kept)))) kept--;
        printf("  %-14.9g %08x %-14.9g %08x  %s\n", v[n], bi, got, bo,
               bi == bo ? "(unchanged)" : "");
        (void)kept;
        /* count trailing zeros of the output mantissa as an upper bound on width */
        int tz = 0; while (tz < 23 && !((bo >> tz) & 1u)) tz++;
        if (23 - tz < minkept) minkept = 23 - tz;
    }
    printf("\nsmallest observed mantissa width in the outputs: %d explicit bits\n", minkept);

    /* second experiment: is only ONE operand rounded? feed v on the B side */
    memset(hA, 0, na * 4); memset(hB, 0, nb * 4);
    for (int k = 0; k < 8; k++) hA[(size_t)0 * K + k] = (k == 0) ? 1.0f : 0.0f;
    for (int n = 0; n < 8; n++) hB[(size_t)0 * N + n] = v[n];   /* B[0][n] = v_n */
    clEnqueueWriteBuffer(q, dA, CL_TRUE, 0, na * 4, hA, 0, 0, 0);
    clEnqueueWriteBuffer(q, dB, CL_TRUE, 0, nb * 4, hB, 0, 0, 0);
    clEnqueueNDRangeKernel(q, ka, 1, 0, &gpa, &lp, 0, 0, 0);
    clEnqueueNDRangeKernel(q, kb, 1, 0, &gpb, &lp, 0, 0, 0);
    clEnqueueNDRangeKernel(q, kg, 1, 0, &gws, &lws, 0, 0, 0);
    clFinish(q);
    clEnqueueReadBuffer(q, dC, CL_TRUE, 0, nc * 4, hC, 0, 0, 0);
    printf("\nsame probe with the values on the B side (A = e_0):\n");
    for (int n = 0; n < 8; n++)
        printf("  %-14.9g %08x -> %-14.9g %08x\n", v[n], bits(v[n]), hC[n], bits(hC[n]));
    return 0;
}
