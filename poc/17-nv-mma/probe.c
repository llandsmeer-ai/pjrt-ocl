/* poc/17 cp.async feasibility host: run every probe.cl variant, report which
 * (if any) delivers the data. Exit 0 iff at least one cp.async form is CORRECT
 * (the go/no-go for a deep async GEMM pipeline on this ICD). §35. */
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(2); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(*n + 1); if (fread(b, 1, *n, f) != *n) exit(2);
    b[*n] = 0; fclose(f); return b;
}

/* one build+run per fresh process avoids in-process JIT-cache collisions that
 * spuriously failed rapid successive builds; we exec ourselves per variant. */
static int run_one(const char *opts) {
    cl_platform_id plats[8]; cl_uint np = 0;
    clGetPlatformIDs(8, plats, &np);
    cl_platform_id plat = 0; char pn[256];
    for (cl_uint i = 0; i < np; i++) {
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof pn, pn, 0);
        if (strstr(pn, "NVIDIA")) { plat = plats[i]; break; }
    }
    if (!plat) { fprintf(stderr, "no NVIDIA platform\n"); exit(2); }
    cl_device_id dev; clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, 0);
    cl_int err; cl_context ctx = clCreateContext(0, 1, &dev, 0, 0, &err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    size_t sl; char *src = slurp("probe.cl", &sl);
    const int N = 256 * 4;
    float *in = malloc(N * 4), *out = malloc(N * 4);
    for (int i = 0; i < N; i++) in[i] = (float)(i * 3 + 1);
    cl_mem gin = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * 4, 0, &err);
    cl_mem gout = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N * 4, 0, &err);
    cl_program prog = clCreateProgramWithSource(ctx, 1,
        (const char **)&src, &sl, &err);
    err = clBuildProgram(prog, 1, &dev, opts, 0, 0);
    if (err != CL_SUCCESS) {
        char log[16384];
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, 0);
        printf("  %-12s BUILD FAILED %d: %.200s\n", opts[0]?opts:"(scalar)", err, log);
        return -1;
    }
    cl_kernel k = clCreateKernel(prog, "probe", &err);
    clSetKernelArg(k, 0, sizeof gin, &gin);
    clSetKernelArg(k, 1, sizeof gout, &gout);
    clEnqueueWriteBuffer(q, gin, CL_TRUE, 0, N * 4, in, 0, 0, 0);
    float z = 0; clEnqueueFillBuffer(q, gout, &z, 4, 0, N * 4, 0, 0, 0);
    clFinish(q);
    size_t g = 256, l = 256;
    if (clEnqueueNDRangeKernel(q, k, 1, 0, &g, &l, 0, 0, 0) != CL_SUCCESS) {
        printf("  %-12s launch failed\n", opts); return -1; }
    clFinish(q);
    clEnqueueReadBuffer(q, gout, CL_TRUE, 0, N * 4, out, 0, 0, 0);
    int bad = 0; for (int i = 0; i < N; i++) if (out[i] != in[i]) bad++;
    printf("  %-12s %s (%d/%d mismatch)\n", opts[0]?opts:"(scalar)",
           bad ? "WRONG" : "CORRECT", bad, N);
    return bad == 0 ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc > 1) return run_one(argv[1]);   /* child: one variant */
    const char *vs[] = { "", "-DV_STSHARED", "-DV_CG", "-DV_CA",
                         "-DV_FENCE", "-DV_SPIN", "-DV_MBAR" };
    int any_async_ok = 0, ctrl_ok = 0;
    char cmd[512];
    for (int i = 0; i < 7; i++) {
        snprintf(cmd, sizeof cmd, "%s '%s'", argv[0], vs[i]);
        int rc = system(cmd);
        rc = WEXITSTATUS(rc);
        if (i >= 2 && rc == 0) any_async_ok = 1;    /* a cp.async form */
        if (i == 1 && rc == 0) ctrl_ok = 1;         /* st.shared control */
    }
    printf("\nCONTROL st.shared through same address: %s\n",
           ctrl_ok ? "CORRECT (address mapping is fine)" : "WRONG");
    printf("VERDICT: %s\n", any_async_ok
        ? "a cp.async form WORKS -> deep async pipeline feasible"
        : "NO cp.async form delivers data -> async pipeline NOT expressible on this ICD");
    return any_async_ok ? 0 : 1;
}
