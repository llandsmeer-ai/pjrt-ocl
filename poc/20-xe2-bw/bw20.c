/* poc/20: what streaming bandwidth is actually ACHIEVABLE on Xe2?
 *
 * Our elementwise/dynamic_slice ops all plateau at ~75 GB/s while LPDDR5X on
 * Lunar Lake has ~136 GB/s of theoretical peak. Before tuning the VM's EW
 * kernel it is worth knowing the real ceiling: a bare triad (c = a + b) swept
 * over vector width (float1/2/4/8/16) and work-group size, plus a pure copy and
 * a read-only reduction to separate read from write cost.
 *
 * Usage: ./bw20 [MiB] [platform-substr]                                     */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

/* One source, three kernels, vector width as a -DVW build option. Grid-stride
 * so the launch geometry is independent of N. */
static const char *SRC =
"#if VW == 1\n"
"#define VT float\n#define VLOAD(p,i) ((p)[i])\n#define VSTORE(p,i,v) ((p)[i]=(v))\n"
"#else\n"
"#define VT CAT(float,VW)\n#define CAT_(a,b) a##b\n#define CAT(a,b) CAT_(a,b)\n"
"#define VLOAD(p,i) (((__global const VT*)(p))[i])\n"
"#define VSTORE(p,i,v) ((((__global VT*)(p))[i])=(v))\n"
"#endif\n"
"__kernel void triad(__global const float *a, __global const float *b,\n"
"                    __global float *c, const uint nv) {\n"
"    for (uint i = get_global_id(0); i < nv; i += get_global_size(0))\n"
"        VSTORE(c, i, VLOAD(a,i) + VLOAD(b,i));\n"
"}\n"
"__kernel void copyk(__global const float *a, __global float *c,\n"
"                    const uint nv) {\n"
"    for (uint i = get_global_id(0); i < nv; i += get_global_size(0))\n"
"        VSTORE(c, i, VLOAD(a,i));\n"
"}\n"
"__kernel void readk(__global const float *a, __global float *c,\n"
"                    const uint nv) {\n"
"    VT acc = (VT)(0.0f);\n"
"    for (uint i = get_global_id(0); i < nv; i += get_global_size(0))\n"
"        acc += VLOAD(a,i);\n"
"    if (get_global_id(0) == 0xffffffffu) VSTORE(c, 0, acc);\n"
"}\n";

int main(int argc, char **argv) {
    size_t mib = argc > 1 ? (size_t)atoi(argv[1]) : 256;
    const char *psub = argc > 2 ? argv[2] : "Intel";
    size_t bytes = mib << 20, n = bytes / 4;

    cl_platform_id plats[8]; cl_uint np = 0;
    clGetPlatformIDs(8, plats, &np);
    cl_platform_id plat = 0; char nm[256];
    for (cl_uint i = 0; i < np; i++) {
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof nm, nm, 0);
        if (strstr(nm, psub)) { plat = plats[i]; break; }
    }
    if (!plat) { fprintf(stderr, "no platform '%s'\n", psub); return 1; }
    cl_device_id dev; clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, 0);
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof nm, nm, 0);
    cl_uint cu = 0, mhz = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, 0);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof mhz, &mhz, 0);
    printf("device: %s (CU=%u, %u MHz) | array %zu MiB\n", nm, cu, mhz, mib);

    cl_int e;
    cl_context ctx = clCreateContext(0, 1, &dev, 0, 0, &e);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &e);
    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, 0, &e);
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, 0, &e);
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, 0, &e);
    float pat = 1.0f;
    clEnqueueFillBuffer(q, dA, &pat, 4, 0, bytes, 0, 0, 0);
    clEnqueueFillBuffer(q, dB, &pat, 4, 0, bytes, 0, 0, 0);
    clFinish(q);

    const int vws[] = {1, 2, 4, 8, 16};
    const size_t lszs[] = {64, 128, 256, 512};
    printf("  %-8s %-6s %10s %10s %10s\n", "kernel", "vw", "lsz", "ms", "GB/s");
    struct { const char *name; int nbuf; } ks[] = {
        {"triad", 3}, {"copyk", 2}, {"readk", 1}};

    for (int ki = 0; ki < 3; ki++) {
        double best_gb = 0; int best_vw = 0; size_t best_l = 0;
        for (int vi = 0; vi < 5; vi++) {
            int vw = vws[vi];
            if (n % (size_t)vw) continue;
            char opts[64]; snprintf(opts, sizeof opts, "-DVW=%d", vw);
            cl_program pr = clCreateProgramWithSource(ctx, 1, &SRC, 0, &e);
            if (clBuildProgram(pr, 1, &dev, opts, 0, 0) != CL_SUCCESS) {
                size_t ln; clGetProgramBuildInfo(pr, dev, CL_PROGRAM_BUILD_LOG, 0,0,&ln);
                char *lg = malloc(ln+1); clGetProgramBuildInfo(pr, dev, CL_PROGRAM_BUILD_LOG, ln, lg, 0);
                lg[ln]=0; printf("  build fail vw=%d: %.150s\n", vw, lg); free(lg);
                clReleaseProgram(pr); continue;
            }
            cl_kernel k = clCreateKernel(pr, ks[ki].name, &e);
            cl_uint nv = (cl_uint)(n / (size_t)vw);
            if (ks[ki].nbuf == 3) {
                clSetKernelArg(k,0,sizeof dA,&dA); clSetKernelArg(k,1,sizeof dB,&dB);
                clSetKernelArg(k,2,sizeof dC,&dC); clSetKernelArg(k,3,sizeof nv,&nv);
            } else {
                clSetKernelArg(k,0,sizeof dA,&dA); clSetKernelArg(k,1,sizeof dC,&dC);
                clSetKernelArg(k,2,sizeof nv,&nv);
            }
            for (int li = 0; li < 4; li++) {
                size_t lsz = lszs[li];
                size_t gsz = (size_t)cu * 8 * lsz;   /* persistent-ish grid */
                if (gsz > nv) gsz = ((nv + lsz - 1) / lsz) * lsz;
                for (int w = 0; w < 3; w++)
                    clEnqueueNDRangeKernel(q,k,1,0,&gsz,&lsz,0,0,0);
                clFinish(q);
                double bestms = 1e30;
                for (int r = 0; r < 10; r++) {
                    double t0 = now_ms();
                    clEnqueueNDRangeKernel(q,k,1,0,&gsz,&lsz,0,0,0);
                    clFinish(q);
                    double dt = now_ms()-t0; if (dt<bestms) bestms=dt;
                }
                double moved = (double)bytes * ks[ki].nbuf;
                double gb = moved / (bestms*1e-3) / 1e9;
                if (gb > best_gb) { best_gb=gb; best_vw=vw; best_l=lsz; }
                if (getenv("BW_VERBOSE"))
                    printf("  %-8s vw=%-3d lsz=%-5zu %8.3f ms %8.1f GB/s\n",
                           ks[ki].name, vw, lsz, bestms, gb);
            }
            clReleaseKernel(k); clReleaseProgram(pr);
        }
        printf("  %-8s best vw=%-3d lsz=%-4zu -> %8.1f GB/s\n",
               ks[ki].name, best_vw, best_l, best_gb);
    }
    return 0;
}
