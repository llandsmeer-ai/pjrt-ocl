/* poc/21 — XMX (DPAS) GEMM feasibility benchmark on Intel Xe2.
 *
 * Sweeps sub-group tile geometries for the bf16 / f16 / tf32
 * cl_intel_subgroup_matrix_multiply_accumulate GEMMs (xmx21.cl, xmx21tf32.cl),
 * times the kernel (min of 8 after warmup) and the f32 -> tile-order packing
 * pre-pass separately, and checks the result against TWO CPU references:
 *   exact   : f64 accumulation of the original f32 operands
 *   rounded : f64 accumulation of the operands after RTE rounding to the
 *             engine's input format (bf16 / f16 / tf32)
 * The gap to `rounded` is the correctness signal (a layout or accumulate bug
 * shows up here); the gap to `exact` is the precision cost of the engine.
 *
 * Usage: ./bench21 [N] [platform-substr]        sweep, sampled check
 *        ./bench21 [N] [platform-substr] full   one config/variant, ALL cells
 *                                               checked against a full O(N^3)
 *                                               f64 reference                */
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
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(n + 1); if (fread(b, 1, n, f) != (size_t)n) exit(2);
    b[n] = 0; fclose(f); return b;
}

/* ---- host-side reference roundings (round-to-nearest-even) ---- */
static float rte_bf16(float f) {                 /* 8 explicit mantissa bits */
    union { float f; unsigned u; } v = { .f = f };
    if (((v.u >> 23) & 0xff) == 0xff) return f;
    unsigned lsb = (v.u >> 16) & 1u;
    v.u = (v.u + 0x7fffu + lsb) & 0xffff0000u;
    return v.f;
}
/* round f32 to `mb` explicit mantissa bits, round-to-nearest-even */
static float rte_mant(float f, int mb) {
    union { float f; unsigned u; } v = { .f = f };
    if (((v.u >> 23) & 0xff) == 0xff || mb >= 23) return f;
    int sh = 23 - mb;
    unsigned lsb = (v.u >> sh) & 1u, half = (1u << (sh - 1)) - 1u;
    v.u = (v.u + half + lsb) & ~((1u << sh) - 1u);
    return v.f;
}
/* truncate f32 to `mb` explicit mantissa bits (round toward zero) */
static float rtz_mant(float f, int mb) {
    union { float f; unsigned u; } v = { .f = f };
    if (((v.u >> 23) & 0xff) == 0xff || mb >= 23) return f;
    v.u &= ~((1u << (23 - mb)) - 1u);
    return v.f;
}
/* Xe2 tf32 DPAS: 10 explicit mantissa bits, TRUNCATED (see tf32bits.c) */
static float rte_tf32(float f) { return rtz_mant(f, 10); }
static float rte_f16(float f) {                  /* 11 bits, exp range 5 bits */
    union { float f; unsigned u; } v = { .f = f };
    unsigned s = v.u >> 31, e = (v.u >> 23) & 0xff, m = v.u & 0x7fffff;
    if (e == 0xff) return f;
    int ne = (int)e - 127 + 15;
    if (ne >= 31) return s ? -INFINITY : INFINITY;
    unsigned hm;
    if (ne <= 0) {
        if (ne < -10) { v.u = s << 31; return v.f; }
        m |= 0x800000u;
        int sh = 14 - ne;
        hm = m >> sh;
        unsigned rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (hm & 1))) hm++;
        double d = ldexp((double)hm, -24);
        return (float)(s ? -d : d);
    }
    hm = m >> 13;
    unsigned rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1))) {
        hm++;
        if (hm == 0x400u) { hm = 0; ne++; if (ne >= 31) return s ? -INFINITY : INFINITY; }
    }
    v.u = (s << 31) | ((unsigned)(ne - 15 + 127) << 23) | (hm << 13);
    return v.f;
}

struct cfg { int sgm, sgn, rm, rn; };
struct var { const char *name, *file, *opt; int elem; float (*rnd)(float); };

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 2048;
    const char *psub = argc > 2 ? argv[2] : "Intel";
    int full = argc > 3 && strstr(argv[3], "full") != 0;
    /* "pos": non-negative operands. The Xe2 tf32 path TRUNCATES its operands
     * (tf32bits.c), so its error is biased toward zero; with sign-symmetric
     * data the bias cancels and error grows like sqrt(K), with all-positive
     * data (post-ReLU activations x positive weights) it grows like K. */
    int posdata = argc > 3 && strstr(argv[3], "pos") != 0;
    int cont = argc > 3 && strstr(argv[3], "cont") != 0;
    int M = N, K = N;

    cl_platform_id plats[8]; cl_uint np = 0;
    clGetPlatformIDs(8, plats, &np);
    cl_platform_id plat = 0; char pn[256];
    for (cl_uint i = 0; i < np; i++) {
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof pn, pn, 0);
        if (strstr(pn, psub)) { plat = plats[i]; break; }
    }
    if (!plat) { fprintf(stderr, "no platform matching '%s'\n", psub); return 2; }
    cl_device_id dev; clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, 0);
    char dn[256]; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof dn, dn, 0);
    cl_uint cu = 0; size_t maxwg = 0; cl_ulong slm = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, 0);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof maxwg, &maxwg, 0);
    clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof slm, &slm, 0);
    printf("device: %s | CUs=%u maxWG=%zu SLM=%lluKB | N=%d%s\n",
           dn, cu, maxwg, (unsigned long long)(slm / 1024), N,
           full ? "  [FULL all-cell verification]" : "");

    cl_int e;
    cl_context ctx = clCreateContext(0, 1, &dev, 0, 0, &e);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &e);

    size_t nelem = (size_t)N * N, fbytes = nelem * 4;
    float *hA = malloc(fbytes), *hB = malloc(fbytes), *hC = malloc(fbytes);
    srand(1);
    for (size_t i = 0; i < nelem; i++) {
        if (cont) {   /* continuous: full 23-bit mantissas, unbiased rounding */
            double ua = rand() / (double)RAND_MAX, ub = rand() / (double)RAND_MAX;
            hA[i] = (float)(posdata ? 1.6 * ua : 1.6 * ua - 0.8);
            hB[i] = (float)(posdata ? 1.6 * ub : 1.6 * ub - 0.8);
        } else {      /* poc/19's discrete set, for speed comparability */
            int o = posdata ? 0 : 8;
            hA[i] = (rand() % 17 - o) * 0.1f; hB[i] = (rand() % 17 - o) * 0.1f;
        }
    }
    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fbytes, hA, &e);
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fbytes, hB, &e);
    cl_mem dAp = clCreateBuffer(ctx, CL_MEM_READ_WRITE, fbytes, 0, &e);  /* 4B/elem worst case */
    cl_mem dBp = clCreateBuffer(ctx, CL_MEM_READ_WRITE, fbytes, 0, &e);
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, fbytes, 0, &e);

    struct var vars[] = {
        { "bf16", "xmx21.cl",     "-DVAR_BF16", 2, rte_bf16 },
        { "f16",  "xmx21.cl",     "-DVAR_F16",  2, rte_f16  },
        { "tf32", "xmx21tf32.cl", "",           4, rte_tf32 },
    };
    const int nvar = 3;

    struct cfg cfgs[] = {
        {2, 4, 4, 2}, {4, 4, 2, 2}, {4, 4, 4, 2}, {4, 4, 2, 4},
        {2, 4, 4, 4}, {2, 2, 4, 4}, {4, 2, 4, 4}, {8, 2, 2, 2},
        {4, 4, 1, 1}, {2, 8, 4, 2}, {4, 4, 4, 4}, {1, 4, 8, 2},
    };
    int ncfg = full ? 1 : (int)(sizeof cfgs / sizeof cfgs[0]);
    if (full) cfgs[0] = (struct cfg){ 4, 4, 4, 2 };   /* best from the sweep */

    /* references */
#define NCHK 256
    int cr[NCHK], cc[NCHK];
    double ref_exact[NCHK], ref_rnd[3][NCHK];
    double *fexact = 0, *frnd[3] = { 0, 0, 0 };
    if (full) {
        fexact = malloc(nelem * sizeof(double));
        for (int v = 0; v < nvar; v++) frnd[v] = malloc(nelem * sizeof(double));
        float *Ar = malloc(fbytes), *Br = malloc(fbytes);
        printf("computing full O(N^3) f64 references (4 passes)...\n");
        for (int v = -1; v < nvar; v++) {
            double *dst = v < 0 ? fexact : frnd[v];
            for (size_t i = 0; i < nelem; i++) {
                Ar[i] = v < 0 ? hA[i] : vars[v].rnd(hA[i]);
                Br[i] = v < 0 ? hB[i] : vars[v].rnd(hB[i]);
            }
            for (int i = 0; i < M; i++) {
                double *cd = dst + (size_t)i * N;
                for (int j = 0; j < N; j++) cd[j] = 0;
                for (int k = 0; k < K; k++) {
                    double a = Ar[(size_t)i * K + k];
                    const float *br = Br + (size_t)k * N;
                    for (int j = 0; j < N; j++) cd[j] += a * br[j];
                }
            }
        }
        free(Ar); free(Br);
    } else {
        for (int t = 0; t < NCHK; t++) { cr[t] = rand() % M; cc[t] = rand() % N; }
        for (int t = 0; t < NCHK; t++) {
            double se = 0, sv[3] = { 0, 0, 0 };
            const float *ar = hA + (size_t)cr[t] * K;
            for (int k = 0; k < K; k++) {
                float a = ar[k], b = hB[(size_t)k * N + cc[t]];
                se += (double)a * (double)b;
                for (int v = 0; v < nvar; v++)
                    sv[v] += (double)vars[v].rnd(a) * (double)vars[v].rnd(b);
            }
            ref_exact[t] = se;
            for (int v = 0; v < nvar; v++) ref_rnd[v][t] = sv[v];
        }
    }

    printf("  %-6s %-26s %8s %9s %8s | %-10s %-8s %-10s %s\n", "type",
           "SGMxSGN RMxRN (WMxWN)", "ms", "GFLOP/s", "pack ms",
           "maxerr/f32", "bias", "err vs rnd", "cells");

    for (int v = 0; v < nvar; v++) {
        char *src = slurp(vars[v].file);
        double packms = -1;
        for (int c = 0; c < ncfg; c++) {
            struct cfg g = cfgs[c];
            int nwi = g.sgm * g.sgn * 16;
            int wm = g.sgm * g.rm * 8, wn = g.sgn * g.rn * 16;
            int kstep = vars[v].elem == 4 ? 8 : 16;
            if (nwi > (int)maxwg) continue;
            if (M % wm || N % wn || K % kstep) continue;

            char opts[320];
            snprintf(opts, sizeof opts,
                     "-cl-std=CL2.0 %s -DSGM=%d -DSGN=%d -DRM=%d -DRN=%d",
                     vars[v].opt, g.sgm, g.sgn, g.rm, g.rn);
            cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **)&src, 0, &e);
            e = clBuildProgram(prog, 1, &dev, opts, 0, 0);
            if (e != CL_SUCCESS) {
                size_t ln = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, 0, &ln);
                char *log = malloc(ln + 1);
                clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, ln, log, 0); log[ln] = 0;
                for (char *p = log; *p; p++) if (*p == '\n') *p = ' ';
                printf("  build FAIL %s %d/%d %d/%d: %.200s\n", vars[v].name,
                       g.sgm, g.sgn, g.rm, g.rn, log);
                free(log); clReleaseProgram(prog); continue;
            }

            cl_kernel ka = clCreateKernel(prog, "pack_a", &e);
            cl_kernel kb = clCreateKernel(prog, "pack_b", &e);
            cl_uint uK = K, uN = N, uM = M;
            clSetKernelArg(ka, 0, sizeof dA, &dA); clSetKernelArg(ka, 1, sizeof dAp, &dAp);
            clSetKernelArg(ka, 2, sizeof uK, &uK);
            clSetKernelArg(kb, 0, sizeof dB, &dB); clSetKernelArg(kb, 1, sizeof dBp, &dBp);
            clSetKernelArg(kb, 2, sizeof uN, &uN);
            size_t gp = nelem, lp = 256;
            for (int w = 0; w < 3; w++) {
                clEnqueueNDRangeKernel(q, ka, 1, 0, &gp, &lp, 0, 0, 0);
                clEnqueueNDRangeKernel(q, kb, 1, 0, &gp, &lp, 0, 0, 0);
            }
            clFinish(q);
            {
                double best = 1e30;
                for (int r = 0; r < 5; r++) {
                    double t0 = now_ms();
                    clEnqueueNDRangeKernel(q, ka, 1, 0, &gp, &lp, 0, 0, 0);
                    clEnqueueNDRangeKernel(q, kb, 1, 0, &gp, &lp, 0, 0, 0);
                    clFinish(q);
                    double dt = now_ms() - t0; if (dt < best) best = dt;
                }
                if (packms < 0) packms = best;
            }

            cl_kernel kg = clCreateKernel(prog, "gemm", &e);
            clSetKernelArg(kg, 0, sizeof dAp, &dAp); clSetKernelArg(kg, 1, sizeof dBp, &dBp);
            clSetKernelArg(kg, 2, sizeof dC, &dC);   clSetKernelArg(kg, 3, sizeof uM, &uM);
            clSetKernelArg(kg, 4, sizeof uN, &uN);   clSetKernelArg(kg, 5, sizeof uK, &uK);
            size_t wgs = (size_t)(M / wm) * (N / wn);
            size_t gws = wgs * nwi, lws = nwi;

            float zero = 0;
            clEnqueueFillBuffer(q, dC, &zero, 4, 0, fbytes, 0, 0, 0);
            int bad = 0;
            /* long warmup: the Lunar Lake iGPU needs ~10s of load to reach a
             * stable clock; 3 iterations gave 2x run-to-run spread. */
            double tw = now_ms();
            while (now_ms() - tw < 300.0) {
                if (clEnqueueNDRangeKernel(q, kg, 1, 0, &gws, &lws, 0, 0, 0) != CL_SUCCESS) {
                    printf("  launch FAIL %s %dx%d %dx%d\n", vars[v].name,
                           g.sgm, g.sgn, g.rm, g.rn);
                    bad = 1; break;
                }
                clFinish(q);
            }
            if (bad) { clReleaseProgram(prog); continue; }
            clFinish(q);
            double best = 1e30;
            for (int r = 0; r < 12; r++) {
                double t0 = now_ms();
                clEnqueueNDRangeKernel(q, kg, 1, 0, &gws, &lws, 0, 0, 0);
                clFinish(q);
                double dt = now_ms() - t0; if (dt < best) best = dt;
            }
            clEnqueueReadBuffer(q, dC, CL_TRUE, 0, fbytes, hC, 0, 0, 0);

            double emax = 0, emax_r = 0, bias = 0, mag = 0;
            size_t ncell;
            if (full) {
                ncell = nelem;
                for (size_t i = 0; i < nelem; i++) {
                    double got = hC[i];
                    double d = fabs(got - fexact[i]), dr = fabs(got - frnd[v][i]);
                    if (d > emax) emax = d;
                    if (dr > emax_r) emax_r = dr;
                    bias += got - fexact[i]; mag += fabs(fexact[i]);
                }
            } else {
                ncell = NCHK;
                for (int t = 0; t < NCHK; t++) {
                    double got = hC[(size_t)cr[t] * N + cc[t]];
                    double d = fabs(got - ref_exact[t]);
                    double dr = fabs(got - ref_rnd[v][t]);
                    if (d > emax) emax = d;
                    if (dr > emax_r) emax_r = dr;
                    bias += got - ref_exact[t]; mag += fabs(ref_exact[t]);
                }
            }
            bias = mag > 0 ? bias / mag : 0;   /* mean SIGNED error / mean |C| */
            double gf = 2.0 * M * N * K / (best * 1e-3) / 1e9;
            char tag[64];
            snprintf(tag, sizeof tag, "%dx%d %dx%d (%dx%d) %dwi",
                     g.sgm, g.sgn, g.rm, g.rn, wm, wn, nwi);
            printf("  %-6s %-26s %8.3f %9.1f %8.3f | %.2e %+.1e %.2e %s %zu\n",
                   vars[v].name, tag, best, gf, packms, emax, bias, emax_r,
                   emax_r < 2e-3 ? "ok " : "BAD", ncell);

            clReleaseKernel(ka); clReleaseKernel(kb); clReleaseKernel(kg);
            clReleaseProgram(prog);
        }
        /* Which input rounding does the engine actually apply? Fit the last
         * config's output against "round f32 operands to mb mantissa bits". */
        if (!full) {
            int bestmb = -1, bestrz = 0; double bestfit = 1e30;
            for (int rz = 0; rz < 2; rz++) for (int mb = 6; mb <= 23; mb++) {
                double emax = 0;
                for (int t = 0; t < NCHK; t++) {
                    double s = 0;
                    const float *ar = hA + (size_t)cr[t] * K;
                    for (int k = 0; k < K; k++) {
                        float a = ar[k], b = hB[(size_t)k * N + cc[t]];
                        s += (double)(rz ? rtz_mant(a, mb) : rte_mant(a, mb)) *
                             (double)(rz ? rtz_mant(b, mb) : rte_mant(b, mb));
                    }
                    double d = fabs(hC[(size_t)cr[t] * N + cc[t]] - s);
                    if (d > emax) emax = d;
                }
                if (emax < bestfit) { bestfit = emax; bestmb = mb; bestrz = rz; }
            }
            printf("    operand-format fit (%s): %d explicit mantissa bits, %s"
                   "  (residual %.2e)\n", vars[v].name, bestmb,
                   bestrz ? "truncated (RTZ)" : "round-to-nearest-even", bestfit);
        }
        free(src);
        printf("\n");
    }

    double refmax = 0;
    if (full) { for (size_t i = 0; i < nelem; i++) if (fabs(fexact[i]) > refmax) refmax = fabs(fexact[i]); }
    else      { for (int t = 0; t < NCHK; t++) if (fabs(ref_exact[t]) > refmax) refmax = fabs(ref_exact[t]); }
    printf("max |C_exact| over the checked cells = %.4f  (divide abs errors by this for relative)\n", refmax);
    return 0;
}
