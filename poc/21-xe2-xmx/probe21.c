/* poc/21 probe host: report the device extension string (XMX-relevant subset),
 * the supported subgroup sizes, and which matrix_mad builtin signatures build.
 * Usage: ./probe21 [platform-substr]   (default "Intel")                     */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CL_DEVICE_SUB_GROUP_SIZES_INTEL 0x4108

static char *slurp(const char *p) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(n + 1); if (fread(b, 1, n, f) != (size_t)n) exit(2);
    b[n] = 0; fclose(f); return b;
}

int main(int argc, char **argv) {
    const char *psub = argc > 1 ? argv[1] : "Intel";
    cl_platform_id plats[8]; cl_uint np = 0;
    clGetPlatformIDs(8, plats, &np);
    cl_platform_id plat = 0; char pn[256];
    for (cl_uint i = 0; i < np; i++) {
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof pn, pn, 0);
        if (strstr(pn, psub)) { plat = plats[i]; break; }
    }
    if (!plat) { fprintf(stderr, "no platform matching '%s'\n", psub); return 2; }
    cl_device_id dev; clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, 0);

    char dname[256], dver[256], drv[256];
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof dname, dname, 0);
    clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof dver, dver, 0);
    clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof drv, drv, 0);
    cl_uint cu = 0; cl_ulong slm = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, 0);
    clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof slm, &slm, 0);
    printf("platform: %s\ndevice:   %s\nversion:  %s | driver %s | CUs=%u SLM=%lluKB\n",
           pn, dname, dver, drv, cu, (unsigned long long)(slm / 1024));

    size_t el = 0; clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, 0, &el);
    char *ext = malloc(el + 1); clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, el, ext, 0);
    ext[el] = 0;
    const char *want[] = { "cl_intel_subgroup_matrix_multiply_accumulate",
                           "cl_intel_subgroup_matrix_multiply_accumulate_tf32",
                           "cl_intel_subgroups", "cl_intel_subgroups_short",
                           "cl_intel_subgroups_char", "cl_intel_required_subgroup_size",
                           "cl_intel_bfloat16_conversions", "cl_khr_fp16",
                           "cl_intel_subgroup_2d_block_io",
                           "cl_intel_subgroup_local_block_io",
                           "cl_intel_subgroup_buffer_prefetch",
                           "cl_khr_integer_dot_product" };
    printf("\nXMX-relevant extensions:\n");
    for (unsigned i = 0; i < sizeof want / sizeof *want; i++) {
        /* exact token match so "..._accumulate" doesn't match "..._accumulate_tf32" */
        int have = 0; const char *p = ext; size_t wl = strlen(want[i]);
        while ((p = strstr(p, want[i]))) {
            char after = p[wl];
            if (after == ' ' || after == 0) { have = 1; break; }
            p += wl;
        }
        printf("  [%s] %s\n", have ? "yes" : " NO", want[i]);
    }

    size_t sgn = 0;
    if (clGetDeviceInfo(dev, CL_DEVICE_SUB_GROUP_SIZES_INTEL, 0, 0, &sgn) == CL_SUCCESS && sgn) {
        size_t *sgs = malloc(sgn);
        clGetDeviceInfo(dev, CL_DEVICE_SUB_GROUP_SIZES_INTEL, sgn, sgs, 0);
        printf("\nsub-group sizes:");
        for (size_t i = 0; i < sgn / sizeof(size_t); i++) printf(" %zu", sgs[i]);
        printf("\n");
    }

    cl_int e; cl_context ctx = clCreateContext(0, 1, &dev, 0, 0, &e);
    char *src = slurp("probe21.cl");
    const char *vs[] = { "", "V_BF16CVT", "V_BF16_M1", "V_BF16_M8", "V_BF16_M8_SHORT",
                         "V_F16_M8", "V_F16_M8_SHORT", "V_F16_M8_HALF",
                         "V_TF32_M8", "V_TF32_M8_F2", "V_I8_M8" };
    int sgv[] = { 8, 16 };
    printf("\nbuild matrix (compile only):\n");
    for (int s = 0; s < 2; s++) for (unsigned i = 0; i < sizeof vs / sizeof *vs; i++) {
        char opts[256];
        snprintf(opts, sizeof opts, "-DSG=%d%s%s", sgv[s], vs[i][0] ? " -D" : "", vs[i]);
        cl_program pr = clCreateProgramWithSource(ctx, 1, (const char **)&src, 0, &e);
        e = clBuildProgram(pr, 1, &dev, opts, 0, 0);
        if (e == CL_SUCCESS) {
            printf("  sg%-2d %-16s OK\n", sgv[s], vs[i][0] ? vs[i] : "(control)");
        } else {
            size_t ln = 0; clGetProgramBuildInfo(pr, dev, CL_PROGRAM_BUILD_LOG, 0, 0, &ln);
            char *log = malloc(ln + 1);
            clGetProgramBuildInfo(pr, dev, CL_PROGRAM_BUILD_LOG, ln, log, 0); log[ln] = 0;
            for (char *c = log; *c; c++) if (*c == '\n') *c = ' ';
            printf("  sg%-2d %-16s FAIL(%d): %.220s\n", sgv[s],
                   vs[i][0] ? vs[i] : "(control)", e, log);
            free(log);
        }
        clReleaseProgram(pr);
    }
    return 0;
}
