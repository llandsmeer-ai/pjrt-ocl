/* poc/05: can two DIFFERENT kernels, launched concurrently on separate
 * in-order queues of the SAME device, stay co-resident and synchronize
 * through atomic flags in a shared buffer? (docs/tile-isa.md ceiling-1:
 * "typed lanes" = per-op-family kernels with separate register budgets,
 * viable only if the driver co-schedules them.)
 *
 * One process = one (platform, GA, GB, rounds) config; run the matrix via
 * run_matrix.sh, each invocation wrapped in `timeout 60`.
 *
 * kernel_a = "mma-ish" (GA groups x 256), kernel_b = "ew-ish" (GB groups x
 * 256). Each round: dummy compute -> SIGNAL own flag -> WAIT on partner's
 * flag (exactly the poc/04 entry_t WAIT/SIGNAL idiom), R=rounds times.
 *
 * Deadlock detection: host polls clGetEventInfo (non-blocking) with a
 * WATCHDOG_S (default 10s) wall-clock budget. On trip we print DEADLOCK and
 * call _exit() immediately -- no further CL calls, so a genuinely stuck
 * spinning kernel cannot hang process teardown. The outer `timeout 60` is
 * the hard backstop if even that fails.
 *
 * Usage: ./poc <platform_substr> <GA> <GB> [rounds=1000]
 * Env: WATCHDOG_S (default 10)
 */
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHK(err, what) \
    do { if ((err) != CL_SUCCESS) { fprintf(stderr, "FAIL %s: %d (line %d)\n", what, (int)(err), __LINE__); exit(1); } } while (0)

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static long envi(const char *k, long d) { const char *v = getenv(k); return v && *v ? atol(v) : d; }

static char *read_file(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(*n + 1); if (fread(b, 1, *n, f) != *n) exit(1);
    b[*n] = 0; fclose(f); return b;
}

/* print the RESULT csv line (also used for the DEADLOCK case) */
static void report(const char *plat, cl_uint ga, cl_uint gb, cl_uint rounds,
                   const char *status, double elapsed_ms, const char *note) {
    double rps = elapsed_ms > 0 ? rounds / (elapsed_ms / 1e3) : 0;
    double ns_per_round = rounds > 0 ? elapsed_ms * 1e6 / rounds : 0;
    printf("RESULT,%s,%u,%u,%u,%s,%.3f,%.1f,%.1f,%s\n",
           plat, ga, gb, rounds, status, elapsed_ms, rps, ns_per_round, note);
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <platform_substr> <GA> <GB> [rounds=1000]\n", argv[0]);
        return 1;
    }
    const char *want = argv[1];
    cl_uint ga = (cl_uint)atol(argv[2]);
    cl_uint gb = (cl_uint)atol(argv[3]);
    cl_uint rounds = (cl_uint)(argc > 4 ? atol(argv[4]) : 1000);
    double watchdog_s = (double)envi("WATCHDOG_S", 10);

    cl_int err;
    cl_platform_id plats[8]; cl_uint np = 0;
    CHK(clGetPlatformIDs(8, plats, &np), "platforms");
    cl_platform_id plat = NULL; char pname[256] = {0};
    for (cl_uint i = 0; i < np; i++) {
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof pname, pname, 0);
        if (strstr(pname, want)) { plat = plats[i]; break; }
    }
    if (!plat) { fprintf(stderr, "no platform '%s'\n", want); return 1; }
    cl_device_id dev;
    CHK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, 0), "device");
    char dname[256]; cl_uint cus;
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof dname, dname, 0);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cus, &cus, 0);
    fprintf(stderr, "%s | %s | %u CUs | GA=%u GB=%u (%.2fx CUs) rounds=%u watchdog=%.0fs\n",
            pname, dname, cus, ga, gb, (double)(ga + gb) / cus, rounds, watchdog_s);

    cl_context ctx = clCreateContext(0, 1, &dev, 0, 0, &err); CHK(err, "ctx");
    /* two SEPARATE in-order queues -- the thing under test. Profiling
     * enabled: host-side wall clock via 2ms-granularity polling (below) is
     * far too coarse for these sub-millisecond kernels -- the real
     * measurement uses device-clock event timestamps. */
    cl_command_queue qa = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err); CHK(err, "qa");
    cl_command_queue qb = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err); CHK(err, "qb");

    size_t srclen; char *src = read_file("kernels.cl", &srclen);
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **)&src, &srclen, &err);
    CHK(err, "prog");
    if (clBuildProgram(prog, 1, &dev, "", 0, 0) != CL_SUCCESS) {
        char log[16384]; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, 0);
        fprintf(stderr, "build:\n%s\n", log); return 1;
    }
    cl_kernel ka = clCreateKernel(prog, "mma_ish", &err); CHK(err, "ka");
    cl_kernel kb = clCreateKernel(prog, "ew_ish", &err); CHK(err, "kb");

    cl_mem scratch_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_float) * (size_t)ga * 256, 0, &err); CHK(err, "scratchA");
    cl_mem scratch_b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_float) * (size_t)gb * 256, 0, &err); CHK(err, "scratchB");
    cl_mem flags_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * 2, 0, &err); CHK(err, "flags");
    cl_uint zero2[2] = {0, 0};
    CHK(clEnqueueWriteBuffer(qa, flags_buf, CL_TRUE, 0, sizeof zero2, zero2, 0, 0, 0), "w flags");

    clSetKernelArg(ka, 0, sizeof scratch_a, &scratch_a);
    clSetKernelArg(ka, 1, sizeof flags_buf, &flags_buf);
    clSetKernelArg(ka, 2, sizeof rounds, &rounds);
    clSetKernelArg(ka, 3, sizeof ga, &ga);
    clSetKernelArg(ka, 4, sizeof gb, &gb);
    clSetKernelArg(kb, 0, sizeof scratch_b, &scratch_b);
    clSetKernelArg(kb, 1, sizeof flags_buf, &flags_buf);
    clSetKernelArg(kb, 2, sizeof rounds, &rounds);
    clSetKernelArg(kb, 3, sizeof ga, &ga);
    clSetKernelArg(kb, 4, sizeof gb, &gb);

    size_t gszA = (size_t)ga * 256, gszB = (size_t)gb * 256, lsz = 256;
    cl_event eva, evb;
    CHK(clEnqueueNDRangeKernel(qa, ka, 1, NULL, &gszA, &lsz, 0, 0, &eva), "launch a");
    CHK(clEnqueueNDRangeKernel(qb, kb, 1, NULL, &gszB, &lsz, 0, 0, &evb), "launch b");
    /* CRITICAL: flush BOTH queues before waiting on either -- an in-order
     * queue's enqueue does not guarantee the command reached the device;
     * without this, waiting on qa's event first can stall submission of qb
     * behind it on some drivers, producing a false deadlock (see NOTES.md). */
    clFlush(qa);
    clFlush(qb);

    double t0 = now_ms();
    cl_int sa = CL_QUEUED, sb = CL_QUEUED;
    for (;;) {
        clGetEventInfo(eva, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof sa, &sa, 0);
        clGetEventInfo(evb, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof sb, &sb, 0);
        if (sa <= CL_COMPLETE && sb <= CL_COMPLETE) break;   /* both done (or error) */
        if (sa < 0 || sb < 0) break;                          /* CL error code */
        double elapsed = now_ms() - t0;
        if (elapsed > watchdog_s * 1e3) {
            fprintf(stderr, "WATCHDOG TRIP after %.0f ms: kernel_a status=%d kernel_b status=%d "
                            "(CL_QUEUED=%d CL_SUBMITTED=%d CL_RUNNING=%d CL_COMPLETE=%d)\n",
                    elapsed, sa, sb, CL_QUEUED, CL_SUBMITTED, CL_RUNNING, CL_COMPLETE);
            report(want, ga, gb, rounds, "DEADLOCK", elapsed, "watchdog-trip");
            /* Do NOT call clFinish/clReleaseContext: a still-spinning
             * workgroup could hang cleanup. _exit() bypasses all of that. */
            _exit(2);
        }
        struct timespec ns = {0, 2 * 1000 * 1000}; /* 2ms poll interval */
        nanosleep(&ns, NULL);
    }
    double elapsed_ms = now_ms() - t0;
    if (sa < 0 || sb < 0) {
        report(want, ga, gb, rounds, "CL_ERROR", elapsed_ms, "event-error");
        return 1;
    }
    clWaitForEvents(1, &eva);
    clWaitForEvents(1, &evb);

    /* real timing: device-clock event timestamps, not the coarse host poll
     * loop above (whose 2ms granularity swamps sub-millisecond kernels).
     * Span = earliest START to latest END across both kernels, since they
     * run concurrently on separate queues. */
    cl_ulong sta, ena, stb, enb;
    clGetEventProfilingInfo(eva, CL_PROFILING_COMMAND_START, sizeof sta, &sta, 0);
    clGetEventProfilingInfo(eva, CL_PROFILING_COMMAND_END, sizeof ena, &ena, 0);
    clGetEventProfilingInfo(evb, CL_PROFILING_COMMAND_START, sizeof stb, &stb, 0);
    clGetEventProfilingInfo(evb, CL_PROFILING_COMMAND_END, sizeof enb, &enb, 0);
    cl_ulong span_start = sta < stb ? sta : stb;
    cl_ulong span_end = ena > enb ? ena : enb;
    double dev_elapsed_ms = (span_end - span_start) / 1e6;

    /* correctness: both flags must have reached exactly ga*rounds/gb*rounds,
     * and every scratch element must show the full round count (no group
     * exited its loop early / diverged). */
    cl_uint flags_final[2];
    CHK(clEnqueueReadBuffer(qa, flags_buf, CL_TRUE, 0, sizeof flags_final, flags_final, 0, 0, 0), "rd flags");
    int ok = (flags_final[0] == ga * rounds) && (flags_final[1] == gb * rounds);
    float *sca = malloc(sizeof(cl_float) * (size_t)ga * 256);
    float *scb = malloc(sizeof(cl_float) * (size_t)gb * 256);
    CHK(clEnqueueReadBuffer(qa, scratch_a, CL_TRUE, 0, sizeof(cl_float) * (size_t)ga * 256, sca, 0, 0, 0), "rd scratchA");
    CHK(clEnqueueReadBuffer(qb, scratch_b, CL_TRUE, 0, sizeof(cl_float) * (size_t)gb * 256, scb, 0, 0, 0), "rd scratchB");
    for (size_t i = 0; ok && i < (size_t)ga * 256; i++) if (sca[i] != (float)rounds) ok = 0;
    for (size_t i = 0; ok && i < (size_t)gb * 256; i++) if (scb[i] != (float)rounds) ok = 0;
    free(sca); free(scb);

    fprintf(stderr, "flags: A=%u (want %u) B=%u (want %u) correctness=%s "
                    "[host-poll %.3f ms, device-clock %.3f ms]\n",
            flags_final[0], ga * rounds, flags_final[1], gb * rounds, ok ? "PASS" : "FAIL",
            elapsed_ms, dev_elapsed_ms);
    report(want, ga, gb, rounds, "OK", dev_elapsed_ms, ok ? "PASS" : "CORRUPT");

    clReleaseMemObject(scratch_a); clReleaseMemObject(scratch_b); clReleaseMemObject(flags_buf);
    clReleaseKernel(ka); clReleaseKernel(kb); clReleaseProgram(prog);
    clReleaseCommandQueue(qa); clReleaseCommandQueue(qb); clReleaseContext(ctx);
    return ok ? 0 : 1;
}
