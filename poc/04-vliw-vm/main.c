/* poc/04 host: tick-synchronous VLIW VM (docs/tile-isa.md).
 *
 * Tests:
 *  A  co-schedule: two independent EW tasks in ONE tick on disjoint lanes
 *  B  matmul as MMA tiles (local-memory staged)
 *  C  reduce: partials tick, combine tick
 *  D  calibration + wide graph: naive level packing vs cost-aware packing;
 *     instrumented mode (one launch per tick + CL event profiling) reports
 *     per-tick time and bubble%.
 *
 * env: OCL_PLATFORM (default "NVIDIA"), VM_LANES (default CUs), VM_LOCAL=256
 */
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHK(err, what) \
    do { if ((err) != CL_SUCCESS) { fprintf(stderr, "FAIL %s: %d (line %d)\n", what, (int)(err), __LINE__); exit(1); } } while (0)

#define EW_TS 16384u
#define MMA_T 16u
enum { T_NOP = 0, T_EW = 1, T_MMA = 2, T_RED_PART = 3, T_RED_COMB = 4, T_FILL = 5 };
enum { EW_ADD = 0, EW_MUL = 1, EW_SUB = 2 };
#define NOPC 0xFFFFFFFFu

typedef struct { cl_uint op, dst, a, b, p0, p1, p2, p3; } task_t;
typedef struct { cl_uint task, tile_lo, tile_hi, pad; } cell_t;
typedef struct { cl_uint task, tile_lo, tile_hi, wait_flag, wait_count, signal_flag, pad0, pad1; } entry_t;
#define FLAG_NONE 0xFFFFFFFFu
#define TASK_BARRIER 0xFFFFFFFEu

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static long envi(const char *k, long d) { const char *v = getenv(k); return v && *v ? atol(v) : d; }
static const char *envs(const char *k, const char *d) { const char *v = getenv(k); return v && *v ? v : d; }

static char *read_file(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(*n + 1); if (fread(b, 1, *n, f) != *n) exit(1);
    b[*n] = 0; fclose(f); return b;
}

/* ---- globals kept simple for a PoC ---- */
static cl_context ctx; static cl_command_queue q, qprof;
static cl_kernel kvliw; static cl_mem arena, tasks_buf, sched_buf, bar_buf, inst_buf;
static cl_uint nlanes; static size_t lsz;

#define MAX_TASKS 64
#define MAX_TICKS 512
static task_t tasks[MAX_TASKS]; static cl_uint n_tasks;
static cell_t sched[MAX_TICKS][1024]; static cl_uint n_ticks;

static void sched_clear(void) {
    n_tasks = 0; n_ticks = 0;
    for (int t = 0; t < MAX_TICKS; ++t)
        for (cl_uint l = 0; l < nlanes; ++l)
            sched[t][l] = (cell_t){NOPC, 0, 0, 0};
}
static cl_uint add_task(task_t t) { tasks[n_tasks] = t; return n_tasks++; }

static void run(int instrumented, double *tick_ms /* [n_ticks] or NULL */) {
    (void)0;
    CHK(clEnqueueWriteBuffer(q, tasks_buf, CL_TRUE, 0, sizeof(task_t) * (n_tasks ? n_tasks : 1), tasks, 0, 0, 0), "w tasks");
    cl_uint barinit[3] = {0, 0, 0};
    CHK(clEnqueueWriteBuffer(q, bar_buf, CL_TRUE, 0, sizeof barinit, barinit, 0, 0, 0), "w bar");
    size_t gsz = nlanes * lsz;
    clSetKernelArg(kvliw, 0, sizeof arena, &arena);
    clSetKernelArg(kvliw, 1, sizeof tasks_buf, &tasks_buf);
    clSetKernelArg(kvliw, 4, sizeof nlanes, &nlanes);
    clSetKernelArg(kvliw, 5, sizeof bar_buf, &bar_buf);
    clSetKernelArg(kvliw, 6, sizeof inst_buf, &inst_buf);
    if (!instrumented) {
        /* one launch, whole schedule: flatten [tick][lane] rows */
        cell_t *flat = malloc(sizeof(cell_t) * n_ticks * nlanes);
        for (cl_uint t = 0; t < n_ticks; ++t)
            memcpy(flat + t * nlanes, sched[t], sizeof(cell_t) * nlanes);
        CHK(clEnqueueWriteBuffer(q, sched_buf, CL_TRUE, 0, sizeof(cell_t) * n_ticks * nlanes, flat, 0, 0, 0), "w sched");
        free(flat);
        clSetKernelArg(kvliw, 2, sizeof sched_buf, &sched_buf);
        clSetKernelArg(kvliw, 3, sizeof n_ticks, &n_ticks);
        CHK(clEnqueueNDRangeKernel(q, kvliw, 1, NULL, &gsz, &lsz, 0, 0, 0), "launch");
        CHK(clFinish(q), "finish");
    } else {
        /* one launch PER TICK on the profiling queue: exact tick durations */
        for (cl_uint t = 0; t < n_ticks; ++t) {
            CHK(clEnqueueWriteBuffer(qprof, sched_buf, CL_TRUE, 0, sizeof(cell_t) * nlanes, sched[t], 0, 0, 0), "w tick");
            cl_uint one = 1;
            clSetKernelArg(kvliw, 2, sizeof sched_buf, &sched_buf);
            clSetKernelArg(kvliw, 3, sizeof one, &one);
            cl_event ev;
            CHK(clEnqueueNDRangeKernel(qprof, kvliw, 1, NULL, &gsz, &lsz, 0, 0, &ev), "launch tick");
            CHK(clFinish(qprof), "finish tick");
            cl_ulong t0, t1;
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof t0, &t0, 0);
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof t1, &t1, 0);
            if (tick_ms) tick_ms[t] = (t1 - t0) / 1e6;
            clReleaseEvent(ev);
            /* barrier state consumed once per launch; reset */
            cl_uint barinit2[3] = {0, 0, 0};
            CHK(clEnqueueWriteBuffer(qprof, bar_buf, CL_TRUE, 0, sizeof barinit2, barinit2, 0, 0, 0), "w bar");
        }
    }
}

/* ---- async engine host side ---- */
static cl_kernel kasync;
static cl_mem streams_buf, lane_tab_buf, flags_buf, ranks_buf;
#define MAX_ENTRIES 8192
static entry_t lane_streams[1024][256]; static cl_uint lane_len[1024];

static void streams_clear(void) {
    memset(lane_len, 0, sizeof lane_len);
    n_tasks = 0;
}
static void emit(cl_uint lane, entry_t e) {
    lane_streams[lane][lane_len[lane]++] = e;
}
static void emit_barrier_all(void) {
    for (cl_uint l = 0; l < nlanes; ++l)
        emit(l, (entry_t){TASK_BARRIER, 0, 0, FLAG_NONE, 0, FLAG_NONE});
}

static double run_async(void) {
    entry_t *flat = malloc(sizeof(entry_t) * MAX_ENTRIES);
    cl_uint *tab = malloc(sizeof(cl_uint) * 2 * nlanes);
    cl_uint off = 0;
    for (cl_uint l = 0; l < nlanes; ++l) {
        tab[2 * l] = off; tab[2 * l + 1] = lane_len[l];
        memcpy(flat + off, lane_streams[l], sizeof(entry_t) * lane_len[l]);
        off += lane_len[l];
    }
    CHK(clEnqueueWriteBuffer(q, tasks_buf, CL_TRUE, 0, sizeof(task_t) * (n_tasks ? n_tasks : 1), tasks, 0, 0, 0), "w tasks");
    CHK(clEnqueueWriteBuffer(q, streams_buf, CL_TRUE, 0, sizeof(entry_t) * (off ? off : 1), flat, 0, 0, 0), "w streams");
    CHK(clEnqueueWriteBuffer(q, lane_tab_buf, CL_TRUE, 0, sizeof(cl_uint) * 2 * nlanes, tab, 0, 0, 0), "w tab");
    cl_uint zeros[64] = {0};
    CHK(clEnqueueWriteBuffer(q, flags_buf, CL_TRUE, 0, sizeof zeros, zeros, 0, 0, 0), "w flags");
    CHK(clEnqueueWriteBuffer(q, bar_buf, CL_TRUE, 0, 12, zeros, 0, 0, 0), "w bar");
    free(flat); free(tab);
    size_t gsz = nlanes * lsz;
    clSetKernelArg(kasync, 0, sizeof arena, &arena);
    clSetKernelArg(kasync, 1, sizeof tasks_buf, &tasks_buf);
    clSetKernelArg(kasync, 2, sizeof streams_buf, &streams_buf);
    clSetKernelArg(kasync, 3, sizeof lane_tab_buf, &lane_tab_buf);
    clSetKernelArg(kasync, 4, sizeof flags_buf, &flags_buf);
    clSetKernelArg(kasync, 5, sizeof bar_buf, &bar_buf);
    clSetKernelArg(kasync, 6, sizeof nlanes, &nlanes);
    clSetKernelArg(kasync, 7, sizeof ranks_buf, &ranks_buf);
    double t0 = now_ms();
    CHK(clEnqueueNDRangeKernel(q, kasync, 1, NULL, &gsz, &lsz, 0, 0, 0), "launch async");
    CHK(clFinish(q), "finish async");
    return now_ms() - t0;
}

static void wr(cl_uint off, const float *v, cl_uint n) {
    CHK(clEnqueueWriteBuffer(q, arena, CL_TRUE, off * 4, n * 4, v, 0, 0, 0), "wr");
}
static void rd(cl_uint off, float *v, cl_uint n) {
    CHK(clEnqueueReadBuffer(q, arena, CL_TRUE, off * 4, n * 4, v, 0, 0, 0), "rd");
}

int main(void) {
    cl_int err;
    cl_platform_id plats[8]; cl_uint np = 0;
    CHK(clGetPlatformIDs(8, plats, &np), "platforms");
    const char *want = envs("OCL_PLATFORM", "NVIDIA");
    cl_platform_id plat = NULL; char pname[256];
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
    nlanes = (cl_uint)envi("VM_LANES", cus);
    if (nlanes > 1024) nlanes = 1024;
    lsz = (size_t)envi("VM_LOCAL", 256);
    printf("%s | %s | %u lanes x %zu threads\n", pname, dname, nlanes, lsz);

    ctx = clCreateContext(0, 1, &dev, 0, 0, &err); CHK(err, "ctx");
    q = clCreateCommandQueue(ctx, dev, 0, &err); CHK(err, "q");
    qprof = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err); CHK(err, "qprof");

    size_t srclen; char *src = read_file("vliw.cl", &srclen);
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **)&src, &srclen, &err);
    CHK(err, "prog");
    if (clBuildProgram(prog, 1, &dev, "", 0, 0) != CL_SUCCESS) {
        char log[16384]; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, 0);
        fprintf(stderr, "build:\n%s\n", log); return 1;
    }
    kvliw = clCreateKernel(prog, "vliw", &err); CHK(err, "kernel");

    const size_t ARENA = 40u << 20; /* 160 MB */
    arena = clCreateBuffer(ctx, CL_MEM_READ_WRITE, ARENA * 4, 0, &err); CHK(err, "arena");
    tasks_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(task_t) * MAX_TASKS, 0, &err); CHK(err, "tasks");
    sched_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(cell_t) * MAX_TICKS * 1024, 0, &err); CHK(err, "sched");
    bar_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 12, 0, &err); CHK(err, "bar");
    inst_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * MAX_TICKS * 1024, 0, &err); CHK(err, "inst");

    float *h = malloc(ARENA * 4);
    int bad, total_bad = 0;

    /* ===== A: two independent EW tasks in one tick, disjoint lanes ===== */
    {
        const cl_uint N = 512 * 1024;                 /* 32 tiles each */
        const cl_uint x = 0, y = N, o1 = 2 * N, o2 = 3 * N;
        for (cl_uint i = 0; i < N; i++) h[i] = (float)(i % 101);
        wr(x, h, N);
        for (cl_uint i = 0; i < N; i++) h[i] = (float)(i % 37) + 1.0f;
        wr(y, h, N);
        sched_clear();
        cl_uint tadd = add_task((task_t){T_EW, o1, x, y, EW_ADD, N});
        cl_uint tmul = add_task((task_t){T_EW, o2, x, y, EW_MUL, N});
        const cl_uint tiles = (N + EW_TS - 1) / EW_TS;   /* 32 */
        const cl_uint half = nlanes / 2;
        for (cl_uint l = 0; l < nlanes; ++l) {
            cl_uint grp = l < half ? tadd : tmul;
            cl_uint idx = l < half ? l : l - half;
            cl_uint span = l < half ? half : nlanes - half;
            cl_uint lo = (cl_uint)((unsigned long long)tiles * idx / span);
            cl_uint hi = (cl_uint)((unsigned long long)tiles * (idx + 1) / span);
            if (lo < hi) sched[0][l] = (cell_t){grp, lo, hi, 0};
        }
        n_ticks = 1;
        run(0, NULL);
        bad = 0;
        float *ha = malloc(N * 4), *hb = malloc(N * 4);
        rd(o1, ha, N); rd(o2, hb, N);
        for (cl_uint i = 0; i < N; i++) {
            float a = (float)(i % 101), b = (float)(i % 37) + 1.0f;
            if (ha[i] != a + b || hb[i] != a * b) bad++;
        }
        printf("A co-scheduled EW (1 tick, 2 ops): %s (%d bad)\n", bad ? "FAIL" : "PASS", bad);
        total_bad += bad; free(ha); free(hb);
    }

    /* ===== B: matmul 128x128x128 as 64 MMA tiles ===== */
    {
        const cl_uint M = 128, N = 128, K = 128;
        const cl_uint A = 4u << 20, B = A + M * K, C = B + K * N;
        for (cl_uint i = 0; i < M * K; i++) h[i] = (float)((int)(i % 7) - 3);
        wr(A, h, M * K);
        for (cl_uint i = 0; i < K * N; i++) h[i] = (float)((int)(i % 5) - 2);
        wr(B, h, K * N);
        sched_clear();
        cl_uint tmm = add_task((task_t){T_MMA, C, A, B, M, N, K});
        const cl_uint tiles = (M / MMA_T) * (N / MMA_T);   /* 64 */
        for (cl_uint tile = 0; tile < tiles; ++tile) {
            cl_uint l = tile % nlanes, t = tile / nlanes;
            sched[t][l] = (cell_t){tmm, tile, tile + 1, 0};
            if (t + 1 > n_ticks) n_ticks = t + 1;
        }
        run(0, NULL);
        rd(C, h, M * N);
        bad = 0;
        for (cl_uint r = 0; r < M && bad < 5; r++)
            for (cl_uint c = 0; c < N; c++) {
                float acc = 0;
                for (cl_uint k = 0; k < K; k++)
                    acc += (float)((int)((r * K + k) % 7) - 3) * (float)((int)((k * N + c) % 5) - 2);
                if (h[r * N + c] != acc) bad++;
            }
        printf("B matmul via MMA tiles: %s (%d bad)\n", bad ? "FAIL" : "PASS", bad);
        total_bad += bad;
    }

    /* ===== C: reduce 1M -> partials tick, combine tick ===== */
    {
        const cl_uint N = 1u << 20, CHUNK = 64 * 1024, parts = N / CHUNK; /* 16 */
        const cl_uint X = 8u << 20, P = X + N, O = P + parts;
        for (cl_uint i = 0; i < N; i++) h[i] = 1.0f;
        wr(X, h, N);
        sched_clear();
        cl_uint tp = add_task((task_t){T_RED_PART, P, X, 0, N, CHUNK});
        cl_uint tc = add_task((task_t){T_RED_COMB, O, P, 0, parts});
        for (cl_uint i = 0; i < parts; ++i) {
            cell_t *c = &sched[0][i % nlanes];
            if (c->task == NOPC)
                *c = (cell_t){tp, i, i + 1, 0};
            else
                c->tile_hi = i + 1;  /* extend range (contiguous when lanes>=parts) */
        }
        sched[1][0] = (cell_t){tc, 0, 1, 0};
        n_ticks = 2;
        run(0, NULL);
        rd(O, h, 1);
        bad = h[0] != (float)N;
        printf("C reduce partial+combine: %s (got %.0f want %u)\n", bad ? "FAIL" : "PASS", h[0], N);
        total_bad += bad;
    }

    /* ===== D: calibration + naive vs cost-aware packing ===== */
    {
        /* wide graph: 1 matmul 512x512x512 (1024 MMA tiles) + 8 independent
         * EW adds of 1M elems (64 tiles each) — all independent */
        const cl_uint M = 512;
        const cl_uint A = 4u << 20, B = A + M * M, C = B + M * M;
        const cl_uint EWN = 1u << 20, ewtiles = EWN / EW_TS; /* 64 */
        const cl_uint EWBASE = 8u << 20;
        for (cl_uint i = 0; i < M * M; i++) h[i] = (float)((int)(i % 7) - 3);
        wr(A, h, M * M); wr(B, h, M * M);
        for (cl_uint i = 0; i < EWN; i++) h[i] = (float)(i % 97);
        for (int e = 0; e < 8; e++) wr(EWBASE + e * 3 * EWN, h, EWN);

        /* --- calibrate: µs per EW tile and per MMA tile --- */
        double tick_ms[MAX_TICKS];
        sched_clear();
        cl_uint tew = add_task((task_t){T_EW, EWBASE + 2 * EWN, EWBASE, EWBASE + EWN, EW_ADD, EWN});
        for (cl_uint l = 0; l < nlanes && l < ewtiles; ++l)
            sched[0][l] = (cell_t){tew, l, l + 1, 0};
        n_ticks = 1;
        run(1, tick_ms);
        double ew_tile_ms = tick_ms[0];
        sched_clear();
        cl_uint tmm = add_task((task_t){T_MMA, C, A, B, M, M, M});
        for (cl_uint l = 0; l < nlanes; ++l)
            sched[0][l] = (cell_t){tmm, l, l + 1, 0};
        n_ticks = 1;
        run(1, tick_ms);
        double mma_tile_ms = tick_ms[0];
        double ratio = mma_tile_ms / ew_tile_ms;
        printf("D calibration: EW tile %.3f ms, MMA tile %.3f ms (ratio %.1f)\n",
               ew_tile_ms, mma_tile_ms, ratio);

        const cl_uint mmtiles = (M / MMA_T) * (M / MMA_T); /* 1024 */
        cl_uint tid_mm, tid_ew[8];
        /* --- naive: each task gets its own tick(s), striped across lanes --- */
        sched_clear();
        tid_mm = add_task((task_t){T_MMA, C, A, B, M, M, M});
        for (int e = 0; e < 8; e++)
            tid_ew[e] = add_task((task_t){T_EW, EWBASE + e * 3 * EWN + 2 * EWN,
                                           EWBASE + e * 3 * EWN,
                                           EWBASE + e * 3 * EWN + EWN, EW_ADD, EWN});
        cl_uint tick = 0;
        for (cl_uint t0 = 0; t0 < mmtiles; t0 += nlanes, ++tick)
            for (cl_uint l = 0; l < nlanes && t0 + l < mmtiles; ++l)
                sched[tick][l] = (cell_t){tid_mm, t0 + l, t0 + l + 1, 0};
        for (int e = 0; e < 8; e++, ++tick)
            for (cl_uint l = 0; l < nlanes && l < ewtiles; ++l)
                sched[tick][l] = (cell_t){tid_ew[e], l, l + 1, 0};
        n_ticks = tick;
        double t0 = now_ms(); run(0, NULL); double naive_ms = now_ms() - t0;
        cl_uint naive_ticks = n_ticks;

        /* --- cost-aware: fill each tick with MMA tiles on most lanes and
         * pack EW tiles (ratio-many per MMA slot) on the rest --- */
        sched_clear();
        tid_mm = add_task((task_t){T_MMA, C, A, B, M, M, M});
        for (int e = 0; e < 8; e++)
            tid_ew[e] = add_task((task_t){T_EW, EWBASE + e * 3 * EWN + 2 * EWN,
                                           EWBASE + e * 3 * EWN,
                                           EWBASE + e * 3 * EWN + EWN, EW_ADD, EWN});
        cl_uint ew_per_cell = ratio < 1 ? 1 : (cl_uint)(ratio + 0.5);
        cl_uint total_ew = 8 * ewtiles;
        /* lanes needed for EW per tick so EW finishes when MMA does */
        cl_uint mm_ticks = (mmtiles + nlanes - 1) / nlanes;  /* upper bound */
        cl_uint ew_lanes = (total_ew + ew_per_cell * mm_ticks - 1) / (ew_per_cell * mm_ticks);
        if (ew_lanes >= nlanes) ew_lanes = nlanes / 4;
        cl_uint mm_lanes = nlanes - ew_lanes;
        cl_uint mi = 0, ei = 0;
        for (tick = 0; (mi < mmtiles || ei < total_ew) && tick < MAX_TICKS; ++tick) {
            for (cl_uint l = 0; l < mm_lanes && mi < mmtiles; ++l, ++mi)
                sched[tick][l] = (cell_t){tid_mm, mi, mi + 1, 0};
            for (cl_uint l = mm_lanes; l < nlanes && ei < total_ew; ++l) {
                cl_uint e = ei / ewtiles, lo = ei % ewtiles;
                cl_uint hi = lo + ew_per_cell;
                if (hi > ewtiles) hi = ewtiles;
                sched[tick][l] = (cell_t){tid_ew[e], lo, hi, 0};
                ei += hi - lo;
            }
        }
        n_ticks = tick;
        t0 = now_ms(); run(0, NULL); double packed_ms = now_ms() - t0;

        printf("D wide graph: naive %u ticks %.2f ms | cost-aware %u ticks %.2f ms | speedup %.2fx\n",
               naive_ticks, naive_ms, n_ticks, packed_ms, naive_ms / packed_ms);

        /* instrumented bubble report for the cost-aware schedule */
        run(1, tick_ms);
        double sum_max = 0, sum_cost = 0;
        for (cl_uint t = 0; t < n_ticks; ++t) {
            sum_max += tick_ms[t];
            for (cl_uint l = 0; l < nlanes; ++l) {
                cell_t c = sched[t][l];
                if (c.task == NOPC) continue;
                double unit = tasks[c.task].op == T_MMA ? mma_tile_ms : ew_tile_ms;
                sum_cost += unit * (c.tile_hi - c.tile_lo);
            }
        }
        printf("D bubbles (cost-aware, instrumented): busy %.0f%%  (ideal-cost %.2f ms / wall %.2f ms x %u lanes)\n",
               100.0 * sum_cost / (sum_max * nlanes), sum_cost, sum_max, nlanes);
    }

    /* ===== E: per-lane streams + scheduler-placed global sync ===== */
    /* lanes [0,4): cooperate on matmul 256^3 (256 MMA tiles, 64/lane)
     * lanes [4,n): 8 EW adds of 1M elems as MANY entries (8 tiles/entry)
     * BARRIER (all lanes)  →  phase 2: Z = C + C on lanes [0,4)          */
    if (nlanes >= 8) {
        kasync = clCreateKernel(prog, "vliw_async", &err); CHK(err, "kasync");
        streams_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(entry_t) * MAX_ENTRIES, 0, &err); CHK(err, "streams");
        lane_tab_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(cl_uint) * 2 * 1024, 0, &err); CHK(err, "tab");
        flags_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 256, 0, &err); CHK(err, "flags");
        ranks_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * 64 * 1024, 0, &err); CHK(err, "ranks");

        const cl_uint M = 256;
        const cl_uint A = 4u << 20, B = A + M * M, C = B + M * M, Z = C + M * M;
        const cl_uint EWN = 1u << 20, ewtiles = EWN / EW_TS;
        const cl_uint EWBASE = 8u << 20;
        for (cl_uint i = 0; i < M * M; i++) h[i] = (float)((int)(i % 7) - 3);
        wr(A, h, M * M); wr(B, h, M * M);
        for (cl_uint i = 0; i < EWN; i++) h[i] = (float)(i % 97);
        for (int e = 0; e < 8; e++) {
            wr(EWBASE + e * 3 * EWN, h, EWN);
            wr(EWBASE + e * 3 * EWN + EWN, h, EWN);   /* operand b too */
        }

        streams_clear();
        cl_uint tmm = add_task((task_t){T_MMA, C, A, B, M, M, M});
        const cl_uint mmtiles = (M / MMA_T) * (M / MMA_T);   /* 256 */
        for (cl_uint l = 0; l < 4; ++l)
            emit(l, (entry_t){tmm, l * mmtiles / 4, (l + 1) * mmtiles / 4, FLAG_NONE, 0, FLAG_NONE});
        const cl_uint ew_lanes = nlanes - 4;
        cl_uint next_lane = 4;
        for (int e = 0; e < 8; e++) {
            cl_uint tew = add_task((task_t){T_EW, EWBASE + e * 3 * EWN + 2 * EWN,
                                             EWBASE + e * 3 * EWN,
                                             EWBASE + e * 3 * EWN + EWN, EW_ADD, EWN});
            for (cl_uint lo = 0; lo < ewtiles; lo += 8) {   /* many small entries */
                cl_uint hi = lo + 8 > ewtiles ? ewtiles : lo + 8;
                emit(next_lane, (entry_t){tew, lo, hi, FLAG_NONE, 0, FLAG_NONE});
                next_lane = 4 + (next_lane - 4 + 1) % ew_lanes;
            }
        }
        emit_barrier_all();
        cl_uint tz = add_task((task_t){T_EW, Z, C, C, EW_ADD, M * M});
        const cl_uint ztiles = (M * M + EW_TS - 1) / EW_TS;   /* 4 */
        for (cl_uint l = 0; l < 4 && l < ztiles; ++l)
            emit(l, (entry_t){tz, l * ztiles / 4, (l + 1) * ztiles / 4, FLAG_NONE, 0, FLAG_NONE});
        double async_ms = run_async();

        /* verify: matmul spot check, Z == 2*C, one EW output */
        rd(C, h, M * M);
        bad = 0;
        for (cl_uint r = 0; r < M && bad < 5; r += 37)
            for (cl_uint c = 0; c < M; c += 41) {
                float acc = 0;
                for (cl_uint k = 0; k < M; k++)
                    acc += (float)((int)((r * M + k) % 7) - 3) * (float)((int)((k * M + c) % 7) - 3);
                if (h[r * M + c] != acc) bad++;
            }
        float *hz = malloc((M * M > EWN ? M * M : EWN) * 4);
        rd(Z, hz, M * M);
        for (cl_uint i = 0; i < M * M; i++)
            if (hz[i] != 2.0f * h[i]) { bad++; break; }
        rd(EWBASE + 2 * EWN, hz, EWN);
        for (cl_uint i = 0; i < EWN; i++)
            if (hz[i] != 2.0f * (float)(i % 97)) { bad++; break; }
        free(hz);

        /* barrier arrival spread: which lanes were last at the sync? */
        cl_uint *ranks = malloc(sizeof(cl_uint) * nlanes);
        CHK(clEnqueueReadBuffer(q, ranks_buf, CL_TRUE, 0, sizeof(cl_uint) * nlanes, ranks, 0, 0, 0), "rd ranks");
        cl_uint last_lane = 0, first_lane = 0;
        for (cl_uint l = 0; l < nlanes; ++l) {
            if (ranks[l] > ranks[last_lane]) last_lane = l;
            if (ranks[l] < ranks[first_lane]) first_lane = l;
        }
        printf("E per-lane streams + global sync: %s (%d bad) [%.2f ms; barrier arrival: first=lane%u(%s) last=lane%u(%s)]\n",
               bad ? "FAIL" : "PASS", bad, async_ms,
               first_lane, first_lane < 4 ? "mma" : "ew",
               last_lane, last_lane < 4 ? "mma" : "ew");
        total_bad += bad;
        free(ranks);
    } else {
        printf("E skipped (need >= 8 lanes)\n");
    }

    /* ===== BENCH_MMA=<M>: matmul throughput via async engine ===== */
    if (envi("BENCH_MMA", 0) && nlanes >= 8) {
        const cl_uint M = (cl_uint)envi("BENCH_MMA", 1024);
        const cl_uint A = 4u << 20, B = A + M * M, C = B + M * M;
        for (cl_uint i = 0; i < M * M; i++) h[i] = (float)((int)(i % 7) - 3);
        wr(A, h, M * M); wr(B, h, M * M);
        const cl_uint tiles = (M / MMA_T) * (M / MMA_T);
        double best = 1e30;
        for (int rep = 0; rep < 5; rep++) {
            streams_clear();
            cl_uint tmm = add_task((task_t){T_MMA, C, A, B, M, M, M});
            for (cl_uint l = 0; l < nlanes; ++l) {
                cl_uint lo = (cl_uint)((unsigned long long)tiles * l / nlanes);
                cl_uint hi = (cl_uint)((unsigned long long)tiles * (l + 1) / nlanes);
                if (lo < hi) emit(l, (entry_t){tmm, lo, hi, FLAG_NONE, 0, FLAG_NONE});
            }
            double ms = run_async();
            if (ms < best) best = ms;
        }
        double gflop = 2.0 * M * M * M / 1e9;
        printf("BENCH MMA %ux%ux%u: %.2f ms best-of-5 = %.1f GFLOP/s\n",
               M, M, M, best, gflop / (best / 1e3));
    }

    printf("%s\n", total_bad ? "SOME TESTS FAILED" : "ALL PASS");
    return total_bad ? 1 : 0;
}
