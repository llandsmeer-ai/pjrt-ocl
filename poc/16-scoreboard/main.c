/* poc/16 — per-tile dependency SCOREBOARD vs the grid-wide barrier (R1, §29/§30)
 *
 * §29 decomposed the transformer base 5.0 ms: 84 % is MATMUL, and the loss is
 * that 107 SERIAL global barriers serialize INDEPENDENT small matmuls (3 QKV,
 * 32 heads, FFN tiles) that each fill only 17–31 % of the 376 lanes → mean lane
 * util 0.31. R1 replaces the grid barrier with point-to-point WAIT/SIGNAL so the
 * independent tasks OVERLAP and fill the grid.
 *
 * This PoC is the HARD GATE (§30): model K independent "matmul-like" tasks + 1
 * dependent task and MEASURE, on real hardware (NVIDIA AND PoCL):
 *   (a) CORRECTNESS — the dependent reads producer DATA gated only by an atomic
 *       flag; device-scope acquire/release must order the non-atomic writes
 *       (poc/07 test E) — stress over many iterations, count stale reads.
 *   (b) OVERLAP — per-WG %globaltimer timestamps: does lane utilization rise
 *       toward the full grid vs the serial-barrier baseline?
 *   (c) SPEEDUP — real wall-clock win on this pattern.
 *
 * If overlap does NOT materialize or correctness can't be made race-free →
 * STOP, do not rewire the scheduler (§14a).
 *
 * MODE 0 (barrier): K producer tasks, each its OWN phase of P tiles packed onto
 *   lanes 0..P-1 (reset per phase — exactly today's under-occupancy), grid
 *   barrier between, then the dependent phase. P/G lanes busy per phase.
 * MODE 1 (scoreboard): all K*P producer tiles spread round-robin across ALL G
 *   lanes; each producer tile atomic_inc's its task flag on completion; the
 *   dependent tiles WAIT until every producer flag == P, then run. Independent
 *   tiles never wait → co-occupy the grid.
 */
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- device source (globaltimer gated by -DHAS_GTIMER, set for NVIDIA) ---- */
static const char *SRC =
"typedef struct { uint kind, task, tile, sig, wmask; } entry_t;\n"
"#ifdef HAS_GTIMER\n"
"static inline uint now(void){ ulong t; asm volatile(\"mov.u64 %0, %%globaltimer;\":\"=l\"(t)); return (uint)t; }\n"
"#else\n"
"static inline uint now(void){ return 0u; }\n"
"#endif\n"
"#ifdef NO_DEV_FENCE\n"
"#define FENCE_REL()\n"
"#define FENCE_ACQ()\n"
"#define LOADU(p) (*(volatile __global uint*)(p))\n"
"#else\n"
"#define FENCE_REL() atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device)\n"
"#define FENCE_ACQ() atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device)\n"
"#define LOADU(p) atomic_load_explicit((volatile __global atomic_uint*)(p), memory_order_relaxed, memory_scope_device)\n"
"#endif\n"
/* grid-wide spin barrier — exactly vmo_barrier (vm_common.cl) */
"static void gbar(volatile __global uint* b, uint ng){\n"
"  barrier(CLK_GLOBAL_MEM_FENCE);\n"
"  if(get_local_id(0)==0){\n"
"    FENCE_REL();\n"
"    uint ph=LOADU(&b[1]);\n"
"    if(atomic_inc(&b[0])==ng-1){ b[0]=0; atomic_inc(&b[1]); }\n"
"    else { while(LOADU(&b[1])==ph); }\n"
"    FENCE_ACQ();\n"
"  }\n"
"  barrier(CLK_GLOBAL_MEM_FENCE);\n"
"}\n"
/* heavy but DCE-proof time sink; returns exactly 0.0f (sink is finite) */
"static float burn(uint seed, uint work){\n"
"  uint j=seed; float sink=0.0f;\n"
"  for(uint i=0;i<work;i++){ j=j*1664525u+1013904223u; sink+=(float)(j&1u); }\n"
"  return sink*0.0f;\n"
"}\n"
"__kernel void run(__global float* outp, __global float* dep,\n"
"                  volatile __global uint* flags, volatile __global uint* bar,\n"
"                  __global const entry_t* entries, __global const uint2* lane_tab,\n"
"                  const uint G, const uint P, const uint K, const uint WORK,\n"
"                  const uint mode,\n"
"                  __global uint* tstart, __global uint* tend, __global uint* tbusy){\n"
"  const uint g=get_group_id(0), lid=get_local_id(0);\n"
"  const uint2 span=lane_tab[g];\n"
"  if(lid==0){ tstart[g]=now(); }\n"
"  uint busy=0;\n"
"  for(uint e=0;e<span.y;e++){\n"
"    const entry_t en=entries[span.x+e];\n"
"    if(en.kind==2u){ gbar(bar,G); continue; }\n"
"    if(en.kind==1u){                       /* dependent tile */\n"
"      if(mode==1u && lid==0){\n"
"        for(uint k=0;k<K;k++) if(en.wmask&(1u<<k)) while(LOADU(&flags[k])<P);\n"
"        FENCE_ACQ();\n"
"      }\n"
"      const uint t0=now();\n"
"      const float z=burn(lid+e*7u, WORK);\n"
"      if(lid==0){\n"
"        float s=0.0f; for(uint k=0;k<K;k++) s+=outp[k*P+en.tile];\n"
"        dep[en.tile]=s+z;\n"
"      }\n"
"      if(lid==0) busy+=now()-t0;\n"
"      continue;\n"
"    }\n"
"    /* producer tile */\n"
"    const uint t0=now();\n"
"    const float z=burn(lid+e*7u, WORK);\n"
"    if(lid==0) outp[en.task*P+en.tile]=(float)(en.task*P+en.tile+1u)+z;\n"
"    if(mode==1u && en.sig!=0xFFFFFFFFu){\n"
"      FENCE_REL();\n"
"      if(lid==0) atomic_inc(&flags[en.sig]);\n"
"    }\n"
"    if(lid==0) busy+=now()-t0;\n"
"  }\n"
"  if(lid==0){ tend[g]=now(); tbusy[g]=busy; }\n"
"}\n";

typedef struct { cl_uint kind, task, tile, sig, wmask; } entry_t;

#define CK(e) do{cl_int _e=(e); if(_e!=CL_SUCCESS){fprintf(stderr,"%s:%d CL err %d\n",__FILE__,__LINE__,_e);exit(1);} }while(0)
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
static int envi(const char*n,int d){ const char*e=getenv(n); return e?atoi(e):d; }

int main(void){
  const char *want = getenv("PJRT_OCL_DEVICE");
  cl_uint nplat; CK(clGetPlatformIDs(0,NULL,&nplat));
  cl_platform_id plats[8]; CK(clGetPlatformIDs(nplat<8?nplat:8,plats,NULL));
  cl_platform_id plat=0; cl_device_id dev=0; char nm[256];
  for(cl_uint i=0;i<nplat;i++){
    CK(clGetPlatformInfo(plats[i],CL_PLATFORM_NAME,sizeof nm,nm,NULL));
    if(want && !strstr(nm,want)) continue;
    cl_uint nd; if(clGetDeviceIDs(plats[i],CL_DEVICE_TYPE_ALL,0,NULL,&nd)!=CL_SUCCESS||!nd) continue;
    cl_device_id ds[8]; CK(clGetDeviceIDs(plats[i],CL_DEVICE_TYPE_ALL,nd<8?nd:8,ds,NULL));
    plat=plats[i]; dev=ds[0]; break;
  }
  if(!dev){ fprintf(stderr,"no device (PJRT_OCL_DEVICE=%s)\n",want?want:"(any)"); return 1; }
  CK(clGetPlatformInfo(plat,CL_PLATFORM_NAME,sizeof nm,nm,NULL));
  char dnm[256]; CK(clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof dnm,dnm,NULL));
  cl_uint cus; CK(clGetDeviceInfo(dev,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof cus,&cus,NULL));
  cl_device_type dtype; CK(clGetDeviceInfo(dev,CL_DEVICE_TYPE,sizeof dtype,&dtype,NULL));
  const int is_gpu = (dtype & CL_DEVICE_TYPE_GPU) != 0;
  printf("platform: %s\ndevice  : %s (%u CUs, %s)\n\n",nm,dnm,cus,is_gpu?"GPU":"CPU");

  /* co-resident lane count: GPU 2*CU (occupancy cap, §9); CPU <= CU (liveness). */
  cl_uint G = envi("G", is_gpu ? 2*cus : cus);
  cl_uint K = envi("K", is_gpu ? 8 : 4);
  cl_uint P = envi("P", G / K);              /* tiles per producer task */
  cl_uint WORK = envi("WORK", is_gpu ? 6000 : 1500);
  cl_uint ITERS = envi("ITERS", 300);
  size_t  lsz = envi("LSZ", is_gpu ? 64 : 1);
  if (P < 1) P = 1;
  const cl_uint NPROD = K * P;               /* total producer tiles */
  printf("[cfg] G=%u lanes  K=%u indep tasks  P=%u tiles/task (=%u prod tiles)  "
         "WORK=%u  ITERS=%u  lsz=%zu\n", G,K,P,NPROD,WORK,ITERS,lsz);
  printf("      barrier-mode occupancy per producer phase = P/G = %.0f%%\n\n",
         100.0*P/G);

  cl_int e;
  cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&e); CK(e);
  cl_command_queue q=clCreateCommandQueue(ctx,dev,0,&e); CK(e);

  /* device-scope fence compile probe → pick build opts */
  char opts[128]="";
  if(is_gpu) strcat(opts,"-DHAS_GTIMER ");
  {
    cl_program p=clCreateProgramWithSource(ctx,1,&SRC,NULL,&e); CK(e);
    char t[160]; snprintf(t,sizeof t,"%s",opts);
    cl_int be=clBuildProgram(p,1,&dev,t,NULL,NULL);
    if(be!=CL_SUCCESS){ strcat(opts,"-DNO_DEV_FENCE "); printf("[warn] device-scope fence rejected → NO_DEV_FENCE (unsafe; timing only)\n"); }
    clReleaseProgram(p);
  }
  cl_program prog=clCreateProgramWithSource(ctx,1,&SRC,NULL,&e); CK(e);
  cl_int be=clBuildProgram(prog,1,&dev,opts,NULL,NULL);
  if(be!=CL_SUCCESS){ char log[8192]; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,sizeof log,log,NULL); fprintf(stderr,"build failed:\n%s\n",log); return 1; }
  cl_kernel k=clCreateKernel(prog,"run",&e); CK(e);

  /* ---- build the two schedules (flat entries + per-lane table) ------------ */
  /* mode 0 (barrier): lane L stream = [prod(task0,L) if L<P] BARRIER
   *   [prod(task1,L) if L<P] BARRIER ... BARRIER [dep(L) if L<P].  Every lane
   *   carries a BARRIER at each phase boundary (all lanes must arrive). */
  entry_t *e0=NULL; size_t n0=0, cap0=0;
  cl_uint *off0=malloc(G*sizeof(cl_uint)), *cnt0=malloc(G*sizeof(cl_uint));
  /* We build per-lane streams then flatten. */
  entry_t **ls0=calloc(G,sizeof(void*)); size_t *ll0=calloc(G,sizeof(size_t)), *lc0=calloc(G,sizeof(size_t));
  #define PUSH(ls,ll,lc,L,EN) do{ if(ll[L]==lc[L]){ lc[L]=lc[L]?lc[L]*2:8; ls[L]=realloc(ls[L],lc[L]*sizeof(entry_t)); } ls[L][ll[L]++]=(EN);}while(0)
  for(cl_uint t=0;t<K;t++){
    for(cl_uint L=0;L<G;L++){
      if(L<P){ entry_t en={0,t,L,t,0}; PUSH(ls0,ll0,lc0,L,en); }
      entry_t bar={2,0,0,0,0}; PUSH(ls0,ll0,lc0,L,bar);   /* phase barrier */
    }
  }
  for(cl_uint L=0;L<G;L++){ if(L<P){ entry_t en={1,0,L,0xFFFFFFFFu,(1u<<K)-1u}; PUSH(ls0,ll0,lc0,L,en);} }
  /* flatten mode 0 */
  for(cl_uint L=0;L<G;L++){ off0[L]=n0; cnt0[L]=ll0[L]; if(n0+ll0[L]>cap0){cap0=(n0+ll0[L])*2+16; e0=realloc(e0,cap0*sizeof(entry_t));} memcpy(e0+n0,ls0[L],ll0[L]*sizeof(entry_t)); n0+=ll0[L]; }

  /* mode 1 (scoreboard): all NPROD producer tiles round-robin across ALL G
   *   lanes; each signals its task flag. Then the P dependent tiles round-robin
   *   across lanes, each WAITs on all K flags. NO barriers. */
  entry_t *e1=NULL; size_t n1=0, cap1=0;
  cl_uint *off1=malloc(G*sizeof(cl_uint)), *cnt1=malloc(G*sizeof(cl_uint));
  entry_t **ls1=calloc(G,sizeof(void*)); size_t *ll1=calloc(G,sizeof(size_t)), *lc1=calloc(G,sizeof(size_t));
  for(cl_uint i=0;i<NPROD;i++){ cl_uint t=i/P, tl=i%P, L=i%G; entry_t en={0,t,tl,t,0}; PUSH(ls1,ll1,lc1,L,en); }
  for(cl_uint tl=0;tl<P;tl++){ cl_uint L=(NPROD+tl)%G; entry_t en={1,0,tl,0xFFFFFFFFu,(1u<<K)-1u}; PUSH(ls1,ll1,lc1,L,en); }
  for(cl_uint L=0;L<G;L++){ off1[L]=n1; cnt1[L]=ll1[L]; if(n1+ll1[L]>cap1){cap1=(n1+ll1[L])*2+16; e1=realloc(e1,cap1*sizeof(entry_t));} memcpy(e1+n1,ls1[L],ll1[L]*sizeof(entry_t)); n1+=ll1[L]; }

  /* ---- buffers ------------------------------------------------------------ */
  cl_mem outp=clCreateBuffer(ctx,CL_MEM_READ_WRITE,NPROD*sizeof(float),NULL,&e); CK(e);
  cl_mem dep =clCreateBuffer(ctx,CL_MEM_READ_WRITE,P*sizeof(float),NULL,&e); CK(e);
  cl_mem flags=clCreateBuffer(ctx,CL_MEM_READ_WRITE,K*sizeof(cl_uint),NULL,&e); CK(e);
  cl_mem bar =clCreateBuffer(ctx,CL_MEM_READ_WRITE,2*sizeof(cl_uint),NULL,&e); CK(e);
  cl_mem tstart=clCreateBuffer(ctx,CL_MEM_READ_WRITE,G*sizeof(cl_uint),NULL,&e); CK(e);
  cl_mem tend  =clCreateBuffer(ctx,CL_MEM_READ_WRITE,G*sizeof(cl_uint),NULL,&e); CK(e);
  cl_mem tbusy =clCreateBuffer(ctx,CL_MEM_READ_WRITE,G*sizeof(cl_uint),NULL,&e); CK(e);

  /* expected dependent value: dep[tl] = sum_k (k*P+tl+1) */
  float *expd=malloc(P*sizeof(float));
  for(cl_uint tl=0;tl<P;tl++){ double s=0; for(cl_uint kk=0;kk<K;kk++) s+=(double)(kk*P+tl+1); expd[tl]=(float)s; }

  cl_uint zero=0;
  size_t gsz=(size_t)G*lsz;
  const char *MODEN[2]={"BARRIER (serial phases, today)","SCOREBOARD (point-to-point)"};

  double res_ms[2]; double res_util[2]; cl_uint res_bad[2];
  for(int mode=0;mode<2;mode++){
    cl_mem ent = mode? clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,n1*sizeof(entry_t),e1,&e)
                     : clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,n0*sizeof(entry_t),e0,&e); CK(e);
    cl_uint *offb = mode?off1:off0, *cntb = mode?cnt1:cnt0;
    cl_uint *ltab=malloc(2*G*sizeof(cl_uint));
    for(cl_uint L=0;L<G;L++){ ltab[2*L]=offb[L]; ltab[2*L+1]=cntb[L]; }
    cl_mem lt=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,2*G*sizeof(cl_uint),ltab,&e); CK(e);

    CK(clSetKernelArg(k,0,sizeof outp,&outp)); CK(clSetKernelArg(k,1,sizeof dep,&dep));
    CK(clSetKernelArg(k,2,sizeof flags,&flags)); CK(clSetKernelArg(k,3,sizeof bar,&bar));
    CK(clSetKernelArg(k,4,sizeof ent,&ent)); CK(clSetKernelArg(k,5,sizeof lt,&lt));
    CK(clSetKernelArg(k,6,sizeof G,&G)); CK(clSetKernelArg(k,7,sizeof P,&P));
    CK(clSetKernelArg(k,8,sizeof K,&K)); CK(clSetKernelArg(k,9,sizeof WORK,&WORK));
    cl_uint m=mode; CK(clSetKernelArg(k,10,sizeof m,&m));
    CK(clSetKernelArg(k,11,sizeof tstart,&tstart)); CK(clSetKernelArg(k,12,sizeof tend,&tend));
    CK(clSetKernelArg(k,13,sizeof tbusy,&tbusy));

    cl_uint bad=0;
    /* warmup */
    CK(clEnqueueFillBuffer(q,bar,&zero,sizeof zero,0,2*sizeof(cl_uint),0,NULL,NULL));
    CK(clEnqueueFillBuffer(q,flags,&zero,sizeof zero,0,K*sizeof(cl_uint),0,NULL,NULL));
    CK(clEnqueueNDRangeKernel(q,k,1,NULL,&gsz,&lsz,0,NULL,NULL)); CK(clFinish(q));

    double t0=now_ms();
    for(cl_uint it=0; it<ITERS; ++it){
      CK(clEnqueueFillBuffer(q,bar,&zero,sizeof zero,0,2*sizeof(cl_uint),0,NULL,NULL));
      CK(clEnqueueFillBuffer(q,flags,&zero,sizeof zero,0,K*sizeof(cl_uint),0,NULL,NULL));
      CK(clEnqueueFillBuffer(q,dep,&zero,sizeof zero,0,P*sizeof(float),0,NULL,NULL));
      CK(clEnqueueFillBuffer(q,outp,&zero,sizeof zero,0,NPROD*sizeof(float),0,NULL,NULL));
      CK(clEnqueueNDRangeKernel(q,k,1,NULL,&gsz,&lsz,0,NULL,NULL));
      /* stress: check the dependent result every iter */
      float *d=malloc(P*sizeof(float));
      CK(clEnqueueReadBuffer(q,dep,CL_TRUE,0,P*sizeof(float),d,0,NULL,NULL));
      for(cl_uint tl=0;tl<P;tl++) if(d[tl]!=expd[tl]) bad++;
      free(d);
    }
    CK(clFinish(q)); double t1=now_ms();
    res_ms[mode]=(t1-t0)/ITERS;
    res_bad[mode]=bad;

    /* utilization from the last iter's per-WG timestamps (GPU only) */
    cl_uint *ts=malloc(G*sizeof(cl_uint)), *te=malloc(G*sizeof(cl_uint)), *tb=malloc(G*sizeof(cl_uint));
    CK(clEnqueueReadBuffer(q,tstart,CL_TRUE,0,G*sizeof(cl_uint),ts,0,NULL,NULL));
    CK(clEnqueueReadBuffer(q,tend,CL_TRUE,0,G*sizeof(cl_uint),te,0,NULL,NULL));
    CK(clEnqueueReadBuffer(q,tbusy,CL_TRUE,0,G*sizeof(cl_uint),tb,0,NULL,NULL));
    double util=-1;
    if(is_gpu){
      cl_uint mn=0xFFFFFFFFu,mx=0; unsigned long long busy=0;
      for(cl_uint L=0;L<G;L++){ if(ts[L]<mn)mn=ts[L]; if(te[L]>mx)mx=te[L]; busy+=tb[L]; }
      double wall=(double)(mx-mn);
      util = wall>0 ? (double)busy/(wall*G) : -1;
      printf("[%d] %-32s wall(kernel span)=%.1f us  busy_sum=%llu us  util=%.3f\n",
             mode, MODEN[mode], wall/1e3, busy/1000ull, util);
    }
    res_util[mode]=util;
    free(ts);free(te);free(tb);free(ltab);
    clReleaseMemObject(ent); clReleaseMemObject(lt);
    printf("[%d] %-32s host wall=%.3f ms/iter   stale/wrong dep reads = %u / %u  %s\n\n",
           mode, MODEN[mode], res_ms[mode], res_bad[mode], ITERS*P,
           res_bad[mode]?"<<< DATA RACE":"clean");
  }

  printf("==== GATE SUMMARY (%s) ====\n", dnm);
  printf("  correctness (scoreboard): %s (%u/%u wrong over %u iters)\n",
         res_bad[1]?"FAIL — RACE":"race-free", res_bad[1], ITERS*P, ITERS);
  if(is_gpu && res_util[0]>0 && res_util[1]>0)
    printf("  overlap:  lane util %.3f (barrier) -> %.3f (scoreboard)  [%.2fx]\n",
           res_util[0], res_util[1], res_util[1]/res_util[0]);
  printf("  speedup:  %.3f ms (barrier) -> %.3f ms (scoreboard)  [%.2fx]\n",
         res_ms[0], res_ms[1], res_ms[0]/res_ms[1]);
  printf("  GATE: %s\n", (!res_bad[1] && (!is_gpu || res_util[1]>res_util[0]*1.3) && res_ms[0]>res_ms[1]*1.15)
                          ? "PASS — overlap real, race-free, faster"
                          : "INSPECT — see numbers above");
  return 0;
}
