/* poc/07 — can we build a CORRECT cross-workgroup barrier?
 *
 * The VLIW megakernel's barrier is a persistent-thread spin over a global
 * atomic arrival counter (poc/01). The while agent found it publishes the
 * *atomic* cond flag fine but lets *non-atomic cross-lane data* go stale under
 * iteration (project risk #1). This PoC pins down WHY and WHAT actually fixes
 * it, per device, with numbers — not theory.
 *
 * Four experiments (run with PJRT_OCL_DEVICE=<substr>, default first GPU):
 *   A. Does this device's OpenCL compiler even ACCEPT device-scope atomics
 *      (`memory_scope_device`)? That is the textbook fix; clinfo says NVIDIA
 *      exposes only work-group scope. Prove it at the compiler.
 *   B. Reproduce the data race in isolation: G persistent workgroups, each
 *      writes arena[i]=iter (plain store), 1.2 spin-barrier, then reads a
 *      NEIGHBOUR's cell and checks it == iter. Count mismatches over T iters.
 *   C. Same, but the consumer reads through `volatile __global` (on NVIDIA this
 *      lowers to an L1-bypassing load → L2, the coherence point). Does it fix B?
 *   D. The portable barrier: kernel-boundary. Two kernels (write, then
 *      read+check) relaunched per iteration on an in-order queue. Correct by
 *      the OpenCL execution model on every vendor, no atomics. Measure the
 *      per-phase overhead so we know Plan B's real cost.
 */
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *SRC = "\n"
/* ---- 1.2 spin-barrier: exactly the vmo_barrier we ship ------------------ */
"static void bar12(volatile __global uint*b,uint ng){\n"
"  barrier(CLK_GLOBAL_MEM_FENCE);\n"
"  if(get_local_id(0)==0){\n"
"    uint ph=atomic_add(&b[1],0);\n"
"    if(atomic_inc(&b[0])==ng-1){ b[0]=0; mem_fence(CLK_GLOBAL_MEM_FENCE); atomic_inc(&b[1]); }\n"
"    else { while(atomic_add(&b[1],0)==ph); }\n"
"  }\n"
"  barrier(CLK_GLOBAL_MEM_FENCE);\n"
"}\n"
/* B: persistent producer/consumer, plain (L1-cacheable) neighbour read ----- */
"__kernel void race_plain(__global int*a,volatile __global uint*b,uint ng,uint T,__global uint*bad){\n"
"  uint g=get_group_id(0);\n"
"  for(uint it=1; it<=T; ++it){\n"
"    a[g]=(int)it;\n"
"    bar12(b,ng);\n"
"    int v=a[(g+1)%ng];\n"
"    if(get_local_id(0)==0 && v!=(int)it) atomic_inc(bad);\n"
"    bar12(b,ng);\n"
"  }\n"
"}\n"
/* Barrier variant: wrap the atomic handshake in DEVICE-SCOPE acquire/release
 * fences (OpenCL 2.0 memory model). If the device honours device scope, the
 * plain (non-atomic) data writes before the release become visible after the
 * acquire — no volatile, no L1-bypass perf cost. */
"static void bar_dev(volatile __global uint*b,uint ng){\n"
"  barrier(CLK_GLOBAL_MEM_FENCE);\n"
"  if(get_local_id(0)==0){\n"
"    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);\n"
"    uint ph=atomic_add(&b[1],0);\n"
"    if(atomic_inc(&b[0])==ng-1){ b[0]=0; atomic_inc(&b[1]); }\n"
"    else { while(atomic_add(&b[1],0)==ph); }\n"
"    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);\n"
"  }\n"
"  barrier(CLK_GLOBAL_MEM_FENCE);\n"
"}\n"
/* E: device-scope-fence barrier, plain read -------------------------------- */
"__kernel void race_devscope(__global int*a,volatile __global uint*b,uint ng,uint T,__global uint*bad){\n"
"  uint g=get_group_id(0);\n"
"  for(uint it=1; it<=T; ++it){\n"
"    a[g]=(int)it;\n"
"    bar_dev(b,ng);\n"
"    int v=a[(g+1)%ng];\n"
"    if(get_local_id(0)==0 && v!=(int)it) atomic_inc(bad);\n"
"    bar_dev(b,ng);\n"
"  }\n"
"}\n"
/* C: same, but neighbour read goes through volatile (L1-bypass on NVIDIA) --- */
"__kernel void race_volatile(__global int*a,volatile __global uint*b,uint ng,uint T,__global uint*bad){\n"
"  uint g=get_group_id(0);\n"
"  volatile __global int*av=a;\n"
"  for(uint it=1; it<=T; ++it){\n"
"    av[g]=(int)it;\n"
"    bar12(b,ng);\n"
"    int v=av[(g+1)%ng];\n"
"    if(get_local_id(0)==0 && v!=(int)it) atomic_inc(bad);\n"
"    bar12(b,ng);\n"
"  }\n"
"}\n"
/* D: kernel-boundary barrier — two tiny kernels relaunched per phase -------- */
"__kernel void d_write(__global int*a,uint it){ a[get_group_id(0)]=(int)it; }\n"
"__kernel void d_check(__global int*a,uint ng,uint it,__global uint*bad){\n"
"  uint g=get_group_id(0);\n"
"  if(a[(g+1)%ng]!=(int)it) atomic_inc(bad);\n"
"}\n";

/* A: standalone source probed for a device-scope atomic fence. */
static const char *SRC_DEVSCOPE =
"__kernel void probe(__global int*a){\n"
"  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);\n"
"  a[get_global_id(0)]=1;\n"
"}\n";

#define CK(e) do{cl_int _e=(e); if(_e!=CL_SUCCESS){fprintf(stderr,"%s:%d CL err %d\n",__FILE__,__LINE__,_e);exit(1);} }while(0)

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }

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
  printf("platform: %s\ndevice  : %s (%u CUs)\n\n",nm,dnm,cus);

  cl_int e;
  cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&e); CK(e);
  cl_command_queue q=clCreateCommandQueue(ctx,dev,0,&e); CK(e);

  /* ---- A: device-scope atomic-fence compile probe --------------------- */
  {
    cl_program p=clCreateProgramWithSource(ctx,1,&SRC_DEVSCOPE,NULL,&e); CK(e);
    cl_int be=clBuildProgram(p,1,&dev,NULL,NULL,NULL);
    printf("[A] device-scope atomic fence (memory_scope_device):\n");
    if(be==CL_SUCCESS){
      printf("    COMPILES — the OpenCL 2.0 memory-model fix is AVAILABLE here.\n\n");
    } else {
      char log[4096]; size_t ls=0;
      clGetProgramBuildInfo(p,dev,CL_PROGRAM_BUILD_LOG,sizeof log,log,&ls);
      if(ls>200) ls=200;
      printf("    REJECTED (build err %d) — device-scope atomics UNAVAILABLE.\n"
             "    log: %.*s\n\n",be,(int)ls,log);
    }
    clReleaseProgram(p);
  }

  /* ---- build the main program (B/C/D) --------------------------------- */
  cl_program prog=clCreateProgramWithSource(ctx,1,&SRC,NULL,&e); CK(e);
  cl_int be=clBuildProgram(prog,1,&dev,NULL,NULL,NULL);
  if(be!=CL_SUCCESS){ char log[8192]; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,sizeof log,log,NULL); fprintf(stderr,"build failed:\n%s\n",log); return 1; }

  /* Use a small, safely-co-resident workgroup count so the spin-barrier does
   * not deadlock on liveness — we are isolating the MEMORY bug, not liveness.
   * One workgroup per CU is the classic persistent-thread residency bound;
   * clamp to 8 so neighbours definitely co-reside on any device. */
  cl_uint G = cus<8?cus:8;
  cl_uint T = 200000;   /* enough iterations to expose a rare stale read */
  size_t lsz=32;
  if(const char*e=getenv("G_ENV")) G=atoi(e);
  if(const char*e=getenv("LSZ_ENV")) lsz=atoi(e);
  if(const char*e=getenv("T_ENV")) T=atoi(e);
  size_t gsz=(size_t)G*lsz;
  printf("[cfg] G=%u lsz=%zu T=%u\n", G, lsz, T);

  cl_mem a=clCreateBuffer(ctx,CL_MEM_READ_WRITE,G*sizeof(int),NULL,&e); CK(e);
  cl_mem b=clCreateBuffer(ctx,CL_MEM_READ_WRITE,2*sizeof(cl_uint),NULL,&e); CK(e);
  cl_mem bad=clCreateBuffer(ctx,CL_MEM_READ_WRITE,sizeof(cl_uint),NULL,&e); CK(e);
  cl_uint zero=0;

  printf("[B/C/D] %u workgroups, %u iterations, neighbour-read cross-WG data\n\n",G,T);

  /* ---- B: plain read, in-kernel 1.2 barrier --------------------------- */
  {
    CK(clEnqueueFillBuffer(q,b,&zero,sizeof zero,0,2*sizeof(cl_uint),0,NULL,NULL));
    CK(clEnqueueFillBuffer(q,bad,&zero,sizeof zero,0,sizeof(cl_uint),0,NULL,NULL));
    CK(clEnqueueFillBuffer(q,a,&zero,sizeof zero,0,G*sizeof(int),0,NULL,NULL));
    cl_kernel k=clCreateKernel(prog,"race_plain",&e); CK(e);
    CK(clSetKernelArg(k,0,sizeof a,&a)); CK(clSetKernelArg(k,1,sizeof b,&b));
    CK(clSetKernelArg(k,2,sizeof G,&G)); CK(clSetKernelArg(k,3,sizeof T,&T));
    CK(clSetKernelArg(k,4,sizeof bad,&bad));
    double t0=now_ms();
    CK(clEnqueueNDRangeKernel(q,k,1,NULL,&gsz,&lsz,0,NULL,NULL));
    CK(clFinish(q)); double t1=now_ms();
    cl_uint nb; CK(clEnqueueReadBuffer(q,bad,CL_TRUE,0,sizeof nb,&nb,0,NULL,NULL));
    printf("[B] in-kernel barrier, PLAIN read : %u stale reads / %u  (%.1f ms)  %s\n",
           nb,T,t1-t0, nb?"<-- DATA RACE":"clean");
    clReleaseKernel(k);
  }

  /* ---- E: device-scope-fence barrier, plain read ---------------------- */
  {
    CK(clEnqueueFillBuffer(q,b,&zero,sizeof zero,0,2*sizeof(cl_uint),0,NULL,NULL));
    CK(clEnqueueFillBuffer(q,bad,&zero,sizeof zero,0,sizeof(cl_uint),0,NULL,NULL));
    CK(clEnqueueFillBuffer(q,a,&zero,sizeof zero,0,G*sizeof(int),0,NULL,NULL));
    cl_kernel k=clCreateKernel(prog,"race_devscope",&e); CK(e);
    CK(clSetKernelArg(k,0,sizeof a,&a)); CK(clSetKernelArg(k,1,sizeof b,&b));
    CK(clSetKernelArg(k,2,sizeof G,&G)); CK(clSetKernelArg(k,3,sizeof T,&T));
    CK(clSetKernelArg(k,4,sizeof bad,&bad));
    double t0=now_ms();
    CK(clEnqueueNDRangeKernel(q,k,1,NULL,&gsz,&lsz,0,NULL,NULL));
    CK(clFinish(q)); double t1=now_ms();
    cl_uint nb; CK(clEnqueueReadBuffer(q,bad,CL_TRUE,0,sizeof nb,&nb,0,NULL,NULL));
    printf("[E] device-scope-fence barrier   : %u stale reads / %u  (%.1f ms)  %s\n",
           nb,T,t1-t0, nb?"<-- device scope NOT honoured":"clean (device scope honoured!)");
    clReleaseKernel(k);
  }

  /* ---- C: volatile read, in-kernel 1.2 barrier ------------------------ */
  {
    CK(clEnqueueFillBuffer(q,b,&zero,sizeof zero,0,2*sizeof(cl_uint),0,NULL,NULL));
    CK(clEnqueueFillBuffer(q,bad,&zero,sizeof zero,0,sizeof(cl_uint),0,NULL,NULL));
    CK(clEnqueueFillBuffer(q,a,&zero,sizeof zero,0,G*sizeof(int),0,NULL,NULL));
    cl_kernel k=clCreateKernel(prog,"race_volatile",&e); CK(e);
    CK(clSetKernelArg(k,0,sizeof a,&a)); CK(clSetKernelArg(k,1,sizeof b,&b));
    CK(clSetKernelArg(k,2,sizeof G,&G)); CK(clSetKernelArg(k,3,sizeof T,&T));
    CK(clSetKernelArg(k,4,sizeof bad,&bad));
    double t0=now_ms();
    CK(clEnqueueNDRangeKernel(q,k,1,NULL,&gsz,&lsz,0,NULL,NULL));
    CK(clFinish(q)); double t1=now_ms();
    cl_uint nb; CK(clEnqueueReadBuffer(q,bad,CL_TRUE,0,sizeof nb,&nb,0,NULL,NULL));
    printf("[C] in-kernel barrier, VOLATILE rd: %u stale reads / %u  (%.1f ms)  %s\n",
           nb,T,t1-t0, nb?"<-- still racing":"clean (L1-bypass fixes it)");
    clReleaseKernel(k);
  }

  /* ---- D: kernel-boundary barrier (host dispatch) --------------------- */
  {
    CK(clEnqueueFillBuffer(q,bad,&zero,sizeof zero,0,sizeof(cl_uint),0,NULL,NULL));
    CK(clEnqueueFillBuffer(q,a,&zero,sizeof zero,0,G*sizeof(int),0,NULL,NULL));
    cl_kernel kw=clCreateKernel(prog,"d_write",&e); CK(e);
    cl_kernel kc=clCreateKernel(prog,"d_check",&e); CK(e);
    cl_uint TD=20000;   /* fewer: 2 launches/iter, still plenty */
    double t0=now_ms();
    for(cl_uint it=1; it<=TD; ++it){
      CK(clSetKernelArg(kw,0,sizeof a,&a)); CK(clSetKernelArg(kw,1,sizeof it,&it));
      CK(clEnqueueNDRangeKernel(q,kw,1,NULL,&gsz,&lsz,0,NULL,NULL));
      CK(clSetKernelArg(kc,0,sizeof a,&a)); CK(clSetKernelArg(kc,1,sizeof G,&G));
      CK(clSetKernelArg(kc,2,sizeof it,&it)); CK(clSetKernelArg(kc,3,sizeof bad,&bad));
      CK(clEnqueueNDRangeKernel(q,kc,1,NULL,&gsz,&lsz,0,NULL,NULL));
    }
    CK(clFinish(q)); double t1=now_ms();
    cl_uint nb; CK(clEnqueueReadBuffer(q,bad,CL_TRUE,0,sizeof nb,&nb,0,NULL,NULL));
    double per_phase_us=(t1-t0)*1e3/(TD*2.0);
    printf("[D] kernel-boundary barrier      : %u stale reads / %u  (%.1f ms, %.2f us/phase)  %s\n",
           nb,TD,t1-t0,per_phase_us, nb?"<-- UNEXPECTED":"clean (guaranteed)");
    clReleaseKernel(kw); clReleaseKernel(kc);
  }

  printf("\nInterpretation:\n"
    "  B racing + A rejected => no in-spec in-kernel fix on this device.\n"
    "  C clean               => volatile L1-bypass is a NVIDIA-specific option (perf cost).\n"
    "  D clean + low us/phase => kernel-boundary (Plan B) is the portable correct barrier.\n");
  return 0;
}
