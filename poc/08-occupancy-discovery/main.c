/* poc/08 — occupancy DISCOVERY: measure co-resident workgroups, don't guess.
 *
 * First Intel Xe2 bring-up (docs/decisions.md #9) proved the megakernel's
 * `ngroups = 2*CL_DEVICE_MAX_COMPUTE_UNITS` heuristic is vendor-poison:
 * CL_DEVICE_MAX_COMPUTE_UNITS counts SMs on NVIDIA but VECTOR ENGINES (XVEs)
 * on Intel, so Arc 140V (64 XVEs, true capacity 32 groups @ local=256) got 128
 * lanes → spin-barrier starvation → clFinish -5. Measured boundary: 32 PASS,
 * 33 FAIL. There is NO portable query for "resident groups for THIS kernel"
 * (register/GRF mode and __local usage change it per kernel & driver).
 *
 * The fix from the literature this repo already cites (Sorensen & Donaldson,
 * OOPSLA 2016): a DISCOVERY protocol — measure the actual co-resident group
 * count at init with a deadlock-free gate/ticket handshake:
 *
 *   d[0]=lock  d[1]=gate(1=open)  d[2]=count
 *   leader:  lock{ if(gate) ticket=count++ }unlock
 *     no ticket        -> exit immediately        (never waits on anyone)
 *     ticket 0         -> poll count until stable, then lock{gate=0}unlock
 *     ticket >0        -> spin while gate open    (HOLDS its residency slot)
 *   result: count == groups that were simultaneously resident. Groups that
 *   were never scheduled never block anyone => deadlock-free by construction,
 *   regardless of how oversized the launch is. Only 1.2 atomics on ONE buffer
 *   (no cross-variable ordering needed) => safe even on strict-1.2 builds
 *   where vmo_barrier's device-scope fences are unavailable.
 *
 * Experiments (PJRT_OCL_DEVICE=<platform substr> selects device, à la poc/07):
 *   A. discovery on a SLIM kernel (tiny footprint) — the optimistic ceiling.
 *   B. discovery on a VM-LIKE kernel (8 KB __local As/Bs like vm2 + register
 *      pressure) — the number that matters; expect <= A.
 *   C. LIVENESS validation: vm-like kernel runs T spin-barrier rounds at
 *      exactly the discovered count — must complete. (Run with count+1 via
 *      `./poc08 --over` to see the failure mode; NOT run by default, it
 *      deadlocks until the driver's hang-check kills it.)
 *   D. what the heuristics would have picked, next to the measured truth.
 */
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *SRC =
/* ---- discovery handshake (1.2 atomics only, single buffer) --------------- */
"#define NOT_RESIDENT 0xFFFFFFFFu\n"
"static uint discover(volatile __global uint *d){\n"
"  uint t = NOT_RESIDENT;\n"
"  while(atomic_cmpxchg(&d[0],0u,1u)!=0u);\n"
"  if(atomic_add(&d[1],0u)==1u) t = atomic_inc(&d[2]);\n"
"  atomic_xchg(&d[0],0u);\n"
"  if(t==0u){\n"
"    uint last=1u, stable=0u;\n"
"    for(uint i=0u; i<50000000u && stable<100000u; ++i){\n"
"      uint c=atomic_add(&d[2],0u);\n"
"      if(c==last) stable++; else { stable=0u; last=c; }\n"
"    }\n"
"    while(atomic_cmpxchg(&d[0],0u,1u)!=0u);\n"
"    atomic_xchg(&d[1],0u);\n"
"    atomic_xchg(&d[0],0u);\n"
"  } else if(t!=NOT_RESIDENT){\n"
"    while(atomic_add(&d[1],0u)==1u);\n"
"  }\n"
"  return t;\n"
"}\n"
/* A: slim probe — minimal registers, no __local ---------------------------- */
"__kernel void probe_slim(volatile __global uint *d){\n"
"  if(get_local_id(0)==0u) discover(d);\n"
"}\n"
/* B: vm-like probe — vm2's __local footprint (As/Bs = 8 KB) + register\n"
 * pressure from a small live accumulator array, so the compiler's GRF choice\n"
 * resembles a real megakernel rather than a toy. */
"#define ASZ 1024\n"
"#define BSZ 1024\n"
"static uint fat_work(__local float *As, __local float *Bs, uint seed){\n"
"  float acc[16];\n"
"  for(int i=0;i<16;i++) acc[i]=(float)(seed+i);\n"
"  for(int k=0;k<16;k++)\n"
"    for(int i=0;i<16;i++)\n"
"      acc[i] = acc[i]*1.0000001f + acc[(i+k)&15];\n"
"  uint lid=get_local_id(0);\n"
"  As[lid&(ASZ-1)]=acc[0]; Bs[lid&(BSZ-1)]=acc[15];\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  return (uint)(As[(lid+1)&(ASZ-1)]+Bs[(lid+2)&(BSZ-1)]);\n"
"}\n"
"__kernel void probe_vmlike(volatile __global uint *d, __global uint *sink, uint never){\n"
"  __local float As[ASZ]; __local float Bs[BSZ];\n"
"  uint v = fat_work(As, Bs, get_group_id(0));\n"
"  if(never==1234567u) sink[get_group_id(0)]=v;   /* keep fat_work live */\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  if(get_local_id(0)==0u) discover(d);\n"
"}\n"
/* C: liveness — same footprint, T rounds of the shipped 1.2 spin-barrier ---- */
"static void bar12(volatile __global uint*b, uint ng){\n"
"  barrier(CLK_GLOBAL_MEM_FENCE);\n"
"  if(get_local_id(0)==0){\n"
"    uint ph=atomic_add(&b[1],0);\n"
"    if(atomic_inc(&b[0])==ng-1){ b[0]=0; mem_fence(CLK_GLOBAL_MEM_FENCE); atomic_inc(&b[1]); }\n"
"    else { while(atomic_add(&b[1],0)==ph); }\n"
"  }\n"
"  barrier(CLK_GLOBAL_MEM_FENCE);\n"
"}\n"
"__kernel void liveness(volatile __global uint *b, __global uint *sink, uint ng, uint T, uint never){\n"
"  __local float As[ASZ]; __local float Bs[BSZ];\n"
"  uint acc=0u;\n"
"  for(uint it=0; it<T; ++it){\n"
"    acc += fat_work(As, Bs, it);\n"
"    bar12(b, ng);\n"
"  }\n"
"  if(never==1234567u) sink[get_group_id(0)]=acc;\n"
"}\n";

#define CK(e) do{cl_int _e=(e); if(_e!=CL_SUCCESS){fprintf(stderr,"%s:%d CL err %d\n",__FILE__,__LINE__,_e);exit(1);} }while(0)

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }

/* Intel residency math via cl_intel_device_attribute_query (if present). */
#define CL_DEVICE_NUM_SLICES_INTEL               0x4252
#define CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL 0x4253
#define CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL    0x4254
#define CL_DEVICE_NUM_THREADS_PER_EU_INTEL       0x4255

static const size_t LOCAL = 256;   /* the VM's fixed local size */

static cl_uint run_discovery(cl_command_queue q, cl_kernel k, cl_mem d,
                             size_t launch_groups, double *ms){
  cl_uint zeros[3] = {0, 1, 0};    /* lock=0, gate=OPEN, count=0 */
  CK(clEnqueueWriteBuffer(q, d, CL_TRUE, 0, sizeof zeros, zeros, 0, NULL, NULL));
  size_t g = launch_groups * LOCAL, l = LOCAL;
  double t0 = now_ms();
  CK(clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL));
  CK(clFinish(q));
  *ms = now_ms() - t0;
  cl_uint out[3];
  CK(clEnqueueReadBuffer(q, d, CL_TRUE, 0, sizeof out, out, 0, NULL, NULL));
  return out[2];
}

int main(int argc, char **argv){
  int over = argc > 1 && !strcmp(argv[1], "--over");
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
  printf("platform: %s\ndevice  : %s (%u \"compute units\")\n\n",nm,dnm,cus);

  cl_int e;
  cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&e); CK(e);
  cl_command_queue q=clCreateCommandQueue(ctx,dev,0,&e); CK(e);
  cl_program p=clCreateProgramWithSource(ctx,1,&SRC,NULL,&e); CK(e);
  if(clBuildProgram(p,1,&dev,NULL,NULL,NULL)!=CL_SUCCESS){
    char log[8192]; size_t ls=0;
    clGetProgramBuildInfo(p,dev,CL_PROGRAM_BUILD_LOG,sizeof log,log,&ls);
    fprintf(stderr,"build failed:\n%.*s\n",(int)ls,log); return 1;
  }
  cl_kernel kslim=clCreateKernel(p,"probe_slim",&e); CK(e);
  cl_kernel kvm  =clCreateKernel(p,"probe_vmlike",&e); CK(e);
  cl_kernel klive=clCreateKernel(p,"liveness",&e); CK(e);

  cl_mem d   =clCreateBuffer(ctx,CL_MEM_READ_WRITE,64,NULL,&e); CK(e);
  cl_mem sink=clCreateBuffer(ctx,CL_MEM_READ_WRITE,4096*4,NULL,&e); CK(e);
  cl_uint never=0;

  /* Per-kernel compiled SIMD (subgroup) width — feeds the Intel formula. */
  size_t simd_slim=0, simd_vm=0;
  CK(clGetKernelWorkGroupInfo(kslim,dev,CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,sizeof simd_slim,&simd_slim,NULL));
  CK(clGetKernelWorkGroupInfo(kvm,dev,CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,sizeof simd_vm,&simd_vm,NULL));

  /* Oversized launch: 4x the CU count, floor 64 — deadlock-free regardless. */
  size_t launch = cus*4 < 64 ? 64 : cus*4;
  if (launch > 4096) launch = 4096;

  double ms;
  CK(clSetKernelArg(kslim,0,sizeof d,&d));
  cl_uint n_slim = run_discovery(q,kslim,d,launch,&ms);
  printf("[A] slim probe   : %4u co-resident groups @ local=%zu (SIMD %zu, launched %zu, %.1f ms)\n",
         n_slim,LOCAL,simd_slim,launch,ms);

  CK(clSetKernelArg(kvm,0,sizeof d,&d));
  CK(clSetKernelArg(kvm,1,sizeof sink,&sink));
  CK(clSetKernelArg(kvm,2,sizeof never,&never));
  cl_uint n_vm = run_discovery(q,kvm,d,launch,&ms);
  printf("[B] vm-like probe: %4u co-resident groups @ local=%zu (SIMD %zu, launched %zu, %.1f ms)\n\n",
         n_vm,LOCAL,simd_vm,launch,ms);

  /* Intel attribute math, when queryable. */
  cl_uint sl=0,ss=0,eu=0,th=0;
  if(clGetDeviceInfo(dev,CL_DEVICE_NUM_SLICES_INTEL,sizeof sl,&sl,NULL)==CL_SUCCESS &&
     clGetDeviceInfo(dev,CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL,sizeof ss,&ss,NULL)==CL_SUCCESS &&
     clGetDeviceInfo(dev,CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL,sizeof eu,&eu,NULL)==CL_SUCCESS &&
     clGetDeviceInfo(dev,CL_DEVICE_NUM_THREADS_PER_EU_INTEL,sizeof th,&th,NULL)==CL_SUCCESS){
    cl_uint hw_threads = sl*ss*eu*th;
    printf("    intel attrs: %u slices x %u subslices x %u EUs x %u thr = %u HW threads\n"
           "    formula    : %u / (%zu/%zu) = %u groups  (vm-like SIMD)\n\n",
           sl,ss,eu,th,hw_threads,hw_threads,LOCAL,simd_vm,
           hw_threads/(cl_uint)(LOCAL/simd_vm));
  }

  /* [C] liveness at the discovered count (and --over: one past it). */
  cl_uint ng = over ? n_vm+1 : n_vm, T = 2000;
  cl_uint bz[2]={0,0};
  CK(clEnqueueWriteBuffer(q,d,CL_TRUE,0,sizeof bz,bz,0,NULL,NULL));
  CK(clSetKernelArg(klive,0,sizeof d,&d));
  CK(clSetKernelArg(klive,1,sizeof sink,&sink));
  CK(clSetKernelArg(klive,2,sizeof ng,&ng));
  CK(clSetKernelArg(klive,3,sizeof T,&T));
  CK(clSetKernelArg(klive,4,sizeof never,&never));
  size_t g=(size_t)ng*LOCAL,l=LOCAL;
  double t0=now_ms();
  CK(clEnqueueNDRangeKernel(q,klive,1,NULL,&g,&l,0,NULL,NULL));
  cl_int fe=clFinish(q);
  double dt=now_ms()-t0;
  printf("[C] liveness @ %u groups x %u barrier rounds: %s (%.1f ms, %.2f us/barrier)\n\n",
         ng,T,fe==CL_SUCCESS?"PASS":"FAIL",dt,dt*1e3/T);
  if(fe!=CL_SUCCESS) printf("    clFinish err %d%s\n",fe,over?" (expected under --over)":"");

  printf("[D] heuristic vs measured (local=%zu):\n",LOCAL);
  printf("    2 x CUs (shipped GPU default) : %u\n",2*cus);
  printf("    CUs / 2                       : %u\n",cus/2);
  printf("    measured (vm-like discovery)  : %u\n",n_vm);
  return fe==CL_SUCCESS?0:1;
}
