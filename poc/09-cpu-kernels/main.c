/* poc/09 — why is XLA CPU beating us on PoCL, per op class?
 *
 * Benchmarks (README has the results table):
 *   [0] machine baseline: single-thread + 8-thread memcpy GB/s (the wall).
 *   [A] EW d=a+b, 16M floats, five kernel patterns on PoCL:
 *       a1 current VM pattern: WG=256, per-WI stride-lsz loop within a 16K tile
 *       a2 classic 1-elem-per-WI (get_global_id), WG=256
 *       a3 contiguous chunk per WI (each WI owns CHUNK consecutive elems)
 *       a4 float8 chunk per WI (explicit 8-wide vectors)
 *       a5 one WI per WG, 8 WGs, big contiguous float8 loop per WG
 *   [B] SGEMM C=A*B, N=1024, three patterns:
 *       b1 the VM's MMA tile shape: 256-WI WG, __local As/Bs staging, barriers
 *       b2 one WI per WG, registers-blocked 4x16 microkernel, float8, no local
 *       b3 like b2 but one row of C per WG (finer-grained, 1024 WGs)
 *   [C] GEMV y=A*x, N=2048: c1 via MMA-tile shape vs c2 one-row-per-WI dot.
 *   [D] dispatch floor: empty-kernel enqueue+finish latency, and 8-WG add(4K).
 *
 * All on the FIRST platform matching PJRT_OCL_DEVICE (default "Portable").
 * cc -O2 -pthread -o poc09 main.c -lOpenCL
 */
#include <CL/cl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CK(e) do{cl_int _e=(e); if(_e!=CL_SUCCESS){fprintf(stderr,"%s:%d CL err %d\n",__FILE__,__LINE__,_e);exit(1);} }while(0)
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }

static const char *SRC =
"#define EW_TS 16384u\n"
/* a1: the VM's current pattern (grid of tiles, WI strided inside a tile) */
"__kernel void a1(__global const float*a, __global const float*b, __global float*d, uint n){\n"
"  uint tile=get_group_id(0), lid=get_local_id(0), lsz=get_local_size(0);\n"
"  uint lo=tile*EW_TS, hi=min(lo+EW_TS,n);\n"
"  for(uint i=lo+lid;i<hi;i+=lsz) d[i]=a[i]+b[i];\n"
"}\n"
/* a1b: same tile interface as a1, but the WI loop restructured so the body is
 * straight-line (uniform outer loop over lsz-strided BLOCKS, WI picks one
 * element per block) — the hypothesis is PoCL's work-group vectorizer handles
 * this where a1's per-WI strided loop defeats it. Memory pattern identical. */
"__kernel void a1b(__global const float*a, __global const float*b, __global float*d, uint n){\n"
"  uint tile=get_group_id(0), lid=get_local_id(0), lsz=get_local_size(0);\n"
"  uint lo=tile*EW_TS, hi=min(lo+EW_TS,n);\n"
"  for(uint base=lo;base<hi;base+=lsz){ uint i=base+lid;\n"
"    if(i<hi) d[i]=a[i]+b[i]; }\n"
"}\n"
/* a1c: like a1b but NO guard — full tiles only (EW_TS % lsz == 0), body is
 * branch-free straight-line. Isolates guard-vs-loop as the vectorizer blocker. */
"__kernel void a1c(__global const float*a, __global const float*b, __global float*d, uint n){\n"
"  uint tile=get_group_id(0), lid=get_local_id(0), lsz=get_local_size(0);\n"
"  uint lo=tile*EW_TS;\n"
"  for(uint base=lo;base<lo+EW_TS;base+=lsz){ uint i=base+lid;\n"
"    d[i]=a[i]+b[i]; }\n"
"}\n"
/* a2: classic 1 element per work-item */
"__kernel void a2(__global const float*a, __global const float*b, __global float*d, uint n){\n"
"  uint i=get_global_id(0); if(i<n) d[i]=a[i]+b[i];\n"
"}\n"
/* a3: contiguous chunk per work-item */
"__kernel void a3(__global const float*a, __global const float*b, __global float*d, uint n, uint chunk){\n"
"  uint lo=get_global_id(0)*chunk, hi=min(lo+chunk,n);\n"
"  for(uint i=lo;i<hi;++i) d[i]=a[i]+b[i];\n"
"}\n"
/* a4: float8 chunk per work-item (n multiple of 8 assumed) */
"__kernel void a4(__global const float8*a, __global const float8*b, __global float8*d, uint n8, uint chunk8){\n"
"  uint lo=get_global_id(0)*chunk8, hi=min(lo+chunk8,n8);\n"
"  for(uint i=lo;i<hi;++i) d[i]=a[i]+b[i];\n"
"}\n"
/* a5: one WI per WG, one big contiguous float8 span per WG */
"__kernel void a5(__global const float8*a, __global const float8*b, __global float8*d, uint n8, uint span8){\n"
"  uint lo=get_group_id(0)*span8, hi=min(lo+span8,n8);\n"
"  for(uint i=lo;i<hi;++i) d[i]=a[i]+b[i];\n"
"}\n"
/* a6: THE candidate for the real VM tile: float8 per WI, WI-coalesced stride
 * (same iteration space as a1, vector-widened). Portable: on GPU this is
 * adjacent 32B vector loads per WI (coalesced); on PoCL the body is already
 * a vector op so no cross-WI vectorization is needed. */
"__kernel void a6(__global const float*a, __global const float*b, __global float*d, uint n){\n"
"  uint tile=get_group_id(0), lid=get_local_id(0), lsz=get_local_size(0);\n"
"  uint lo8=tile*(EW_TS/8), hi8=min(lo8+EW_TS/8, n/8);\n"
"  for(uint j=lo8+lid;j<hi8;j+=lsz)\n"
"    vstore8(vload8(j,a)+vload8(j,b), j, d);\n"
"}\n"
/* b1: the VM's MMA tile shape — 64x64 C tile, BK=16 panels staged in local,\n"
 * 16x16 WI grid, 4x4 register microtile, barriers per K panel. */
"#define TM 64\n#define TN 64\n#define BK 16\n#define TD 16\n"
"__kernel void b1(__global const float*A, __global const float*B, __global float*C, uint N){\n"
"  __local float As[BK*TM]; __local float Bs[BK*TN];\n"
"  uint tx=get_local_id(0)%TD, ty=get_local_id(0)/TD;\n"
"  uint tiles=N/TN, tile=get_group_id(0);\n"
"  uint tm=(tile/tiles)*TM, tn=(tile%tiles)*TN;\n"
"  float acc[4][4]; for(int i=0;i<4;i++)for(int j=0;j<4;j++)acc[i][j]=0.0f;\n"
"  for(uint k0=0;k0<N;k0+=BK){\n"
"    for(uint t=get_local_id(0); t<BK*TM; t+=TD*TD){\n"
"      uint r=t/BK, c=t%BK; As[r*BK+c]=A[(tm+r)*N+k0+c]; }\n"
"    for(uint t=get_local_id(0); t<BK*TN; t+=TD*TD){\n"
"      uint r=t/TN, c=t%TN; Bs[r*TN+c]=B[(k0+r)*N+tn+c]; }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for(uint k=0;k<BK;k++)\n"
"      for(int i=0;i<4;i++){ float av=As[(ty*4+i)*BK+k];\n"
"        for(int j=0;j<4;j++) acc[i][j]=mad(av,Bs[k*TN+tx*4+j],acc[i][j]); }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"  }\n"
"  for(int i=0;i<4;i++)for(int j=0;j<4;j++) C[(tm+ty*4+i)*N+tn+tx*4+j]=acc[i][j];\n"
"}\n"
/* b2: CPU-shaped SGEMM — one WI per WG, 4-row x 2*float8-col register block,\n"
 * no local memory, no barriers; WG count = row-blocks (N/4). */
"__kernel void b2(__global const float*A, __global const float*B, __global float*C, uint N){\n"
"  uint r0=get_group_id(0)*4;\n"
"  for(uint c0=0;c0<N;c0+=16){\n"
"    float8 acc[4][2];\n"
"    for(int i=0;i<4;i++){acc[i][0]=(float8)(0.0f);acc[i][1]=(float8)(0.0f);}\n"
"    for(uint k=0;k<N;k++){\n"
"      float8 b0=vload8(0,B+k*N+c0), b1=vload8(0,B+k*N+c0+8);\n"
"      for(int i=0;i<4;i++){ float av=A[(r0+i)*N+k];\n"
"        acc[i][0]=mad((float8)(av),b0,acc[i][0]);\n"
"        acc[i][1]=mad((float8)(av),b1,acc[i][1]); }\n"
"    }\n"
"    for(int i=0;i<4;i++){ vstore8(acc[i][0],0,C+(r0+i)*N+c0);\n"
"                          vstore8(acc[i][1],0,C+(r0+i)*N+c0+8); }\n"
"  }\n"
"}\n"
/* b3: b2's 4x16 microkernel EMBEDDED in the megakernel tile interface —
 * 256-WI workgroup, one 64x64 output tile, WIs 0..63 each own a 4-row x
 * 16-col block, WIs 64..255 idle. Measures whether idle WIs in the CPU
 * work-item loop cost anything vs b2's 1-WI/WG shape. */
"__kernel void b3(__global const float*A, __global const float*B, __global float*C, uint N){\n"
"  uint tiles_n=N/64u, tile=get_group_id(0);\n"
"  uint row0=(tile/tiles_n)*64u, col0=(tile%tiles_n)*64u;\n"
"  uint lid=get_local_id(0);\n"
"  if(lid<64u){\n"
"    uint r0=row0+(lid/4u)*4u, c0=col0+(lid%4u)*16u;\n"
"    float8 a0[4],a1[4];\n"
"    for(int i=0;i<4;i++){a0[i]=(float8)(0.0f);a1[i]=(float8)(0.0f);}\n"
"    for(uint k=0;k<N;k++){\n"
"      float8 b0=vload8(0,B+k*N+c0), b1=vload8(0,B+k*N+c0+8u);\n"
"      for(int i=0;i<4;i++){ float8 av=(float8)(A[(r0+i)*N+k]);\n"
"        a0[i]=mad(av,b0,a0[i]); a1[i]=mad(av,b1,a1[i]); }\n"
"    }\n"
"    for(int i=0;i<4;i++){ vstore8(a0[i],0,C+(r0+i)*N+c0); vstore8(a1[i],0,C+(r0+i)*N+c0+8u); }\n"
"  }\n"
"}\n"
/* c1: GEMV via the MMA-tile shape is approximated by b1 with N-col=1 padded\n"
 * tile — measured separately in the harness by calling b1 on Nx64 (wasteful\n"
 * width) is not representative; instead c1 = classic 1-row-per-WI scalar dot,\n"
 * c2 = 1-row-per-WI float8 dot. */
"__kernel void c1(__global const float*A, __global const float*x, __global float*y, uint N){\n"
"  uint r=get_global_id(0); if(r>=N) return; float s=0.0f;\n"
"  for(uint k=0;k<N;k++) s=mad(A[r*N+k],x[k],s); y[r]=s;\n"
"}\n"
"__kernel void c2(__global const float*A, __global const float*x, __global float*y, uint N){\n"
"  uint r=get_global_id(0); if(r>=N) return; float8 s=(float8)(0.0f);\n"
"  for(uint k=0;k<N;k+=8) s=mad(vload8(0,A+r*N+k),vload8(0,x+k),s);\n"
"  y[r]=s.s0+s.s1+s.s2+s.s3+s.s4+s.s5+s.s6+s.s7;\n"
"}\n"
"__kernel void empty(__global float*d){ (void)d; }\n";

/* ---- [0] memcpy baseline ---------------------------------------------------- */
typedef struct { float *dst; const float *src; size_t n; } cpyarg;
static void *cpy_thread(void *p){ cpyarg*a=p; memcpy(a->dst,a->src,a->n*4); return 0; }

static void baseline(size_t n){
  float *a=malloc(n*4), *b=malloc(n*4);
  memset(a,1,n*4); memset(b,2,n*4);
  double best1=1e30, best8=1e30;
  for(int r=0;r<5;r++){ double t0=now_ms(); memcpy(b,a,n*4); double dt=now_ms()-t0; if(dt<best1)best1=dt; }
  pthread_t th[8]; cpyarg args[8]; size_t per=n/8;
  for(int r=0;r<5;r++){
    double t0=now_ms();
    for(int i=0;i<8;i++){ args[i]=(cpyarg){b+i*per,a+i*per,per}; pthread_create(&th[i],0,cpy_thread,&args[i]); }
    for(int i=0;i<8;i++) pthread_join(th[i],0);
    double dt=now_ms()-t0; if(dt<best8)best8=dt;
  }
  /* memcpy traffic = read n + write n (write-allocate makes it ~3n; report nominal 2n) */
  printf("[0] memcpy %zuMB: 1-thread %.1f GB/s, 8-thread %.1f GB/s (nominal r+w)\n",
         n*4>>20, 2.0*n*4/best1/1e6, 2.0*n*4/best8/1e6);
  free(a); free(b);
}

static cl_kernel K(cl_program p, const char*name){ cl_int e; cl_kernel k=clCreateKernel(p,name,&e); CK(e); return k; }

static double run(cl_command_queue q, cl_kernel k, size_t g, size_t l, int iters){
  size_t gg=g, ll=l;
  CK(clEnqueueNDRangeKernel(q,k,1,NULL,&gg,l?&ll:NULL,0,NULL,NULL)); CK(clFinish(q)); /* warm */
  double best=1e30;
  for(int r=0;r<5;r++){
    double t0=now_ms();
    for(int i=0;i<iters;i++) CK(clEnqueueNDRangeKernel(q,k,1,NULL,&gg,l?&ll:NULL,0,NULL,NULL));
    CK(clFinish(q));
    double dt=(now_ms()-t0)/iters; if(dt<best)best=dt;
  }
  return best;
}

int main(void){
  const char *want = getenv("PJRT_OCL_DEVICE"); if(!want) want="Portable";
  cl_uint np; CK(clGetPlatformIDs(0,NULL,&np));
  cl_platform_id ps[8]; CK(clGetPlatformIDs(np<8?np:8,ps,NULL));
  cl_device_id dev=0; char nm[256];
  for(cl_uint i=0;i<np;i++){
    CK(clGetPlatformInfo(ps[i],CL_PLATFORM_NAME,sizeof nm,nm,NULL));
    if(!strstr(nm,want)) continue;
    cl_uint nd; if(clGetDeviceIDs(ps[i],CL_DEVICE_TYPE_ALL,0,NULL,&nd)!=CL_SUCCESS||!nd) continue;
    cl_device_id ds[4]; CK(clGetDeviceIDs(ps[i],CL_DEVICE_TYPE_ALL,1,ds,NULL)); dev=ds[0]; break;
  }
  if(!dev){ fprintf(stderr,"no device matching %s\n",want); return 1; }
  char dnm[256]; CK(clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof dnm,dnm,NULL));
  printf("device: %s\n\n", dnm);

  const size_t N = 16u<<20;         /* 16M floats */
  baseline(N);

  cl_int e;
  cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&e); CK(e);
  cl_command_queue q=clCreateCommandQueue(ctx,dev,0,&e); CK(e);
  cl_program p=clCreateProgramWithSource(ctx,1,&SRC,NULL,&e); CK(e);
  if(clBuildProgram(p,1,&dev,NULL,NULL,NULL)!=CL_SUCCESS){
    char log[16384]; size_t ls=0;
    clGetProgramBuildInfo(p,dev,CL_PROGRAM_BUILD_LOG,sizeof log,log,&ls);
    fprintf(stderr,"build failed:\n%.*s\n",(int)ls,log); return 1; }

  cl_mem A=clCreateBuffer(ctx,CL_MEM_READ_WRITE,N*4,NULL,&e); CK(e);
  cl_mem B=clCreateBuffer(ctx,CL_MEM_READ_WRITE,N*4,NULL,&e); CK(e);
  cl_mem D=clCreateBuffer(ctx,CL_MEM_READ_WRITE,N*4,NULL,&e); CK(e);
  { /* touch to commit pages */
    cl_kernel k=K(p,"a2"); cl_uint n=N;
    CK(clSetKernelArg(k,0,8,&A)); CK(clSetKernelArg(k,1,8,&B)); CK(clSetKernelArg(k,2,8,&D)); CK(clSetKernelArg(k,3,4,&n));
    size_t g=N; CK(clEnqueueNDRangeKernel(q,k,1,NULL,&g,NULL,0,NULL,NULL)); CK(clFinish(q)); clReleaseKernel(k);
  }
  const double GB = 3.0*N*4/1e6;    /* ms -> GB/s for 2 reads + 1 write */
  cl_uint n=N, n8=N/8;

  { cl_kernel k=K(p,"a1"); CK(clSetKernelArg(k,0,8,&A));CK(clSetKernelArg(k,1,8,&B));CK(clSetKernelArg(k,2,8,&D));CK(clSetKernelArg(k,3,4,&n));
    double ms=run(q,k,(N/16384)*256,256,3);
    printf("[a1] VM tile/stride  : %7.2f ms  %6.1f GB/s\n",ms,GB/ms); clReleaseKernel(k); }
  { cl_kernel k=K(p,"a1b"); CK(clSetKernelArg(k,0,8,&A));CK(clSetKernelArg(k,1,8,&B));CK(clSetKernelArg(k,2,8,&D));CK(clSetKernelArg(k,3,4,&n));
    double ms=run(q,k,(N/16384)*256,256,3);
    printf("[a1b] VM tile restru : %7.2f ms  %6.1f GB/s\n",ms,GB/ms); clReleaseKernel(k); }
  { cl_kernel k=K(p,"a1c"); CK(clSetKernelArg(k,0,8,&A));CK(clSetKernelArg(k,1,8,&B));CK(clSetKernelArg(k,2,8,&D));CK(clSetKernelArg(k,3,4,&n));
    double ms=run(q,k,(N/16384)*256,256,3);
    printf("[a1c] restru noguard : %7.2f ms  %6.1f GB/s\n",ms,GB/ms); clReleaseKernel(k); }
  { cl_kernel k=K(p,"a2"); CK(clSetKernelArg(k,0,8,&A));CK(clSetKernelArg(k,1,8,&B));CK(clSetKernelArg(k,2,8,&D));CK(clSetKernelArg(k,3,4,&n));
    double ms=run(q,k,N,256,3);
    printf("[a2] 1-elem/WI       : %7.2f ms  %6.1f GB/s\n",ms,GB/ms); clReleaseKernel(k); }
  { cl_kernel k=K(p,"a3"); cl_uint chunk=16384; cl_uint nwi=(N+chunk-1)/chunk;
    CK(clSetKernelArg(k,0,8,&A));CK(clSetKernelArg(k,1,8,&B));CK(clSetKernelArg(k,2,8,&D));CK(clSetKernelArg(k,3,4,&n));CK(clSetKernelArg(k,4,4,&chunk));
    double ms=run(q,k,nwi,128,3);
    printf("[a3] chunk/WI scalar : %7.2f ms  %6.1f GB/s\n",ms,GB/ms); clReleaseKernel(k); }
  { cl_kernel k=K(p,"a4"); cl_uint chunk8=2048; cl_uint nwi=(n8+chunk8-1)/chunk8;
    CK(clSetKernelArg(k,0,8,&A));CK(clSetKernelArg(k,1,8,&B));CK(clSetKernelArg(k,2,8,&D));CK(clSetKernelArg(k,3,4,&n8));CK(clSetKernelArg(k,4,4,&chunk8));
    double ms=run(q,k,nwi,128,3);
    printf("[a4] chunk/WI float8 : %7.2f ms  %6.1f GB/s\n",ms,GB/ms); clReleaseKernel(k); }
  { cl_kernel k=K(p,"a5"); cl_uint span8=n8/8;
    CK(clSetKernelArg(k,0,8,&A));CK(clSetKernelArg(k,1,8,&B));CK(clSetKernelArg(k,2,8,&D));CK(clSetKernelArg(k,3,4,&n8));CK(clSetKernelArg(k,4,4,&span8));
    double ms=run(q,k,8,1,3);
    printf("[a5] 8 WG x 1 WI f8  : %7.2f ms  %6.1f GB/s\n\n",ms,GB/ms); clReleaseKernel(k); }

  { cl_kernel k=K(p,"a6"); CK(clSetKernelArg(k,0,8,&A));CK(clSetKernelArg(k,1,8,&B));CK(clSetKernelArg(k,2,8,&D));CK(clSetKernelArg(k,3,4,&n));
    double ms=run(q,k,(N/16384)*256,256,3);
    printf("[a6]  f8 coalesced   : %7.2f ms  %6.1f GB/s\n",ms,GB/ms); clReleaseKernel(k); }

  /* [B] SGEMM N=1024 */
  const cl_uint M=1024; const double GF=2.0*(double)M*M*M/1e6; /* ms -> GFLOP/s */
  { cl_kernel k=K(p,"b1"); CK(clSetKernelArg(k,0,8,&A));CK(clSetKernelArg(k,1,8,&B));CK(clSetKernelArg(k,2,8,&D));CK(clSetKernelArg(k,3,4,&M));
    double ms=run(q,k,(M/64)*(M/64)*256,256,1);
    printf("[b1] MMA-tile shape  : %7.2f ms  %6.1f GFLOP/s\n",ms,GF/ms); clReleaseKernel(k); }
  { cl_kernel k=K(p,"b2"); CK(clSetKernelArg(k,0,8,&A));CK(clSetKernelArg(k,1,8,&B));CK(clSetKernelArg(k,2,8,&D));CK(clSetKernelArg(k,3,4,&M));
    double ms=run(q,k,M/4,1,1);
    printf("[b2] CPU 4x16 f8     : %7.2f ms  %6.1f GFLOP/s\n\n",ms,GF/ms); clReleaseKernel(k); }

  { cl_kernel k=K(p,"b3"); CK(clSetKernelArg(k,0,8,&A));CK(clSetKernelArg(k,1,8,&B));CK(clSetKernelArg(k,2,8,&D));CK(clSetKernelArg(k,3,4,&M));
    double ms=run(q,k,(size_t)(M/64)*(M/64)*256,256,1);
    printf("[b3] b2-in-tile-iface: %7.2f ms  %6.1f GFLOP/s\n\n",ms,GF/ms); clReleaseKernel(k); }
  /* [C] GEMV N=2048 */
  { const cl_uint G2=2048; const double GBv=(double)G2*G2*4/1e6;
    cl_kernel k=K(p,"c1"); CK(clSetKernelArg(k,0,8,&A));CK(clSetKernelArg(k,1,8,&B));CK(clSetKernelArg(k,2,8,&D));CK(clSetKernelArg(k,3,4,&G2));
    double ms=run(q,k,G2,256,3);
    printf("[c1] GEMV row/WI     : %7.2f ms  %6.1f GB/s\n",ms,GBv/ms); clReleaseKernel(k);
    k=K(p,"c2"); CK(clSetKernelArg(k,0,8,&A));CK(clSetKernelArg(k,1,8,&B));CK(clSetKernelArg(k,2,8,&D));CK(clSetKernelArg(k,3,4,&G2));
    ms=run(q,k,G2,256,3);
    printf("[c2] GEMV row/WI f8  : %7.2f ms  %6.1f GB/s\n\n",ms,GBv/ms); clReleaseKernel(k); }

  /* [D] dispatch floor */
  { cl_kernel k=K(p,"empty"); CK(clSetKernelArg(k,0,8,&D));
    double ms=run(q,k,8*256,256,100);
    printf("[d1] empty 8WGx256   : %7.3f ms/launch (enqueue+finish)\n",ms); clReleaseKernel(k); }
  { cl_kernel k=K(p,"a2"); cl_uint sn=4096;
    CK(clSetKernelArg(k,0,8,&A));CK(clSetKernelArg(k,1,8,&B));CK(clSetKernelArg(k,2,8,&D));CK(clSetKernelArg(k,3,4,&sn));
    double ms=run(q,k,4096,256,100);
    printf("[d2] add 4K          : %7.3f ms/launch\n",ms); clReleaseKernel(k); }
  return 0;
}
