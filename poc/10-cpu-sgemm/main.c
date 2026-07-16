/* poc/10 — CPU SGEMM ladder: how far does cache blocking take OpenCL-on-PoCL?
 *
 * poc/09 established the barrier-free register-blocked shape (b2: 4x16 float8,
 * 1 WI/WG): ~60-77 GFLOP/s, ~8x off XLA/Eigen (~620). Its k-loop reads B at
 * stride N*4B — a fresh cache line (and page) every iteration. The classic fix
 * ladder, each step measured here:
 *   v0  b2 baseline (stride-N B reads)
 *   v1  PACKED B: a parallel pre-pass reorders B into column-panel-major
 *       (panel = 16 columns, K x 16 contiguous), so the k-loop is sequential.
 *       Pack cost is included in every timing (it's O(K*N) vs O(M*N*K) work).
 *   v2  v1 + 6x16 register block (12 float8 accs + 2 B + 1 A broadcast = 15
 *       of 16 AVX2 ymm regs) — 1.5x less B traffic than 4 rows.
 *   v3  v2 + KC blocking (KC=512): C accumulated across KC sweeps so each
 *       sweep's B panel slice stays L2/L3-hot across row blocks.
 * Verification: every variant is checked against a scalar reference at N=256.
 *
 * PJRT_OCL_DEVICE selects the platform (default "Portable").
 * cc -O2 -Wall -o poc10 main.c -lOpenCL && ./poc10
 */
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define CK(e) do{cl_int _e=(e); if(_e!=CL_SUCCESS){fprintf(stderr,"%s:%d CL err %d\n",__FILE__,__LINE__,_e);exit(1);} }while(0)
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }

static const char *SRC =
/* pack B (KxN row-major) into panels: Bp[p*K*16 + k*16 + j] = B[k*N + p*16+j].
 * One WI per (panel,k) pair; two float8 moves. */
"__kernel void packB(__global const float*B, __global float*Bp, uint N, uint K){\n"
"  uint gid=get_global_id(0); uint np=N/16u;\n"
"  uint p=gid/K, k=gid%K; if(p>=np) return;\n"
"  float8 lo=vload8(0,B+k*N+p*16u), hi=vload8(0,B+k*N+p*16u+8u);\n"
"  vstore8(lo,0,Bp+(size_t)p*K*16u+k*16u);\n"
"  vstore8(hi,0,Bp+(size_t)p*K*16u+k*16u+8u);\n"
"}\n"
/* v0: poc/09 b2 — 4x16, stride-N B reads */
"__kernel void v0(__global const float*A, __global const float*B, __global float*C, uint M, uint N, uint K){\n"
"  uint r0=get_group_id(0)*4u; if(r0>=M) return;\n"
"  for(uint c0=0;c0<N;c0+=16u){\n"
"    float8 a0[4],a1[4];\n"
"    for(int i=0;i<4;i++){a0[i]=(float8)(0.0f);a1[i]=(float8)(0.0f);}\n"
"    for(uint k=0;k<K;k++){\n"
"      float8 b0=vload8(0,B+k*N+c0), b1=vload8(0,B+k*N+c0+8u);\n"
"      for(int i=0;i<4;i++){ float8 av=(float8)(A[(r0+i)*K+k]);\n"
"        a0[i]=mad(av,b0,a0[i]); a1[i]=mad(av,b1,a1[i]); }\n"
"    }\n"
"    for(int i=0;i<4;i++){ vstore8(a0[i],0,C+(r0+i)*N+c0); vstore8(a1[i],0,C+(r0+i)*N+c0+8u); }\n"
"  }\n"
"}\n"
/* v1: 4x16 on packed B */
"__kernel void v1(__global const float*A, __global const float*Bp, __global float*C, uint M, uint N, uint K){\n"
"  uint r0=get_group_id(0)*4u; if(r0>=M) return;\n"
"  for(uint p=0;p<N/16u;p++){\n"
"    __global const float*panel=Bp+(size_t)p*K*16u;\n"
"    float8 a0[4],a1[4];\n"
"    for(int i=0;i<4;i++){a0[i]=(float8)(0.0f);a1[i]=(float8)(0.0f);}\n"
"    for(uint k=0;k<K;k++){\n"
"      float8 b0=vload8(0,panel+k*16u), b1=vload8(0,panel+k*16u+8u);\n"
"      for(int i=0;i<4;i++){ float8 av=(float8)(A[(r0+i)*K+k]);\n"
"        a0[i]=mad(av,b0,a0[i]); a1[i]=mad(av,b1,a1[i]); }\n"
"    }\n"
"    for(int i=0;i<4;i++){ vstore8(a0[i],0,C+(r0+i)*N+p*16u); vstore8(a1[i],0,C+(r0+i)*N+p*16u+8u); }\n"
"  }\n"
"}\n"
/* v2: 6x16 on packed B (12 accs + 2 B + 1 A = 15 ymm) */
"__kernel void v2(__global const float*A, __global const float*Bp, __global float*C, uint M, uint N, uint K){\n"
"  uint r0=get_group_id(0)*6u; if(r0>=M) return;\n"
"  uint nr=min(6u,M-r0);\n"
"  if(nr==6u){\n"
"   for(uint p=0;p<N/16u;p++){\n"
"    __global const float*panel=Bp+(size_t)p*K*16u;\n"
"    float8 a0[6],a1[6];\n"
"    for(int i=0;i<6;i++){a0[i]=(float8)(0.0f);a1[i]=(float8)(0.0f);}\n"
"    for(uint k=0;k<K;k++){\n"
"      float8 b0=vload8(0,panel+k*16u), b1=vload8(0,panel+k*16u+8u);\n"
"      for(int i=0;i<6;i++){ float8 av=(float8)(A[(r0+i)*K+k]);\n"
"        a0[i]=mad(av,b0,a0[i]); a1[i]=mad(av,b1,a1[i]); }\n"
"    }\n"
"    for(int i=0;i<6;i++){ vstore8(a0[i],0,C+(r0+i)*N+p*16u); vstore8(a1[i],0,C+(r0+i)*N+p*16u+8u); }\n"
"   }\n"
"  } else {\n"
"   for(uint c0=0;c0<N;c0++) for(uint i=0;i<nr;i++){ float s=0.0f;\n"
"     for(uint k=0;k<K;k++) s=mad(A[(r0+i)*K+k], Bp[(size_t)(c0/16u)*K*16u+k*16u+(c0%16u)], s);\n"
"     C[(r0+i)*N+c0]=s; }\n"
"  }\n"
"}\n"
/* v3: v2 + KC blocking; C += per sweep (kc0>0 accumulates from memory) */
"__kernel void v3(__global const float*A, __global const float*Bp, __global float*C, uint M, uint N, uint K, uint kc0, uint kc1){\n"
"  uint r0=get_group_id(0)*6u; if(r0>=M) return;\n"
"  uint nr=min(6u,M-r0);\n"
"  if(nr==6u){\n"
"   for(uint p=0;p<N/16u;p++){\n"
"    __global const float*panel=Bp+(size_t)p*K*16u;\n"
"    float8 a0[6],a1[6];\n"
"    if(kc0==0u){ for(int i=0;i<6;i++){a0[i]=(float8)(0.0f);a1[i]=(float8)(0.0f);} }\n"
"    else for(int i=0;i<6;i++){ a0[i]=vload8(0,C+(r0+i)*N+p*16u); a1[i]=vload8(0,C+(r0+i)*N+p*16u+8u); }\n"
"    for(uint k=kc0;k<kc1;k++){\n"
"      float8 b0=vload8(0,panel+k*16u), b1=vload8(0,panel+k*16u+8u);\n"
"      for(int i=0;i<6;i++){ float8 av=(float8)(A[(r0+i)*K+k]);\n"
"        a0[i]=mad(av,b0,a0[i]); a1[i]=mad(av,b1,a1[i]); }\n"
"    }\n"
"    for(int i=0;i<6;i++){ vstore8(a0[i],0,C+(r0+i)*N+p*16u); vstore8(a1[i],0,C+(r0+i)*N+p*16u+8u); }\n"
"   }\n"
"  } else if(kc0==0u){\n"
"   for(uint c0=0;c0<N;c0++) for(uint i=0;i<nr;i++){ float s=0.0f;\n"
"     for(uint k=0;k<K;k++) s=mad(A[(r0+i)*K+k], Bp[(size_t)(c0/16u)*K*16u+k*16u+(c0%16u)], s);\n"
"     C[(r0+i)*N+c0]=s; }\n"
"  }\n"
"}\n";

typedef struct { cl_context ctx; cl_command_queue q; cl_program p; } Env;

static void ref_mm(const float*A,const float*B,float*C,int M,int N,int K){
  for(int i=0;i<M;i++) for(int j=0;j<N;j++){ float s=0;
    for(int k=0;k<K;k++) s+=A[i*K+k]*B[k*N+j]; C[i*N+j]=s; }
}

static double run_variant(Env*e, const char*name, cl_mem A, cl_mem B, cl_mem Bp,
                          cl_mem C, cl_uint M, cl_uint N, cl_uint K,
                          int pack, int rows, cl_uint KC, int iters){
  cl_int err;
  cl_kernel kp = pack ? clCreateKernel(e->p,"packB",&err) : NULL;
  cl_kernel k = clCreateKernel(e->p,name,&err); CK(err);
  double best=1e30;
  for(int r=0;r<3;r++){
    double t0=now_ms();
    for(int it=0;it<iters;it++){
      if(pack){
        CK(clSetKernelArg(kp,0,8,&B)); CK(clSetKernelArg(kp,1,8,&Bp));
        CK(clSetKernelArg(kp,2,4,&N)); CK(clSetKernelArg(kp,3,4,&K));
        size_t g=(size_t)(N/16)*K;
        CK(clEnqueueNDRangeKernel(e->q,kp,1,NULL,&g,NULL,0,NULL,NULL));
      }
      cl_mem bsrc = pack ? Bp : B;
      CK(clSetKernelArg(k,0,8,&A)); CK(clSetKernelArg(k,1,8,&bsrc));
      CK(clSetKernelArg(k,2,8,&C)); CK(clSetKernelArg(k,3,4,&M));
      CK(clSetKernelArg(k,4,4,&N)); CK(clSetKernelArg(k,5,4,&K));
      size_t g=(M+rows-1)/rows, l=1;
      if(KC){
        for(cl_uint kc=0;kc<K;kc+=KC){
          cl_uint kc1=kc+KC<K?kc+KC:K;
          CK(clSetKernelArg(k,6,4,&kc)); CK(clSetKernelArg(k,7,4,&kc1));
          CK(clEnqueueNDRangeKernel(e->q,k,1,NULL,&g,&l,0,NULL,NULL));
        }
      } else {
        CK(clEnqueueNDRangeKernel(e->q,k,1,NULL,&g,&l,0,NULL,NULL));
      }
    }
    CK(clFinish(e->q));
    double dt=(now_ms()-t0)/iters; if(dt<best)best=dt;
  }
  if(kp)clReleaseKernel(kp); clReleaseKernel(k);
  return best;
}

static int verify(Env*e, const char*name, int pack, int rows, cl_uint KC){
  const int M=192, N=256, K=224;   /* deliberately non-square, M%6!=0 */
  float *A=malloc(M*K*4),*B=malloc(K*N*4),*C=malloc(M*N*4),*R=malloc(M*N*4);
  srand(7);
  for(int i=0;i<M*K;i++)A[i]=(rand()%17-8)/4.0f;
  for(int i=0;i<K*N;i++)B[i]=(rand()%17-8)/4.0f;
  ref_mm(A,B,R,M,N,K);
  cl_int err;
  cl_mem dA=clCreateBuffer(e->ctx,CL_MEM_COPY_HOST_PTR,M*K*4,A,&err);CK(err);
  cl_mem dB=clCreateBuffer(e->ctx,CL_MEM_COPY_HOST_PTR,K*N*4,B,&err);CK(err);
  cl_mem dBp=clCreateBuffer(e->ctx,CL_MEM_READ_WRITE,(size_t)K*N*4,NULL,&err);CK(err);
  cl_mem dC=clCreateBuffer(e->ctx,CL_MEM_READ_WRITE,M*N*4,NULL,&err);CK(err);
  run_variant(e,name,dA,dB,dBp,dC,M,N,K,pack,rows,KC,1);
  CK(clEnqueueReadBuffer(e->q,dC,CL_TRUE,0,M*N*4,C,0,NULL,NULL));
  int bad=0;
  for(int i=0;i<M*N;i++) if(fabsf(C[i]-R[i])>1e-3f*(1.0f+fabsf(R[i]))) bad++;
  clReleaseMemObject(dA);clReleaseMemObject(dB);clReleaseMemObject(dBp);clReleaseMemObject(dC);
  free(A);free(B);free(C);free(R);
  return bad;
}

int main(void){
  const char *want=getenv("PJRT_OCL_DEVICE"); if(!want)want="Portable";
  cl_uint np; CK(clGetPlatformIDs(0,NULL,&np));
  cl_platform_id ps[8]; CK(clGetPlatformIDs(np<8?np:8,ps,NULL));
  cl_device_id dev=0; char nm[256];
  for(cl_uint i=0;i<np;i++){
    CK(clGetPlatformInfo(ps[i],CL_PLATFORM_NAME,sizeof nm,nm,NULL));
    if(!strstr(nm,want)) continue;
    cl_uint nd; if(clGetDeviceIDs(ps[i],CL_DEVICE_TYPE_ALL,0,NULL,&nd)!=CL_SUCCESS||!nd) continue;
    cl_device_id ds[4]; CK(clGetDeviceIDs(ps[i],CL_DEVICE_TYPE_ALL,1,ds,NULL)); dev=ds[0]; break;
  }
  if(!dev){fprintf(stderr,"no device\n");return 1;}
  char dnm[256]; CK(clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof dnm,dnm,NULL));
  printf("device: %s\n\n",dnm);
  Env e; cl_int err;
  e.ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&err);CK(err);
  e.q=clCreateCommandQueue(e.ctx,dev,0,&err);CK(err);
  e.p=clCreateProgramWithSource(e.ctx,1,&SRC,NULL,&err);CK(err);
  if(clBuildProgram(e.p,1,&dev,NULL,NULL,NULL)!=CL_SUCCESS){
    char log[16384]; size_t ls=0;
    clGetProgramBuildInfo(e.p,dev,CL_PROGRAM_BUILD_LOG,sizeof log,log,&ls);
    fprintf(stderr,"build:\n%.*s\n",(int)ls,log);return 1;}

  struct { const char*name; int pack, rows; cl_uint KC; } vs[] = {
    {"v0",0,4,0}, {"v1",1,4,0}, {"v2",1,6,0}, {"v3",1,6,512},
  };
  for(int i=0;i<4;i++){
    int bad=verify(&e,vs[i].name,vs[i].pack,vs[i].rows,vs[i].KC);
    printf("[%s] verify: %s\n",vs[i].name,bad?"FAIL":"ok");
    if(bad){ printf("  (%d bad)\n",bad); continue; }
  }
  printf("\n");
  for(cl_uint Nn=1024; Nn<=2048; Nn*=2){
    cl_uint M=Nn,N=Nn,K=Nn;
    cl_mem A=clCreateBuffer(e.ctx,CL_MEM_READ_WRITE,(size_t)M*K*4,NULL,&err);CK(err);
    cl_mem B=clCreateBuffer(e.ctx,CL_MEM_READ_WRITE,(size_t)K*N*4,NULL,&err);CK(err);
    cl_mem Bp=clCreateBuffer(e.ctx,CL_MEM_READ_WRITE,(size_t)K*N*4,NULL,&err);CK(err);
    cl_mem C=clCreateBuffer(e.ctx,CL_MEM_READ_WRITE,(size_t)M*N*4,NULL,&err);CK(err);
    const double GF=2.0*(double)M*N*K/1e6;
    printf("N=%u:\n",Nn);
    for(int i=0;i<4;i++){
      double ms=run_variant(&e,vs[i].name,A,B,Bp,C,M,N,K,vs[i].pack,vs[i].rows,vs[i].KC,1);
      printf("  [%s]%s %8.2f ms  %6.1f GFLOP/s\n",vs[i].name,
             vs[i].pack?" +pack":"      ",ms,GF/ms);
    }
    clReleaseMemObject(A);clReleaseMemObject(B);clReleaseMemObject(Bp);clReleaseMemObject(C);
  }
  return 0;
}
