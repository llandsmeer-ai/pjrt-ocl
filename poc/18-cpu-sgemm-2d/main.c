#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define CK(e) do{cl_int _e=(e); if(_e!=CL_SUCCESS){fprintf(stderr,"%s:%d CL err %d\n",__FILE__,__LINE__,_e);exit(1);} }while(0)
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }

static const char *SRC =
"__kernel void packB(__global const float*B, __global float*Bp, uint N, uint K){\n"
"  uint gid=get_global_id(0); uint np=N/16u;\n"
"  uint p=gid/K, k=gid%K; if(p>=np) return;\n"
"  float8 lo=vload8(0,B+k*N+p*16u), hi=vload8(0,B+k*N+p*16u+8u);\n"
"  vstore8(lo,0,Bp+(size_t)p*K*16u+k*16u);\n"
"  vstore8(hi,0,Bp+(size_t)p*K*16u+k*16u+8u);\n"
"}\n"
/* v3 reference: 6x16 array-accumulator KC */
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
"}\n"
/* v4: 6x16, FULLY UNROLLED named accumulators (no arrays) */
"#define ACC6(b, A0,A1,A2,A3,A4,A5) do{ \\\n"
"  float8 av0=(float8)(A0),av1=(float8)(A1),av2=(float8)(A2),av3=(float8)(A3),av4=(float8)(A4),av5=(float8)(A5); \\\n"
"  c00=mad(av0,b0,c00); c10=mad(av0,b1,c10); \\\n"
"  c01=mad(av1,b0,c01); c11=mad(av1,b1,c11); \\\n"
"  c02=mad(av2,b0,c02); c12=mad(av2,b1,c12); \\\n"
"  c03=mad(av3,b0,c03); c13=mad(av3,b1,c13); \\\n"
"  c04=mad(av4,b0,c04); c14=mad(av4,b1,c14); \\\n"
"  c05=mad(av5,b0,c05); c15=mad(av5,b1,c15); }while(0)\n"
"__kernel void v4(__global const float*A, __global const float*Bp, __global float*C, uint M, uint N, uint K, uint kc0, uint kc1){\n"
"  uint r0=get_group_id(0)*6u; if(r0>=M) return;\n"
"  if(r0+6u>M){ for(uint c0=0;c0<N&&kc0==0u;c0++) for(uint i=0;i<M-r0;i++){ float s=0.0f;\n"
"     for(uint k=0;k<K;k++) s=mad(A[(r0+i)*K+k], Bp[(size_t)(c0/16u)*K*16u+k*16u+(c0%16u)], s);\n"
"     C[(r0+i)*N+c0]=s; } return; }\n"
"  __global const float*a0p=A+(r0+0)*K,*a1p=A+(r0+1)*K,*a2p=A+(r0+2)*K,*a3p=A+(r0+3)*K,*a4p=A+(r0+4)*K,*a5p=A+(r0+5)*K;\n"
"  for(uint p=0;p<N/16u;p++){\n"
"    __global const float*panel=Bp+(size_t)p*K*16u;\n"
"    float8 c00,c01,c02,c03,c04,c05,c10,c11,c12,c13,c14,c15;\n"
"    if(kc0==0u){ c00=c01=c02=c03=c04=c05=c10=c11=c12=c13=c14=c15=(float8)(0.0f); }\n"
"    else { __global float*cp=C+p*16u; \n"
"      c00=vload8(0,cp+(r0+0)*N);c10=vload8(0,cp+(r0+0)*N+8u);\n"
"      c01=vload8(0,cp+(r0+1)*N);c11=vload8(0,cp+(r0+1)*N+8u);\n"
"      c02=vload8(0,cp+(r0+2)*N);c12=vload8(0,cp+(r0+2)*N+8u);\n"
"      c03=vload8(0,cp+(r0+3)*N);c13=vload8(0,cp+(r0+3)*N+8u);\n"
"      c04=vload8(0,cp+(r0+4)*N);c14=vload8(0,cp+(r0+4)*N+8u);\n"
"      c05=vload8(0,cp+(r0+5)*N);c15=vload8(0,cp+(r0+5)*N+8u); }\n"
"    for(uint k=kc0;k<kc1;k++){\n"
"      float8 b0=vload8(0,panel+k*16u), b1=vload8(0,panel+k*16u+8u);\n"
"      ACC6(b, a0p[k],a1p[k],a2p[k],a3p[k],a4p[k],a5p[k]);\n"
"    }\n"
"    __global float*cp=C+p*16u;\n"
"    vstore8(c00,0,cp+(r0+0)*N);vstore8(c10,0,cp+(r0+0)*N+8u);\n"
"    vstore8(c01,0,cp+(r0+1)*N);vstore8(c11,0,cp+(r0+1)*N+8u);\n"
"    vstore8(c02,0,cp+(r0+2)*N);vstore8(c12,0,cp+(r0+2)*N+8u);\n"
"    vstore8(c03,0,cp+(r0+3)*N);vstore8(c13,0,cp+(r0+3)*N+8u);\n"
"    vstore8(c04,0,cp+(r0+4)*N);vstore8(c14,0,cp+(r0+4)*N+8u);\n"
"    vstore8(c05,0,cp+(r0+5)*N);vstore8(c15,0,cp+(r0+5)*N+8u);\n"
"  }\n"
"}\n"
"#ifndef PC\n#define PC 4u\n#endif\n"
"__kernel void v6(__global const float*A, __global const float*Bp, __global float*C, uint M, uint N, uint K, uint kc0, uint kc1){\n"
"  uint r0=get_group_id(0)*6u; uint pg=get_group_id(1)*PC;\n"
"  uint np=N/16u; if(r0>=M||pg>=np) return;\n"
"  uint pe=min(pg+PC,np);\n"
"  if(r0+6u>M){ for(uint p=pg;p<pe&&kc0==0u;p++) for(uint c=p*16u;c<p*16u+16u;c++) for(uint i=0;i<M-r0;i++){ float s=0.0f;\n"
"     for(uint k=0;k<K;k++) s=mad(A[(r0+i)*K+k], Bp[(size_t)p*K*16u+k*16u+(c%16u)], s);\n"
"     C[(r0+i)*N+c]=s; } return; }\n"
"  __global const float*a0p=A+(r0+0)*K,*a1p=A+(r0+1)*K,*a2p=A+(r0+2)*K,*a3p=A+(r0+3)*K,*a4p=A+(r0+4)*K,*a5p=A+(r0+5)*K;\n"
"  for(uint p=pg;p<pe;p++){\n"
"    __global const float*panel=Bp+(size_t)p*K*16u;\n"
"    float8 c00,c01,c02,c03,c04,c05,c10,c11,c12,c13,c14,c15;\n"
"    if(kc0==0u){ c00=c01=c02=c03=c04=c05=c10=c11=c12=c13=c14=c15=(float8)(0.0f); }\n"
"    else { __global float*cp=C+p*16u;\n"
"      c00=vload8(0,cp+(r0+0)*N);c10=vload8(0,cp+(r0+0)*N+8u);\n"
"      c01=vload8(0,cp+(r0+1)*N);c11=vload8(0,cp+(r0+1)*N+8u);\n"
"      c02=vload8(0,cp+(r0+2)*N);c12=vload8(0,cp+(r0+2)*N+8u);\n"
"      c03=vload8(0,cp+(r0+3)*N);c13=vload8(0,cp+(r0+3)*N+8u);\n"
"      c04=vload8(0,cp+(r0+4)*N);c14=vload8(0,cp+(r0+4)*N+8u);\n"
"      c05=vload8(0,cp+(r0+5)*N);c15=vload8(0,cp+(r0+5)*N+8u); }\n"
"    for(uint k=kc0;k<kc1;k++){\n"
"      float8 b0=vload8(0,panel+k*16u), b1=vload8(0,panel+k*16u+8u);\n"
"      ACC6(b, a0p[k],a1p[k],a2p[k],a3p[k],a4p[k],a5p[k]); }\n"
"    __global float*cp=C+p*16u;\n"
"    vstore8(c00,0,cp+(r0+0)*N);vstore8(c10,0,cp+(r0+0)*N+8u);\n"
"    vstore8(c01,0,cp+(r0+1)*N);vstore8(c11,0,cp+(r0+1)*N+8u);\n"
"    vstore8(c02,0,cp+(r0+2)*N);vstore8(c12,0,cp+(r0+2)*N+8u);\n"
"    vstore8(c03,0,cp+(r0+3)*N);vstore8(c13,0,cp+(r0+3)*N+8u);\n"
"    vstore8(c04,0,cp+(r0+4)*N);vstore8(c14,0,cp+(r0+4)*N+8u);\n"
"    vstore8(c05,0,cp+(r0+5)*N);vstore8(c15,0,cp+(r0+5)*N+8u);\n"
"  }\n"
"}\n"
;

typedef struct { cl_context ctx; cl_command_queue q; cl_program p; } Env;
static void ref_mm(const float*A,const float*B,float*C,int M,int N,int K){
  for(int i=0;i<M;i++) for(int j=0;j<N;j++){ float s=0;
    for(int k=0;k<K;k++) s+=A[i*K+k]*B[k*N+j]; C[i*N+j]=s; }
}
static double run_variant(Env*e,const char*name,cl_mem A,cl_mem B,cl_mem Bp,cl_mem C,
    cl_uint M,cl_uint N,cl_uint K,int rows,cl_uint KC,int iters){
  cl_int err;
  cl_kernel kp=clCreateKernel(e->p,"packB",&err);CK(err);
  cl_kernel k=clCreateKernel(e->p,name,&err);CK(err);
  double best=1e30;
  for(int r=0;r<4;r++){ double t0=now_ms();
    for(int it=0;it<iters;it++){
      CK(clSetKernelArg(kp,0,8,&B));CK(clSetKernelArg(kp,1,8,&Bp));CK(clSetKernelArg(kp,2,4,&N));CK(clSetKernelArg(kp,3,4,&K));
      size_t g=(size_t)(N/16)*K; CK(clEnqueueNDRangeKernel(e->q,kp,1,NULL,&g,NULL,0,NULL,NULL));
      CK(clSetKernelArg(k,0,8,&A));CK(clSetKernelArg(k,1,8,&Bp));CK(clSetKernelArg(k,2,8,&C));
      CK(clSetKernelArg(k,3,4,&M));CK(clSetKernelArg(k,4,4,&N));CK(clSetKernelArg(k,5,4,&K));
      size_t g2=(M+rows-1)/rows,l=1;
      for(cl_uint kc=0;kc<K;kc+=KC){ cl_uint kc1=kc+KC<K?kc+KC:K;
        CK(clSetKernelArg(k,6,4,&kc));CK(clSetKernelArg(k,7,4,&kc1));
        CK(clEnqueueNDRangeKernel(e->q,k,1,NULL,&g2,&l,0,NULL,NULL)); }
    }
    CK(clFinish(e->q)); double dt=(now_ms()-t0)/iters; if(dt<best)best=dt; }
  clReleaseKernel(kp);clReleaseKernel(k); return best;
}
static double run2d(Env*e,const char*name,cl_mem A,cl_mem B,cl_mem Bp,cl_mem C,
    cl_uint M,cl_uint N,cl_uint K,cl_uint KC,cl_uint PC,int iters){
  cl_int err; cl_kernel kp=clCreateKernel(e->p,"packB",&err);CK(err);
  cl_kernel k=clCreateKernel(e->p,name,&err);CK(err); double best=1e30;
  for(int r=0;r<4;r++){ double t0=now_ms();
    for(int it=0;it<iters;it++){
      CK(clSetKernelArg(kp,0,8,&B));CK(clSetKernelArg(kp,1,8,&Bp));CK(clSetKernelArg(kp,2,4,&N));CK(clSetKernelArg(kp,3,4,&K));
      size_t g=(size_t)(N/16)*K; CK(clEnqueueNDRangeKernel(e->q,kp,1,NULL,&g,NULL,0,NULL,NULL));
      CK(clSetKernelArg(k,0,8,&A));CK(clSetKernelArg(k,1,8,&Bp));CK(clSetKernelArg(k,2,8,&C));
      CK(clSetKernelArg(k,3,4,&M));CK(clSetKernelArg(k,4,4,&N));CK(clSetKernelArg(k,5,4,&K));
      cl_uint np=N/16u, ngp=(np+PC-1)/PC;
      size_t gs[2]={(M+5)/6, ngp}, ls[2]={1,1};
      for(cl_uint kc=0;kc<K;kc+=KC){ cl_uint kc1=kc+KC<K?kc+KC:K;
        CK(clSetKernelArg(k,6,4,&kc));CK(clSetKernelArg(k,7,4,&kc1));
        CK(clEnqueueNDRangeKernel(e->q,k,2,NULL,gs,ls,0,NULL,NULL)); } }
    CK(clFinish(e->q)); double dt=(now_ms()-t0)/iters; if(dt<best)best=dt; }
  clReleaseKernel(kp);clReleaseKernel(k); return best; }
static int verify(Env*e,const char*name,int rows,cl_uint KC){
  const int M=192,N=256,K=224;
  float*A=malloc(M*K*4),*B=malloc(K*N*4),*C=malloc(M*N*4),*R=malloc(M*N*4);
  srand(7); for(int i=0;i<M*K;i++)A[i]=(rand()%17-8)/4.0f; for(int i=0;i<K*N;i++)B[i]=(rand()%17-8)/4.0f;
  ref_mm(A,B,R,M,N,K); cl_int err;
  cl_mem dA=clCreateBuffer(e->ctx,CL_MEM_COPY_HOST_PTR,M*K*4,A,&err);CK(err);
  cl_mem dB=clCreateBuffer(e->ctx,CL_MEM_COPY_HOST_PTR,K*N*4,B,&err);CK(err);
  cl_mem dBp=clCreateBuffer(e->ctx,CL_MEM_READ_WRITE,(size_t)K*N*4,NULL,&err);CK(err);
  cl_mem dC=clCreateBuffer(e->ctx,CL_MEM_READ_WRITE,M*N*4,NULL,&err);CK(err);
  run_variant(e,name,dA,dB,dBp,dC,M,N,K,rows,KC,1);
  CK(clEnqueueReadBuffer(e->q,dC,CL_TRUE,0,M*N*4,C,0,NULL,NULL));
  int bad=0; for(int i=0;i<M*N;i++) if(fabsf(C[i]-R[i])>1e-3f*(1.0f+fabsf(R[i]))) bad++;
  clReleaseMemObject(dA);clReleaseMemObject(dB);clReleaseMemObject(dBp);clReleaseMemObject(dC);
  free(A);free(B);free(C);free(R); return bad;
}
int main(void){
  const char*want=getenv("PJRT_OCL_DEVICE"); if(!want)want="Portable";
  cl_uint np;CK(clGetPlatformIDs(0,NULL,&np));cl_platform_id ps[8];CK(clGetPlatformIDs(np<8?np:8,ps,NULL));
  cl_device_id dev=0;char nm[256];
  for(cl_uint i=0;i<np;i++){CK(clGetPlatformInfo(ps[i],CL_PLATFORM_NAME,sizeof nm,nm,NULL));
    if(!strstr(nm,want))continue; cl_uint nd; if(clGetDeviceIDs(ps[i],CL_DEVICE_TYPE_ALL,0,NULL,&nd)!=CL_SUCCESS||!nd)continue;
    cl_device_id ds[4];CK(clGetDeviceIDs(ps[i],CL_DEVICE_TYPE_ALL,1,ds,NULL));dev=ds[0];break;}
  if(!dev){fprintf(stderr,"no device\n");return 1;}
  Env e;cl_int err; e.ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&err);CK(err);
  e.q=clCreateCommandQueue(e.ctx,dev,0,&err);CK(err);
  cl_uint PC=getenv("PC")?atoi(getenv("PC")):4; cl_uint KC=getenv("KC")?atoi(getenv("KC")):1024;
  char opts[128]; snprintf(opts,sizeof opts,"-DPC=%uu",PC);
  e.p=clCreateProgramWithSource(e.ctx,1,&SRC,NULL,&err);CK(err);
  if(clBuildProgram(e.p,1,&dev,opts,NULL,NULL)!=CL_SUCCESS){char log[16384];size_t ls=0;
    clGetProgramBuildInfo(e.p,dev,CL_PROGRAM_BUILD_LOG,sizeof log,log,&ls);fprintf(stderr,"build:\n%.*s\n",(int)ls,log);return 1;}
  printf("PC=%u KC=%u\n",PC,KC);
  for(cl_uint Nn=512;Nn<=2048;Nn*=2){cl_uint M=Nn,N=Nn,K=Nn;
    cl_mem A=clCreateBuffer(e.ctx,CL_MEM_READ_WRITE,(size_t)M*K*4,NULL,&err);CK(err);
    cl_mem B=clCreateBuffer(e.ctx,CL_MEM_READ_WRITE,(size_t)K*N*4,NULL,&err);CK(err);
    cl_mem Bp=clCreateBuffer(e.ctx,CL_MEM_READ_WRITE,(size_t)K*N*4,NULL,&err);CK(err);
    cl_mem C=clCreateBuffer(e.ctx,CL_MEM_READ_WRITE,(size_t)M*N*4,NULL,&err);CK(err);
    const double GF=2.0*(double)M*N*K/1e6; int iters=Nn>=2048?2:4;
    double ms=run2d(&e,"v6",A,B,Bp,C,M,N,K,KC,PC,iters);
    printf("  N=%u v6 %8.2f ms  %6.1f GFLOP/s\n",Nn,ms,GF/ms);
    clReleaseMemObject(A);clReleaseMemObject(B);clReleaseMemObject(Bp);clReleaseMemObject(C);}
  return 0;
}
