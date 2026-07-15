/* Does a cross-workgroup barrier called from an UNSTRUCTURED for(;;) loop
 * (like vm2's interpreter) deadlock on PoCL, while the same barrier in a simple
 * bounded loop (poc/07) does not? Isolates the vm2 PoCL hang. */
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static const char *SRC =
"static void bar12(volatile __global uint*b,uint ng){\n"
"  barrier(CLK_GLOBAL_MEM_FENCE);\n"
"  if(get_local_id(0)==0){ uint ph=atomic_add(&b[1],0);\n"
"    if(atomic_inc(&b[0])==ng-1){ b[0]=0; mem_fence(CLK_GLOBAL_MEM_FENCE); atomic_inc(&b[1]); }\n"
"    else { while(atomic_add(&b[1],0)==ph); } }\n"
"  barrier(CLK_GLOBAL_MEM_FENCE);\n"
"}\n"
/* structured: barrier in a plain counted loop (poc/07 style) */
"__kernel void structured(volatile __global uint*b,uint ng,uint T,__global uint*o){\n"
"  for(uint i=0;i<T;i++){ o[get_group_id(0)]++; bar12(b,ng); }\n"
"}\n"
/* unstructured: barrier inside for(;;) with data-driven break/continue, reading\n"
" * an 'entry stream' from global — exactly vm2's frame-walk shape. */
"__kernel void unstructured(__global const uint*ent,uint cnt,volatile __global uint*b,uint ng,uint T,__global uint*o){\n"
"  __local float As[1024]; __local float Bs[1024];\n"   /* vm2's 8KB __local */
"  uint lid=get_local_id(0);\n"
"  uint st[8]; float acc[16];\n"                        /* vm2 frame stack + MMA-ish regs */
"  for(int r=0;r<16;r++) acc[r]=(float)lid;\n"
"  for(uint it=0; it<T; ++it){\n"
"    uint pc=0;\n"
"    for(;;){\n"
"      if(pc>=cnt) break;\n"
"      uint e=ent[pc];\n"
"      if(e==0xFFFFFFFEu){ bar12(b,ng); pc++; continue; }\n"
"      /* IMBALANCE: only lane 0 does heavy pre-barrier work (like vm2's EW\n"
"       * tile landing on one lane while others idle to the barrier). */\n"
"      if(get_group_id(0)==0){ for(uint z=0; z<200000u; z++) acc[0]+=As[lid]*0.5f; }\n"
"      for(int d=0;d<8;d++) st[d]=pc+d;\n"
"      for(int r=0;r<16;r++) acc[r]=acc[r]*1.001f+As[lid];\n"
"      As[lid]=(float)it; Bs[lid]=As[lid]*2.0f;\n"
"      o[get_group_id(0)]+=st[0]+(uint)(Bs[lid]+acc[0]+acc[15]); pc++;\n"
"    }\n"
"  }\n"
"}\n";
#define CK(e) do{cl_int _e=(e);if(_e!=CL_SUCCESS){fprintf(stderr,"%d: err %d\n",__LINE__,_e);exit(1);}}while(0)
int main(void){
  const char*want=getenv("PJRT_OCL_DEVICE");
  cl_uint np;clGetPlatformIDs(0,NULL,&np);cl_platform_id ps[8];clGetPlatformIDs(np<8?np:8,ps,NULL);
  cl_platform_id pl=0;cl_device_id dv=0;char nm[256];
  for(cl_uint i=0;i<np;i++){clGetPlatformInfo(ps[i],CL_PLATFORM_NAME,sizeof nm,nm,NULL);if(want&&!strstr(nm,want))continue;cl_uint nd;if(clGetDeviceIDs(ps[i],CL_DEVICE_TYPE_ALL,0,NULL,&nd)!=CL_SUCCESS||!nd)continue;cl_device_id ds[8];clGetDeviceIDs(ps[i],CL_DEVICE_TYPE_ALL,nd<8?nd:8,ds,NULL);pl=ps[i];dv=ds[0];break;}
  if(!dv){fprintf(stderr,"no dev\n");return 1;}
  clGetPlatformInfo(pl,CL_PLATFORM_NAME,sizeof nm,nm,NULL);printf("platform: %s\n",nm);
  cl_int e;cl_context c=clCreateContext(NULL,1,&dv,NULL,NULL,&e);CK(e);
  cl_command_queue q=clCreateCommandQueue(c,dv,0,&e);CK(e);
  cl_program p=clCreateProgramWithSource(c,1,&SRC,NULL,&e);CK(e);
  cl_int be=clBuildProgram(p,1,&dv,NULL,NULL,NULL);
  if(be){char lg[4096];clGetProgramBuildInfo(p,dv,CL_PROGRAM_BUILD_LOG,sizeof lg,lg,NULL);fprintf(stderr,"build:%s\n",lg);return 1;}
  cl_uint G = getenv("G_ENV")?atoi(getenv("G_ENV")):4;
  size_t lsz=256, gsz=(size_t)G*lsz;
  cl_uint T=2000, zero=0;
  cl_mem b=clCreateBuffer(c,CL_MEM_READ_WRITE,2*sizeof(cl_uint),NULL,&e);
  cl_mem o=clCreateBuffer(c,CL_MEM_READ_WRITE,G*sizeof(cl_uint),NULL,&e);
  /* entry stream: work, BARRIER, work */
  cl_uint ent[3]={1u,0xFFFFFFFEu,1u};
  cl_mem em=clCreateBuffer(c,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof ent,ent,&e);

  printf("G=%u\n",G);
  { CK(clEnqueueFillBuffer(q,b,&zero,4,0,8,0,0,0)); CK(clEnqueueFillBuffer(q,o,&zero,4,0,G*4,0,0,0));
    cl_kernel k=clCreateKernel(p,"structured",&e);CK(e);
    clSetKernelArg(k,0,sizeof b,&b);clSetKernelArg(k,1,4,&G);clSetKernelArg(k,2,4,&T);clSetKernelArg(k,3,sizeof o,&o);
    CK(clEnqueueNDRangeKernel(q,k,1,0,&gsz,&lsz,0,0,0)); CK(clFinish(q));
    printf("[structured]   barrier in counted loop : DONE\n"); }
  { CK(clEnqueueFillBuffer(q,b,&zero,4,0,8,0,0,0)); CK(clEnqueueFillBuffer(q,o,&zero,4,0,G*4,0,0,0));
    cl_kernel k=clCreateKernel(p,"unstructured",&e);CK(e); cl_uint cnt=3;
    clSetKernelArg(k,0,sizeof em,&em);clSetKernelArg(k,1,4,&cnt);clSetKernelArg(k,2,sizeof b,&b);clSetKernelArg(k,3,4,&G);clSetKernelArg(k,4,4,&T);clSetKernelArg(k,5,sizeof o,&o);
    CK(clEnqueueNDRangeKernel(q,k,1,0,&gsz,&lsz,0,0,0)); CK(clFinish(q));
    printf("[unstructured] barrier in for(;;)+break: DONE\n"); }
  printf("both completed — no deadlock\n");
  return 0;
}
