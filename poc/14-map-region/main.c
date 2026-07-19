/* poc/13 — map-region fusion: does keeping intermediates ON-CHIP across a run
 * of pure-map ops actually beat today's round-tripping elementwise chain?
 *
 * §23 thesis: our "fusion" (§11) removes the barrier, NOT the round-trips —
 * every EW tile-op still does arena[a] -> compute -> arena[dst]. A K-op GELU
 * chain = K global writes + K reads. The fix (§23 / ideas-for-v2 Idea A) is a
 * fused map-region tile-op: load the region's input ONCE, interpret the op
 * sub-list over on-chip scratch, store ONCE.
 *
 * This PoC hand-emits the GELU tail (tanh approx, 9 pure-map micro-ops that
 * reuse x 4x — a real DAG) as a single region interpreted three ways that
 * differ ONLY in the scratch address space:
 *   [G] region_global : scratch R[] in GLOBAL  -> models today's round-trips
 *   [L] region_local  : scratch R[] in __LOCAL -> the fused region (step 1)
 *   [R] region_reg    : scratch in switch-addressed REGISTERS (step 2 ceiling)
 * plus two references:
 *   [E] ewchain       : the honest "today" path — 9 separate vectorized EW
 *                        kernel launches, ping-ponging global buffers (§11 chain)
 *   [H] gelu_hard     : one hardcoded vectorized gelu kernel (absolute ceiling)
 *
 * Same micro-op program, same numerics for G/L/R — the delta L(or R) vs G/E is
 * the go/no-go for building the general mechanism.  Data = (4,128,2048) f32 =
 * 1,048,576 elems (base transformer FFN GELU size).
 *
 * cc -O2 -Wall -o poc13 main.c -lOpenCL ; ./poc13
 * Device via PJRT_OCL_DEVICE (default "NVIDIA"; use "Portable" for PoCL CPU).
 */
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CK(e) do{cl_int _e=(e); if(_e!=CL_SUCCESS){fprintf(stderr,"%s:%d CL err %d\n",__FILE__,__LINE__,_e);exit(1);} }while(0)
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }

/* ---- the region micro-op program (host + device share this struct) -------- */
enum { M_MUL=0, M_ADD=1, M_AFF=2, M_TANH=3, M_SUB=4 };  /* AFF: dst = a*s + t */
typedef struct { cl_uint kind, dst, a, b; cl_float s, t; } mop;  /* 24 bytes */

/* GELU tanh approx: 0.5*x*(1 + tanh(0.7978845608*(x + 0.044715*x^3)))
 * slot 0 = x (region input); output = slot 3; uses slots {0,1,2,3}. */
static const mop GELU[] = {
    {M_MUL, 1, 0, 0, 0.f, 0.f},               /* x2 = x*x            */
    {M_MUL, 2, 1, 0, 0.f, 0.f},               /* x3 = x2*x           */
    {M_AFF, 3, 2, 0, 0.044715f, 0.f},         /* 0.044715 * x3       */
    {M_ADD, 3, 3, 0, 0.f, 0.f},               /* + x                 */
    {M_AFF, 3, 3, 0, 0.7978845608f, 0.f},     /* * sqrt(2/pi)        */
    {M_TANH,3, 3, 0, 0.f, 0.f},               /* tanh(...)           */
    {M_AFF, 3, 3, 0, 1.f, 1.f},               /* * 1 + 1  (=+1)      */
    {M_MUL, 3, 3, 0, 0.f, 0.f},               /* * x                 */
    {M_AFF, 3, 3, 0, 0.5f, 0.f},              /* * 0.5               */
};
#define NPROG ((cl_uint)(sizeof(GELU)/sizeof(GELU[0])))
#define OUTSLOT 3u
#define NSLOTS 8u      /* scratch register-file size (bounds region width) */

static double gelu_ref(double x) {
    double inner = 0.7978845608 * (x + 0.044715 * x*x*x);
    return 0.5 * x * (1.0 + tanh(inner));
}

static const char *SRC =
"#define EW_TS 4096u\n"       /* GPU EW tile (§22); host tiles must match */
"#define NSLOTS 8u\n"
"typedef struct { uint kind, dst, a, b; float s, t; } mop;\n"
"static float mop_apply(uint kind, float a, float b, float s, float t){\n"
"  switch(kind){\n"
"    case 0: return a*b;         /* MUL  */\n"
"    case 1: return a+b;         /* ADD  */\n"
"    case 2: return a*s+t;       /* AFF  */\n"
"    case 3: return tanh(a);     /* TANH */\n"
"    case 4: return a-b;         /* SUB  */\n"
"    default: return a; }\n"
"}\n"
/* ---- [G] scratch in GLOBAL: models today's per-op round-trips ------------- *
 * gR is a global scratch of nslots planes (gR[slot*n + i]).  Every micro-op
 * reads/writes global -> K loads + K stores per element, exactly the traffic
 * §11 chained EW ops pay (measured §22: ~15us/tile, latency-bound). */
"__kernel void region_global(__global const float* x, __global float* out,\n"
"    __global float* gR, uint n, __constant mop* prog, uint nprog){\n"
"  uint tile=get_group_id(0), lid=get_local_id(0), lsz=get_local_size(0);\n"
"  uint lo=tile*EW_TS, hi=min(lo+EW_TS,n);\n"
"  for(uint i=lo+lid;i<hi;i+=lsz){\n"
"    gR[0u*n+i]=x[i];\n"                       /* load region input   */
"    for(uint p=0;p<nprog;p++){ mop m=prog[p];\n"
"      float a=gR[m.a*n+i], b=gR[m.b*n+i];\n"
"      gR[m.dst*n+i]=mop_apply(m.kind,a,b,m.s,m.t); }\n"
"    out[i]=gR[3u*n+i];\n"                     /* store region output */
"  }\n"
"}\n"
/* ---- [L] scratch in __LOCAL: the fused region, step 1 (local staging) ------ *
 * R[slot][lid] in SLM/L1 — indexable by the runtime slot id (the crux: a
 * private array with a runtime index would spill; __local stays on-chip).
 * One global load of x, one global store; all intermediates in local. */
"__kernel void region_local(__global const float* x, __global float* out,\n"
"    uint n, __constant mop* prog, uint nprog){\n"
"  __local float R[NSLOTS][256];\n"
"  uint tile=get_group_id(0), lid=get_local_id(0), lsz=get_local_size(0);\n"
"  uint lo=tile*EW_TS, hi=min(lo+EW_TS,n);\n"
"  for(uint i=lo+lid;i<hi;i+=lsz){\n"
"    R[0][lid]=x[i];\n"
"    for(uint p=0;p<nprog;p++){ mop m=prog[p];\n"
"      float a=R[m.a][lid], b=R[m.b][lid];\n"
"      R[m.dst][lid]=mop_apply(m.kind,a,b,m.s,m.t); }\n"
"    out[i]=R[3][lid];\n"
"  }\n"
"}\n"
/* ---- [R] scratch in switch-addressed REGISTERS: step 2 ceiling ------------- *
 * r0..r7 are true registers; RD/WR dispatch the runtime slot via a switch
 * (the interpreter's dispatch cost, amortized over the element).  No local, no
 * global intermediates.  NSLOTS bounded (region width) before local spill. */
"#define RD(i) ((i)==0?r0:(i)==1?r1:(i)==2?r2:(i)==3?r3:(i)==4?r4:(i)==5?r5:(i)==6?r6:r7)\n"
"#define WR(i,v) do{ switch(i){ case 0:r0=v;break; case 1:r1=v;break; \\\n"
"  case 2:r2=v;break; case 3:r3=v;break; case 4:r4=v;break; case 5:r5=v;break; \\\n"
"  case 6:r6=v;break; default:r7=v;break; } }while(0)\n"
"__kernel void region_reg(__global const float* x, __global float* out,\n"
"    uint n, __constant mop* prog, uint nprog){\n"
"  uint tile=get_group_id(0), lid=get_local_id(0), lsz=get_local_size(0);\n"
"  uint lo=tile*EW_TS, hi=min(lo+EW_TS,n);\n"
"  for(uint i=lo+lid;i<hi;i+=lsz){\n"
"    float r0=x[i],r1=0,r2=0,r3=0,r4=0,r5=0,r6=0,r7=0;\n"
"    for(uint p=0;p<nprog;p++){ mop m=prog[p];\n"
"      float a=RD(m.a), b=RD(m.b);\n"
"      WR(m.dst, mop_apply(m.kind,a,b,m.s,m.t)); }\n"
"    out[i]=RD(3);\n"
"  }\n"
"}\n"
/* ---- [E] the honest 'today' EW chain: separate vectorized launches --------- *
 * Each op is a full grid-stride pass reading global operands, writing global —
 * float4 + 2x unroll (the shipped GPU EW fast path, §22).  Host ping-pongs two
 * buffers and relaunches 9x (chain fusion removes the barrier, not the passes). */
"__kernel void ew_mul(__global const float* a, __global const float* b, __global float* d, uint n){\n"
"  uint tile=get_group_id(0), lid=get_local_id(0), lsz=get_local_size(0);\n"
"  uint lo=tile*EW_TS, hi=min(lo+EW_TS,n);\n"
"  __global const float4* a4=(__global const float4*)a; __global const float4* b4=(__global const float4*)b;\n"
"  __global float4* d4=(__global float4*)d; uint lo4=lo>>2, hi4=lo4+((hi-lo)>>2);\n"
"  for(uint j=lo4+lid;j<hi4;j+=lsz) d4[j]=a4[j]*b4[j];\n"
"}\n"
"__kernel void ew_add(__global const float* a, __global const float* b, __global float* d, uint n){\n"
"  uint tile=get_group_id(0), lid=get_local_id(0), lsz=get_local_size(0);\n"
"  uint lo=tile*EW_TS, hi=min(lo+EW_TS,n);\n"
"  __global const float4* a4=(__global const float4*)a; __global const float4* b4=(__global const float4*)b;\n"
"  __global float4* d4=(__global float4*)d; uint lo4=lo>>2, hi4=lo4+((hi-lo)>>2);\n"
"  for(uint j=lo4+lid;j<hi4;j+=lsz) d4[j]=a4[j]+b4[j];\n"
"}\n"
"__kernel void ew_aff(__global const float* a, __global float* d, float s, float t, uint n){\n"
"  uint tile=get_group_id(0), lid=get_local_id(0), lsz=get_local_size(0);\n"
"  uint lo=tile*EW_TS, hi=min(lo+EW_TS,n);\n"
"  __global const float4* a4=(__global const float4*)a; __global float4* d4=(__global float4*)d;\n"
"  uint lo4=lo>>2, hi4=lo4+((hi-lo)>>2); float4 s4=(float4)(s), t4=(float4)(t);\n"
"  for(uint j=lo4+lid;j<hi4;j+=lsz) d4[j]=mad(a4[j],s4,t4);\n"
"}\n"
"__kernel void ew_tanh(__global const float* a, __global float* d, uint n){\n"
"  uint tile=get_group_id(0), lid=get_local_id(0), lsz=get_local_size(0);\n"
"  uint lo=tile*EW_TS, hi=min(lo+EW_TS,n);\n"
"  __global const float4* a4=(__global const float4*)a; __global float4* d4=(__global float4*)d;\n"
"  uint lo4=lo>>2, hi4=lo4+((hi-lo)>>2);\n"
"  for(uint j=lo4+lid;j<hi4;j+=lsz) d4[j]=tanh(a4[j]);\n"
"}\n"
/* ---- [H] hardcoded vectorized gelu: absolute ceiling (1 load, 1 store) ----- */
"__kernel void gelu_hard(__global const float* x, __global float* out, uint n){\n"
"  uint tile=get_group_id(0), lid=get_local_id(0), lsz=get_local_size(0);\n"
"  uint lo=tile*EW_TS, hi=min(lo+EW_TS,n);\n"
"  __global const float4* x4=(__global const float4*)x; __global float4* o4=(__global float4*)out;\n"
"  uint lo4=lo>>2, hi4=lo4+((hi-lo)>>2);\n"
"  for(uint j=lo4+lid;j<hi4;j+=lsz){ float4 v=x4[j];\n"
"    float4 in=0.7978845608f*(v+0.044715f*v*v*v);\n"
"    o4[j]=0.5f*v*(1.0f+tanh(in)); }\n"
"}\n"
"static float4 mop_apply4(uint kind, float4 a, float4 b, float s, float t){\n"
"  switch(kind){ case 0: return a*b; case 1: return a+b; case 2: return a*s+t;\n"
"    case 3: return tanh(a); case 4: return a-b; default: return a; }\n"
"}\n"
/* ---- [E1] faithful 'today' megakernel chain: 9 VECTORIZED float4 passes in ONE
 * launch over global ping-pong scratch (gR planes). Same as §11 chain fusion:
 * one launch, no barrier, but each op still round-trips its plane through global.
 * This is the honest baseline to beat (not 9 separate launches). */
"__kernel void ewchain1(__global const float* x, __global float* out,\n"
"    __global float* gR, uint n, __constant mop* prog, uint nprog){\n"
"  uint tile=get_group_id(0), lid=get_local_id(0), lsz=get_local_size(0);\n"
"  uint lo=tile*EW_TS, hi=min(lo+EW_TS,n); uint lo4=lo>>2, hi4=lo4+((hi-lo)>>2);\n"
"  uint n4=n>>2;\n"
"  __global const float4* x4=(__global const float4*)x;\n"
"  __global float4* g4=(__global float4*)gR; __global float4* o4=(__global float4*)out;\n"
"  for(uint j=lo4+lid;j<hi4;j+=lsz) g4[0u*n4+j]=x4[j];\n"      /* seed slot 0 */
"  for(uint p=0;p<nprog;p++){ mop m=prog[p];\n"
"    for(uint j=lo4+lid;j<hi4;j+=lsz)\n"
"      g4[m.dst*n4+j]=mop_apply4(m.kind,g4[m.a*n4+j],g4[m.b*n4+j],m.s,m.t); }\n"
"  for(uint j=lo4+lid;j<hi4;j+=lsz) o4[j]=g4[3u*n4+j];\n"
"}\n"
/* ---- [L4] float4-vectorized local-staged region: the fair fused competitor -- */
"__kernel void region_local4(__global const float* x, __global float* out,\n"
"    uint n, __constant mop* prog, uint nprog){\n"
"  __local float4 R[NSLOTS][256];\n"
"  uint tile=get_group_id(0), lid=get_local_id(0), lsz=get_local_size(0);\n"
"  uint lo=tile*EW_TS, hi=min(lo+EW_TS,n); uint lo4=lo>>2, hi4=lo4+((hi-lo)>>2);\n"
"  __global const float4* x4=(__global const float4*)x; __global float4* o4=(__global float4*)out;\n"
"  for(uint j=lo4+lid;j<hi4;j+=lsz){\n"
"    R[0][lid]=x4[j];\n"
"    for(uint p=0;p<nprog;p++){ mop m=prog[p];\n"
"      R[m.dst][lid]=mop_apply4(m.kind,R[m.a][lid],R[m.b][lid],m.s,m.t); }\n"
"    o4[j]=R[3][lid];\n"
"  }\n"
"}\n"
/* ---- [R4] float4-vectorized switch-register region: the ceiling of interp ---- */
"#define RD4(i) ((i)==0?r0:(i)==1?r1:(i)==2?r2:(i)==3?r3:(i)==4?r4:(i)==5?r5:(i)==6?r6:r7)\n"
"#define WR4(i,v) do{ switch(i){ case 0:r0=v;break; case 1:r1=v;break; \\\n"
"  case 2:r2=v;break; case 3:r3=v;break; case 4:r4=v;break; case 5:r5=v;break; \\\n"
"  case 6:r6=v;break; default:r7=v;break; } }while(0)\n"
"__kernel void region_reg4(__global const float* x, __global float* out,\n"
"    uint n, __constant mop* prog, uint nprog){\n"
"  uint tile=get_group_id(0), lid=get_local_id(0), lsz=get_local_size(0);\n"
"  uint lo=tile*EW_TS, hi=min(lo+EW_TS,n); uint lo4=lo>>2, hi4=lo4+((hi-lo)>>2);\n"
"  __global const float4* x4=(__global const float4*)x; __global float4* o4=(__global float4*)out;\n"
"  for(uint j=lo4+lid;j<hi4;j+=lsz){\n"
"    float4 r0=x4[j],r1=0,r2=0,r3=0,r4=0,r5=0,r6=0,r7=0;\n"
"    for(uint p=0;p<nprog;p++){ mop m=prog[p];\n"
"      WR4(m.dst, mop_apply4(m.kind,RD4(m.a),RD4(m.b),m.s,m.t)); }\n"
"    o4[j]=RD4(3);\n"
"  }\n"
"}\n";

static cl_platform_id pick_platform(void){
    const char* want=getenv("PJRT_OCL_DEVICE"); if(!want||!*want) want="NVIDIA";
    cl_uint np=0; clGetPlatformIDs(0,NULL,&np);
    cl_platform_id* ps=malloc(np*sizeof(*ps)); clGetPlatformIDs(np,ps,NULL);
    cl_platform_id chosen=ps[0]; char name[256];
    for(cl_uint i=0;i<np;i++){ clGetPlatformInfo(ps[i],CL_PLATFORM_NAME,sizeof(name),name,NULL);
        if(strstr(name,want)){ chosen=ps[i]; break; } }
    clGetPlatformInfo(chosen,CL_PLATFORM_NAME,sizeof(name),name,NULL);
    printf("platform: %s\n",name); free(ps); return chosen;
}

int main(int argc, char** argv){
    cl_platform_id plat=pick_platform();
    cl_device_id dev; cl_uint nd=0;
    if(clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&dev,&nd)!=CL_SUCCESS||nd==0)
        CK(clGetDeviceIDs(plat,CL_DEVICE_TYPE_ALL,1,&dev,&nd));
    char dname[256]; clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof(dname),dname,NULL);
    printf("device:   %s\n",dname);
    cl_int e; cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&e); CK(e);
    cl_command_queue q=clCreateCommandQueue(ctx,dev,0,&e); CK(e);
    cl_program prog=clCreateProgramWithSource(ctx,1,&SRC,NULL,&e); CK(e);
    if(clBuildProgram(prog,1,&dev,"",NULL,NULL)!=CL_SUCCESS){
        size_t ls=0; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,0,NULL,&ls);
        char* log=malloc(ls+1); clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,ls,log,NULL);
        log[ls]=0; fprintf(stderr,"BUILD LOG:\n%s\n",log); return 1; }

    cl_uint n=4*128*2048;                    /* 1,048,576 default */
    if(argc>1){ n=(cl_uint)strtoul(argv[1],NULL,10); n&=~3u; if(n<4) n=4; }
    const size_t EW_TS=4096, WG=256;
    const size_t ntiles=(n+EW_TS-1)/EW_TS;
    const size_t gsz=ntiles*WG, lsz=WG;

    float* hx=malloc(n*sizeof(float));
    for(cl_uint i=0;i<n;i++) hx[i]=((float)((i*2654435761u)>>8)/(float)(1u<<24))*8.0f-4.0f;
    float* hout=malloc(n*sizeof(float));

    cl_mem dx =clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,n*sizeof(float),hx,&e); CK(e);
    cl_mem dout=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,n*sizeof(float),NULL,&e); CK(e);
    cl_mem dgR =clCreateBuffer(ctx,CL_MEM_READ_WRITE,(size_t)NSLOTS*n*sizeof(float),NULL,&e); CK(e);
    cl_mem dprog=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(GELU),(void*)GELU,&e); CK(e);
    /* ping-pong buffers for the EW-chain reference */
    cl_mem dpA=clCreateBuffer(ctx,CL_MEM_READ_WRITE,n*sizeof(float),NULL,&e); CK(e);
    cl_mem dpB=clCreateBuffer(ctx,CL_MEM_READ_WRITE,n*sizeof(float),NULL,&e); CK(e);

    cl_kernel kG=clCreateKernel(prog,"region_global",&e); CK(e);
    cl_kernel kL=clCreateKernel(prog,"region_local",&e); CK(e);
    cl_kernel kR=clCreateKernel(prog,"region_reg",&e); CK(e);
    cl_kernel kH=clCreateKernel(prog,"gelu_hard",&e); CK(e);
    cl_kernel kMul=clCreateKernel(prog,"ew_mul",&e); CK(e);
    cl_kernel kAdd=clCreateKernel(prog,"ew_add",&e); CK(e);
    cl_kernel kAff=clCreateKernel(prog,"ew_aff",&e); CK(e);
    cl_kernel kTanh=clCreateKernel(prog,"ew_tanh",&e); CK(e);
    cl_kernel kE1=clCreateKernel(prog,"ewchain1",&e); CK(e);
    cl_kernel kL4=clCreateKernel(prog,"region_local4",&e); CK(e);
    cl_kernel kR4=clCreateKernel(prog,"region_reg4",&e); CK(e);

    /* max abs err vs double reference */
    double maxerr=0;
    #define CHECK(buf) do{ CK(clEnqueueReadBuffer(q,buf,CL_TRUE,0,n*sizeof(float),hout,0,NULL,NULL)); \
        double me=0; for(cl_uint i=0;i<n;i++){ double d=fabs((double)hout[i]-gelu_ref(hx[i])); if(d>me)me=d; } \
        maxerr=me; }while(0)

    #define RUN1(K) do{ CK(clEnqueueNDRangeKernel(q,K,1,NULL,&gsz,&lsz,0,NULL,NULL)); }while(0)

    /* [G] region_global */
    CK(clSetKernelArg(kG,0,sizeof(cl_mem),&dx)); CK(clSetKernelArg(kG,1,sizeof(cl_mem),&dout));
    CK(clSetKernelArg(kG,2,sizeof(cl_mem),&dgR)); CK(clSetKernelArg(kG,3,sizeof(cl_uint),&n));
    CK(clSetKernelArg(kG,4,sizeof(cl_mem),&dprog)); { cl_uint np=NPROG; CK(clSetKernelArg(kG,5,sizeof(cl_uint),&np)); }
    /* [L] region_local */
    CK(clSetKernelArg(kL,0,sizeof(cl_mem),&dx)); CK(clSetKernelArg(kL,1,sizeof(cl_mem),&dout));
    CK(clSetKernelArg(kL,2,sizeof(cl_uint),&n)); CK(clSetKernelArg(kL,3,sizeof(cl_mem),&dprog));
    { cl_uint np=NPROG; CK(clSetKernelArg(kL,4,sizeof(cl_uint),&np)); }
    /* [R] region_reg */
    CK(clSetKernelArg(kR,0,sizeof(cl_mem),&dx)); CK(clSetKernelArg(kR,1,sizeof(cl_mem),&dout));
    CK(clSetKernelArg(kR,2,sizeof(cl_uint),&n)); CK(clSetKernelArg(kR,3,sizeof(cl_mem),&dprog));
    { cl_uint np=NPROG; CK(clSetKernelArg(kR,4,sizeof(cl_uint),&np)); }
    /* [H] gelu_hard */
    CK(clSetKernelArg(kH,0,sizeof(cl_mem),&dx)); CK(clSetKernelArg(kH,1,sizeof(cl_mem),&dout));
    CK(clSetKernelArg(kH,2,sizeof(cl_uint),&n));
    /* [E1] ewchain1 (1 launch, vectorized, global scratch) */
    CK(clSetKernelArg(kE1,0,sizeof(cl_mem),&dx)); CK(clSetKernelArg(kE1,1,sizeof(cl_mem),&dout));
    CK(clSetKernelArg(kE1,2,sizeof(cl_mem),&dgR)); CK(clSetKernelArg(kE1,3,sizeof(cl_uint),&n));
    CK(clSetKernelArg(kE1,4,sizeof(cl_mem),&dprog)); { cl_uint np=NPROG; CK(clSetKernelArg(kE1,5,sizeof(cl_uint),&np)); }
    /* [L4] region_local4 */
    CK(clSetKernelArg(kL4,0,sizeof(cl_mem),&dx)); CK(clSetKernelArg(kL4,1,sizeof(cl_mem),&dout));
    CK(clSetKernelArg(kL4,2,sizeof(cl_uint),&n)); CK(clSetKernelArg(kL4,3,sizeof(cl_mem),&dprog));
    { cl_uint np=NPROG; CK(clSetKernelArg(kL4,4,sizeof(cl_uint),&np)); }
    /* [R4] region_reg4 */
    CK(clSetKernelArg(kR4,0,sizeof(cl_mem),&dx)); CK(clSetKernelArg(kR4,1,sizeof(cl_mem),&dout));
    CK(clSetKernelArg(kR4,2,sizeof(cl_uint),&n)); CK(clSetKernelArg(kR4,3,sizeof(cl_mem),&dprog));
    { cl_uint np=NPROG; CK(clSetKernelArg(kR4,4,sizeof(cl_uint),&np)); }

    /* EW-chain: builds gelu from ping-pong buffers (dpA/dpB + dx reused). */
    void enqueue_ewchain(void){
        cl_uint z=n;
        #define A2(K,x0,x1) do{ CK(clSetKernelArg(K,0,sizeof(cl_mem),x0)); CK(clSetKernelArg(K,1,sizeof(cl_mem),x1)); }while(0)
        /* mul(x,x)->pB */ A2(kMul,&dx,&dx); CK(clSetKernelArg(kMul,2,sizeof(cl_mem),&dpB)); CK(clSetKernelArg(kMul,3,sizeof(cl_uint),&z)); RUN1(kMul);
        /* mul(pB,x)->pA */ A2(kMul,&dpB,&dx); CK(clSetKernelArg(kMul,2,sizeof(cl_mem),&dpA)); RUN1(kMul);
        /* aff(pA,0.044715,0)->pB */ CK(clSetKernelArg(kAff,0,sizeof(cl_mem),&dpA)); CK(clSetKernelArg(kAff,1,sizeof(cl_mem),&dpB)); { float s=0.044715f,t=0.f; CK(clSetKernelArg(kAff,2,sizeof(float),&s)); CK(clSetKernelArg(kAff,3,sizeof(float),&t)); CK(clSetKernelArg(kAff,4,sizeof(cl_uint),&z)); } RUN1(kAff);
        /* add(pB,x)->pA */ A2(kAdd,&dpB,&dx); CK(clSetKernelArg(kAdd,2,sizeof(cl_mem),&dpA)); CK(clSetKernelArg(kAdd,3,sizeof(cl_uint),&z)); RUN1(kAdd);
        /* aff(pA,c,0)->pB */ CK(clSetKernelArg(kAff,0,sizeof(cl_mem),&dpA)); CK(clSetKernelArg(kAff,1,sizeof(cl_mem),&dpB)); { float s=0.7978845608f,t=0.f; CK(clSetKernelArg(kAff,2,sizeof(float),&s)); CK(clSetKernelArg(kAff,3,sizeof(float),&t)); } RUN1(kAff);
        /* tanh(pB)->pA */ CK(clSetKernelArg(kTanh,0,sizeof(cl_mem),&dpB)); CK(clSetKernelArg(kTanh,1,sizeof(cl_mem),&dpA)); CK(clSetKernelArg(kTanh,2,sizeof(cl_uint),&z)); RUN1(kTanh);
        /* aff(pA,1,1)->pB */ CK(clSetKernelArg(kAff,0,sizeof(cl_mem),&dpA)); CK(clSetKernelArg(kAff,1,sizeof(cl_mem),&dpB)); { float s=1.f,t=1.f; CK(clSetKernelArg(kAff,2,sizeof(float),&s)); CK(clSetKernelArg(kAff,3,sizeof(float),&t)); } RUN1(kAff);
        /* mul(pB,x)->pA */ A2(kMul,&dpB,&dx); CK(clSetKernelArg(kMul,2,sizeof(cl_mem),&dpA)); RUN1(kMul);
        /* aff(pA,0.5,0)->out */ CK(clSetKernelArg(kAff,0,sizeof(cl_mem),&dpA)); CK(clSetKernelArg(kAff,1,sizeof(cl_mem),&dout)); { float s=0.5f,t=0.f; CK(clSetKernelArg(kAff,2,sizeof(float),&s)); CK(clSetKernelArg(kAff,3,sizeof(float),&t)); } RUN1(kAff);
        #undef A2
    }

    const int ROUNDS=20, ITERS=50;
    #define BENCH(label, body, checkbuf) do{ \
        body; CK(clFinish(q)); \
        CHECK(checkbuf); \
        double best=1e30; \
        for(int r=0;r<ROUNDS;r++){ double t0=now_ms(); \
            for(int it=0;it<ITERS;it++){ body; } CK(clFinish(q)); \
            double dt=(now_ms()-t0)/ITERS; if(dt<best) best=dt; } \
        double gb = 2.0*n*sizeof(float)/1e9 / (best/1e3); \
        printf("  %-14s %8.4f ms   %7.1f GB/s (1R+1W)   maxerr=%.2e\n", label, best, gb, maxerr); \
    }while(0)

    printf("\nGELU tail (%u pure-map micro-ops), n=%u f32 = %.1f MiB, tiles=%zu WG=%zu\n",
           NPROG, n, n*4.0/(1<<20), ntiles, WG);
    printf("  same interpreter, scratch address space differs (G=global L=local R=reg):\n");
    printf("  --- scalar interpreter vs multi-launch chain ---\n");
    BENCH("[E] ewchain9x", enqueue_ewchain(),  dout);
    BENCH("[G] region_gbl",RUN1(kG),           dout);
    BENCH("[L] region_lcl",RUN1(kL),           dout);
    BENCH("[R] region_reg",RUN1(kR),           dout);
    printf("  --- VECTORIZED (float4): faithful today vs fair fused region ---\n");
    BENCH("[E1] ewchain1x",RUN1(kE1),          dout);
    BENCH("[L4] region_l4", RUN1(kL4),         dout);
    BENCH("[R4] region_r4", RUN1(kR4),         dout);
    BENCH("[H] gelu_hard", RUN1(kH),           dout);
    printf("go/no-go: region_l4/r4 (fused) vs ewchain1x (today, 1-launch vectorized).\n");
    return 0;
}
