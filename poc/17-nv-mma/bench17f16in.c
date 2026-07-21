/* poc/17 §39 — f16-INPUT tensor-core ceiling. Drives mma17_f16in.cl. A,B are
 * uploaded as fp16 (half the global bytes of the f32 arena); tests whether the
 * staging-bound 92 TF/s wall (STAGE decomposition: global reads are the largest
 * addable cost) breaks once we stop reading 2x-wide f32 inputs. C stays f32.
 * A,B,C share one f32 buffer; aoff/boff/coff are FLOAT offsets. Verify = exact
 * small ints (exact in f16). env: OCL_PLATFORM, VM_LANES, ONLY, EXTRA. */
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

static void chk(cl_int e,const char*m){if(e){fprintf(stderr,"ERR %d @ %s\n",e,m);exit(1);}}
static long envi(const char*k,long d){const char*v=getenv(k);return v&&*v?atol(v):d;}
static const char*envs(const char*k,const char*d){const char*v=getenv(k);return v&&*v?v:d;}
static char*slurp(const char*p,size_t*n){FILE*f=fopen(p,"rb");if(!f){perror(p);exit(1);}fseek(f,0,2);*n=ftell(f);fseek(f,0,0);char*b=malloc(*n+1);if(fread(b,1,*n,f)!=*n)exit(1);b[*n]=0;fclose(f);return b;}

/* f32 -> IEEE half bits (round-to-nearest-even, handles our small ints exactly) */
static uint16_t f2h(float f){
    uint32_t x; memcpy(&x,&f,4);
    uint32_t sign=(x>>16)&0x8000u; int32_t exp=((x>>23)&0xff)-127+15; uint32_t man=x&0x7fffffu;
    if(((x>>23)&0xff)==0xff) return sign|0x7c00u|(man?0x200u:0);
    if(exp<=0){ if(exp<-10) return sign; man|=0x800000u; uint32_t sh=14-exp; uint32_t h=man>>sh;
        if((man>>(sh-1))&1) h++; return sign|h; }
    if(exp>=0x1f) return sign|0x7c00u;
    uint16_t h=sign|(exp<<10)|(man>>13); if((man>>12)&1) h++; return h;
}

static cl_context ctx; static cl_command_queue q; static cl_device_id dev;
static char*src; static size_t sl; static cl_mem arena; static cl_uint nlanes;

static cl_kernel build(const char*opts){
    cl_int e; cl_program p=clCreateProgramWithSource(ctx,1,(const char**)&src,&sl,&e);chk(e,"prog");
    if(clBuildProgram(p,1,&dev,opts,0,0)!=CL_SUCCESS){char log[16384];clGetProgramBuildInfo(p,dev,CL_PROGRAM_BUILD_LOG,sizeof log,log,0);fprintf(stderr,"build failed (%s):\n%s\n",opts,log);exit(1);}
    cl_kernel k=clCreateKernel(p,"mm",&e);chk(e,"kernel");return k;
}
static size_t klocal(cl_kernel k){size_t rq[3]={0,0,0};clGetKernelWorkGroupInfo(k,dev,CL_KERNEL_COMPILE_WORK_GROUP_SIZE,sizeof rq,rq,0);return rq[0]?rq[0]:256;}

/* layout in floats: A halves at [0, M*K/2), B halves at [M*K/2, +K*N/2), C f32 after */
static cl_uint AO,BO,CO;
static void offsets(cl_uint M,cl_uint N,cl_uint K){AO=0;BO=(M*K+1)/2;CO=BO+(K*N+1)/2;}

static double run_once(cl_kernel k,cl_uint M,cl_uint N,cl_uint K){
    offsets(M,N,K);
    clSetKernelArg(k,0,sizeof arena,&arena);
    clSetKernelArg(k,1,sizeof AO,&AO); clSetKernelArg(k,2,sizeof BO,&BO);
    clSetKernelArg(k,3,sizeof CO,&CO); clSetKernelArg(k,4,sizeof M,&M);
    clSetKernelArg(k,5,sizeof N,&N);   clSetKernelArg(k,6,sizeof K,&K);
    clSetKernelArg(k,7,sizeof nlanes,&nlanes);
    size_t l=klocal(k), g=(size_t)nlanes*l;
    struct timespec a,b; clFinish(q); clock_gettime(CLOCK_MONOTONIC,&a);
    chk(clEnqueueNDRangeKernel(q,k,1,0,&g,&l,0,0,0),"launch");
    chk(clFinish(q),"finish"); clock_gettime(CLOCK_MONOTONIC,&b);
    return (b.tv_sec-a.tv_sec)*1e3+(b.tv_nsec-a.tv_nsec)/1e6;
}
static void upload(cl_uint M,cl_uint N,cl_uint K,float*A,float*B){
    offsets(M,N,K);
    uint16_t*Ah=malloc((size_t)M*K*2),*Bh=malloc((size_t)K*N*2);
    for(size_t i=0;i<(size_t)M*K;i++)Ah[i]=f2h(A[i]);
    for(size_t i=0;i<(size_t)K*N;i++)Bh[i]=f2h(B[i]);
    chk(clEnqueueWriteBuffer(q,arena,CL_TRUE,(size_t)AO*4,(size_t)M*K*2,Ah,0,0,0),"wA");
    chk(clEnqueueWriteBuffer(q,arena,CL_TRUE,(size_t)BO*4,(size_t)K*N*2,Bh,0,0,0),"wB");
    free(Ah);free(Bh);
}
static int verify(cl_kernel k){
    const cl_uint S=512; offsets(S,S,S);
    float*A=malloc(S*S*4),*B=malloc(S*S*4),*C=malloc(S*S*4);
    for(cl_uint i=0;i<S;i++)for(cl_uint j=0;j<S;j++){A[i*S+j]=(float)((i+2*j)%5);B[i*S+j]=(float)((3*i+j)%4);}
    upload(S,S,S,A,B);
    float z0=0.f; chk(clEnqueueFillBuffer(q,arena,&z0,4,(size_t)CO*4,S*S*4,0,0,0),"z"); chk(clFinish(q),"zf");
    run_once(k,S,S,S);
    chk(clEnqueueReadBuffer(q,arena,CL_TRUE,(size_t)CO*4,S*S*4,C,0,0,0),"rC");
    int bad=0;
    for(cl_uint i=0;i<S&&bad<4;i++)for(cl_uint j=0;j<S;j++){
        float ref=0;for(cl_uint p=0;p<S;p++)ref+=A[i*S+p]*B[p*S+j];
        if(fabsf(C[i*S+j]-ref)>1e-2f){if(bad<4)fprintf(stderr,"  bad @(%u,%u) got %g want %g\n",i,j,C[i*S+j],ref);bad++;break;}}
    free(A);free(B);free(C);return bad==0;
}
static void accuracy(cl_kernel k,cl_uint S){
    offsets(S,S,S);
    float*A=malloc((size_t)S*S*4),*B=malloc((size_t)S*S*4),*C=malloc((size_t)S*S*4);
    unsigned seed=12345;
    for(size_t i=0;i<(size_t)S*S;i++){seed=seed*1103515245u+12345u;A[i]=((seed>>9)/8388608.0f-0.5f);
        seed=seed*1103515245u+12345u;B[i]=((seed>>9)/8388608.0f-0.5f);}
    upload(S,S,S,A,B);
    float z0=0.f;clEnqueueFillBuffer(q,arena,&z0,4,(size_t)CO*4,(size_t)S*S*4,0,0,0);clFinish(q);
    run_once(k,S,S,S);
    chk(clEnqueueReadBuffer(q,arena,CL_TRUE,(size_t)CO*4,(size_t)S*S*4,C,0,0,0),"rC");
    double maxrel=0,maxabs=0;
    for(cl_uint i=0;i<64;i++)for(cl_uint j=0;j<S;j++){
        double ref=0;for(cl_uint p=0;p<S;p++)ref+=(double)A[i*S+p]*B[p*S+j];
        double d=fabs(C[i*S+j]-ref),rel=d/(fabs(ref)+1e-6);
        if(d>maxabs)maxabs=d;if(rel>maxrel)maxrel=rel;}
    free(A);free(B);free(C);
    printf("    accuracy S=%u: max_abs=%.2e max_rel=%.2e\n",S,maxabs,maxrel);
}
static double bench(cl_kernel k,cl_uint S){
    /* upload once (random-ish ints so verify-like data present) then time */
    float*A=malloc((size_t)S*S*4),*B=malloc((size_t)S*S*4);
    for(size_t i=0;i<(size_t)S*S;i++){A[i]=(float)(i%5);B[i]=(float)(i%4);}
    upload(S,S,S,A,B); free(A);free(B);
    double best=1e30;for(int r=0;r<7;r++){double ms=run_once(k,S,S,S);if(ms<best)best=ms;}
    return 2.0*(double)S*S*S/1e9/(best/1e3);
}

int main(void){
    cl_int e; cl_platform_id pl[8]; cl_uint np=0; chk(clGetPlatformIDs(8,pl,&np),"plat");
    const char*want=envs("OCL_PLATFORM","NVIDIA"); cl_platform_id p=0; char pn[256];
    for(cl_uint i=0;i<np;i++){clGetPlatformInfo(pl[i],CL_PLATFORM_NAME,sizeof pn,pn,0);if(strstr(pn,want)){p=pl[i];break;}}
    if(!p){fprintf(stderr,"no platform '%s'\n",want);return 1;}
    chk(clGetDeviceIDs(p,CL_DEVICE_TYPE_ALL,1,&dev,0),"dev");
    char dn[256];cl_uint cus;clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof dn,dn,0);
    clGetDeviceInfo(dev,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof cus,&cus,0);
    nlanes=(cl_uint)envi("VM_LANES",2*cus);
    printf("%s | %s | %u CUs | %u lanes\n",pn,dn,cus,nlanes);
    ctx=clCreateContext(0,1,&dev,0,0,&e);chk(e,"ctx");
    q=clCreateCommandQueue(ctx,dev,0,&e);chk(e,"q");
    src=slurp("mma17_f16in.cl",&sl);
    const size_t AR=3ull*4096*4096; arena=clCreateBuffer(ctx,CL_MEM_READ_WRITE,AR*4,0,&e);chk(e,"arena");

    struct{const char*name,*opts;}steps[]={
      {"F16IN 256x128 W8x4 BK32 NBUF3 * WINNER","-DTM=256 -DTN=128 -DBK=32 -DNBUF=3 -DWM=8 -DWN=4"},
      {"F16IN 256x128 W8x4 BK32 NBUF2 (1024t)","-DTM=256 -DTN=128 -DBK=32 -DNBUF=2 -DWM=8 -DWN=4"},
      {"F16IN 256x128 W8x4 BK16 NBUF2 (1024t)","-DTM=256 -DTN=128 -DBK=16 -DNBUF=2 -DWM=8 -DWN=4"},
      {"F16IN 256x128 W16x2 BK32 NBUF2(1024t)","-DTM=256 -DTN=128 -DBK=32 -DNBUF=2 -DWM=16 -DWN=2"},
      {"F16IN 128x128 W4x4 BK32 NBUF2 (512t)", "-DTM=128 -DTN=128 -DBK=32 -DNBUF=2 -DWM=4 -DWN=4"},
      {"F16IN 128x128 W4x4 BK64 NBUF2 (512t)", "-DTM=128 -DTN=128 -DBK=64 -DNBUF=2 -DWM=4 -DWN=4"},
      {"F16IN 256x256 W8x4 BK16 NBUF2 (1024t)","-DTM=256 -DTN=256 -DBK=16 -DNBUF=2 -DWM=8 -DWN=4"},
    };
    int ns=sizeof steps/sizeof steps[0]; const char*only=getenv("ONLY");
    if(envi("WARMUP",1)){cl_kernel wk=build("-DTM=256 -DTN=128 -DBK=16 -DNBUF=2 -DWM=8 -DWN=4");
        for(int r=0;r<15;r++)run_once(wk,4096,4096,4096);clReleaseKernel(wk);}
    printf("\n%-40s %5s %9s %9s\n","config","512","2048 TF","4096 TF");
    printf("-----------------------------------------------------------------------------\n");
    for(int i=0;i<ns;i++){
        if(only&&!strstr(steps[i].name,only))continue;
        char ob[512];snprintf(ob,sizeof ob,"%s %s",steps[i].opts,envs("EXTRA",""));
        cl_kernel k=build(ob); int ok=verify(k);
        double g2=bench(k,2048)/1e3,g4=bench(k,4096)/1e3;
        printf("%-40s %5s %9.1f %9.1f\n",steps[i].name,ok?"ok":"NO",g2,g4);
        if(getenv("ACC"))accuracy(k,1024);
        clReleaseKernel(k);
    }
    printf("\nref: f32-arena wmma/mma f16 ~92 TF/s (§38); cuBLAS tf32 116-133; cuBLAS f16 ~260\n");
    return 0;
}
