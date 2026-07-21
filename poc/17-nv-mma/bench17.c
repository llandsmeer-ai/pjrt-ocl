/* poc/17 standalone tensor-core matmul bench: sweep tile size / buffering /
 * lanes, verify at 512^3 (small-int, tf32-exact), report best-of-N TFLOP/s at
 * 2048^3 and 4096^3. Anchors the §35 ceiling of the OpenCL->PTX WMMA path vs
 * the in-megakernel ~18-20 and cuBLAS 116-133. env: OCL_PLATFORM, VM_LANES. */
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static void chk(cl_int e, const char *m){ if(e){fprintf(stderr,"ERR %d @ %s\n",e,m);exit(1);} }
static long envi(const char*k,long d){const char*v=getenv(k);return v&&*v?atol(v):d;}
static const char*envs(const char*k,const char*d){const char*v=getenv(k);return v&&*v?v:d;}
static char*slurp(const char*p,size_t*n){FILE*f=fopen(p,"rb");if(!f){perror(p);exit(1);}fseek(f,0,2);*n=ftell(f);fseek(f,0,0);char*b=malloc(*n+1);if(fread(b,1,*n,f)!=*n)exit(1);b[*n]=0;fclose(f);return b;}

static cl_context ctx; static cl_command_queue q; static cl_device_id dev;
static char*src; static size_t sl; static cl_mem arena; static cl_uint nlanes;

static cl_kernel build(const char*opts){
    cl_int e;
    cl_program p=clCreateProgramWithSource(ctx,1,(const char**)&src,&sl,&e);chk(e,"prog");
    if(clBuildProgram(p,1,&dev,opts,0,0)!=CL_SUCCESS){
        char log[16384];clGetProgramBuildInfo(p,dev,CL_PROGRAM_BUILD_LOG,sizeof log,log,0);
        fprintf(stderr,"build failed (%s):\n%s\n",opts,log);exit(1);
    }
    cl_kernel k=clCreateKernel(p,"mm",&e);chk(e,"kernel");return k;
}
static cl_uint g_regs(cl_kernel k){
    cl_uint r=0; clGetKernelWorkGroupInfo(k,dev,0x11B3/*NV_REGISTERS*/,sizeof r,&r,0); return r;
}
static double run_once(cl_kernel k,cl_uint M,cl_uint N,cl_uint K){
    cl_uint ao=0,bo=M*K,co=M*K+K*N;
    clSetKernelArg(k,0,sizeof arena,&arena);
    clSetKernelArg(k,1,sizeof ao,&ao); clSetKernelArg(k,2,sizeof bo,&bo);
    clSetKernelArg(k,3,sizeof co,&co); clSetKernelArg(k,4,sizeof M,&M);
    clSetKernelArg(k,5,sizeof N,&N);   clSetKernelArg(k,6,sizeof K,&K);
    clSetKernelArg(k,7,sizeof nlanes,&nlanes);
    size_t g=(size_t)nlanes*256,l=256;
    struct timespec a,b; clFinish(q);
    clock_gettime(CLOCK_MONOTONIC,&a);
    chk(clEnqueueNDRangeKernel(q,k,1,0,&g,&l,0,0,0),"launch");
    chk(clFinish(q),"finish");
    clock_gettime(CLOCK_MONOTONIC,&b);
    return (b.tv_sec-a.tv_sec)*1e3+(b.tv_nsec-a.tv_nsec)/1e6;
}
static int verify(cl_kernel k){
    const cl_uint S=512; const cl_uint ao=0,bo=S*S,co=2*S*S;
    float*A=malloc(S*S*4),*B=malloc(S*S*4),*C=malloc(S*S*4);
    for(cl_uint i=0;i<S;i++)for(cl_uint j=0;j<S;j++){
        A[i*S+j]=(float)((i+2*j)%5); B[i*S+j]=(float)((3*i+j)%4); }
    chk(clEnqueueWriteBuffer(q,arena,CL_TRUE,ao*4,S*S*4,A,0,0,0),"wA");
    chk(clEnqueueWriteBuffer(q,arena,CL_TRUE,bo*4,S*S*4,B,0,0,0),"wB");
    run_once(k,S,S,S);
    chk(clEnqueueReadBuffer(q,arena,CL_TRUE,co*4,S*S*4,C,0,0,0),"rC");
    int bad=0;
    for(cl_uint i=0;i<S&&bad<4;i++)for(cl_uint j=0;j<S;j++){
        float ref=0; for(cl_uint p=0;p<S;p++)ref+=A[i*S+p]*B[p*S+j];
        if(fabsf(C[i*S+j]-ref)>1e-2f){ if(bad<4)fprintf(stderr,"  bad @(%u,%u) got %g want %g\n",i,j,C[i*S+j],ref); bad++; break; } }
    free(A);free(B);free(C); return bad==0;
}
static double bench(cl_kernel k,cl_uint S){
    double best=1e30; for(int r=0;r<7;r++){double ms=run_once(k,S,S,S);if(ms<best)best=ms;}
    return 2.0*S*S*S/1e9 / (best/1e3); /* GFLOP/s */
}

int main(void){
    cl_int e; cl_platform_id pl[8]; cl_uint np=0; chk(clGetPlatformIDs(8,pl,&np),"plat");
    const char*want=envs("OCL_PLATFORM","NVIDIA"); cl_platform_id p=0; char pn[256];
    for(cl_uint i=0;i<np;i++){clGetPlatformInfo(pl[i],CL_PLATFORM_NAME,sizeof pn,pn,0);if(strstr(pn,want)){p=pl[i];break;}}
    if(!p){fprintf(stderr,"no platform '%s'\n",want);return 1;}
    chk(clGetDeviceIDs(p,CL_DEVICE_TYPE_ALL,1,&dev,0),"dev");
    char dn[256]; cl_uint cus; clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof dn,dn,0);
    clGetDeviceInfo(dev,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof cus,&cus,0);
    nlanes=(cl_uint)envi("VM_LANES",2*cus);
    printf("%s | %s | %u CUs | %u lanes x256\n",pn,dn,cus,nlanes);
    ctx=clCreateContext(0,1,&dev,0,0,&e);chk(e,"ctx");
    q=clCreateCommandQueue(ctx,dev,0,&e);chk(e,"q");
    src=slurp("mma17.cl",&sl);
    const size_t AR=3ull*4096*4096; arena=clCreateBuffer(ctx,CL_MEM_READ_WRITE,AR*4,0,&e);chk(e,"arena");

    struct { const char*name,*opts; } steps[] = {
      {"64x64  BK16 NBUF1 (== in-mega tile)", "-DTM=64  -DTN=64  -DBK=16 -DNBUF=1"},
      {"64x64  BK16 NBUF2 (sync dbl-buf)",    "-DTM=64  -DTN=64  -DBK=16 -DNBUF=2"},
      {"128x64 BK16 NBUF1",                   "-DTM=128 -DTN=64  -DBK=16 -DNBUF=1"},
      {"128x64 BK16 NBUF2",                   "-DTM=128 -DTN=64  -DBK=16 -DNBUF=2"},
      {"128x128 BK16 NBUF1",                  "-DTM=128 -DTN=128 -DBK=16 -DNBUF=1"},
      {"128x128 BK16 NBUF2",                  "-DTM=128 -DTN=128 -DBK=16 -DNBUF=2"},
      {"128x128 BK32 NBUF2",                  "-DTM=128 -DTN=128 -DBK=32 -DNBUF=2"},
      {"256x128 BK16 NBUF1",                  "-DTM=256 -DTN=128 -DBK=16 -DNBUF=1"},
      {"256x128 BK16 NBUF2",                  "-DTM=256 -DTN=128 -DBK=16 -DNBUF=2"},
    };
    int ns=sizeof steps/sizeof steps[0];
    const char*only=getenv("ONLY");

    if(envi("WARMUP",1)){ cl_kernel wk=build("-DTM=128 -DTN=128 -DBK=16 -DNBUF=2");
        for(int r=0;r<15;r++) run_once(wk,4096,4096,4096); clReleaseKernel(wk); }

    printf("\n%-38s %5s %5s %9s %9s\n","config","512","regs","2048 TF","4096 TF");
    printf("---------------------------------------------------------------------------\n");
    for(int i=0;i<ns;i++){
        if(only&&!strstr(steps[i].name,only))continue;
        cl_kernel k=build(steps[i].opts);
        int ok=verify(k); cl_uint regs=g_regs(k);
        double g2=bench(k,2048)/1e3, g4=bench(k,4096)/1e3;
        printf("%-38s %5s %5u %9.1f %9.1f\n",steps[i].name,ok?"ok":"NO",regs,g2,g4);
        clReleaseKernel(k);
    }
    printf("\nref: in-megakernel tile ~18-20 TF/s (§10d/§31); cuBLAS 116-133 TF/s\n");
    return 0;
}
