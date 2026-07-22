/* poc/13 — async / prefetched DRAM loads host harness.
 *
 * Benchmarks two representative tile loops, three variants each, on the device
 * selected by PJRT_OCL_DEVICE (platform-name substring, like poc/07/08):
 *   LOOP A streaming EW: ew_scalar / ew_regdb / ew_async
 *   LOOP B matmul K-loop: mma_single / mma_double / mma_async
 * Plus an occupancy-discovery pass (poc/08 handshake) at 8 KB vs 16 KB __local
 * to price the co-residency cost of double-buffered/async-local staging (§10c).
 *
 * Persistent-grid faithful: FIXED grid = 2*CU groups, grid-stride over tiles.
 * Emits CSV rows on stdout; human summary on stderr.
 *
 *   make && PJRT_OCL_DEVICE=NVIDIA ./poc13 > results_nvidia.csv
 *            PJRT_OCL_DEVICE=Portable ./poc13 > results_pocl.csv
 */
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define CK(e) do{cl_int _e=(e); if(_e!=CL_SUCCESS){fprintf(stderr,"%s:%d CL err %d\n",__FILE__,__LINE__,_e);exit(1);} }while(0)
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }

static const size_t LOCAL = 256;
static cl_context ctx; static cl_command_queue q; static cl_device_id dev;
static size_t g_grid_groups; /* fixed persistent grid (2*CU) */

static char *read_file(const char *path){
    FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){perror("read");exit(1);}
    b[n]=0; fclose(f); return b;
}

static cl_program build(const char *src, const char *opts){
    cl_int e; cl_program p=clCreateProgramWithSource(ctx,1,&src,NULL,&e); CK(e);
    if(clBuildProgram(p,1,&dev,opts,NULL,NULL)!=CL_SUCCESS){
        char log[16384]; size_t ls=0;
        clGetProgramBuildInfo(p,dev,CL_PROGRAM_BUILD_LOG,sizeof log,log,&ls);
        fprintf(stderr,"build failed (%s):\n%.*s\n",opts?opts:"",(int)ls,log); exit(1);
    }
    return p;
}

/* ---- timing: best-of-N wall ms for a prepared kernel over a fixed grid ---- */
static double bench(cl_kernel k, int reps){
    size_t g=g_grid_groups*LOCAL, l=LOCAL;
    /* warmup */
    CK(clEnqueueNDRangeKernel(q,k,1,NULL,&g,&l,0,NULL,NULL)); CK(clFinish(q));
    double best=1e30;
    for(int r=0;r<reps;r++){
        double t0=now_ms();
        CK(clEnqueueNDRangeKernel(q,k,1,NULL,&g,&l,0,NULL,NULL));
        CK(clFinish(q));
        double dt=now_ms()-t0; if(dt<best) best=dt;
    }
    return best;
}

/* =============================== LOOP A: EW =============================== */
static void run_ew(cl_program prog){
    const uint sizes[]={262144u,1048576u,4194304u,16777216u};
    const char *names[]={"ew_scalar","ew_regdb","ew_async"};
    cl_int e;
    for(size_t si=0;si<sizeof sizes/sizeof*sizes;si++){
        uint n=sizes[si];
        float *ha=malloc((size_t)n*4), *ref=malloc((size_t)n*4);
        for(uint i=0;i<n;i++) ha[i]=(float)((i*1103515245u+12345u)&0xffff)/65536.0f;
        float s=1.5f,t=-0.25f;
        for(uint i=0;i<n;i++) ref[i]=ha[i]*s+t;
        cl_mem da=clCreateBuffer(ctx,CL_MEM_READ_ONLY,(size_t)n*4,NULL,&e);CK(e);
        cl_mem dd=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,(size_t)n*4,NULL,&e);CK(e);
        CK(clEnqueueWriteBuffer(q,da,CL_TRUE,0,(size_t)n*4,ha,0,NULL,NULL));
        for(int v=0;v<3;v++){
            cl_kernel k=clCreateKernel(prog,names[v],&e);CK(e);
            CK(clSetKernelArg(k,0,sizeof da,&da));
            CK(clSetKernelArg(k,1,sizeof dd,&dd));
            CK(clSetKernelArg(k,2,sizeof n,&n));
            CK(clSetKernelArg(k,3,sizeof s,&s));
            CK(clSetKernelArg(k,4,sizeof t,&t));
            double ms=bench(k,20);
            /* correctness */
            float *out=malloc((size_t)n*4);
            CK(clEnqueueReadBuffer(q,dd,CL_TRUE,0,(size_t)n*4,out,0,NULL,NULL));
            double maxerr=0; for(uint i=0;i<n;i++){double d=fabs(out[i]-ref[i]); if(d>maxerr)maxerr=d;}
            free(out);
            double gbs = (double)n*8.0/1e9 / (ms/1e3);   /* 4B read + 4B write */
            printf("ew,%s,%u,%.4f,%.1f,%.2e\n",names[v]+3,n,ms,gbs,maxerr);
            fprintf(stderr,"  EW %-7s n=%-9u  %8.4f ms  %6.1f GB/s  err=%.1e\n",
                    names[v]+3,n,ms,gbs,maxerr);
            clReleaseKernel(k);
        }
        clReleaseMemObject(da); clReleaseMemObject(dd); free(ha); free(ref);
    }
}

/* ============================== LOOP B: MMA =============================== */
static uint padup(uint x,uint m){ return ((x+m-1)/m)*m; }
static void run_mma(cl_program prog){
    /* base-representative shapes (transformer base: D=1024, ff=4096, M=512). */
    const uint shapes[][3]={{512,512,512},{512,2048,512},{1024,1024,1024},
                            {512,512,2048},{2048,2048,2048}};
    const char *names[]={"mma_single","mma_double","mma_async"};
    cl_int e;
    for(size_t si=0;si<sizeof shapes/sizeof*shapes;si++){
        uint M=padup(shapes[si][0],64),N=padup(shapes[si][1],64),K=padup(shapes[si][2],16);
        float *hA=malloc((size_t)M*K*4), *hB=malloc((size_t)K*N*4);
        float *hBcm=malloc((size_t)N*K*4), *ref=malloc((size_t)M*N*4);
        for(size_t i=0;i<(size_t)M*K;i++) hA[i]=(float)((i*2654435761u)&0x3ff)/1024.0f-0.5f;
        for(size_t i=0;i<(size_t)K*N;i++) hB[i]=(float)((i*40503u+7u)&0x3ff)/1024.0f-0.5f;
        for(uint kk=0;kk<K;kk++) for(uint nn=0;nn<N;nn++) hBcm[(size_t)nn*K+kk]=hB[(size_t)kk*N+nn];
        /* host reference (single-threaded, only for the two smaller shapes) */
        int check = ((size_t)M*N*K <= (size_t)512*512*512*4);
        if(check){
            for(uint m=0;m<M;m++) for(uint nn=0;nn<N;nn++){
                double acc=0; for(uint kk=0;kk<K;kk++) acc+=(double)hA[(size_t)m*K+kk]*hB[(size_t)kk*N+nn];
                ref[(size_t)m*N+nn]=(float)acc;
            }
        }
        cl_mem dA=clCreateBuffer(ctx,CL_MEM_READ_ONLY,(size_t)M*K*4,NULL,&e);CK(e);
        cl_mem dB=clCreateBuffer(ctx,CL_MEM_READ_ONLY,(size_t)N*K*4,NULL,&e);CK(e);
        cl_mem dC=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,(size_t)M*N*4,NULL,&e);CK(e);
        CK(clEnqueueWriteBuffer(q,dA,CL_TRUE,0,(size_t)M*K*4,hA,0,NULL,NULL));
        CK(clEnqueueWriteBuffer(q,dB,CL_TRUE,0,(size_t)N*K*4,hBcm,0,NULL,NULL));
        float *base_out=malloc((size_t)M*N*4); int have_base=0;
        for(int v=0;v<3;v++){
            cl_kernel k=clCreateKernel(prog,names[v],&e);CK(e);
            CK(clSetKernelArg(k,0,sizeof dA,&dA));
            CK(clSetKernelArg(k,1,sizeof dB,&dB));
            CK(clSetKernelArg(k,2,sizeof dC,&dC));
            CK(clSetKernelArg(k,3,sizeof M,&M));
            CK(clSetKernelArg(k,4,sizeof N,&N));
            CK(clSetKernelArg(k,5,sizeof K,&K));
            double ms=bench(k,10);
            float *out=malloc((size_t)M*N*4);
            CK(clEnqueueReadBuffer(q,dC,CL_TRUE,0,(size_t)M*N*4,out,0,NULL,NULL));
            double maxerr=0; const char *cmp;
            if(check){ for(size_t i=0;i<(size_t)M*N;i++){double d=fabs(out[i]-ref[i]);if(d>maxerr)maxerr=d;} cmp="vs-host"; }
            else if(!have_base){ memcpy(base_out,out,(size_t)M*N*4); have_base=1; maxerr=0; cmp="base"; }
            else { for(size_t i=0;i<(size_t)M*N;i++){double d=fabs(out[i]-base_out[i]);if(d>maxerr)maxerr=d;} cmp="vs-single"; }
            free(out);
            double gflops = 2.0*(double)M*N*K/1e9 / (ms/1e3);
            printf("mma,%s,%ux%ux%u,%.4f,%.1f,%.2e,%s\n",names[v]+4,M,N,K,ms,gflops,maxerr,cmp);
            fprintf(stderr,"  MMA %-6s %ux%ux%u  %8.4f ms  %7.1f GFLOP/s  err=%.1e (%s)\n",
                    names[v]+4,M,N,K,ms,gflops,maxerr,cmp);
            clReleaseKernel(k);
        }
        free(base_out);
        clReleaseMemObject(dA);clReleaseMemObject(dB);clReleaseMemObject(dC);
        free(hA);free(hB);free(hBcm);free(ref);
    }
}

/* ===================== occupancy discovery (poc/08) ====================== */
static cl_uint discover_groups(cl_program prog){
    cl_int e; cl_kernel k=clCreateKernel(prog,"probe",&e);CK(e);
    cl_mem d=clCreateBuffer(ctx,CL_MEM_READ_WRITE,64,NULL,&e);CK(e);
    cl_mem sink=clCreateBuffer(ctx,CL_MEM_READ_WRITE,8192*4,NULL,&e);CK(e);
    cl_uint never=0;
    CK(clSetKernelArg(k,0,sizeof d,&d));
    CK(clSetKernelArg(k,1,sizeof sink,&sink));
    CK(clSetKernelArg(k,2,sizeof never,&never));
    cl_uint cus; CK(clGetDeviceInfo(dev,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof cus,&cus,NULL));
    size_t launch=cus*4<64?64:cus*4; if(launch>4096)launch=4096;
    cl_uint zeros[3]={0,1,0};
    CK(clEnqueueWriteBuffer(q,d,CL_TRUE,0,sizeof zeros,zeros,0,NULL,NULL));
    size_t g=launch*LOCAL,l=LOCAL;
    CK(clEnqueueNDRangeKernel(q,k,1,NULL,&g,&l,0,NULL,NULL)); CK(clFinish(q));
    cl_uint out[3]; CK(clEnqueueReadBuffer(q,d,CL_TRUE,0,sizeof out,out,0,NULL,NULL));
    clReleaseKernel(k); clReleaseMemObject(d); clReleaseMemObject(sink);
    return out[2];
}

int main(void){
    const char *want=getenv("PJRT_OCL_DEVICE");
    cl_uint nplat; CK(clGetPlatformIDs(0,NULL,&nplat));
    cl_platform_id plats[8]; CK(clGetPlatformIDs(nplat<8?nplat:8,plats,NULL));
    cl_platform_id plat=0; char nm[256];
    for(cl_uint i=0;i<nplat;i++){
        CK(clGetPlatformInfo(plats[i],CL_PLATFORM_NAME,sizeof nm,nm,NULL));
        if(want&&!strstr(nm,want)) continue;
        cl_uint nd; if(clGetDeviceIDs(plats[i],CL_DEVICE_TYPE_ALL,0,NULL,&nd)!=CL_SUCCESS||!nd) continue;
        cl_device_id ds[8]; CK(clGetDeviceIDs(plats[i],CL_DEVICE_TYPE_ALL,nd<8?nd:8,ds,NULL));
        plat=plats[i]; dev=ds[0]; break;
    }
    if(!dev){fprintf(stderr,"no device (PJRT_OCL_DEVICE=%s)\n",want?want:"(any)");return 1;}
    CK(clGetPlatformInfo(plat,CL_PLATFORM_NAME,sizeof nm,nm,NULL));
    char dnm[256]; CK(clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof dnm,dnm,NULL));
    cl_uint cus; CK(clGetDeviceInfo(dev,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof cus,&cus,NULL));
    cl_ulong lmem; CK(clGetDeviceInfo(dev,CL_DEVICE_LOCAL_MEM_SIZE,sizeof lmem,&lmem,NULL));
    fprintf(stderr,"platform: %s\ndevice  : %s (%u CU, %lu KB local)\n",nm,dnm,cus,(unsigned long)lmem/1024);

    cl_int e; ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&e);CK(e);
    q=clCreateCommandQueue(ctx,dev,0,&e);CK(e);
    g_grid_groups = 2*cus;                         /* the megakernel launch cap */
    fprintf(stderr,"grid    : %zu groups x %zu (persistent, grid-stride)\n\n",g_grid_groups,LOCAL);

    /* GPU EW tile size 4096 per §22; PoCL keeps 16384 (host-tile overhead). */
    int is_cpu = strstr(nm,"Portable")||strstr(dnm,"CPU")||strstr(dnm,"cpu");
    char opts[128]; snprintf(opts,sizeof opts,"-DEW_TS=%d",is_cpu?16384:4096);
    char *src=read_file("kernels.cl");
    cl_program prog=build(src,opts);

    printf("kind,variant,size,ms,throughput,maxerr,cmp\n");
    fprintf(stderr,"== LOOP A: streaming elementwise (d=a*s+t) ==\n");
    run_ew(prog);
    fprintf(stderr,"\n== LOOP B: matmul K-loop global->local stage ==\n");
    run_mma(prog);

    /* occupancy: 8 KB (single) vs 16 KB (double/async-local) __local. */
    fprintf(stderr,"\n== occupancy discovery (co-resident groups) ==\n");
    cl_program p8 =build(src,"-DLOCAL_FLOATS=2048");   /*  8 KB */
    cl_program p16=build(src,"-DLOCAL_FLOATS=4096");   /* 16 KB */
    cl_uint g8=discover_groups(p8), g16=discover_groups(p16);
    fprintf(stderr,"   8 KB local (single-buffer) : %u co-resident groups\n",g8);
    fprintf(stderr,"  16 KB local (double/async)  : %u co-resident groups  (cap=%zu)\n",g16,g_grid_groups);
    printf("occ,8KB,-, -,%u,-,-\n",g8);
    printf("occ,16KB,-, -,%u,-,-\n",g16);
    fprintf(stderr,"\nmegakernel cap = 2*CU = %zu; occupancy matters only if it drops BELOW the cap.\n",g_grid_groups);
    free(src);
    return 0;
}
