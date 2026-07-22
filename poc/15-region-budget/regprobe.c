/* regprobe: build the real vm2 megakernel with -cl-nv-verbose and print the
 * per-kernel register / smem / spill usage from the NVIDIA build log. A
 * DETERMINISTIC instrument (unlike the noisy spin residency probe) for the
 * §27 register-budget investigation. Usage: regprobe [extra build opts...]
 * Include path must point at the generated build-dev/vm_cl_source.h. */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vm_cl_source.h"   /* defines kVmClSource */

int main(int argc, char **argv) {
  char opts[8192] = "-cl-std=CL3.0 -cl-nv-verbose";
  for (int i = 1; i < argc; i++) { strcat(opts, " "); strcat(opts, argv[i]); }

  cl_platform_id plats[8]; cl_uint np = 0;
  clGetPlatformIDs(8, plats, &np);
  cl_platform_id plat = 0; cl_device_id dev = 0;
  for (cl_uint i = 0; i < np; i++) {
    char name[256] = {0};
    clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof(name), name, 0);
    if (strstr(name, "NVIDIA")) { plat = plats[i]; break; }
  }
  if (!plat) { fprintf(stderr, "no NVIDIA platform\n"); return 1; }
  clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, 0);
  cl_int err;
  cl_context ctx = clCreateContext(0, 1, &dev, 0, 0, &err);
  const char *src = kVmClSource;
  cl_program prog = clCreateProgramWithSource(ctx, 1, &src, 0, &err);
  err = clBuildProgram(prog, 1, &dev, opts, 0, 0);
  size_t logsz = 0;
  clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, 0, &logsz);
  char *log = malloc(logsz + 1);
  clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logsz, log, 0);
  log[logsz] = 0;
  printf("=== opts: %s | status %d ===\n", opts, err);
  char *save = 0;
  char *line = strtok_r(log, "\n", &save);
  while (line) {
    if (strstr(line, "vm2") || strstr(line, "egisters") ||
        strstr(line, "tack") || strstr(line, "pill") ||
        strstr(line, "Function") || strstr(line, "rror"))
      printf("%s\n", line);
    line = strtok_r(0, "\n", &save);
  }
  free(log);
  return 0;
}
