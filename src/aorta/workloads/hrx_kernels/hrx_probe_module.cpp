// hrx_probe_module.cpp - dynamic-module-launch variant of hrx_probe.
//
// Same computation and verdict logic as hrx_probe (out[i] = in[i] + 100, in=7,
// out pre-set to 0), but the kernel is loaded at runtime from a SEPARATE code
// object (hrx_probe_module_kernel.code) via the HIP driver/module API
// (hipModuleLoad -> hipModuleGetFunction -> hipModuleLaunchKernel) instead of the
// statically-registered hipLaunchKernelGGL path.
//
// Verdict from out[0]:
//   107 -> FULLY_WORKS (read + write + copies all OK)
//   100 -> INPUT_READ_ZERO (H2D / input-arg broken)
//     0 -> OUTPUT_NOT_WRITTEN (module-launch kernel-write / out-arg broken)
//
// Loads the .code from next to the executable, so it runs from any cwd.
// Exits 0 on 107 (works), 1 otherwise.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <libgen.h>
#include <string>

#define CK(c) do{ hipError_t e=(c); if(e!=hipSuccess){ \
  printf("HIP_ERR %s at %d: %s\n",#c,__LINE__,hipGetErrorString(e)); return 2;} }while(0)

static std::string exe_dir() {
  char buf[4096];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n < 0) return ".";
  buf[n] = '\0';
  return std::string(dirname(buf));
}

int main() {
  const unsigned long N = 1000000;
  const size_t bytes = N * sizeof(float);
  float* in_h  = (float*)malloc(bytes);
  float* out_h = (float*)malloc(bytes);
  for (unsigned long i = 0; i < N; i++) { in_h[i] = 7.0f; out_h[i] = -1.0f; }

  float *in_d, *out_d;
  CK(hipMalloc(&in_d, bytes));
  CK(hipMalloc(&out_d, bytes));
  CK(hipMemset(out_d, 0, bytes));                               // out_d = 0
  CK(hipMemcpy(in_d, in_h, bytes, hipMemcpyHostToDevice));      // in_d  = 7
  CK(hipDeviceSynchronize());

  std::string code_path = exe_dir() + "/hrx_probe_module_kernel.code";
  printf("info: hipModuleLoad %s\n", code_path.c_str());
  hipModule_t module;
  hipFunction_t func;
  CK(hipModuleLoad(&module, code_path.c_str()));
  CK(hipModuleGetFunction(&func, module, "add100"));

  struct { void* out; void* in; unsigned long n; } args;
  args.out = out_d;
  args.in  = in_d;
  args.n   = N;
  size_t arg_size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
                    HIP_LAUNCH_PARAM_END};

  const unsigned block = 256;
  const unsigned grid  = (N + block - 1) / block;
  printf("info: hipModuleLaunchKernel 'add100'\n");
  CK(hipModuleLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, 0, NULL, (void**)&config));
  CK(hipDeviceSynchronize());

  CK(hipMemcpy(out_h, out_d, bytes, hipMemcpyDeviceToHost));
  CK(hipDeviceSynchronize());

  printf("out[0]=%g (expect 107)\n", out_h[0]);
  if      (out_h[0] == 107.0f) printf("VERDICT=FULLY_WORKS\n");
  else if (out_h[0] == 100.0f) printf("VERDICT=INPUT_READ_ZERO (H2D/in-arg broken)\n");
  else if (out_h[0] == 0.0f)   printf("VERDICT=OUTPUT_NOT_WRITTEN (module-launch out-arg broken)\n");
  else                         printf("VERDICT=GARBAGE (address mismatch)\n");

  return out_h[0] == 107.0f ? 0 : 1;
}
