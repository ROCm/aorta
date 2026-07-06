// hrx_probe_graph_setparams.cpp - isolates the hipGraphKernelNodeSetParams
// "extra" (HIP_LAUNCH_PARAM pre-packed buffer) path.
//
// The node is FIRST created with kernelParams (the ARGS_ARRAY path, a different
// and known-good mechanism), then updated via hipGraphKernelNodeSetParams using
// the `extra` buffer - so this repro exercises ONLY the SetParams-with-extra
// path, independent of hipGraphAddKernelNode-with-extra. The graph is
// instantiated AFTER SetParams so the launch reflects the updated params.
//
// add100: out[i] = in[i] + 100, in=7, out pre-set to 0.
//   107 -> FULLY_WORKS
//   100 -> INPUT_READ_ZERO
//     0 -> OUTPUT_NOT_WRITTEN (SetParams extra/out-arg not resolved)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

__global__ void add100(float* out, const float* in, unsigned long n) {
  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] + 100.0f;
}

#define CK(c) do{ hipError_t e=(c); if(e!=hipSuccess){ \
  printf("HIP_ERR %s at %d: %s\n",#c,__LINE__,hipGetErrorString(e)); return 2;} }while(0)

struct Args { void* out; void* in; unsigned long n; };

static const char* verdict(float v) {
  if (v == 107.0f) return "FULLY_WORKS";
  if (v == 100.0f) return "INPUT_READ_ZERO";
  if (v == 0.0f)   return "OUTPUT_NOT_WRITTEN (SetParams extra/out-arg broken)";
  return "GARBAGE";
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
  CK(hipMemset(out_d, 0, bytes));
  CK(hipMemcpy(in_d, in_h, bytes, hipMemcpyHostToDevice));
  CK(hipDeviceSynchronize());

  const unsigned block = 256;
  const unsigned grid  = (N + block - 1) / block;

  hipGraph_t graph;
  CK(hipGraphCreate(&graph, 0));

  // --- Create the node via kernelParams (known-good ARGS_ARRAY path) --------
  unsigned long n_val = N;
  void* kernel_params[] = {&out_d, &in_d, &n_val};
  hipKernelNodeParams kp_add;
  memset(&kp_add, 0, sizeof(kp_add));
  kp_add.func = (void*)add100;
  kp_add.gridDim = dim3(grid, 1, 1);
  kp_add.blockDim = dim3(block, 1, 1);
  kp_add.sharedMemBytes = 0;
  kp_add.kernelParams = kernel_params;
  kp_add.extra = NULL;
  hipGraphNode_t knode;
  CK(hipGraphAddKernelNode(&knode, graph, NULL, 0, &kp_add));

  // --- Update the SAME node via SetParams using `extra` (path under test) ---
  Args args; args.out = out_d; args.in = in_d; args.n = N;
  size_t arg_size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
                    HIP_LAUNCH_PARAM_END};
  hipKernelNodeParams kp_set;
  memset(&kp_set, 0, sizeof(kp_set));
  kp_set.func = (void*)add100;
  kp_set.gridDim = dim3(grid, 1, 1);
  kp_set.blockDim = dim3(block, 1, 1);
  kp_set.sharedMemBytes = 0;
  kp_set.kernelParams = NULL;
  kp_set.extra = config;  // <-- the path under test
  CK(hipGraphKernelNodeSetParams(knode, &kp_set));

  // Instantiate AFTER SetParams so the launch reflects the updated params.
  hipGraphExec_t exec;
  CK(hipGraphInstantiate(&exec, graph, NULL, NULL, 0));
  CK(hipGraphLaunch(exec, 0));
  CK(hipDeviceSynchronize());
  CK(hipMemcpy(out_h, out_d, bytes, hipMemcpyDeviceToHost));
  CK(hipDeviceSynchronize());

  printf("out[0]=%g (expect 107)\n", out_h[0]);
  printf("VERDICT=%s\n", verdict(out_h[0]));
  return out_h[0] == 107.0f ? 0 : 1;
}
