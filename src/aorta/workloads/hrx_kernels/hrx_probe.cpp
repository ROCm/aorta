// hrx_probe.cpp — localizes the HRX wrong-result bug.
// out[i] = in[i] + 100, with in=7 and out pre-set to 0 via hipMemset.
//   107 -> fully works (read + write + copies all OK)
//   100 -> input read as 0 (H2D / input-arg broken)
//     0 -> output write never reached host buffer (kernel-write / out-arg broken)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void add100(float* out, const float* in, unsigned long n) {
  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] + 100.0f;
}

#define CK(c) do{ hipError_t e=(c); if(e!=hipSuccess){ \
  printf("HIP_ERR %s at %d: %s\n",#c,__LINE__,hipGetErrorString(e)); return 2;} }while(0)

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

  add100<<<(N + 255) / 256, 256>>>(out_d, in_d, N);
  CK(hipDeviceSynchronize());
  CK(hipMemcpy(out_h, out_d, bytes, hipMemcpyDeviceToHost));
  CK(hipDeviceSynchronize());

  printf("out[0]=%g (expect 107)\n", out_h[0]);
  if      (out_h[0] == 107.0f) printf("VERDICT=FULLY_WORKS\n");
  else if (out_h[0] == 100.0f) printf("VERDICT=INPUT_READ_ZERO (H2D/in-arg broken)\n");
  else if (out_h[0] == 0.0f)   printf("VERDICT=OUTPUT_NOT_WRITTEN (kernel-write/out-arg broken)\n");
  else                         printf("VERDICT=GARBAGE (address mismatch)\n");
  // Exit 0 on 107 (works), 1 otherwise -- mirrors the other probe variants so
  // downstream tooling can tell success from failure by exit code even if
  // stdout parsing changes.
  return out_h[0] == 107.0f ? 0 : 1;
}
