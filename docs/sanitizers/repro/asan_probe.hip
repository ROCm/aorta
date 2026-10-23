// GPU AddressSanitizer capability probe for aorta (gfx950 / MI350).
//
// One binary, one defect per mode, so each run answers one question:
//
//   host_oob    host heap overflow. Pure host, no GPU touched -- runnable on
//               the head node to confirm the host ASan runtime is live.
//   inbounds    control: a correct kernel. Must run clean under ASan, or the
//               instrumented build itself is the defect.
//   device_oob  off-by-one store one int past a hipMalloc'd buffer of n ints.
//               The headline: does device-side ASan see it on this stack?
//   device_far  one store 1 MiB past the buffer. ASan only flags accesses that
//               land in a tracked redzone, so this is expected to be MISSED or
//               to fault as a plain page fault -- the contrast case that shows
//               ASan is not a general bounds checker.
//   device_uaf  launch a kernel on a buffer after hipFree (use-after-free).
//   lds_oob     out-of-bounds store to __shared__ (LDS) memory. ROCm documents
//               LDS/scratch as not instrumented; this checks that on our stack.
//   read_oob    off-by-one LOAD past a hipMalloc'd buffer (the stores above
//               are all writes; a reader is how a stale-value bug looks).
//   pinned_oob  off-by-one store past a hipHostMalloc'd (pinned host) buffer,
//               the allocator data loaders use.
//   managed_oob off-by-one store past a hipMallocManaged buffer.
//
// n = 1000 ints = 4000 bytes, deliberately not a multiple of any allocation
// granularity, so an off-by-one lands in rounding slack unless the allocator's
// redzone tracks the requested size exactly.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(x)                                                                  \
  do {                                                                            \
    hipError_t e_ = (x);                                                          \
    if (e_ != hipSuccess) {                                                       \
      fprintf(stderr, "PROBE_HIP_ERROR %s -> %s (%s)\n", #x, hipGetErrorName(e_), \
              hipGetErrorString(e_));                                             \
      return 2;                                                                   \
    }                                                                             \
  } while (0)

__global__ void write_n(int* a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = i;
}

// The guard admits i == n: exactly one element past the allocation.
__global__ void write_n_plus_one(int* a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= n) a[i] = i;
}

__global__ void write_far(int* a, int n, int stride) {
  if (blockIdx.x == 0 && threadIdx.x == 0) a[n + stride] = 42;
}

// The guard admits i == n on the source: one element read past the allocation.
__global__ void read_n_plus_one(const int* a, int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= n) out[i % n] = a[i];
}

// Launched with 65 threads against a 64-entry __shared__ array: thread 64
// stores one past the end of LDS.
__global__ void lds_oob(int* out, int n) {
  __shared__ int s[64];
  int t = threadIdx.x;
  s[t] = t;
  __syncthreads();
  out[t % n] = s[t % 64];
}

static int host_oob() {
  int n = 1000;
  int* h = static_cast<int*>(malloc(n * sizeof(int)));
  volatile int* hv = h;
  hv[n] = 7;  // one int past the host allocation
  int r = hv[0];
  free(h);
  printf("PROBE_DONE mode=host_oob r=%d\n", r);
  return 0;
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "inbounds";

  // Host-only mode returns before any HIP call so it runs without a GPU.
  if (!strcmp(mode, "host_oob")) return host_oob();

  const int n = 1000;
  int* d = nullptr;
  CHECK(hipMalloc(&d, n * sizeof(int)));

  if (!strcmp(mode, "inbounds")) {
    hipLaunchKernelGGL(write_n, dim3((n + 255) / 256), dim3(256), 0, 0, d, n);
  } else if (!strcmp(mode, "device_oob")) {
    hipLaunchKernelGGL(write_n_plus_one, dim3((n + 256) / 256), dim3(256), 0, 0, d, n);
  } else if (!strcmp(mode, "device_far")) {
    hipLaunchKernelGGL(write_far, dim3(1), dim3(1), 0, 0, d, n, (1 << 20) / 4);
  } else if (!strcmp(mode, "device_uaf")) {
    CHECK(hipFree(d));
    hipLaunchKernelGGL(write_n, dim3((n + 255) / 256), dim3(256), 0, 0, d, n);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    printf("PROBE_DONE mode=%s\n", mode);
    return 0;
  } else if (!strcmp(mode, "lds_oob")) {
    hipLaunchKernelGGL(lds_oob, dim3(1), dim3(65), 0, 0, d, n);
  } else if (!strcmp(mode, "read_oob")) {
    int* out = nullptr;
    CHECK(hipMalloc(&out, n * sizeof(int)));
    hipLaunchKernelGGL(read_n_plus_one, dim3((n + 256) / 256), dim3(256), 0, 0, d, out, n);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    CHECK(hipFree(out));
  } else if (!strcmp(mode, "pinned_oob") || !strcmp(mode, "managed_oob")) {
    int* p = nullptr;
    if (mode[0] == 'p') {
      CHECK(hipHostMalloc(&p, n * sizeof(int), hipHostMallocDefault));
    } else {
      CHECK(hipMallocManaged(&p, n * sizeof(int)));
    }
    hipLaunchKernelGGL(write_n_plus_one, dim3((n + 256) / 256), dim3(256), 0, 0, p, n);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    CHECK(mode[0] == 'p' ? hipHostFree(p) : hipFree(p));
  } else {
    fprintf(stderr, "unknown mode %s\n", mode);
    return 64;
  }
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipFree(d));
  printf("PROBE_DONE mode=%s\n", mode);
  return 0;
}
