// A host-only stand-in for <hip/hip_runtime.h>, for tests/reward/adapters/test_standin_harness.py: enough of the
// HIP API for the generated harness to compile with a host C++ compiler and run on the CPU. Device memory is host
// memory, and a launch only records its arguments, so a test can read back what the harness would pass a kernel.
#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>

#define __global__
#define __host__
#define __device__
#define __shared__
#define __launch_bounds__(...)

enum hipError_t { hipSuccess = 0, hipErrorOutOfMemory = 2 };
enum hipMemcpyKind { hipMemcpyHostToDevice = 1 };
struct dim3 {
  unsigned x, y, z;
  dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};

inline hipError_t hipMalloc(void** p, size_t n) {
  *p = std::malloc(n);
  return *p ? hipSuccess : hipErrorOutOfMemory;
}
inline hipError_t hipMemset(void* p, int v, size_t n) {
  std::memset(p, v, n);
  return hipSuccess;
}
inline hipError_t hipMemcpy(void* d, const void* s, size_t n, hipMemcpyKind) {
  std::memcpy(d, s, n);
  return hipSuccess;
}
inline hipError_t hipFree(void* p) {
  std::free(p);
  return hipSuccess;
}
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }
inline const char* hipGetErrorString(hipError_t) { return "fake"; }

// The last launch: the kernel, its block, and argv (valid only during the launch; a test's kernel-side hook reads it)
struct FakeLaunch {
  const void* kernel = nullptr;
  unsigned grid = 0, block = 0;
  size_t lds = 0;
  void** argv = nullptr;
};
inline FakeLaunch& fake_last_launch() {
  static FakeLaunch l;
  return l;
}
using fake_launch_hook = void (*)(const FakeLaunch&);
inline fake_launch_hook& fake_on_launch() {
  static fake_launch_hook h = nullptr;
  return h;
}
inline hipError_t hipLaunchKernel(const void* f, dim3 g, dim3 b, void** argv, size_t lds, void*) {
  FakeLaunch& l = fake_last_launch();
  l.kernel = f;
  l.grid = g.x;
  l.block = b.x;
  l.lds = lds;
  l.argv = argv;
  if (fake_on_launch()) fake_on_launch()(l);
  return hipSuccess;
}
