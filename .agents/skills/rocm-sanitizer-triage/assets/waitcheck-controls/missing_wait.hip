// Offline Waitcheck development control; do not execute the compiled program.
// Adapted from ROCm/rocm-systems, revision d03c6fe410eed62b211d56bb7878dd1f494269fe,
// emulation/rocjitsu/docs/waitcheck/README.md, HIP C++ example.
// The scalar load deliberately lacks the wait before its result is consumed.
#include <hip/hip_runtime.h>

__global__ void waitcheck_control() {
  asm volatile("s_load_dword s4, s[0:1], 0\n\t"
               "s_mov_b32 s5, s4" ::: "s4", "s5", "memory");
}

// Retain the upstream launch structure for compilation only. The preparation
// runner passes this executable as DATA to rj_waitcheck; it never launches it.
int main() {
  waitcheck_control<<<1, 1>>>();
  return hipDeviceSynchronize();
}
