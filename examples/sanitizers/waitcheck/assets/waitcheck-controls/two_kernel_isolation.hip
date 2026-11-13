// Offline Waitcheck isolation control; do not execute the compiled program.
// Adapted from ROCm/rocm-systems, revision d03c6fe410eed62b211d56bb7878dd1f494269fe,
// emulation/rocjitsu/docs/waitcheck/README.md, HIP C++ example.
// See LICENSE.upstream.txt in this directory.
// Build both kernels into one gfx1250 device image, then independently inspect
// its actual image index and each descriptor's .text entry offset before scanning.
#include <hip/hip_runtime.h>

extern "C" __global__ void waitcheck_missing() {
  asm volatile("s_load_dword s4, s[0:1], 0\n\t"
               "s_mov_b32 s5, s4" ::: "s4", "s5", "memory");
}

extern "C" __global__ void waitcheck_corrected() {
  asm volatile("s_load_dword s4, s[0:1], 0\n\t"
               "s_wait_kmcnt 0\n\t"
               "s_mov_b32 s5, s4" ::: "s4", "s5", "memory");
}

// Compilation-only host stub: retain launch references to both kernels.
// Pass the resulting file as DATA to the offline scanner; NEVER launch it.
// The intentionally malformed load is not a valid runtime memory-access test.
int main() {
  waitcheck_missing<<<1, 1>>>();
  waitcheck_corrected<<<1, 1>>>();
  return hipDeviceSynchronize();
}
