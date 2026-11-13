// Offline diagnostic-limit control; NEVER execute the compiled program.
// Adapted from ROCm/rocm-systems, revision d03c6fe410eed62b211d56bb7878dd1f494269fe,
// emulation/rocjitsu/docs/waitcheck/README.md, HIP C++ example.
// See LICENSE.upstream.txt in this directory.
// These intentional scalar load/use hazards are for static analysis only.
// Confirm the actual compiled entry and a high-cap result with more than one
// diagnostic before using it to investigate cap1 versus cap64 behavior.
#include <hip/hip_runtime.h>

extern "C" __global__ void waitcheck_multiple_missing() {
  asm volatile("s_load_dword s4, s[0:1], 0\n\t"
               "s_mov_b32 s5, s4\n\t"
               "s_wait_kmcnt 0\n\t"
               "s_load_dword s6, s[0:1], 4\n\t"
               "s_mov_b32 s7, s6\n\t"
               "s_wait_kmcnt 0\n\t"
               "s_load_dword s8, s[0:1], 8\n\t"
               "s_mov_b32 s9, s8\n\t"
               "s_wait_kmcnt 0"
               ::: "s4", "s5", "s6", "s7", "s8", "s9", "memory");
}

// Compilation-only retention stub. The invalid loads are not runtime test data.
// Give the resulting file to rj_waitcheck as DATA; never launch this executable.
int main() {
  waitcheck_multiple_missing<<<1, 1>>>();
  return hipDeviceSynchronize();
}
