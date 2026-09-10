// Small single-kernel object WITH shared-memory (LDS) reads/writes and barriers,
// written race-free, so ConSan has real sites to analyze (contrast with
// tiny_vecadd, which has zero LDS/atomic/barrier sites -> coverage_incomplete).
#include <hip/hip_runtime.h>

extern "C" __global__ void lds_reduce(const float* in, float* out, int n) {
    __shared__ float s[256];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    s[t] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            s[t] += s[t + stride];
        }
        __syncthreads();
    }
    if (t == 0) {
        out[blockIdx.x] = s[0];
    }
}
