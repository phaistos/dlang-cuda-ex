#include <cuda_runtime.h>
#include <stdio.h>

__global__
void cuincr(double *data, int sz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < sz) {
    data[i] += 1;
    i += gridDim.x * blockDim.x;
  }
}

extern "C" void incr(double* data, int sz) {
  cuincr<<<256, 256>>>(data, sz);
}
