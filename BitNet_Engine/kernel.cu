// Pure CUDA kernel — NO C++ standard library headers to avoid cudafe++ crash
// on CUDA 12.8 + MSVC 14.44.
#include <cuda_runtime.h>
#include <stdint.h>   // C header, not <cstdint> — cudafe++ handles this fine

__global__ void ternary_matmul_kernel(
    const float* __restrict__ inputs,
    const int8_t* __restrict__ weights,
    float* __restrict__ output,
    int M,
    int K,
    int N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            int8_t w = weights[col * K + k];

            if (w == 1) {
                sum += inputs[row * K + k];
            }
            else if (w == -1) {
                sum -= inputs[row * K + k];
            }
        }

        output[row * N + col] = sum;
    }
}

// Plain C-linkage launcher — Binding.cpp calls this with raw pointers
extern "C" void launch_ternary_matmul(
    const float* inputs,
    const int8_t* weights,
    float* output,
    int M, int K, int N,
    cudaStream_t stream
) {
    dim3 threads(16, 16);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    ternary_matmul_kernel<<<blocks, threads, 0, stream>>>(
        inputs, weights, output, M, K, N
    );
}
