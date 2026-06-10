#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

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

// Function called by Binding.cpp
at::Tensor ternary_matmul_cuda_forward(
    at::Tensor inputs,
    at::Tensor weights
) {
    int M = static_cast<int>(inputs.size(0));
    int K = static_cast<int>(inputs.size(1));
    int N = static_cast<int>(weights.size(0));
    auto output = at::zeros({M, N}, inputs.options());

    dim3 threads(16, 16);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    ternary_matmul_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        inputs.data_ptr<float>(),
        weights.data_ptr<int8_t>(),
        output.data_ptr<float>(),
        M,
        K,
        N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
