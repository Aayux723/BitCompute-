#include <torch/extension.h>
#include <cuda_runtime.h>

// Tiny CUDA kernel
__global__ void add_one_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        data[idx] += 1.0f;
    }
}

// Function called by Binding.cpp
torch::Tensor ternary_matmul_cuda_forward(
    torch::Tensor inputs,
    torch::Tensor weights
) {
    auto output = inputs.clone();

    int size = output.numel();

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    add_one_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        size
    );

    cudaDeviceSynchronize();

    return output;
}