#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cstdint>
#include <limits>
#include <vector>

torch::Tensor ternary_matmul_cuda_forward(torch::Tensor inputs, torch::Tensor weights);

// --- THE MULTI-THREADED CPU WORKER LOOP ---
torch::Tensor ternary_matmul_cpu_forward(torch::Tensor inputs, torch::Tensor weights) {
    int64_t M = inputs.size(0);
    int64_t K = inputs.size(1);
    int64_t N = weights.size(0);

    auto options = torch::TensorOptions()
        .dtype(inputs.dtype())
        .device(inputs.device());

    auto output = torch::zeros({M, N}, options);

    auto inputs_ptr = inputs.data_ptr<float>();
    auto weights_ptr = weights.data_ptr<int8_t>();
    auto output_ptr = output.data_ptr<float>();

    at::parallel_for(0, M, 0, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;

                for (int64_t k = 0; k < K; ++k) {
                    int8_t w = weights_ptr[j * K + k];

                    if (w == 1) {
                        sum += inputs_ptr[i * K + k];
                    }
                    else if (w == -1) {
                        sum -= inputs_ptr[i * K + k];
                    }
                }

                output_ptr[i * N + j] = sum;
            }
        }
    });

    return output;
}

// --- CPU-ONLY VERSION ---
torch::Tensor ternary_matmul(torch::Tensor inputs, torch::Tensor weights) {
    TORCH_CHECK(inputs.dim() == 2, "Inputs must be a 2D matrix");
    TORCH_CHECK(weights.dim() == 2, "Weights must be a 2D matrix");
    TORCH_CHECK(inputs.size(1) == weights.size(1),
                "Inner matrix dimensions (K) must match");
    TORCH_CHECK(inputs.scalar_type() == torch::kFloat32,
                "Inputs must be float32");
    TORCH_CHECK(weights.scalar_type() == torch::kInt8,
                "Weights must be int8 ternary values encoded as -1, 0, or 1");
    TORCH_CHECK(inputs.device() == weights.device(),
                "Inputs and weights must be on the same device");
    TORCH_CHECK(inputs.numel() <= std::numeric_limits<int>::max(),
                "Input tensor is too large for the current CUDA kernel index type");
    TORCH_CHECK(weights.numel() <= std::numeric_limits<int>::max(),
                "Weight tensor is too large for the current CUDA kernel index type");
    TORCH_CHECK(inputs.size(0) <= std::numeric_limits<int>::max() &&
                inputs.size(1) <= std::numeric_limits<int>::max() &&
                weights.size(0) <= std::numeric_limits<int>::max(),
                "Matrix dimensions are too large for the current CUDA kernel index type");
    TORCH_CHECK(inputs.size(0) == 0 ||
                weights.size(0) <= std::numeric_limits<int>::max() / inputs.size(0),
                "Output tensor is too large for the current CUDA kernel index type");

    inputs = inputs.contiguous();
    weights = weights.contiguous();

    if (inputs.is_cuda()) {
        return ternary_matmul_cuda_forward(inputs, weights);
    }

    return ternary_matmul_cpu_forward(inputs, weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "ternary_matmul",
        &ternary_matmul,
        "Ternary Matrix Multiplication Engine"
    );
}
