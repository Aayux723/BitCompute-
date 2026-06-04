#include <torch/extension.h>
#include <vector>
#include <cstdint>

// --- THE CPU WORKER LOOP ---
torch::Tensor ternary_matmul_cpu_forward(torch::Tensor inputs, torch::Tensor weights) {
    int M = inputs.size(0);
    int K = inputs.size(1);
    int N = weights.size(0);

    auto options = torch::TensorOptions()
        .dtype(inputs.dtype())
        .device(inputs.device());

    auto output = torch::zeros({M, N}, options);

    auto inputs_ptr = inputs.data_ptr<float>();
    auto weights_ptr = weights.data_ptr<int32_t>();
    auto output_ptr = output.data_ptr<float>();

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                int32_t w = weights_ptr[j * K + k];

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

    return output;
}

// --- CPU-ONLY VERSION ---
torch::Tensor ternary_matmul(torch::Tensor inputs, torch::Tensor weights) {
    TORCH_CHECK(inputs.dim() == 2, "Inputs must be a 2D matrix");
    TORCH_CHECK(weights.dim() == 2, "Weights must be a 2D matrix");
    TORCH_CHECK(inputs.size(1) == weights.size(1),
                "Inner matrix dimensions (K) must match");

    return ternary_matmul_cpu_forward(inputs, weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "ternary_matmul",
        &ternary_matmul,
        "Ternary Matrix Multiplication Engine"
    );
}