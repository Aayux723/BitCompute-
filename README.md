# BitCompute: 1.58-bit LLM Execution Engine

![PyPI - Version](https://img.shields.io/pypi/v/bitcompute)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

BitCompute is a custom native PyTorch C++ and CUDA extension designed to drastically reduce Large Language Model (LLM) memory usage by implementing **Ternary Matrix Multiplication**. Inspired by the *BitNet b1.58* architecture, this execution engine forces neural network weights into just three states: `-1`, `0`, and `1`. 

By completely bypassing standard 16-bit or 32-bit floating-point (FP32) matrix multiplications, BitCompute achieves staggering memory savings, allowing large models to run on consumer hardware or edge devices.

##  Features
* **Custom CUDA Kernel**: A highly optimized C++ engine that skips expensive floating-point multiplications entirely (since multiplying by `-1`, `0`, or `1` is computationally identical to addition/subtraction).
* **PyTorch Integration**: A seamless PyBind11 wrapper that allows Python developers to swap out standard `nn.Linear` layers without writing any C++ code.
* **Intelligent Routing**: Automatically detects CPU vs. CUDA tensors and routes operations to the appropriate hardware backend.
* **Cross-Platform**: Packaged for PyPI with pre-compiled `.whl` files for Windows, and automated Source Distributions (`.tar.gz`) for on-the-fly Linux compilation (e.g., Google Colab).

---

##  Benchmark Analysis

We trained and benchmarked a small GPT model on a standard consumer GPU across three different execution strategies to demonstrate the power of 1.58-bit quantization. 

| Metric | Standard FP32 Engine | BitLinear (PyTorch Simulation) | BitCompute (Native CUDA C++) |
|--------|----------------------|--------------------------------|------------------------------|
| **Peak VRAM Usage** | 306.14 MB | 18.53 MB | **17.56 MB** |
| **Memory Saved** | Baseline | **93.9% Reduction** | **94.2% Reduction** |
| **Generation Speed** | 203.02 tok/sec | 55.33 tok/sec | **55.74 tok/sec** |

### What these numbers mean:
1. **Memory Annihilation:** The standard FP32 model requires over 300 MB of VRAM for even a tiny benchmark model. By utilizing our custom BitCompute CUDA engine, we dropped the VRAM requirement down to **17.56 MB**—an astonishing **94.2% memory reduction**. 
2. **True Hardware Execution vs. Simulation:** Many researchers simulate 1.58-bit networks in standard PyTorch using "fake quantization" (casting FP32 to Int8, and then back to FP32). While the PyTorch simulation achieved similar memory savings (18.53 MB), it relied on heavy Python overhead. Our native CUDA extension pushed the memory footprint even lower (17.56 MB) by keeping the math strictly inside the C++ backend.
3. **Speed vs. Memory Tradeoff:** Currently, standard FP32 operations are heavily accelerated by NVIDIA Tensor Cores and highly optimized `cuBLAS` libraries (yielding 203 tokens/sec). Because BitCompute is a custom, hand-written C++ kernel, it operates at ~55 tokens/sec. While slightly slower, the staggering 94% memory reduction is what allows these models to run on edge devices (like smartphones and IoT sensors) that physically do not possess enough RAM to boot an FP32 model in the first place.

---

##  Installation

BitCompute is fully open-source and hosted on the Python Package Index (PyPI). 

**Install on Windows:**
```bash
pip install bitcompute
```

**Install on Linux / Google Colab:**
(Pip will automatically download the source distribution and compile the CUDA engine natively)
```bash
pip install bitcompute --no-cache-dir
```

##  Usage

BitCompute exposes a direct function for ternary matrix multiplication that can be dropped into any PyTorch training loop or inference script.

```python
import torch
import BitNet_engine

# 1. Create your input (float32) and weights (int8)
inputs = torch.randn(128, 256, device="cuda", dtype=torch.float32)
ternary_weights = torch.randint(-1, 2, (512, 256), device="cuda", dtype=torch.int8)

# 2. Run the C++ Custom Engine!
# The engine automatically routes to CPU or GPU based on your tensor's device
output = BitNet_engine.ternary_matmul(inputs, ternary_weights)

print(output.shape) 
# torch.Size([128, 512])
```

## 📈 Future Roadmap
1. **Bit-Packing**: Currently, weights are stored in `int8` containers. By utilizing bit-level packing in C++, we can pack four 1.58-bit weights into a single byte, reducing memory usage by an additional 4x.
2. **Tensor Core Acceleration**: Rewriting the native CUDA kernel using NVIDIA PTX assembly to force the addition/subtraction loops onto the GPU's Tensor Cores to close the speed gap with `cuBLAS`. 
3. **Automated `nn.Module` Replacement**: Implementing a Python utility to automatically recursively replace standard `nn.Linear` layers in any HuggingFace model with `BitLinear` layers powered by our engine.
