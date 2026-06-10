import argparse
import importlib
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
ENGINE_DIR = PROJECT_ROOT / "BitNet_Engine"

if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))


def load_engine():
    try:
        return importlib.import_module("BitNet_engine")
    except ImportError as exc:
        raise SystemExit(
            "Could not import BitNet_engine. Build it first with:\n"
            '  cd "C:\\Users\\Aayush Anand\\Desktop\\Bit-1\\BitNet_Engine"\n'
            '  python setup.py build_ext --inplace\n'
        ) from exc


def format_bytes(num_bytes):
    units = ["B", "KB", "MB", "GB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0


def tensor_bytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_op(fn, iterations, device, warmup):
    for _ in range(warmup):
        fn()
    sync_if_cuda(device)

    start = time.perf_counter()
    for _ in range(iterations):
        result = fn()
    sync_if_cuda(device)
    elapsed = time.perf_counter() - start
    return elapsed / iterations, result


def make_table(headers, rows):
    widths = [
        max(len(str(header)), *(len(str(row[index])) for row in rows))
        for index, header in enumerate(headers)
    ]
    sep = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [sep]
    lines.append("| " + " | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)) + " |")
    lines.append(sep)
    for row in rows:
        lines.append("| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |")
    lines.append(sep)
    return "\n".join(lines)


def run_benchmark(m, n, k, iterations, warmup, seed):
    torch.manual_seed(seed)
    engine = load_engine()

    inputs_cpu = torch.randn(m, k, dtype=torch.float32).contiguous()
    weights_int8_cpu = torch.randint(-1, 2, (n, k), dtype=torch.int8).contiguous()
    weights_fp32_cpu = weights_int8_cpu.float().contiguous()

    rows = []

    fp32_cpu_time, fp32_cpu_out = time_op(
        lambda: inputs_cpu @ weights_fp32_cpu.t(),
        iterations,
        torch.device("cpu"),
        warmup,
    )
    rows.append([
        "PyTorch FP32 CPU",
        f"{fp32_cpu_time * 1000:.4f} ms",
        "1.00x",
        format_bytes(tensor_bytes(inputs_cpu, weights_fp32_cpu, fp32_cpu_out)),
        "baseline",
    ])

    custom_cpu_time, custom_cpu_out = time_op(
        lambda: engine.ternary_matmul(inputs_cpu, weights_int8_cpu),
        iterations,
        torch.device("cpu"),
        warmup,
    )
    custom_cpu_diff = (fp32_cpu_out - custom_cpu_out).abs().max().item()
    rows.append([
        "Custom INT8 CPU",
        f"{custom_cpu_time * 1000:.4f} ms",
        f"{fp32_cpu_time / custom_cpu_time:.2f}x",
        format_bytes(tensor_bytes(inputs_cpu, weights_int8_cpu, custom_cpu_out)),
        f"max diff {custom_cpu_diff:.6g}",
    ])

    if torch.cuda.is_available():
        device = torch.device("cuda")
        inputs_gpu = inputs_cpu.to(device)
        weights_int8_gpu = weights_int8_cpu.to(device)
        weights_fp32_gpu = weights_fp32_cpu.to(device)

        torch.cuda.reset_peak_memory_stats(device)
        fp32_gpu_time, fp32_gpu_out = time_op(
            lambda: inputs_gpu @ weights_fp32_gpu.t(),
            iterations,
            device,
            warmup,
        )
        fp32_gpu_peak = torch.cuda.max_memory_allocated(device)
        rows.append([
            "PyTorch FP32 CUDA",
            f"{fp32_gpu_time * 1000:.4f} ms",
            f"{fp32_cpu_time / fp32_gpu_time:.2f}x",
            format_bytes(fp32_gpu_peak),
            f"max diff {(fp32_cpu_out - fp32_gpu_out.cpu()).abs().max().item():.6g}",
        ])

        torch.cuda.reset_peak_memory_stats(device)
        custom_cuda_time, custom_cuda_out = time_op(
            lambda: engine.ternary_matmul(inputs_gpu, weights_int8_gpu),
            iterations,
            device,
            warmup,
        )
        custom_cuda_peak = torch.cuda.max_memory_allocated(device)
        custom_cuda_diff = (custom_cpu_out - custom_cuda_out.cpu()).abs().max().item()
        rows.append([
            "Custom INT8 CUDA",
            f"{custom_cuda_time * 1000:.4f} ms",
            f"{fp32_cpu_time / custom_cuda_time:.2f}x",
            format_bytes(custom_cuda_peak),
            f"max diff {custom_cuda_diff:.6g}",
        ])

    print(f"BitNet operator benchmark: M={m}, N={n}, K={k}, iters={iterations}, warmup={warmup}")
    print(make_table(
        ["Backend", "Latency", "Speed vs CPU FP32", "Memory / Peak VRAM", "Correctness"],
        rows,
    ))


def main():
    parser = argparse.ArgumentParser(description="BitNet custom execution engine benchmark")
    parser.add_argument("--m", type=int, default=128, help="Input rows, often batch * sequence")
    parser.add_argument("--n", type=int, default=512, help="Output features")
    parser.add_argument("--k", type=int, default=512, help="Input features")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    run_benchmark(args.m, args.n, args.k, args.iters, args.warmup, args.seed)


if __name__ == "__main__":
    main()
