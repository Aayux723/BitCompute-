import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ENGINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENGINE_DIR.parent
DATA_PATH = PROJECT_ROOT / "Dataset" / "shakespeare.txt"

if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import BitNet_engine


class WeightQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, eps=1e-5):
        mean = weight.mean()
        weight_centered = weight - mean
        gamma = weight_centered.abs().mean()
        weight_scaled = weight_centered / (gamma + eps)
        weight_clipped = torch.clamp(weight_scaled, min=-1.0, max=1.0)
        return torch.round(weight_clipped)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class ActivationQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps=1e-5):
        qmax = 127.0
        abs_max = x.abs().max()
        scale = qmax / (abs_max + eps)
        x_scaled = x * scale
        x_quantized = torch.round(torch.clamp(x_scaled, -qmax, qmax))
        return x_quantized / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class EngineTernaryLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_2d, weight, bias, eps):
        quantized_x = ActivationQuantizeSTE.apply(x_2d, eps).contiguous()
        quantized_w = WeightQuantizeSTE.apply(weight, eps).contiguous()
        weight_int8 = quantized_w.to(torch.int8).contiguous()

        out = BitNet_engine.ternary_matmul(quantized_x, weight_int8)
        if bias is not None:
            out = out + bias

        ctx.save_for_backward(quantized_x, quantized_w)
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_output):
        quantized_x, quantized_w = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        grad_x = grad_output.matmul(quantized_w)
        grad_weight = grad_output.transpose(0, 1).matmul(quantized_x)
        grad_bias = grad_output.sum(0) if ctx.has_bias else None
        return grad_x, grad_weight, grad_bias, None


class EngineBitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, eps=1e-5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * (1.0 / in_features**0.5)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        original_shape = x.shape[:-1]
        x_2d = x.reshape(-1, self.in_features).contiguous()
        out_2d = EngineTernaryLinearFn.apply(x_2d, self.weight, self.bias, self.eps)
        return out_2d.reshape(*original_shape, self.out_features)


class CharDataset:
    def __init__(self, block_size, batch_size, device):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        text = DATA_PATH.read_text(encoding="utf-8")
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def decode(self, idxs):
        return "".join(self.itos[int(i)] for i in idxs)

    def get_batch(self, split):
        data_source = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data_source) - self.block_size, (self.batch_size,))
        x = torch.stack([data_source[i : i + self.block_size] for i in ix])
        y = torch.stack([data_source[i + 1 : i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = EngineBitLinear(n_embd, head_size, bias=False)
        self.query = EngineBitLinear(n_embd, head_size, bias=False)
        self.value = EngineBitLinear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        _, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(n_head)]
        )
        self.proj = EngineBitLinear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            EngineBitLinear(n_embd, 4 * n_embd),
            nn.ReLU(),
            EngineBitLinear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class EngineBitGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, device):
        super().__init__()
        self.block_size = block_size
        self.device_name = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = EngineBitLinear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device_name)
        )
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model, dataset, eval_iters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = dataset.get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def benchmark_generation(model, device, tokens_to_generate):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _ = model.generate(context, max_new_tokens=10)
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device == "cuda"
        else 0.0
    )

    for _ in range(2):
        _ = model.generate(context, max_new_tokens=20)
    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    generated = model.generate(context, max_new_tokens=tokens_to_generate)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    return {
        "generation_time_seconds": round(elapsed, 4),
        "tokens_per_second": round(tokens_to_generate / elapsed, 2),
        "peak_vram_mb": round(peak_memory_mb, 2),
        "sample_token_ids": generated[0].detach().cpu().tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a character transformer with EngineBitLinear layers"
    )
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=300)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-embd", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--tokens-to-generate", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint", default="tiny_gpt_engine_bitnet.pth")
    parser.add_argument("--metrics", default="engine_bitnet_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    dataset = CharDataset(args.block_size, args.batch_size, device)
    model = EngineBitGPT(
        dataset.vocab_size,
        args.block_size,
        args.n_embd,
        args.n_head,
        args.n_layer,
        device,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(f"--- Training EngineBitGPT on {device} ---")
    print(f"custom engine: {BitNet_engine.__file__}")
    print(f"vocab_size={dataset.vocab_size}, parameters={sum(p.numel() for p in model.parameters())}")

    train_start = time.perf_counter()
    final_losses = None
    for step in range(args.max_iters):
        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            final_losses = estimate_loss(model, dataset, args.eval_iters)
            print(
                f"step {step}: train loss {final_losses['train']:.4f}, "
                f"val loss {final_losses['val']:.4f}"
            )

        xb, yb = dataset.get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    train_seconds = time.perf_counter() - train_start

    checkpoint_path = ENGINE_DIR / args.checkpoint
    metrics_path = ENGINE_DIR / args.metrics
    torch.save(model.state_dict(), checkpoint_path)

    model.eval()
    generation = benchmark_generation(model, device, args.tokens_to_generate)
    sample_text = dataset.decode(generation.pop("sample_token_ids"))

    metrics = {
        "model_type": "EngineBitGPT_Custom_CUDA_Ternary",
        "device": device,
        "custom_engine_path": str(Path(BitNet_engine.__file__).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "train_seconds": round(train_seconds, 2),
        "final_train_loss": round(final_losses["train"], 4) if final_losses else None,
        "final_val_loss": round(final_losses["val"], 4) if final_losses else None,
        **generation,
        "sample_text": sample_text,
    }

    metrics_path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    print(f"--- Saved checkpoint: {checkpoint_path} ---")
    print(f"--- Saved metrics: {metrics_path} ---")
    print("\n=== ENGINE BITNET METRICS ===")
    for key, value in metrics.items():
        if key != "sample_text":
            print(f"{key}: {value}")
    print("\n=== SAMPLE ===")
    print(sample_text)


if __name__ == "__main__":
    main()
