import torch
import BitNet_engine

x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

w = torch.tensor([
    [1, 0],
    [-1, 1]
], dtype=torch.int32)

y = BitNet_engine.ternary_matmul(x, w)

print("Input:")
print(x)

print("Weights:")
print(w)

print("Output:")
print(y)