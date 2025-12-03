import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Matrix multiplication kernel: C = A @ B
    A: [M, K], B: [K, N], C: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers to first block of A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load with masking
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
        
        # Compute matmul block
        acc += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


@triton.jit
def gelu_mul_kernel(
    gate_ptr, up_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused GELU and Mul kernel: output = GELU(gate) * up
    Using exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load gate and up
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt_2_inv = 0.7071067811865476  # 1 / sqrt(2)
    gelu_gate = gate * 0.5 * (1.0 + tl.math.erf(gate * sqrt_2_inv))
    
    # Multiply with up
    result = gelu_gate * up
    
    # Store result
    tl.store(output_ptr + offsets, result.to(tl.float16), mask=mask)


def triton_matmul(a, b):
    """
    Triton matrix multiplication: C = A @ B
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Incompatible dimensions"
    
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    return c


class Model(nn.Module):
    """
    Full Triton implementation using matmul, GELU, and multiplication.
    Input shape: [batch_size, in_features]
    Output shape: [batch_size, out_features]
    
    This performs: output = GELU(x @ W1) * (x @ W2)
    where W1, W2 are [in_features, out_features]
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        # Store weights as Parameters (transposed for efficient matmul)
        self.weight1 = nn.Parameter(torch.empty(in_features, out_features, dtype=torch.float16))
        self.weight2 = nn.Parameter(torch.empty(in_features, out_features, dtype=torch.float16))
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))

    def forward(self, x):
        # x: [batch_size, in_features]
        # Compute gate = x @ W1 using Triton matmul
        gate = triton_matmul(x, self.weight1)  # [batch_size, out_features]
        # Compute up = x @ W2 using Triton matmul
        up = triton_matmul(x, self.weight2)    # [batch_size, out_features]
        
        # Fused GELU and multiplication using Triton
        n_elements = gate.numel()
        output = torch.empty_like(gate)
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        gelu_mul_kernel[grid](
            gate, up, output, n_elements, BLOCK_SIZE
        )
        
        return output


batch_size = 1024
in_features = 4096
out_features = 4096


def get_inputs():
    return [torch.rand(batch_size, in_features, dtype=torch.float16, device='cuda')]


def get_init_inputs():
    return [in_features, out_features]
