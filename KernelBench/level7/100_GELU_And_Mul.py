import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def gelu_and_mul_kernel(
    input_ptr,
    output_ptr,
    d: tl.constexpr,
    stride_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused GELU and Mul kernel: output = GELU(input[:, :d]) * input[:, d:]
    Using exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < d
    
    # Load gate (first half) and up (second half)
    gate_ptr = input_ptr + row_idx * stride_row + col_offsets
    up_ptr = input_ptr + row_idx * stride_row + d + col_offsets
    
    gate = tl.load(gate_ptr, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr, mask=mask, other=0.0).to(tl.float32)
    
    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt_2_inv = 0.7071067811865476  # 1 / sqrt(2)
    gelu_gate = gate * 0.5 * (1.0 + tl.math.erf(gate * sqrt_2_inv))
    
    # Multiply with up
    result = gelu_gate * up
    
    # Store result
    output_ptr_row = output_ptr + row_idx * d + col_offsets
    tl.store(output_ptr_row, result.to(tl.float16), mask=mask)


class Model(nn.Module):
    """
    Triton implementation: GELU(input[:, :d]) * input[:, d:]
    Input shape: [batch_size, 2 * out_features]
    Output shape: [batch_size, out_features]
    """
    def __init__(self, out_features):
        super(Model, self).__init__()
        self.out_features = out_features

    def forward(self, x):
        # x: [batch_size, 2 * out_features]
        batch_size = x.shape[0]
        d = self.out_features
        
        output = torch.empty(batch_size, d, dtype=x.dtype, device=x.device)
        
        # Calculate block size (next power of 2 >= d)
        BLOCK_SIZE = triton.next_power_of_2(d)
        
        # Launch kernel
        grid = (batch_size,)
        gelu_and_mul_kernel[grid](
            x, output, d, x.stride(0), BLOCK_SIZE
        )
        
        return output


batch_size = 64 * 1024
out_features = 8192


def get_inputs():
    """Generate input tensor of shape [batch_size, 2 * out_features]"""
    return [torch.rand(batch_size, 2 * out_features, dtype=torch.float16, device='cuda')]


def get_init_inputs():
    return [out_features]
