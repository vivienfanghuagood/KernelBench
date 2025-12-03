"""
SiLU and Mul Activation (SwiGLU)

This kernel implements the SwiGLU activation function which computes:
output = SiLU(x[:d]) * x[d:]

where SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

This is a gated activation commonly used in transformer MLP layers.

Source: https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/elementwise/activation.cu
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    d: tl.constexpr,
    stride_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SiLU and Mul kernel: output = SiLU(input[..., :d]) * input[..., d:]
    where SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < d
    
    # Load gate (first half) and up (second half)
    gate_ptr = input_ptr + row_idx * stride_row + col_offsets
    up_ptr = input_ptr + row_idx * stride_row + d + col_offsets
    
    gate = tl.load(gate_ptr, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr, mask=mask, other=0.0).to(tl.float32)
    
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    
    # Multiply with up
    result = silu_gate * up
    
    # Store result
    output_ptr_row = output_ptr + row_idx * d + col_offsets
    tl.store(output_ptr_row, result.to(tl.float16), mask=mask)


class Model(nn.Module):
    """
    SiLU and Mul activation function for SwiGLU (Triton implementation).
    
    Computes output[i] = silu(input[i, :d]) * input[i, d:2*d]
    where d = input.size(-1) // 2
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (num_tokens, 2*d)
        
        Returns:
            Output tensor of shape (num_tokens, d)
        """
        d = x.shape[-1] // 2
        num_tokens = x.shape[0]
        
        output = torch.empty(num_tokens, d, dtype=x.dtype, device=x.device)
        
        # Calculate block size (next power of 2 >= d)
        BLOCK_SIZE = triton.next_power_of_2(d)
        
        # Launch kernel
        grid = (num_tokens,)
        silu_and_mul_kernel[grid](
            x, output, d, x.stride(0), BLOCK_SIZE
        )
        
        return output


def get_inputs():
    num_tokens = 2048
    d = 4096
    x = torch.randn(num_tokens, 2 * d, dtype=torch.float16, device='cuda')
    return [x]


def get_init_inputs():
    return []
