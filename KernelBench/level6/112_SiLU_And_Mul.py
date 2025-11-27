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
import torch.nn.functional as F


class Model(nn.Module):
    """
    SiLU and Mul activation function for SwiGLU.
    
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
        return F.silu(x[..., :d]) * x[..., d:]


def get_inputs():
    num_tokens = 2048
    d = 4096
    x = torch.randn(num_tokens, 2 * d, dtype=torch.float16, device='cuda')
    return [x]


def get_init_inputs():
    return []
