"""
GELU Tanh Approximation and Mul Activation (GeGLU with Tanh)

This kernel implements the GeGLU activation function with tanh approximation:
output = GELU_tanh(x[:d]) * x[d:]

where GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

This is a faster approximation of GELU commonly used in transformer models.

Source: https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/elementwise/activation.cu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    GELU Tanh and Mul activation function for GeGLU.
    
    Computes output[i] = gelu_tanh(input[i, :d]) * input[i, d:2*d]
    where d = input.size(-1) // 2
    
    GELU with tanh approximation:
    gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
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
        return F.gelu(x[..., :d], approximate='tanh') * x[..., d:]


def get_inputs():
    num_tokens = 2048
    d = 4096
    x = torch.randn(num_tokens, 2 * d, dtype=torch.float16, device='cuda')
    return [x]


def get_init_inputs():
    return []
