# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

block_shape = (128, 128)

class Model(nn.Module):
    """Reference implementation using PyTorch operations."""
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor, weight: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
        """
        Args:
            x: [m, k] int8 input
            weight: [n, k] int8 weight
            x_scale: [m, scale_k] fp32 activation scales
            w_scale: [scale_n, scale_k] fp32 weight scales
        Returns:
            output: [m, n] in specified dtype
        """
        block_shape_n, block_shape_k = block_shape
        m, k = x.shape
        n = weight.shape[0]
        scale_n = (n + block_shape_n - 1) // block_shape_n
        scale_k = (k + block_shape_k - 1) // block_shape_k
        
        # Apply block-wise scaling to input
        x = x.to(x_scale.dtype).view(
            m, k // block_shape[1], block_shape[1]
        ) * x_scale.unsqueeze(-1)
        x = x.view(m, k)

        # Apply block-wise scaling to weight
        w_scale = rearrange(
            w_scale.view(-1, 1)
            .repeat(1, block_shape_n * block_shape_k)
            .view(scale_n, scale_k, block_shape_n, block_shape_k),
            "num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)",
        )
        w_scale = w_scale[:n, :k]
        weight = weight.to(w_scale.dtype) * w_scale

        out = F.linear(x.to(torch.float32), weight.to(torch.float32))
        return out.to(dtype)


m = 1024
n = 4096
k = 4096
block_shape_n, block_shape_k = block_shape
scale_n = (n + block_shape_n - 1) // block_shape_n
scale_k = (k + block_shape_k - 1) // block_shape_k

# Test configuration
m = 1024
n = 4096
k = 4096
block_shape_n, block_shape_k = block_shape
scale_n = (n + block_shape_n - 1) // block_shape_n
scale_k = (k + block_shape_k - 1) // block_shape_k

def get_inputs():
    # Support both fp8 formats
    fp8_dtype = getattr(torch, 'float8_e4m3fnuz', torch.float8_e4m3fn)
    x = (torch.rand((m, k), dtype=torch.float16, device="cuda") / 10).to(fp8_dtype)
    weight = (torch.rand((n, k), dtype=torch.float16, device="cuda") / 10).to(fp8_dtype)
    x_scale = torch.rand([m, scale_k], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")
    return [x, weight, x_scale, w_scale]

def get_init_inputs():
    return []
