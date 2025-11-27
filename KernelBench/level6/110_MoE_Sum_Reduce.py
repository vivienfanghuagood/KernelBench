import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Original PyTorch implementation: MoE Sum Reduction
    Reduces expert outputs by summing across the topk dimension with optional scaling
    
    Input: input [num_tokens, topk, hidden_size] - outputs from topk experts per token
    Output: output [num_tokens, hidden_size] - summed expert outputs
    
    Based on sglang's moe_sum operator from:
    sgl-kernel/csrc/moe/moe_sum.cu
    
    Applies routed_scaling_factor to the summed output if provided
    """
    def __init__(self, routed_scaling_factor=1.0):
        super(Model, self).__init__()
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: [num_tokens, topk, hidden_size] expert outputs
        
        Returns:
            output: [num_tokens, hidden_size] summed and scaled outputs
        """
        # Sum across topk dimension
        output = input_tensor.sum(dim=1)
        
        # Apply scaling factor if not 1.0
        if self.routed_scaling_factor != 1.0:
            output = output * self.routed_scaling_factor
        
        return output

num_tokens = 8192
topk = 2
hidden_size = 4096

def get_inputs():
    """Generate input tensor of shape [num_tokens, topk, hidden_size]"""
    return [torch.randn(num_tokens, topk, hidden_size, dtype=torch.float16)]

def get_init_inputs():
    return [1.0]  # routed_scaling_factor
