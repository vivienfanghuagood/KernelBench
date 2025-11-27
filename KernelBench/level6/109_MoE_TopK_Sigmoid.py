import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Original PyTorch implementation: MoE TopK Sigmoid routing
    Applies sigmoid activation to gating logits and selects top-k experts per token
    
    Input: gating_output [num_tokens, num_experts]
    Output: topk_weights [num_tokens, topk], topk_indices [num_tokens, topk]
    
    Based on sglang's topk_sigmoid operator from:
    sgl-kernel/csrc/moe/moe_topk_sigmoid_kernels.cu
    
    Unlike softmax routing, sigmoid allows multiple experts to be activated independently
    """
    def __init__(self, topk, renormalize=True):
        super(Model, self).__init__()
        self.topk = topk
        self.renormalize = renormalize

    def forward(self, gating_output):
        """
        Args:
            gating_output: [num_tokens, num_experts] router logits
        
        Returns:
            topk_weights: [num_tokens, topk] weights for top-k experts (after sigmoid)
            topk_indices: [num_tokens, topk] indices of top-k experts
        """
        # Apply sigmoid activation to each expert score independently
        routing_weights = torch.sigmoid(gating_output)
        
        # Select top-k experts based on sigmoid scores
        topk_weights, topk_indices = torch.topk(routing_weights, k=self.topk, dim=-1, sorted=False)
        
        # Optionally renormalize top-k weights
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        return topk_weights, topk_indices

num_tokens = 8192
num_experts = 64
topk = 2

def get_inputs():
    """Generate gating output tensor of shape [num_tokens, num_experts]"""
    return [torch.randn(num_tokens, num_experts, dtype=torch.float32)]

def get_init_inputs():
    return [topk, True]  # topk, renormalize
