import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Original PyTorch implementation: MoE TopK Softmax routing
    Computes softmax over gating logits and selects top-k experts per token
    
    Input: gating_output [num_tokens, num_experts]
    Output: topk_weights [num_tokens, topk], topk_indices [num_tokens, topk]
    
    Based on sglang's topk_softmax operator from:
    sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu
    """
    def __init__(self, topk, renormalize=True, moe_softcapping=0.0):
        super(Model, self).__init__()
        self.topk = topk
        self.renormalize = renormalize
        self.moe_softcapping = moe_softcapping

    def forward(self, gating_output):
        """
        Args:
            gating_output: [num_tokens, num_experts] router logits
        
        Returns:
            topk_weights: [num_tokens, topk] normalized weights for top-k experts
            topk_indices: [num_tokens, topk] indices of top-k experts
        """
        # Apply tanh softcapping if enabled
        if self.moe_softcapping > 0.0:
            gating_output = self.moe_softcapping * torch.tanh(gating_output / self.moe_softcapping)
        
        # Compute softmax over all experts
        routing_weights = F.softmax(gating_output, dim=-1, dtype=torch.float32)
        
        # Select top-k experts
        topk_weights, topk_indices = torch.topk(routing_weights, k=self.topk, dim=-1, sorted=False)
        
        # Renormalize top-k weights if requested
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
    return [topk, True, 0.0]  # topk, renormalize, moe_softcapping
